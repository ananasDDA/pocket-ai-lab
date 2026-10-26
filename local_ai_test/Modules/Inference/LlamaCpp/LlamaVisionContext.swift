//
//  LlamaVisionContext.swift
//  local_ai_test
//
//  Actor that owns a llama.cpp text context *plus* a libmtmd multimodal
//  projector. Mirrors the internal `LlamaContext` actor from
//  `LlamaCppTextEngine.swift`, but replaces the text-only prompt-decode
//  step with `mtmd_tokenize` and per-chunk evaluation, which handles both
//  text and image chunks (and the per-architecture specifics like
//  non-causal attention for Gemma 3 or M-RoPE for Qwen2-VL).
//
//  Public API is deliberately small:
//    • `create(modelPath:mmprojPath:)` — one-shot async loader.
//    • `generate(prompt:images:parameters:onPiece:)` — streams tokens.
//    • `reset()` / `cleanup()` — KV cache + resource management.
//
//  The prompt passed to `generate` must already contain one `<__media__>`
//  marker per image (the callers, i.e. `LlamaCppVisionEngine`, splice
//  them in). libmtmd replaces each marker with a sequence of image tokens
//  that share the chunk-level positional embedding.
//
//  KV prefix reuse: the caller re-sends the whole conversation every turn.
//  The actor remembers what its KV cache holds as a chunk list
//  (`KVChunk`: text token runs and images identified by a content hash),
//  cuts the cache back to the longest prefix the new prompt shares with it
//  and evaluates only the rest, so a follow-up turn costs its own tokens
//  rather than the whole transcript. See `VisionKVCache.swift`.
//
//  Reference: tools/mtmd/mtmd-cli.cpp and tools/server in llama.cpp.
//

import Foundation

#if canImport(llama)

import llama

/// Owns llama_model + llama_context + mtmd_context, plus a sampler chain.
/// Serialised via Swift's actor model — every call into the C API is
/// guaranteed to happen on the actor's executor, matching libmtmd's
/// "not thread-safe" contract for the eval helpers.
actor LlamaVisionContext {

    // MARK: - Stored state

    private var model: OpaquePointer?
    private var lctx: OpaquePointer?
    private var mctx: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    /// Next KV position: everything below it is in the cache.
    private var nPast: llama_pos = 0

    /// What the KV cache holds for sequence 0, in order. Always
    /// `KVPrefix.totalPositions(cached) == nPast` between calls.
    private var cached: [KVChunk] = []

    /// How the cache can be cut back. Attention caches forget any suffix;
    /// a recurrent state (Mamba-style layers, including the hybrid Qwen 3.5)
    /// only exists for its last position, so it is snapshotted near the end
    /// of every prompt and restored from there instead.
    private enum Rollback { case partial, checkpoints }
    private let rollback: Rollback

    /// Recurrent-state snapshots: the state after `position` positions.
    private struct Checkpoint {
        let position: Int
        let data: [UInt8]
    }
    private var checkpoints: [Checkpoint] = []
    /// Two snapshots: this turn's and the previous one, which an edited or
    /// regenerated last message can still rewind to. A snapshot is a few
    /// tens of megabytes on the 2B-4B hybrids.
    private static let maxCheckpoints = 2
    private static let maxCheckpointBytes = 96 * 1024 * 1024
    private static let checkpointOffsetFromEnd = 4

    /// Sliding-window size, 0 when the model has none. Used to refuse a
    /// rewind behind what a sliding-window cache still remembers.
    private let nSwa: Int32
    /// M-RoPE models place an image on fewer positions than it has
    /// tokens; KV positions are then not consecutive across an image.
    private let usesMRoPE: Bool

    /// Context window requested from llama.cpp. Vision tokens are heavy
    /// (Gemma 3 maps one image to 256 tokens), so the callers budget the
    /// conversation against this number before it reaches the C API.
    static let contextTokens: UInt32 = 8192
    /// Batch size of the context; text is decoded in slices of this size.
    private static let batchSize: Int32 = 512

    // MARK: - Lifecycle

    /// Load both the text GGUF and the mmproj projector.
    ///
    /// - Parameters:
    ///   - modelPath: absolute path to the main `.gguf`.
    ///   - mmprojPath: absolute path to the companion `mmproj-*.gguf`.
    static func create(modelPath: String, mmprojPath: String) async throws -> LlamaVisionContext {
        _ = LlamaRuntime.bootstrap

        // --- Text model ------------------------------------------------------
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #else
        modelParams.n_gpu_layers = 999
        #endif

        let logMark = LlamaLogCapture.shared.mark()
        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw LlamaRuntime.loadError(since: logMark, stage: .model)
        }

        // --- Text context ----------------------------------------------------
        // Vision models produce many more tokens per turn than text-only
        // (an image can expand to ~256–1024 tokens), so we bump the context
        // window. 8K is a conservative default that fits on a modern iPhone.
        var ctxParams = llama_context_default_params()
        // Vision tokens are HEAVY: Gemma 3 maps every image to 256 tokens,
        // and an 8-frame video = 2048 tokens by itself. We need substantial
        // headroom to fit a few turns of multimodal chat without thrashing.
        // 8192 is the practical sweet spot on iPhone (≈500 MB Metal compute
        // buffer). For audio-heavy workloads bump to 16384 if RAM allows.
        ctxParams.n_ctx = contextTokens
        // n_batch must be ≥ the per-image token count (256 for Gemma 3),
        // otherwise image batches get split and can hit "no KV slot"
        // (rc=1) corner cases. 512 is the upstream default.
        ctxParams.n_batch = UInt32(batchSize)
        ctxParams.n_threads = Int32(max(1, ProcessInfo.processInfo.processorCount - 1))
        ctxParams.n_threads_batch = ctxParams.n_threads
        // Full-size sliding-window cache (llama.cpp's default, spelled out
        // because prefix reuse depends on it): a trimmed SWA cache forgets
        // old positions, and cutting back behind them would need a
        // snapshot as well.
        ctxParams.swa_full = true

        guard let lctx = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw LlamaRuntime.loadError(since: logMark, stage: .context)
        }

        // --- Multimodal projector -------------------------------------------
        // IMPORTANT: always start from `mtmd_context_params_default()` —
        // upstream adds fields across releases (e.g. `flash_attn_type`,
        // `image_min_tokens`) and we only want to override the handful we
        // actually care about. Zero-init would corrupt future builds.
        var mparams = mtmd_context_params_default()
        #if targetEnvironment(simulator)
        mparams.use_gpu = false
        #else
        mparams.use_gpu = true
        #endif
        mparams.print_timings = false
        mparams.n_threads = Int32(max(1, ProcessInfo.processInfo.processorCount - 1))

        guard let mctx = mtmd_init_from_file(mmprojPath, model, mparams) else {
            llama_free(lctx)
            llama_model_free(model)
            throw InferenceError.generationFailed(
                "llama.cpp: failed to init mtmd projector from \(mmprojPath)"
            )
        }

        return LlamaVisionContext(model: model, lctx: lctx, mctx: mctx)
    }

    private init(model: OpaquePointer, lctx: OpaquePointer, mctx: OpaquePointer) {
        self.model = model
        self.lctx = lctx
        self.mctx = mctx
        self.rollback = (llama_model_is_recurrent(model) || llama_model_is_hybrid(model))
            ? .checkpoints : .partial
        self.nSwa = max(0, llama_model_n_swa(model))
        self.usesMRoPE = mtmd_decode_use_mrope(mctx)
    }

    /// Whether `generate()` is currently running on this actor.
    /// Used to defer `cleanup()` until the C pipeline is fully unwound —
    /// prevents Metal command buffers from being freed while still in flight.
    private var generating = false
    private var cleanupPending = false

    /// The chat template stored in the GGUF, if it has one.
    func embeddedChatTemplate() -> String? {
        guard let model else { return nil }
        return LlamaChatTemplate.embedded(in: model)
    }

    /// Wipes the KV cache and resets `nPast`. The model weights stay loaded.
    func reset() {
        forgetSequence()
    }

    /// Drops sequence 0 from the KV cache, the record of what it held and
    /// every snapshot.
    ///
    /// Two details matter here. `llama_synchronize` first: the previous
    /// turn's Metal command buffers may still be in flight, and mutating
    /// cache state under them cost us a GPU fault ("victim of GPU
    /// error/recovery") that took out the next image encode. And
    /// `llama_memory_seq_rm` over the whole sequence rather than
    /// `llama_memory_clear(mem, true)`: removing a whole sequence never
    /// fails and only touches metadata, where clearing with `data: true`
    /// zeroes about a gigabyte of KV buffers nothing will ever read.
    private func forgetSequence() {
        if let lctx {
            llama_synchronize(lctx)
            if let mem = llama_get_memory(lctx) {
                llama_memory_seq_rm(mem, 0, -1, -1)
            }
        }
        nPast = 0
        cached = []
        checkpoints = []
    }

    /// Frees every C resource. Safe to call multiple times.
    /// If called while `generate()` is running on this actor, the actual
    /// deallocation is deferred until generation exits — this prevents the
    /// Metal backend from accessing buffers that llama_free() has already
    /// returned to the allocator while a GPU command buffer is still live.
    func cleanup() {
        guard !generating else {
            cleanupPending = true
            return
        }
        performCleanup()
    }

    private func performCleanup() {
        // `llama_synchronize` flushes any pending Metal command buffers and
        // waits for the GPU to idle before we hand memory back to the OS.
        // Without this, EXC_BAD_ACCESS can occur in ggml_metal_buffer_is_shared
        // when the GPU accesses a buffer whose Swift/C owner was already freed.
        if let lctx { llama_synchronize(lctx) }
        if let sampler { llama_sampler_free(sampler) }
        if let mctx { mtmd_free(mctx) }
        if let lctx { llama_free(lctx) }
        if let model { llama_model_free(model) }
        sampler = nil
        mctx = nil
        lctx = nil
        model = nil
        cached = []
        checkpoints = []
        nPast = 0
        cleanupPending = false
    }

    // MARK: - Generation

    /// Tokenize the prompt + images via libmtmd, evaluate what the KV cache
    /// does not already hold, then run a standard llama.cpp sampling loop
    /// streaming pieces to `onPiece`. Calls `onPiece(text, tokenIndex)` per
    /// UTF-8-complete chunk and returns the number of tokens generated with
    /// this call's prompt figures. (They are returned rather than kept on
    /// the actor: a caller reading actor state after its `await` could see
    /// the figures of a generation that ran in between.)
    ///
    /// The `prompt` string **must** contain exactly as many `<__media__>`
    /// markers as `images.count`, or mtmd returns code 1 and we throw.
    ///
    /// The body must stay free of suspension points: the whole rewind -
    /// evaluate - sample sequence has to run as one actor turn, or a second
    /// call interleaves with it on the same KV cache. Everything it calls is
    /// either C or `nonisolated` Swift for that reason (see
    /// `VisionKVCache.swift`), and the guard below turns any future
    /// regression into an error instead of a corrupted cache.
    func generate(
        prompt: String,
        images: [Data],
        parameters: GenerationParameters,
        onPiece: @Sendable (String, Int) -> Void
    ) async throws -> LlamaVisionGeneration {
        guard let model, let lctx, let mctx else { throw InferenceError.modelNotLoaded }
        guard !generating else {
            throw InferenceError.generationFailed(
                "llama.cpp: another reply is still being generated with this model."
            )
        }
        generating = true
        defer {
            generating = false
            if cleanupPending { performCleanup() }
        }

        let vocab = llama_model_get_vocab(model)

        // --- Rebuild sampler chain ------------------------------------------
        // Free any chain left over from a previous turn — the actor owns it,
        // there are no dangling references.
        if let existing = sampler { llama_sampler_free(existing) }
        sampler = nil
        var sparams = llama_sampler_chain_default_params()
        sparams.no_perf = true
        guard let chain = llama_sampler_chain_init(sparams) else {
            throw InferenceError.generationFailed("llama.cpp: sampler chain init failed")
        }
        let seed = parameters.seed.map { UInt32(truncatingIfNeeded: $0) }
            ?? UInt32.random(in: 0...UInt32.max)
        LlamaSamplerChainBuilder.add(LlamaSamplerPlan.steps(for: parameters), to: chain, seed: seed)
        sampler = chain

        // --- Build bitmaps ---------------------------------------------------
        // `mtmd_helper_bitmap_init_from_buf` decodes JPEG/PNG/etc via the
        // same stb_image path as mtmd-cli. On success libmtmd owns the
        // bitmap; ownership transfers to the chunks list after tokenize.
        //
        // Store as `[OpaquePointer?]` so we can hand the buffer straight to
        // `mtmd_tokenize`, whose `const mtmd_bitmap **` parameter maps to
        // `UnsafeMutablePointer<OpaquePointer?>` on the Swift side.
        var bitmaps: [OpaquePointer?] = []
        bitmaps.reserveCapacity(images.count)
        func freeBitmapsOnError() {
            for bmp in bitmaps { if let b = bmp { mtmd_bitmap_free(b) } }
        }
        for (idx, data) in images.enumerated() {
            let bmp: OpaquePointer? = data.withUnsafeBytes { raw -> OpaquePointer? in
                guard let base = raw.bindMemory(to: UInt8.self).baseAddress else { return nil }
                return mtmd_helper_bitmap_init_from_buf(mctx, base, data.count)
            }
            guard let bmp else {
                freeBitmapsOnError()
                throw InferenceError.generationFailed(
                    "llama.cpp: failed to decode image #\(idx) (\(data.count) bytes). " +
                    "Supported: JPEG, PNG, BMP, GIF."
                )
            }
            // The id travels into the image chunk(s) libmtmd makes from
            // this bitmap; it is what the cache recognises the picture by
            // on the next turn.
            mtmd_bitmap_set_id(bmp, Self.contentID(of: bmp))
            bitmaps.append(bmp)
        }

        // --- Allocate chunks & tokenize -------------------------------------
        guard let chunks = mtmd_input_chunks_init() else {
            freeBitmapsOnError()
            throw InferenceError.generationFailed("llama.cpp: mtmd_input_chunks_init failed")
        }
        // From here on, whether tokenize succeeds or fails, chunks must be
        // freed. Bitmap ownership transfers to chunks only on success (rc=0).
        defer { mtmd_input_chunks_free(chunks) }

        let rc: Int32 = prompt.withCString { cPrompt -> Int32 in
            var text = mtmd_input_text(
                text: cPrompt,
                // The prompt is always the whole conversation from its first
                // token, so it always starts a sequence and wants a BOS
                // token. The same setting every turn is what keeps the
                // cached prefix and the new tokenization in agreement.
                add_special: true,
                parse_special: true
            )
            return bitmaps.withUnsafeMutableBufferPointer { buf -> Int32 in
                // `mtmd_tokenize` takes `const mtmd_bitmap **` which Swift
                // imports as `UnsafeMutablePointer<OpaquePointer?>`. Pass
                // the Array's own mutable storage; libmtmd only reads it.
                mtmd_tokenize(mctx, chunks, &text, buf.baseAddress, buf.count)
            }
        }

        switch rc {
        case 0:
            // Bitmaps are now owned by `chunks`; do NOT free them ourselves.
            break
        case 1:
            freeBitmapsOnError()
            throw InferenceError.generationFailed(
                "llama.cpp: <__media__> marker count (\(prompt.components(separatedBy: "<__media__>").count - 1)) " +
                "does not match image count (\(bitmaps.count))."
            )
        case 2:
            freeBitmapsOnError()
            throw InferenceError.generationFailed(
                "llama.cpp: image preprocessing failed (unsupported resolution or format?)."
            )
        default:
            freeBitmapsOnError()
            throw InferenceError.generationFailed("llama.cpp: mtmd_tokenize returned \(rc)")
        }

        // --- Describe the prompt as KV chunks --------------------------------
        let promptChunks = describe(chunks, bos: llama_vocab_bos(vocab))
        let promptPositions = KVPrefix.totalPositions(promptChunks)
        guard promptPositions > 0 else {
            throw InferenceError.generationFailed("llama.cpp: the prompt tokenized to nothing")
        }

        // --- Rewind the cache to what it shares with this prompt ------------
        let rewind = rewindCache(to: promptChunks)
        guard rewind.reused < promptPositions,
              let start = KVPrefix.cut(promptChunks, atPosition: rewind.reused),
              start.chunk < promptChunks.count
        else {
            // Cannot happen - the rewind only stops on chunk boundaries -
            // but a wrong cut would decode garbage, so start over instead.
            forgetSequence()
            throw InferenceError.generationFailed("llama.cpp: internal error (bad cache cut)")
        }
        let reusedChunks = KVPrefix.prefix(promptChunks, positions: rewind.reused) ?? []

        // --- Evaluate the rest ----------------------------------------------
        // Text is decoded here in batch-sized slices (the same batches
        // `mtmd_helper_eval_chunk_single` builds, but able to start in the
        // middle of a run and to stop where a snapshot is due); media goes
        // through libmtmd, which encodes it and applies the per-arch decode
        // tweaks (non-causal attention, M-RoPE positions, ...).
        let checkpointAt = rollback == .checkpoints
            ? KVPrefix.checkpointPosition(
                in: promptChunks,
                reused: rewind.reused,
                offsetFromEnd: Self.checkpointOffsetFromEnd
            )
            : nil
        let prefillStart = Date()
        var evaluatedMedia = 0
        for index in start.chunk..<promptChunks.count {
            let isLast = index == promptChunks.count - 1
            let evalRC: Int32
            switch promptChunks[index] {
            case .text(let tokens):
                let from = index == start.chunk ? start.offset : 0
                evalRC = decodeText(
                    Array(tokens[from...]),
                    logitsLast: isLast,
                    checkpointAt: checkpointAt
                )
            case .media(let media):
                guard let chunk = mtmd_input_chunks_get(chunks, index) else {
                    evalRC = -1
                    break
                }
                var newNPast: llama_pos = nPast
                evalRC = mtmd_helper_eval_chunk_single(
                    mctx, lctx, chunk, nPast, /*seq_id*/ 0, Self.batchSize,
                    /*logits_last*/ isLast, &newNPast
                )
                if evalRC == 0 {
                    nPast += llama_pos(media.positions)
                    cached.append(.media(media))
                    evaluatedMedia += 1
                }
            }
            guard evalRC == 0 else {
                // CRITICAL: any non-zero rc leaves the KV cache in a partially-
                // written state — the record no longer matches what's actually
                // in memory. If we leave it as-is, every subsequent generate()
                // call hits llama_decode with bad positions and returns rc=-1,
                // breaking the whole chat session.
                //
                // Wipe the KV cache so the next message starts clean. The user
                // loses prior context (acceptable: the alternative is a dead chat).
                forgetSequence()

                // Translate libllama codes into something readable.
                let promptTokens = KVPrefix.totalTokens(promptChunks)
                let reason: String
                switch evalRC {
                case 1:
                    reason = "the conversation no longer fits the \(Self.contextTokens)-token "
                        + "context window (\(promptTokens) tokens with media). "
                        + "Clear chat, or send fewer / smaller images."
                case 2:
                    reason = "compute error (out of GPU memory?)"
                default:
                    reason = "internal error rc=\(evalRC)"
                }
                throw InferenceError.generationFailed("llama.cpp: \(reason)")
            }
            if Task.isCancelled { throw InferenceError.cancelled }
        }

        let promptTokens = KVPrefix.totalTokens(promptChunks)
        let reusedTokens = KVPrefix.totalTokens(reusedChunks)
        let prefill = LlamaVisionPrefill(
            report: PromptCacheReport(
                promptTokens: promptTokens,
                reusedTokens: reusedTokens,
                evaluatedTokens: promptTokens - reusedTokens,
                reusedMedia: KVPrefix.mediaCount(reusedChunks),
                evaluatedMedia: evaluatedMedia,
                rewind: rewind.kind
            ),
            ms: Date().timeIntervalSince(prefillStart) * 1000
        )

        // --- Sampling loop (text only; mirrors LlamaContext.completion) -----
        // Multibyte characters are often split across tokens; accumulate raw
        // bytes and emit only complete UTF-8 sequences (see LlamaContext).
        // Every sampled token that is decoded joins the cache record, so the
        // next turn can reuse the reply as well (when it re-tokenizes the
        // same way).
        var produced = 0
        var pendingUTF8: [UInt8] = []
        var generated: [Int32] = []
        defer { cached = KVPrefix.appending(generated, to: cached) }
        while produced < parameters.maxTokens {
            if Task.isCancelled { throw InferenceError.cancelled }

            let tokenNew = llama_sampler_sample(chain, lctx, -1)
            if llama_vocab_is_eog(vocab, tokenNew) { break }

            var piece = [CChar](repeating: 0, count: 128)
            var nChars = llama_token_to_piece(vocab, tokenNew, &piece, Int32(piece.count), 0, false)
            if nChars < 0 {
                piece = [CChar](repeating: 0, count: Int(-nChars))
                nChars = llama_token_to_piece(vocab, tokenNew, &piece, Int32(piece.count), 0, false)
            }
            if nChars > 0 {
                pendingUTF8.append(contentsOf: piece.prefix(Int(nChars)).map { UInt8(bitPattern: $0) })
                let complete = LlamaContext.takeCompleteUTF8Prefix(&pendingUTF8)
                if !complete.isEmpty { onPiece(complete, produced) }
            }

            var single = llama_batch_init(1, 0, 1)
            single.token[0] = tokenNew
            single.pos[0] = nPast
            single.n_seq_id[0] = 1
            single.seq_id[0]?[0] = 0
            single.logits[0] = 1
            single.n_tokens = 1
            let decRC = llama_decode(lctx, single)
            llama_batch_free(single)
            guard decRC == 0 else {
                // Same recovery story as the prompt path: a failed decode
                // mid-stream means the KV cache is now inconsistent.
                generated = []
                forgetSequence()
                let reason = (decRC == 1)
                    ? "context window full mid-generation (try Clear chat)"
                    : "decode failed rc=\(decRC)"
                throw InferenceError.generationFailed("llama.cpp: \(reason)")
            }
            nPast += 1
            generated.append(tokenNew)
            produced += 1
        }

        // Flush whatever is left (lossy for a genuinely truncated tail).
        if !pendingUTF8.isEmpty {
            onPiece(String(decoding: pendingUTF8, as: UTF8.self), produced)
        }
        return LlamaVisionGeneration(tokens: produced, prefill: prefill)
    }

    // MARK: - Prompt description

    /// The chunk list libmtmd produced, as the cache records it: token ids
    /// for text, content id and size for media. One entry per libmtmd chunk,
    /// so an index here is an index into `chunks`.
    private func describe(_ chunks: OpaquePointer, bos: llama_token) -> [KVChunk] {
        let count = mtmd_input_chunks_size(chunks)
        var out: [KVChunk] = []
        out.reserveCapacity(count)
        for index in 0..<count {
            guard let chunk = mtmd_input_chunks_get(chunks, index) else {
                out.append(.text([]))
                continue
            }
            if mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT {
                var n = 0
                var tokens: [Int32] = []
                if let raw = mtmd_input_chunk_get_tokens_text(chunk, &n), n > 0 {
                    tokens = Array(UnsafeBufferPointer(start: raw, count: n))
                }
                if index == 0 {
                    tokens = KVPrefix.droppingDuplicateLeadingBOS(tokens, bos: bos)
                }
                out.append(.text(tokens))
            } else {
                // An image without an id (never the case: every bitmap gets
                // one above) must never match anything.
                var id = mtmd_input_chunk_get_id(chunk).map { String(cString: $0) } ?? ""
                if id.isEmpty { id = "unidentified-\(UUID().uuidString)" }
                out.append(.media(KVMedia(
                    id: id,
                    tokens: mtmd_input_chunk_get_n_tokens(chunk),
                    positions: Int(mtmd_input_chunk_get_n_pos(chunk))
                )))
            }
        }
        return out
    }

    /// Content hash of a decoded bitmap (its pixels, not the file bytes:
    /// the same picture re-encoded is still the same picture).
    private static func contentID(of bitmap: OpaquePointer) -> String {
        let width = mtmd_bitmap_get_nx(bitmap)
        let height = mtmd_bitmap_get_ny(bitmap)
        let count = mtmd_bitmap_get_n_bytes(bitmap)
        guard let data = mtmd_bitmap_get_data(bitmap), count > 0 else {
            return "unidentified-\(UUID().uuidString)"
        }
        return KVPrefix.mediaID(
            width: width,
            height: height,
            bytes: UnsafeRawBufferPointer(start: data, count: count)
        )
    }

    // MARK: - Cache rewind

    /// Cuts the KV cache back to the longest prefix it shares with
    /// `prompt`, leaving `nPast`, `cached` and the snapshots describing
    /// exactly what is left. Returns how many positions are reused.
    ///
    /// Every path that cannot prove the memory holds the reused prefix ends
    /// in a full clear: decoding the whole prompt again is slow, decoding
    /// on top of a wrong prefix is a wrong answer.
    private func rewindCache(
        to prompt: [KVChunk]
    ) -> (reused: Int, kind: PromptCacheReport.Rewind) {
        guard let lctx, let mem = llama_get_memory(lctx) else {
            forgetSequence()
            return (0, .cleared)
        }
        guard !cached.isEmpty else {
            forgetSequence()
            return (0, .empty)
        }
        var reuse = KVPrefix.reusablePositions(cached: cached, new: prompt)
        guard reuse > 0 else {
            forgetSequence()
            return (0, .cleared)
        }

        // The previous turn's Metal command buffers can still be in flight;
        // changing cache state under them is what a GPU fault is made of.
        llama_synchronize(lctx)

        var kind = PromptCacheReport.Rewind.prefix
        if reuse < KVPrefix.totalPositions(cached),
           !llama_memory_seq_rm(mem, 0, llama_pos(reuse), -1) {
            // The memory refused to forget a suffix: a recurrent state only
            // exists for its last position. Restore the newest snapshot
            // inside the shared prefix and cut there.
            guard let at = KVPrefix.checkpoint(
                    atOrBefore: reuse,
                    among: checkpoints.map(\.position)
                  ),
                  let checkpoint = checkpoints.last(where: { $0.position == at }),
                  restore(checkpoint),
                  llama_memory_seq_rm(mem, 0, llama_pos(at), -1)
            else {
                forgetSequence()
                return (0, .cleared)
            }
            reuse = at
            kind = .checkpoint
        }

        // A sliding-window cache may have dropped positions the first new
        // token still has to attend to. With `swa_full` it never should,
        // but a model that hits it must start over rather than attend to
        // holes.
        if nSwa > 0, llama_memory_seq_pos_min(mem, 0) > max(0, Int32(reuse) - nSwa) {
            forgetSequence()
            return (0, .cleared)
        }
        // The memory must now end right before the first position this call
        // decodes. M-RoPE images leave gaps (an image's cells all sit on its
        // first position), so there it can only be checked as "below".
        let posMax = llama_memory_seq_pos_max(mem, 0)
        let expected = Int32(reuse) - 1
        if posMax < 0 || posMax > expected || (!usesMRoPE && posMax != expected) {
            forgetSequence()
            return (0, .cleared)
        }

        cached = KVPrefix.prefix(cached, positions: reuse) ?? []
        checkpoints.removeAll { $0.position > reuse }
        nPast = llama_pos(reuse)
        guard KVPrefix.totalPositions(cached) == reuse else {
            forgetSequence()
            return (0, .cleared)
        }
        return (reuse, kind)
    }

    // MARK: - Text decode and snapshots

    /// Decodes `tokens` from `nPast` on, in batch-sized slices, appending
    /// each decoded slice to the record. Asks for logits on the very last
    /// token when `logitsLast`. When `checkpointAt` falls inside this run a
    /// slice ends exactly there and the recurrent state is snapshotted
    /// before the rest is decoded. Returns llama_decode's code.
    private func decodeText(
        _ tokens: [Int32],
        logitsLast: Bool,
        checkpointAt: Int?
    ) -> Int32 {
        guard let lctx, !tokens.isEmpty else { return 0 }
        let batchSize = Int(Self.batchSize)
        var batch = llama_batch_init(Self.batchSize, 0, 1)
        defer { llama_batch_free(batch) }

        var offset = 0
        while offset < tokens.count {
            var end = min(offset + batchSize, tokens.count)
            if let checkpointAt {
                let local = checkpointAt - Int(nPast) + offset
                if local > offset, local < end { end = local }
            }
            let slice = tokens[offset..<end]
            let isLastSlice = end == tokens.count
            for (i, token) in slice.enumerated() {
                batch.token[i] = token
                batch.pos[i] = nPast + llama_pos(i)
                batch.n_seq_id[i] = 1
                batch.seq_id[i]?[0] = 0
                batch.logits[i] = (logitsLast && isLastSlice && i == slice.count - 1) ? 1 : 0
            }
            batch.n_tokens = Int32(slice.count)
            let rc = llama_decode(lctx, batch)
            guard rc == 0 else { return rc }
            nPast += llama_pos(slice.count)
            cached = KVPrefix.appending(Array(slice), to: cached)
            offset = end
            if let checkpointAt, Int(nPast) == checkpointAt {
                saveCheckpoint()
            }
        }
        return 0
    }

    /// Snapshots the recurrent part of sequence 0 at `nPast`. Best effort:
    /// a snapshot that cannot be taken only costs the next turn a full
    /// decode.
    private func saveCheckpoint() {
        guard let lctx else { return }
        let flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
        llama_synchronize(lctx)
        let size = llama_state_seq_get_size_ext(lctx, 0, flags)
        guard size > 0, size <= Self.maxCheckpointBytes else { return }
        var data = [UInt8](repeating: 0, count: size)
        let written = data.withUnsafeMutableBufferPointer { buffer in
            llama_state_seq_get_data_ext(lctx, buffer.baseAddress, size, 0, flags)
        }
        guard written == size else { return }
        let position = Int(nPast)
        checkpoints.removeAll { $0.position >= position }
        checkpoints.append(Checkpoint(position: position, data: data))
        if checkpoints.count > Self.maxCheckpoints {
            checkpoints.removeFirst(checkpoints.count - Self.maxCheckpoints)
        }
    }

    /// Puts a snapshot's recurrent state back. The attention part of the
    /// cache is untouched and is cut by the caller.
    private func restore(_ checkpoint: Checkpoint) -> Bool {
        guard let lctx else { return false }
        let flags = llama_state_seq_flags(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
        let read = checkpoint.data.withUnsafeBufferPointer { buffer in
            llama_state_seq_set_data_ext(lctx, buffer.baseAddress, buffer.count, 0, flags)
        }
        return read == checkpoint.data.count
    }
}

/// What one `generate` call produced: the number of tokens and the prompt
/// figures of that same call.
nonisolated struct LlamaVisionGeneration: Sendable {
    let tokens: Int
    let prefill: LlamaVisionPrefill
}

/// The vision context's prompt figures for one `generate` call.
nonisolated struct LlamaVisionPrefill: Sendable {
    var report = PromptCacheReport(promptTokens: 0, reusedTokens: 0, evaluatedTokens: 0, rewind: .empty)
    /// Time spent decoding the part of the prompt that was not reused.
    var ms: Double = 0
}

#endif // canImport(llama)
