//
//  LlamaVisionContext.swift
//  local_ai_test
//
//  Actor that owns a llama.cpp text context *plus* a libmtmd multimodal
//  projector. Mirrors the internal `LlamaContext` actor from
//  `LlamaCppTextEngine.swift`, but replaces the text-only prompt-decode
//  step with `mtmd_tokenize` → `mtmd_helper_eval_chunks`, which handles
//  both text and image chunks (and the per-architecture specifics like
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
//  Reference: tools/mtmd/mtmd-cli.cpp in llama.cpp.
//

import Foundation

#if canImport(llama)

import llama

/// Owns llama_model + llama_context + mtmd_context, plus a sampler chain.
/// Serialised via Swift's actor model — every call into the C API is
/// guaranteed to happen on the actor's executor, matching libmtmd's
/// "not thread-safe" contract for `mtmd_helper_eval_chunks`.
actor LlamaVisionContext {

    // MARK: - Stored state

    private var model: OpaquePointer?
    private var lctx: OpaquePointer?
    private var mctx: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    /// KV-cache cursor. Persisted across turns; reset via `reset()`.
    private var nPast: llama_pos = 0

    // MARK: - Lifecycle

    /// Load both the text GGUF and the mmproj projector.
    ///
    /// - Parameters:
    ///   - modelPath: absolute path to the main `.gguf`.
    ///   - mmprojPath: absolute path to the companion `mmproj-*.gguf`.
    static func create(modelPath: String, mmprojPath: String) async throws -> LlamaVisionContext {
        llama_backend_init()

        // --- Text model ------------------------------------------------------
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #else
        modelParams.n_gpu_layers = 999
        #endif

        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw InferenceError.generationFailed("llama.cpp: failed to load model at \(modelPath)")
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
        ctxParams.n_ctx = 8192
        // n_batch must be ≥ the per-image token count (256 for Gemma 3),
        // otherwise mtmd_helper_eval_chunks splits image batches and can
        // hit "no KV slot" (rc=1) corner cases. 512 is the upstream default.
        ctxParams.n_batch = 512
        ctxParams.n_threads = Int32(max(1, ProcessInfo.processInfo.processorCount - 1))
        ctxParams.n_threads_batch = ctxParams.n_threads

        guard let lctx = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw InferenceError.generationFailed("llama.cpp: failed to create context")
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
    }

    /// Whether `generate()` is currently running on this actor.
    /// Used to defer `cleanup()` until the C pipeline is fully unwound —
    /// prevents Metal command buffers from being freed while still in flight.
    private var generating = false
    private var cleanupPending = false

    /// Wipes the KV cache and resets `nPast`. The model weights stay loaded.
    func reset() {
        guard let lctx else { return }
        if let mem = llama_get_memory(lctx) {
            llama_memory_clear(mem, true)
        }
        nPast = 0
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
        cleanupPending = false
    }

    // MARK: - Generation

    /// Tokenize the prompt + images via libmtmd, evaluate all chunks, then
    /// run a standard llama.cpp sampling loop streaming pieces to `onPiece`.
    /// Calls `onPiece(text, tokenIndex)` per UTF-8-complete chunk and
    /// returns the number of tokens generated.
    ///
    /// The `prompt` string **must** contain exactly as many `<__media__>`
    /// markers as `images.count`, or mtmd returns code 1 and we throw.
    func generate(
        prompt: String,
        images: [Data],
        parameters: GenerationParameters,
        onPiece: @Sendable (String, Int) -> Void
    ) async throws -> Int {
        guard let model, let lctx, let mctx else { throw InferenceError.modelNotLoaded }
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
        var sparams = llama_sampler_chain_default_params()
        sparams.no_perf = true
        guard let chain = llama_sampler_chain_init(sparams) else {
            throw InferenceError.generationFailed("llama.cpp: sampler chain init failed")
        }
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, parameters.repetitionPenalty, 0.0, 0.0))
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(parameters.topK)))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(parameters.topP, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_temp(parameters.temperature))
        let seed = parameters.seed.map { UInt32(truncatingIfNeeded: $0) }
            ?? UInt32.random(in: 0...UInt32.max)
        llama_sampler_chain_add(chain, llama_sampler_init_dist(seed))
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
                // add_special=true only on the very first decode so BOS is
                // prepended exactly once per conversation.
                add_special: nPast == 0,
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

        // --- Evaluate chunks -------------------------------------------------
        // `mtmd_helper_eval_chunks` is the canonical entry point: it walks
        // the chunk list, running `llama_decode` for text and
        // `mtmd_encode` + `llama_decode` for images, applying per-arch
        // tweaks (non-causal attention, M-RoPE positions, ...). We ask for
        // `logits_last=true` so the next sampling step has the right logits.
        var newNPast: llama_pos = 0
        let evalRC = mtmd_helper_eval_chunks(
            mctx,
            lctx,
            chunks,
            nPast,
            /*seq_id*/ 0,
            /*n_batch*/ 512,
            /*logits_last*/ true,
            &newNPast
        )
        guard evalRC == 0 else {
            // CRITICAL: any non-zero rc leaves the KV cache in a partially-
            // written state — `nPast` no longer matches what's actually in
            // memory. If we leave it as-is, every subsequent generate() call
            // hits llama_decode with bad positions and returns rc=-1, breaking
            // the whole chat session.
            //
            // Wipe the KV cache so the next message starts clean. The user
            // loses prior context (acceptable: the alternative is a dead chat).
            if let mem = llama_get_memory(lctx) {
                llama_memory_clear(mem, true)
            }
            nPast = 0

            // Translate libllama codes into something readable.
            let reason: String
            switch evalRC {
            case 1:
                reason = "context window full (try Clear chat, or send fewer / smaller media)"
            case 2:
                reason = "compute error (out of GPU memory?)"
            default:
                reason = "internal error rc=\(evalRC)"
            }
            throw InferenceError.generationFailed("llama.cpp: \(reason)")
        }
        nPast = newNPast

        // --- Sampling loop (text only; mirrors LlamaContext.completion) -----
        // Multibyte characters are often split across tokens; accumulate raw
        // bytes and emit only complete UTF-8 sequences (see LlamaContext).
        var produced = 0
        var pendingUTF8: [UInt8] = []
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
                // Same recovery story as the eval-chunks path: a failed
                // decode mid-stream means KV cache is now inconsistent.
                if let mem = llama_get_memory(lctx) {
                    llama_memory_clear(mem, true)
                }
                nPast = 0
                let reason = (decRC == 1)
                    ? "context window full mid-generation (try Clear chat)"
                    : "decode failed rc=\(decRC)"
                throw InferenceError.generationFailed("llama.cpp: \(reason)")
            }
            nPast += 1
            produced += 1
        }

        // Flush whatever is left (lossy for a genuinely truncated tail).
        if !pendingUTF8.isEmpty {
            onPiece(String(decoding: pendingUTF8, as: UTF8.self), produced)
        }
        return produced
    }
}

#endif // canImport(llama)
