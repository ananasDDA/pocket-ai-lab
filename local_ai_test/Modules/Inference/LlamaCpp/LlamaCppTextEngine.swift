//
//  LlamaCppTextEngine.swift
//  local_ai_test
//
//  GGUF text-generation through llama.cpp via its Swift Package
//  (https://github.com/ggerganov/llama.cpp - contains a Package.swift since
//  mid-2024). Alternative: drop-in `llama.xcframework` built via
//  `Scripts/build_llama_xcframework.sh`.
//
//  Swift 5/6 adopts C-API through `import llama` module. An internal
//  `LlamaContext` actor isolates llama_context_* calls and surfaces a
//  streaming `AsyncThrowingStream<String, Error>`.
//
//  Chat templates are resolved through `llama_chat_apply_template`; for
//  models that do not embed a template we fall back to `PromptFormatter`.
//

import Foundation

#if canImport(llama)

import llama

final class LlamaCppTextEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .textOnly

    private let modelDirectory: URL
    private let model: AIModel
    private var context: LlamaContext?

    var isLoaded: Bool { context != nil }

    init(modelDirectory: URL, model: AIModel) {
        self.modelDirectory = modelDirectory
        self.model = model
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        progress(.loadingWeights(0.0))
        let ggufURL = try locateGGUF()
        context = try await LlamaContext.create(modelPath: ggufURL.path)
        progress(.ready)
    }

    func unload() async {
        let ctx = context
        context = nil
        await ctx?.cleanup()
    }

    func embeddedChatTemplate() async -> String? {
        await context?.embeddedChatTemplate()
    }

    func resetConversation() async {
        await context?.reset()
    }

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state (`context`, `model`) is main-actor
            // isolated; the heavy decode work runs inside the LlamaContext
            // actor, so this task only awaits and forwards stream events.
            let task = Task { @MainActor [weak self] in
                guard let self, let ctx = self.context else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                do {
                    let modelID = self.model.id
                    let embedded = await ctx.embeddedChatTemplate()
                    // Fit the conversation into the window by whole turns
                    // before tokenizing (see `HistoryTrim`): the system turn
                    // and the question stay, older exchanges go in batches
                    // so the KV prefix cache keeps hitting, and only a turn
                    // that is too long on its own is cut inside. Measured
                    // with the model's own tokenizer on the exact prompt.
                    let room = ContextBudget.promptRoom(
                        maxTokens: parameters.maxTokens,
                        contextSize: Int(LlamaContext.contextSize)
                    )
                    let fitted = await HistoryTrim.fit(
                        turns: turns,
                        room: room,
                        count: { await ctx.tokenCount(of: $0, asPrompt: false) },
                        measure: { @MainActor kept in
                            let text = ChatTemplateResolver.render(
                                turns: kept, modelID: modelID, embedded: embedded
                            ).text
                            return await ctx.tokenCount(of: text, asPrompt: true)
                        }
                    )
                    let rendered = ChatTemplateResolver.render(
                        turns: fitted.kept,
                        modelID: modelID,
                        embedded: embedded
                    )
                    continuation.yield(.diagnostic(.promptRendered(rendered)))
                    let prompt = rendered.text

                    let start = Date()

                    // Token index is passed from the actor, so no shared
                    // mutable state is captured across concurrency domains.
                    let tokensGenerated = try await ctx.completion(
                        prompt: prompt,
                        maxTokens: parameters.maxTokens,
                        sampling: LlamaSamplerPlan.steps(for: parameters),
                        seed: parameters.seed
                    ) { piece, index in
                        if index == 0 {
                            let ms = Date().timeIntervalSince(start) * 1000
                            continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                        }
                        continuation.yield(.textDelta(piece))
                    }

                    let elapsed = Date().timeIntervalSince(start)
                    let prefill = await ctx.lastPrefill
                    continuation.yield(.diagnostic(.promptProcessed(
                        tokenCount: prefill.tokens,
                        timeMs: prefill.ms
                    )))
                    continuation.yield(.diagnostic(.promptCache(PromptCacheReport(
                        promptTokens: prefill.tokens,
                        reusedTokens: prefill.reused,
                        evaluatedTokens: prefill.tokens - prefill.reused,
                        droppedFrontTokens: prefill.droppedFront,
                        droppedTurns: fitted.droppedTurns,
                        truncatedTokens: fitted.truncatedTokens,
                        firstUserTurnIntact: fitted.firstUserTurnIntact,
                        rewind: prefill.rewind
                    ))))
                    continuation.yield(.diagnostic(.finished(
                        tokensGenerated: tokensGenerated,
                        tokensPerSecond: elapsed > 0 ? Double(tokensGenerated) / elapsed : 0
                    )))
                    continuation.finish()
                } catch {
                    continuation.finish(
                        throwing: Task.isCancelled
                            ? InferenceError.cancelled
                            : InferenceError.generationFailed(error.localizedDescription)
                    )
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func locateGGUF() throws -> URL {
        if let preferred = model.preferredGGUFFilename {
            let candidate = modelDirectory.appendingPathComponent(preferred)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        let items = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        guard let gguf = items.first(where: { GGUFNaming.isWeights($0.lastPathComponent) }) else {
            throw InferenceError.generationFailed("No .gguf file found in \(modelDirectory.path)")
        }
        return gguf
    }
}

/// One-time process setup for everything built on ggml.
///
/// Metal residency sets (on by default in this llama.cpp revision) keep
/// every buffer the recent graphs touched resident for three minutes. On a
/// phone that is memory this app does not have, and worse: long multimodal
/// chats died mid-image-encode with "Discarded (victim of GPU
/// error/recovery)", after which the Metal backend stays in an error state
/// until it is recreated, so every later turn fails too. The same build
/// trips a residency-set assertion (`[rsets->data count] == 0`) when the
/// process exits. Turning the feature off costs nothing measurable here -
/// the weights are memory-mapped and the working set is small enough that
/// the allocator keeps up.
enum LlamaRuntime {

    /// Idempotent: the `Void` initialiser runs on first access and never
    /// again. Both llama.cpp contexts touch it before they load anything.
    static let bootstrap: Void = {
        setenv("GGML_METAL_NO_RESIDENCY", "1", 1)
        // Before the backend starts logging: a failed load reports its
        // reason only here, and `LlamaLoadFailure` reads it back.
        llama_log_set({ level, text, _ in
            guard let text else { return }
            // Same output the default logger gives, so console logs stay.
            fputs(text, stderr)
            LlamaLogCapture.shared.append(level: Int(level.rawValue), text: String(cString: text))
        }, nil)
        llama_backend_init()
    }()

    /// Builds the error for a load that returned NULL, from what llama.cpp
    /// logged since `mark`.
    static func loadError(since mark: Int, stage: LlamaLoadFailure.Stage) -> LlamaLoadError {
        let log = LlamaLogCapture.shared.lines(since: mark)
        return LlamaLoadError(
            cause: LlamaLoadFailure.classify(log: log, stage: stage),
            logExcerpt: LlamaLoadFailure.excerpt(from: log)
        )
    }
}

/// Turns a `LlamaSamplerPlan` into llama.cpp samplers (b8882 signatures).
/// Shared with `LlamaVisionContext`.
nonisolated enum LlamaSamplerChainBuilder {
    static func add(
        _ steps: [LlamaSamplerStep],
        to chain: UnsafeMutablePointer<llama_sampler>?,
        seed: UInt32
    ) {
        for step in steps {
            let sampler: UnsafeMutablePointer<llama_sampler>?
            switch step {
            case let .penalties(lastN, rep, freq, presence):
                sampler = llama_sampler_init_penalties(lastN, rep, freq, presence)
            case let .topK(k):
                sampler = llama_sampler_init_top_k(k)
            case let .topP(p, minKeep):
                sampler = llama_sampler_init_top_p(p, minKeep)
            case let .minP(p, minKeep):
                sampler = llama_sampler_init_min_p(p, minKeep)
            case let .temperature(t):
                sampler = llama_sampler_init_temp(t)
            case .dist:
                sampler = llama_sampler_init_dist(seed)
            }
            llama_sampler_chain_add(chain, sampler)
        }
    }
}

/// Actor isolating all direct calls into the llama.cpp C API.
///
/// Implemented against the public headers in llama.cpp/llama.h. Implementors
/// of this file should keep the surface area minimal: loading, batched decode,
/// sampler chain (`LlamaSamplerPlan`),
/// token streaming, reset.
actor LlamaContext {
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var nPast: Int32 = 0
    /// The exact token sequence currently held in the KV cache, so a new
    /// prompt can reuse the part of it that has not changed. Always the
    /// same length as `nPast`.
    private var cachedTokens: [llama_token] = []
    /// Prompt size (after fitting it into the window), the time spent
    /// decoding the part the cache did not already hold, how much of it the
    /// cache did hold and how many tokens were cut from the front to fit,
    /// for the last completion. Read by the engine for its diagnostics.
    private(set) var lastPrefill: (
        tokens: Int, ms: Double, reused: Int, droppedFront: Int, rewind: PromptCacheReport.Rewind
    ) = (0, 0, 0, 0, .empty)

    /// Context window requested from llama.cpp. Prompts longer than what is
    /// left of it are trimmed from the front (oldest history first).
    static let contextSize: UInt32 = 4096
    /// Largest number of tokens one `llama_decode` call may carry. llama.cpp
    /// aborts the process (GGML_ASSERT, not a recoverable error) when a
    /// batch exceeds it, so the prompt is always fed in slices of this size.
    static let batchSize: UInt32 = 512

    static func create(modelPath: String) async throws -> LlamaContext {
        _ = LlamaRuntime.bootstrap
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

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = contextSize
        ctxParams.n_batch = batchSize
        ctxParams.n_threads = Int32(max(1, ProcessInfo.processInfo.processorCount - 1))
        ctxParams.n_threads_batch = ctxParams.n_threads

        guard let ctx = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw LlamaRuntime.loadError(since: logMark, stage: .context)
        }

        return LlamaContext(model: model, ctx: ctx)
    }

    private init(model: OpaquePointer, ctx: OpaquePointer) {
        self.model = model
        self.ctx = ctx
    }

    func reset() {
        dropCache()
    }

    /// The chat template stored in the GGUF, if it has one.
    func embeddedChatTemplate() -> String? {
        guard let model else { return nil }
        return LlamaChatTemplate.embedded(in: model)
    }

    func cleanup() {
        if let sampler { llama_sampler_free(sampler) }
        if let ctx { llama_free(ctx) }
        if let model { llama_model_free(model) }
        sampler = nil
        ctx = nil
        model = nil
    }

    /// Runs one completion. Calls `onPiece(text, tokenIndex)` for every
    /// decoded UTF-8-complete chunk and returns the number of tokens
    /// generated. The index lets callers detect the first token without
    /// sharing mutable state across concurrency domains.
    ///
    /// `sampling` is the chain to build (`LlamaSamplerPlan.steps(for:)`).
    /// `seed` fixes the distribution sampler's RNG; `nil` keeps the previous
    /// behaviour of drawing a fresh random seed per completion.
    func completion(
        prompt: String,
        maxTokens: Int,
        sampling: [LlamaSamplerStep],
        seed: UInt64? = nil,
        onPiece: @Sendable (String, Int) -> Void
    ) async throws -> Int {
        guard let model, let ctx else { throw InferenceError.modelNotLoaded }

        let vocab = llama_model_get_vocab(model)

        // Build sampler chain
        if sampler != nil { llama_sampler_free(sampler) }
        var sparams = llama_sampler_chain_default_params()
        sparams.no_perf = true
        let chain = llama_sampler_chain_init(sparams)
        // A caller-supplied seed makes the sampler reproducible (the benchmark
        // relies on it); without one, every completion draws a fresh seed, as
        // before. llama.cpp reserves 0xFFFFFFFF for "pick a random seed", so a
        // supplied seed that truncates onto it is nudged by one rather than
        // silently becoming random.
        let samplerSeed: UInt32
        if let seed {
            // 0xFFFFFFFF is llama.cpp's LLAMA_DEFAULT_SEED sentinel; spelled
            // out here because the C macro's imported Swift type is not
            // guaranteed to be UInt32.
            let truncated = UInt32(truncatingIfNeeded: seed)
            samplerSeed = truncated == 0xFFFF_FFFF ? truncated &- 1 : truncated
        } else {
            samplerSeed = UInt32(truncatingIfNeeded: UInt64.random(in: 0...UInt64(UInt32.max)))
        }
        LlamaSamplerChainBuilder.add(sampling, to: chain, seed: samplerSeed)
        sampler = chain

        // Tokenize prompt
        guard var tokens = tokenize(prompt, asPrompt: true), !tokens.isEmpty else {
            throw InferenceError.generationFailed("llama.cpp: tokenize failed")
        }

        // Fit the prompt into the context window, keeping room for the
        // requested completion. The engine has already fitted the
        // conversation by whole turns (`HistoryTrim`), so this is a safety
        // net for a prompt that still does not fit: it loses its oldest
        // tokens (the BOS token is kept so the model still sees a proper
        // sequence start), and the cut is reported as `droppedFrontTokens`.
        //
        // The budget is measured against the whole window, NOT against what
        // is left after `nPast`: every engine in this app is stateless and
        // the caller re-sends the entire conversation on each turn, so the
        // history is already part of `tokens`. Subtracting `nPast` as well
        // charged the transcript twice per message and ended a chat with
        // "context window full" after a handful of turns.
        let contextSize = Int(Self.contextSize)
        let room = ContextBudget.promptRoom(maxTokens: maxTokens, contextSize: contextSize)
        guard room >= 16 else {
            throw InferenceError.generationFailed(
                "llama.cpp: the context window (\(contextSize) tokens) leaves no room for a reply."
            )
        }
        let renderedCount = tokens.count
        if tokens.count > room {
            let keepFromEnd = room - 1
            tokens = [tokens[0]] + Array(tokens.suffix(keepFromEnd))
        }

        // Rewind the KV cache to the longest prefix it still shares with
        // this prompt, so only the new tail has to be decoded.
        let hadCache = !cachedTokens.isEmpty
        try rewindCache(to: tokens)
        let reused = Int(nPast)
        let rewind: PromptCacheReport.Rewind = reused > 0 ? .prefix : (hadCache ? .cleared : .empty)

        // Decode the prompt in slices no larger than the batch the context
        // was created with. Only the last token of the last slice needs
        // logits.
        let batchSize = Int(Self.batchSize)
        var batch = llama_batch_init(Int32(batchSize), 0, 1)
        defer { llama_batch_free(batch) }
        // Everything before `nPast` is already in the cache after the
        // rewind above, so the loop starts there.
        var offset = Int(nPast)
        let prefillStart = Date()
        while offset < tokens.count {
            let slice = tokens[offset..<min(offset + batchSize, tokens.count)]
            let isLastSlice = offset + slice.count == tokens.count
            for (i, tok) in slice.enumerated() {
                batch.token[i] = tok
                batch.pos[i] = nPast + Int32(i)
                batch.n_seq_id[i] = 1
                batch.seq_id[i]?[0] = 0
                batch.logits[i] = (isLastSlice && i == slice.count - 1) ? 1 : 0
            }
            batch.n_tokens = Int32(slice.count)
            guard llama_decode(ctx, batch) == 0 else {
                // A failed decode leaves the cache holding an unknown
                // number of the batch's tokens, so the mirror can no longer
                // be trusted. Drop everything: the next turn re-decodes the
                // conversation instead of building on a bad prefix.
                dropCache()
                throw InferenceError.generationFailed("llama.cpp: decode(prompt) failed")
            }
            nPast += Int32(slice.count)
            cachedTokens.append(contentsOf: slice)
            offset += slice.count
            if Task.isCancelled { throw InferenceError.cancelled }
        }
        lastPrefill = (
            tokens.count,
            Date().timeIntervalSince(prefillStart) * 1000,
            reused,
            renderedCount - tokens.count,
            rewind
        )

        // Generation loop.
        // Multibyte characters (Cyrillic, CJK, emoji) are often split across
        // several tokens; decoding each piece separately with String(cString:)
        // yields U+FFFD garbage. Accumulate raw bytes and emit only complete
        // UTF-8 sequences.
        var produced = 0
        var pendingUTF8: [UInt8] = []
        while produced < maxTokens {
            if Task.isCancelled { throw InferenceError.cancelled }

            let tokenNew = llama_sampler_sample(chain, ctx, -1)
            if llama_vocab_is_eog(vocab, tokenNew) { break }

            // Piece -> raw bytes (piece may legitimately fill the whole
            // buffer with no null terminator; negative return = buffer too
            // small, retry with the exact required size).
            var piece = [CChar](repeating: 0, count: 128)
            var nChars = llama_token_to_piece(vocab, tokenNew, &piece, Int32(piece.count), 0, false)
            if nChars < 0 {
                piece = [CChar](repeating: 0, count: Int(-nChars))
                nChars = llama_token_to_piece(vocab, tokenNew, &piece, Int32(piece.count), 0, false)
            }
            if nChars > 0 {
                pendingUTF8.append(contentsOf: piece.prefix(Int(nChars)).map { UInt8(bitPattern: $0) })
                let complete = Self.takeCompleteUTF8Prefix(&pendingUTF8)
                if !complete.isEmpty { onPiece(complete, produced) }
            }

            var single = llama_batch_init(1, 0, 1)
            single.token[0] = tokenNew
            single.pos[0] = nPast
            single.n_seq_id[0] = 1
            single.seq_id[0]?[0] = 0
            single.logits[0] = 1
            single.n_tokens = 1
            let rc = llama_decode(ctx, single)
            llama_batch_free(single)
            guard rc == 0 else {
                dropCache()
                throw InferenceError.generationFailed("llama.cpp: decode(token) failed")
            }
            nPast += 1
            cachedTokens.append(tokenNew)
            produced += 1
        }

        // Flush whatever is left (lossy for a genuinely truncated tail).
        if !pendingUTF8.isEmpty {
            onPiece(String(decoding: pendingUTF8, as: UTF8.self), produced)
        }
        return produced
    }

    /// Number of tokens `text` decodes to: as a whole prompt (`asPrompt`,
    /// with the BOS handling `completion` applies) or as a piece of one.
    /// The engine measures candidate prompts with it while fitting the
    /// conversation into the window.
    func tokenCount(of text: String, asPrompt: Bool) -> Int {
        tokenize(text, asPrompt: asPrompt)?.count ?? (text.utf8.count / 2 + 1)
    }

    /// Tokenizes `text` with special tokens parsed. `nil` when the model is
    /// not loaded or tokenization fails.
    ///
    /// `add_special` is off: `PromptFormatter` already writes the template's
    /// own BOS marker for the families that have one (Llama 3's
    /// "<|begin_of_text|>"), and asking llama_tokenize to add it as well
    /// started every prompt with BOS twice - llama.cpp complains about
    /// exactly that in check_double_bos_eos. For a whole prompt, the
    /// families whose template carries no marker get the token prepended.
    private func tokenize(_ text: String, asPrompt: Bool) -> [llama_token]? {
        guard let model else { return nil }
        let vocab = llama_model_get_vocab(model)
        let cString = Array(text.utf8CString)
        let capacity = Int32(text.utf8.count + 64)
        var tokens = [llama_token](repeating: 0, count: Int(capacity))
        let n = llama_tokenize(vocab, cString, Int32(cString.count - 1), &tokens, capacity, false, true)
        guard n >= 0 else { return nil }
        tokens = Array(tokens.prefix(Int(n)))
        if asPrompt {
            let bos = llama_vocab_bos(vocab)
            if llama_vocab_get_add_bos(vocab), tokens.first != bos {
                tokens.insert(bos, at: 0)
            }
        }
        return tokens
    }

    /// Drops every cached token from the first one that differs from
    /// `tokens`, leaving `nPast` on the shared prefix. The last prompt token
    /// is never reused: the sampler needs logits, and those only exist for a
    /// token that was actually decoded in this call.
    ///
    /// A partial removal can be refused (sliding-window caches cannot always
    /// forget a suffix); starting from an empty cache is always valid, so
    /// that is the fallback.
    private func rewindCache(to tokens: [llama_token]) throws {
        guard let ctx else { throw InferenceError.modelNotLoaded }
        let reusable = min(
            Self.commonPrefixLength(cachedTokens, tokens),
            max(tokens.count - 1, 0)
        )
        guard reusable < cachedTokens.count else { return }
        guard let mem = llama_get_memory(ctx) else {
            throw InferenceError.generationFailed("llama.cpp: no memory handle")
        }
        // The previous turn's Metal command buffers can still be in flight;
        // changing cache state under them is what a GPU fault is made of.
        llama_synchronize(ctx)
        if reusable == 0 || !llama_memory_seq_rm(mem, 0, llama_pos(reusable), -1) {
            llama_memory_seq_rm(mem, 0, -1, -1)
            nPast = 0
            cachedTokens = []
            return
        }
        nPast = Int32(reusable)
        cachedTokens.removeLast(cachedTokens.count - reusable)
    }

    /// Forgets the whole cache. Used when a decode fails and the cache no
    /// longer matches the mirror.
    private func dropCache() {
        if let ctx {
            llama_synchronize(ctx)
            if let mem = llama_get_memory(ctx) {
                // Removing the whole sequence never fails and, unlike
                // `llama_memory_clear(mem, true)`, does not zero the KV
                // buffers - nothing ever reads them again.
                llama_memory_seq_rm(mem, 0, -1, -1)
            }
        }
        nPast = 0
        cachedTokens = []
    }

    /// Length of the longest common prefix of two token sequences.
    static func commonPrefixLength(_ lhs: [llama_token], _ rhs: [llama_token]) -> Int {
        var i = 0
        let limit = min(lhs.count, rhs.count)
        while i < limit, lhs[i] == rhs[i] { i += 1 }
        return i
    }

    /// Removes and returns the longest prefix of `buffer` that forms complete
    /// UTF-8 sequences, leaving a (possibly empty) incomplete tail in place.
    static func takeCompleteUTF8Prefix(_ buffer: inout [UInt8]) -> String {
        var end = buffer.count
        var i = buffer.count - 1
        var continuationBytes = 0
        // Walk back over up to 3 continuation bytes to find the last lead byte.
        while i >= 0 && continuationBytes < 3 {
            let byte = buffer[i]
            if byte & 0b1100_0000 == 0b1000_0000 {
                continuationBytes += 1
                i -= 1
                continue
            }
            // `byte` is a lead byte (or ASCII / invalid — both are "complete").
            let sequenceLength: Int
            if byte & 0b1000_0000 == 0 { sequenceLength = 1 }
            else if byte & 0b1110_0000 == 0b1100_0000 { sequenceLength = 2 }
            else if byte & 0b1111_0000 == 0b1110_0000 { sequenceLength = 3 }
            else if byte & 0b1111_1000 == 0b1111_0000 { sequenceLength = 4 }
            else { sequenceLength = 1 } // invalid lead — pass through as-is
            if buffer.count - i < sequenceLength {
                end = i // sequence still incomplete — keep it buffered
            }
            break
        }
        guard end > 0 else { return "" }
        let out = String(decoding: buffer[0..<end], as: UTF8.self)
        buffer.removeFirst(end)
        return out
    }
}

#else

final class LlamaCppTextEngine: InferenceEngine, @unchecked Sendable {
    let capabilities: ModelCapabilities = .textOnly
    var isLoaded: Bool { false }
    init(modelDirectory: URL, model: AIModel) {}
    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        throw InferenceError.backendUnavailable(
            "llama.cpp backend requires the 'llama' module. Add the llama.cpp Swift package or llama.xcframework (see Scripts/build_llama_xcframework.sh)."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("llama.cpp not linked")) }
    }
}

#endif
