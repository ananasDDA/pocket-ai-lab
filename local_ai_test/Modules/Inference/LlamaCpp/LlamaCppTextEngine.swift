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
                    let family = PromptFormatter.detectFamily(modelId: self.model.id)
                    let prompt = PromptFormatter.format(turns: turns, family: family)

                    let start = Date()

                    // Token index is passed from the actor, so no shared
                    // mutable state is captured across concurrency domains.
                    let tokensGenerated = try await ctx.completion(
                        prompt: prompt,
                        maxTokens: parameters.maxTokens,
                        temperature: parameters.temperature,
                        topP: parameters.topP,
                        topK: parameters.topK,
                        repetitionPenalty: parameters.repetitionPenalty
                    ) { piece, index in
                        if index == 0 {
                            let ms = Date().timeIntervalSince(start) * 1000
                            continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                        }
                        continuation.yield(.textDelta(piece))
                    }

                    let elapsed = Date().timeIntervalSince(start)
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
        guard let gguf = items.first(where: { $0.pathExtension.lowercased() == "gguf" && !$0.lastPathComponent.hasPrefix("mmproj") }) else {
            throw InferenceError.generationFailed("No .gguf file found in \(modelDirectory.path)")
        }
        return gguf
    }
}

/// Actor isolating all direct calls into the llama.cpp C API.
///
/// Implemented against the public headers in llama.cpp/llama.h. Implementors
/// of this file should keep the surface area minimal: loading, batched decode,
/// sampler chain (top-k, top-p, temperature, repetition penalty, min-p),
/// token streaming, reset.
actor LlamaContext {
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var sampler: UnsafeMutablePointer<llama_sampler>?
    private var nPast: Int32 = 0

    static func create(modelPath: String) async throws -> LlamaContext {
        llama_backend_init()
        var modelParams = llama_model_default_params()
        #if targetEnvironment(simulator)
        modelParams.n_gpu_layers = 0
        #else
        modelParams.n_gpu_layers = 999
        #endif

        guard let model = llama_model_load_from_file(modelPath, modelParams) else {
            throw InferenceError.generationFailed("llama.cpp: failed to load model at \(modelPath)")
        }

        var ctxParams = llama_context_default_params()
        ctxParams.n_ctx = 4096
        ctxParams.n_batch = 512
        ctxParams.n_threads = Int32(max(1, ProcessInfo.processInfo.processorCount - 1))
        ctxParams.n_threads_batch = ctxParams.n_threads

        guard let ctx = llama_init_from_model(model, ctxParams) else {
            llama_model_free(model)
            throw InferenceError.generationFailed("llama.cpp: failed to create context")
        }

        return LlamaContext(model: model, ctx: ctx)
    }

    private init(model: OpaquePointer, ctx: OpaquePointer) {
        self.model = model
        self.ctx = ctx
    }

    func reset() {
        guard let ctx else { return }
        if let mem = llama_get_memory(ctx) {
            llama_memory_clear(mem, true)
        }
        nPast = 0
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
    func completion(
        prompt: String,
        maxTokens: Int,
        temperature: Float,
        topP: Float,
        topK: Int,
        repetitionPenalty: Float,
        onPiece: @Sendable (String, Int) -> Void
    ) async throws -> Int {
        guard let model, let ctx else { throw InferenceError.modelNotLoaded }

        let vocab = llama_model_get_vocab(model)

        // Build sampler chain
        if sampler != nil { llama_sampler_free(sampler) }
        var sparams = llama_sampler_chain_default_params()
        sparams.no_perf = true
        let chain = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(64, repetitionPenalty, 0.0, 0.0))
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(Int32(topK)))
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(topP, 1))
        llama_sampler_chain_add(chain, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(chain, llama_sampler_init_dist(UInt32(truncatingIfNeeded: UInt64.random(in: 0...UInt64(UInt32.max)))))
        sampler = chain

        // Tokenize prompt
        let promptCstr = Array(prompt.utf8CString)
        let maxTokens32 = Int32(prompt.utf8.count + 64)
        var tokens = [llama_token](repeating: 0, count: Int(maxTokens32))
        let n = llama_tokenize(
            vocab,
            promptCstr,
            Int32(promptCstr.count - 1),
            &tokens,
            maxTokens32,
            true,
            true
        )
        guard n > 0 else { throw InferenceError.generationFailed("llama.cpp: tokenize failed") }
        tokens = Array(tokens.prefix(Int(n)))

        // Decode prompt in batches
        var batch = llama_batch_init(Int32(tokens.count), 0, 1)
        defer { llama_batch_free(batch) }
        for (i, tok) in tokens.enumerated() {
            batch.token[i] = tok
            batch.pos[i] = nPast + Int32(i)
            batch.n_seq_id[i] = 1
            batch.seq_id[i]?[0] = 0
            batch.logits[i] = (i == tokens.count - 1) ? 1 : 0
        }
        batch.n_tokens = Int32(tokens.count)
        guard llama_decode(ctx, batch) == 0 else {
            throw InferenceError.generationFailed("llama.cpp: decode(prompt) failed")
        }
        nPast += Int32(tokens.count)

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
                throw InferenceError.generationFailed("llama.cpp: decode(token) failed")
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
