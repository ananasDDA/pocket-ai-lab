//
//  CoreMLTextEngine.swift
//  local_ai_test
//
//  Generic Core ML LLM runner built on top of huggingface/swift-transformers
//  (https://github.com/huggingface/swift-transformers). Finds the model on
//  disk, hands it to `MLModel`, and drives it token-by-token through
//  `LanguageModel`.
//
//  TWO WAYS A MODEL GETS HERE. Catalog entries download an already-compiled
//  `.mlmodelc` from the Hub and load it as-is - no on-device compile at all.
//  A `.mlpackage` imported from Files still goes through `CoreMLCompiler`
//  once. `load` handles both and prefers whatever is already compiled.
//
//  WHAT A COMPATIBLE MODEL LOOKS LIKE. swift-transformers hard-codes the Core
//  ML feature names it drives: input `inputIds`, optional `causalMask`, output
//  `logits`, and (for the KV-cached path) states `keyCache` and `valueCache`.
//  `LanguageModel` force-unwraps `inputIds`, so a model exported with the more
//  common snake_case names (`input_ids`, `key_cache`) does not merely fail -
//  it traps. `validateIO` below turns that into a thrown error, but a catalog
//  entry should be checked against these names BEFORE it is added.
//
//  NOTE ON MODULE NAMES: swift-transformers exposes a *product* called
//  "Transformers", but that product has no module of its own — it vends the
//  three targets `Tokenizers`, `Generation` and `Models`. `canImport(Transformers)`
//  is therefore always false, which silently compiled this whole engine out of
//  every build (including App Store releases) until it was caught by
//  BackendLinkageTests. Guard on the real module names instead.
//

import Foundation

#if canImport(Models) && canImport(Generation) && canImport(Tokenizers)

import CoreML
import Generation
import Models
import Tokenizers

final class CoreMLTextEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .textOnly

    private let modelDirectory: URL
    private let modelId: String
    private var languageModel: LanguageModel?
    private var tokenizer: (any Tokenizer)?

    var isLoaded: Bool { languageModel != nil && tokenizer != nil }

    init(modelDirectory: URL, modelId: String) {
        self.modelDirectory = modelDirectory
        self.modelId = modelId
    }

    // MARK: - Load / unload

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        let compiledURL: URL
        if let prebuilt = CoreMLCompiler.locateCompiledModel(in: modelDirectory) {
            // The repo shipped an already-compiled `.mlmodelc`, or a previous
            // load compiled one. Either way there is nothing to do: hand it
            // straight to Core ML. This is the path every catalog entry takes.
            progress(.compiling(1.0))
            compiledURL = prebuilt
        } else {
            // Import path: the user brought a bare `.mlpackage` in from Files.
            let mlpackage = try locateMLPackage()
            progress(.compiling(0.0))
            // Same size-scaled estimate the downloader uses, so the bar moves
            // at the same rate whichever path triggered the compile. Measured
            // on an iPhone Air: a 3.8 GB package compiles in well under a
            // second, so the estimate is deliberately pessimistic rather than
            // wrong in the direction that matters.
            let packageBytes = CoreMLCompiler.directorySize(of: mlpackage)
            compiledURL = try await CoreMLCompiler.compileIfNeeded(
                mlpackageURL: mlpackage,
                cacheDirectory: modelDirectory,
                estimatedDurationSeconds: 90 + 120 * Double(packageBytes) / 1_073_741_824
            ) { p in progress(.compiling(p)) }
        }

        // The tokenizer files (tokenizer.json / tokenizer_config.json /
        // config.json) are downloaded next to the .mlpackage by
        // `ModelDownloader.filterFiles(_:for:)`, so the model directory *is*
        // the "model folder" swift-transformers expects. Loading it locally
        // also keeps `LanguageModel`'s own lazy `tokenizer` (which would hit
        // the network via LanguageModelConfigurationFromHub) out of the path.
        progress(.loadingWeights(0.1))
        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)

        progress(.loadingWeights(0.5))
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndGPU
        // `LanguageModel.modelName` calls fatalError() when the model has
        // neither the `co.huggingface.exporters.name` metadata key nor a
        // display name, and `init` reads it eagerly. Supplying our catalog id
        // makes that path total.
        configuration.modelDisplayName = modelId

        let mlModel = try MLModel(contentsOf: compiledURL, configuration: configuration)
        try Self.validateIO(of: mlModel)

        self.tokenizer = tokenizer
        self.languageModel = Self.makeLanguageModel(for: mlModel)
        progress(.ready)
    }

    func unload() async {
        languageModel = nil
        tokenizer = nil
    }

    /// Stateless engine — the full history arrives in `turns` and the Core ML
    /// KV-cache state is reset at the start of every `generate`.
    func resetConversation() async {}

    // MARK: - Generation

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated. The Core ML
            // prediction itself runs off-thread inside `MLModel.prediction`,
            // this task only drives the sampling loop.
            let task = Task { @MainActor [weak self] in
                guard let self, let model = self.languageModel, let tokenizer = self.tokenizer else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                do {
                    let contextLimit = max(model.maxContextLength, 1)
                    var tokens = Self.promptTokens(turns: turns, tokenizer: tokenizer, modelId: self.modelId)
                    guard !tokens.isEmpty else {
                        throw InferenceError.generationFailed("Prompt tokenized to zero tokens.")
                    }
                    // Drop the oldest history when the prompt does not fit,
                    // keeping room for an actual answer rather than a single
                    // token.
                    tokens = Self.trimToContext(
                        tokens,
                        contextLimit: contextLimit,
                        maxNewTokens: parameters.maxTokens,
                        bosTokenId: tokenizer.bosTokenId
                    )
                    let promptTokenCount = tokens.count
                    let budget = min(parameters.maxTokens, contextLimit - promptTokenCount)
                    guard budget > 0 else {
                        throw InferenceError.generationFailed(
                            "Prompt uses the model's entire \(contextLimit)-token context window."
                        )
                    }

                    var config = GenerationConfig(
                        maxLength: promptTokenCount + budget,
                        maxNewTokens: budget,
                        doSample: parameters.temperature > 0,
                        temperature: Double(parameters.temperature),
                        topK: max(1, parameters.topK),
                        topP: Double(parameters.topP),
                        repetitionPenalty: Double(parameters.repetitionPenalty)
                    )
                    config.eosTokenId = tokenizer.eosTokenId
                    config.bosTokenId = tokenizer.bosTokenId

                    // Required by `LanguageModelWithStatefulKVCache`, which
                    // fatalErrors on an uninitialized MLState.
                    await model.resetState()

                    let promptStart = Date()
                    var sampler = TokenSampler(parameters: parameters, seed: parameters.seed)
                    var generated: [Int] = []
                    var emitted = ""
                    var firstTokenReported = false
                    var promptReported = false
                    let start = Date()

                    while generated.count < budget {
                        if Task.isCancelled { throw InferenceError.cancelled }

                        let input = MLTensor(tokens.map { Int32($0) }).expandingShape(at: 0)
                        let scores = await Self.shieldedPrediction(model, input: input, config: config)
                        let logits = await scores.cast(to: Float.self).shapedArray(of: Float.self).scalars

                        if !promptReported {
                            continuation.yield(.diagnostic(.promptProcessed(
                                tokenCount: promptTokenCount,
                                timeMs: Date().timeIntervalSince(promptStart) * 1000
                            )))
                            promptReported = true
                        }
                        if Task.isCancelled { throw InferenceError.cancelled }

                        let next = sampler.selectToken(logits: logits, penalizing: tokens)
                        if let eos = config.eosTokenId, next == eos { break }

                        tokens.append(next)
                        generated.append(next)

                        if !firstTokenReported {
                            continuation.yield(.diagnostic(.firstTokenLatency(
                                ms: Date().timeIntervalSince(start) * 1000
                            )))
                            firstTokenReported = true
                        }

                        // Re-decode the whole generated run each step: a single
                        // Cyrillic/CJK/emoji character is frequently split over
                        // several tokens, and decoding those in isolation yields
                        // U+FFFD. An incomplete tail is held back until the next
                        // token completes it.
                        let full = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
                        guard full.hasPrefix(emitted) else {
                            emitted = full
                            continue
                        }
                        let delta = String(full.dropFirst(emitted.count))
                        if delta.isEmpty || delta.hasSuffix("\u{FFFD}") { continue }
                        emitted = full
                        continuation.yield(.textDelta(delta))
                    }

                    // Flush a held-back tail (lossy only for a genuinely
                    // truncated multi-byte sequence).
                    let finalText = tokenizer.decode(tokens: generated, skipSpecialTokens: true)
                    if finalText.hasPrefix(emitted), finalText.count > emitted.count {
                        continuation.yield(.textDelta(String(finalText.dropFirst(emitted.count))))
                    }

                    let elapsed = Date().timeIntervalSince(start)
                    continuation.yield(.diagnostic(.finished(
                        tokensGenerated: generated.count,
                        tokensPerSecond: elapsed > 0 ? Double(generated.count) / elapsed : 0
                    )))
                    continuation.finish()
                } catch {
                    continuation.finish(
                        throwing: Task.isCancelled || error is CancellationError
                            ? InferenceError.cancelled
                            : (error as? InferenceError) ?? InferenceError.generationFailed(error.localizedDescription)
                    )
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: - Context budgeting

    /// Trims a prompt to the model's context window while RESERVING room for
    /// the reply.
    ///
    /// Trimming to `contextLimit - 1` (what this used to do) is technically
    /// "it fits", but it buys a one-token answer — and the shipped Core ML
    /// entry has a 512-token window, so a normal chat history reaches that
    /// cliff in a couple of turns and the model appears to answer with a
    /// single word. The reservation is `min(maxNewTokens, contextLimit / 2)`:
    /// the caller's own reply budget when it is modest, capped at half the
    /// window so a 1024-token default cannot gut a 512-token prompt.
    ///
    /// A leading BOS token survives the trim — the sequence is re-rooted on
    /// it rather than starting mid-history — because a chat template puts it
    /// first and models trained with it degrade without it.
    nonisolated static func trimToContext(
        _ tokens: [Int],
        contextLimit: Int,
        maxNewTokens: Int,
        bosTokenId: Int?
    ) -> [Int] {
        let limit = max(contextLimit, 1)
        let reserved = max(1, min(maxNewTokens, limit / 2))
        let keep = max(1, limit - reserved)
        guard tokens.count > keep else { return tokens }

        if let bos = bosTokenId, tokens.first == bos, keep > 1 {
            return [bos] + tokens.suffix(keep - 1)
        }
        return Array(tokens.suffix(keep))
    }

    // MARK: - Prompt building

    /// Prefers the model's own chat template (from `tokenizer_config.json` /
    /// `chat_template.jinja`); falls back to our hand-rolled `PromptFormatter`
    /// for Core ML exports that ship no template.
    ///
    /// swift-transformers 1.0.0 returns token *ids* from `applyChatTemplate`,
    /// not a rendered string, so the two paths converge on `[Int]`.
    private static func promptTokens(
        turns: [ChatTurn],
        tokenizer: any Tokenizer,
        modelId: String
    ) -> [Int] {
        if tokenizer.hasChatTemplate {
            let messages: [Tokenizers.Message] = turns.map {
                ["role": $0.role.rawValue, "content": $0.content]
            }
            if let ids = try? tokenizer.applyChatTemplate(messages: messages), !ids.isEmpty {
                return ids
            }
        }
        let family = PromptFormatter.detectFamily(modelId: modelId)
        return tokenizer.encode(text: PromptFormatter.format(turns: turns, family: family))
    }

    // MARK: - Cancellation shield

    /// Runs one prediction where task cancellation cannot reach it.
    ///
    /// swift-transformers wraps `MLModel.prediction` in `try!`
    /// (Models/LanguageModel.swift, both the plain and the stateful path).
    /// An async Core ML prediction awaited from a cancelled task throws
    /// `CancellationError`, and `try!` turns that into a fatal error - so
    /// every cancelled Core ML generation killed the app: a benchmark task
    /// timing out (two overnight sweeps died on exactly that), or the Stop
    /// button in the chat.
    ///
    /// A detached task inherits no cancellation, so the prediction always
    /// completes; the sampling loop checks `Task.isCancelled` right after
    /// and stops then. The cost is one extra decode step (well under a
    /// second) before a cancel takes effect.
    private static func shieldedPrediction(
        _ model: LanguageModel,
        input: MLTensor,
        config: GenerationConfig
    ) async -> MLTensor {
        await Task.detached(priority: .userInitiated) {
            await model.predictNextTokenScores(input, config: config)
        }.value
    }

    // MARK: - Model construction

    /// Mirrors `LanguageModel.loadCompiled(url:computeUnits:)`, which we cannot
    /// use directly because it builds its own `MLModelConfiguration` (dropping
    /// `modelDisplayName`) and because its KV-cache probe calls `fatalError()`
    /// for shapes it does not support. Here the same decision is made from the
    /// public model description and unsupported models throw instead.
    private static func makeLanguageModel(for mlModel: MLModel) -> LanguageModel {
        let base = LanguageModel(model: mlModel)
        let description = mlModel.modelDescription
        let hasStatefulKVCache = description.stateDescriptionsByName["keyCache"] != nil
            && description.stateDescriptionsByName["valueCache"] != nil
        // `LanguageModelWithStatefulKVCache.init` requires a ranged query
        // sequence length so it can pre-fill and then extend one token at a time.
        guard hasStatefulKVCache, base.maxContextLength - base.minContextLength > 1 else {
            return base
        }
        return LanguageModelWithStatefulKVCache(model: mlModel)
    }

    /// Turns the library's force-unwraps and `fatalError`s over model I/O into
    /// recoverable errors — a malformed download must not crash the app.
    private static func validateIO(of mlModel: MLModel) throws {
        let description = mlModel.modelDescription
        guard let inputIds = description.inputDescriptionsByName["inputIds"] else {
            throw InferenceError.generationFailed(
                "This Core ML model has no `inputIds` input and cannot be used for text generation."
            )
        }
        guard inputIds.multiArrayConstraint != nil else {
            throw InferenceError.generationFailed(
                "This Core ML model does not declare a shape for `inputIds`."
            )
        }
        guard description.outputDescriptionsByName["logits"] != nil else {
            throw InferenceError.generationFailed(
                "This Core ML model has no `logits` output and cannot be used for text generation."
            )
        }
        // KV cache passed through model I/O (rather than Core ML state) is not
        // implemented by swift-transformers 1.0.0; running it would produce a
        // missing-input crash on the first prediction.
        let hasStatefulKVCache = description.stateDescriptionsByName["keyCache"] != nil
            && description.stateDescriptionsByName["valueCache"] != nil
        if !hasStatefulKVCache, description.inputDescriptionsByName["keyCache"] != nil {
            throw InferenceError.generationFailed(
                "This Core ML model uses an input/output KV cache, which this app's Core ML engine does not support."
            )
        }
    }

    /// Finds the downloaded package by scanning — the directory name comes
    /// from the repo, so it is never hardcoded. The choice is delegated to
    /// `CoreMLCompiler.locatePackage` so that the downloader compiles exactly
    /// the package this engine goes on to load; the two used to disagree
    /// whenever a directory held more than one variant.
    private func locateMLPackage() throws -> URL {
        guard let package = CoreMLCompiler.locatePackage(in: modelDirectory) else {
            throw InferenceError.generationFailed("No .mlpackage found in \(modelDirectory.path)")
        }
        return package
    }
}

// MARK: - Sampling

/// Next-token selection for the Core ML path.
///
/// swift-transformers 1.0.0 samples inside its own `Generation` loop, which
/// has no cancellation check, ignores `topP` and `repetitionPenalty`, and
/// hands the callback the decoded prompt *plus* completion on every step. We
/// therefore drive `predictNextTokenScores` ourselves and sample here, which
/// also lets `GenerationParameters.seed` make a run reproducible.
private struct TokenSampler {
    private let temperature: Float
    private let topK: Int
    private let topP: Float
    private let repetitionPenalty: Float
    private let greedy: Bool
    private var state: UInt64

    init(parameters: GenerationParameters, seed: UInt64?) {
        temperature = parameters.temperature
        topK = max(1, parameters.topK)
        topP = min(max(parameters.topP, 0), 1)
        repetitionPenalty = parameters.repetitionPenalty
        greedy = parameters.temperature <= 0 || parameters.topK == 1
        state = seed ?? UInt64.random(in: 1...UInt64.max)
    }

    /// `penalizing` is the full token history (prompt + completion), matching
    /// the HuggingFace `RepetitionPenaltyLogitsProcessor` convention.
    mutating func selectToken(logits: [Float], penalizing history: [Int]) -> Int {
        guard !logits.isEmpty else { return 0 }
        var scores = logits

        if repetitionPenalty != 1.0 {
            for token in Set(history) where token >= 0 && token < scores.count {
                let score = scores[token]
                scores[token] = score > 0 ? score / repetitionPenalty : score * repetitionPenalty
            }
        }

        guard !greedy else { return argmax(scores) }

        // Temperature, then top-k, then nucleus (top-p), then multinomial draw.
        let scaled = scores.map { $0 / temperature }
        var candidates = scaled.enumerated()
            .sorted { $0.element > $1.element }
            .prefix(min(topK, scaled.count))
            .map { (token: $0.offset, logit: $0.element) }
        guard let best = candidates.first?.logit else { return argmax(scores) }

        var probabilities = candidates.map { Float(exp(Double($0.logit - best))) }
        let total = probabilities.reduce(0, +)
        guard total > 0 else { return argmax(scores) }
        probabilities = probabilities.map { $0 / total }

        if topP < 1 {
            var cumulative: Float = 0
            var cutoff = probabilities.count
            for (index, p) in probabilities.enumerated() {
                cumulative += p
                if cumulative >= topP { cutoff = index + 1; break }
            }
            candidates = Array(candidates.prefix(cutoff))
            probabilities = Array(probabilities.prefix(cutoff))
        }

        let draw = Float(nextUnitRandom()) * probabilities.reduce(0, +)
        var running: Float = 0
        for (index, p) in probabilities.enumerated() {
            running += p
            if draw <= running { return candidates[index].token }
        }
        return candidates.last?.token ?? argmax(scores)
    }

    private func argmax(_ scores: [Float]) -> Int {
        var bestIndex = 0
        var bestValue = scores[0]
        for index in 1..<scores.count where scores[index] > bestValue {
            bestValue = scores[index]
            bestIndex = index
        }
        return bestIndex
    }

    /// SplitMix64 — small, seedable and self-contained, so a supplied
    /// `GenerationParameters.seed` reproduces a run exactly.
    private mutating func nextUnitRandom() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z = z ^ (z >> 31)
        return Double(z >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }
}

#else

final class CoreMLTextEngine: InferenceEngine, @unchecked Sendable {
    let capabilities: ModelCapabilities = .textOnly
    var isLoaded: Bool { false }
    init(modelDirectory: URL, modelId: String) {}
    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        throw InferenceError.backendUnavailable(
            "CoreMLTextEngine requires the 'swift-transformers' SPM package. " +
            "Add https://github.com/huggingface/swift-transformers and link its Transformers product " +
            "(modules Models + Generation + Tokenizers)."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("swift-transformers not linked")) }
    }
}

#endif
