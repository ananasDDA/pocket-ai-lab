//
//  CoreMLTextEngine.swift
//  local_ai_test
//
//  Generic Core ML LLM runner built on top of huggingface/swift-transformers
//  (https://github.com/huggingface/swift-transformers). Downloads .mlpackage
//  through our own `ModelDownloader`, compiles via `CoreMLCompiler`, and
//  generates tokens through `LanguageModel.generate`.
//

import Foundation

#if canImport(Transformers)

import CoreML
import Transformers
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

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        let mlpackage = try locateMLPackage()
        progress(.compiling(0.0))
        let compiledURL = try await CoreMLCompiler.compileIfNeeded(
            mlpackageURL: mlpackage,
            cacheDirectory: modelDirectory,
            estimatedDurationSeconds: 90
        ) { p in progress(.compiling(p)) }

        progress(.loadingWeights(0.1))
        // Actually apply the compute-units configuration — constructing
        // LanguageModel(url:) directly would silently drop it.
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndGPU

        let tokenizer = try await AutoTokenizer.from(modelFolder: modelDirectory)
        let mlModel = try MLModel(contentsOf: compiledURL, configuration: config)
        let model = LanguageModel(model: mlModel)

        self.tokenizer = tokenizer
        self.languageModel = model
        progress(.ready)
    }

    func unload() async {
        languageModel = nil
        tokenizer = nil
    }

    func resetConversation() async {}

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated.
            let task = Task { @MainActor [weak self] in
                guard let self, let model = self.languageModel, let tokenizer = self.tokenizer else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                do {
                    let prompt: String
                    if let templated = try? tokenizer.applyChatTemplate(messages: turns.map {
                        ["role": $0.role.rawValue, "content": $0.content]
                    }) {
                        prompt = templated
                    } else {
                        let family = PromptFormatter.detectFamily(modelId: self.modelId)
                        prompt = PromptFormatter.format(turns: turns, family: family)
                    }

                    let config = GenerationConfig(
                        maxNewTokens: parameters.maxTokens,
                        doSample: parameters.temperature > 0,
                        temperature: Double(parameters.temperature),
                        topP: Double(parameters.topP),
                        repetitionPenalty: Double(parameters.repetitionPenalty)
                    )

                    let start = Date()
                    var previous = ""
                    var tokensGenerated = 0
                    var firstTokenReported = false

                    _ = try await model.generate(config: config, prompt: prompt) { partial in
                        let delta = String(partial.dropFirst(previous.count))
                        if !delta.isEmpty {
                            if !firstTokenReported {
                                let ms = Date().timeIntervalSince(start) * 1000
                                continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                                firstTokenReported = true
                            }
                            tokensGenerated += 1
                            continuation.yield(.textDelta(delta))
                        }
                        previous = partial
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

    private func locateMLPackage() throws -> URL {
        let items = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        guard let pkg = items.first(where: { $0.pathExtension == "mlpackage" }) else {
            throw InferenceError.generationFailed("No .mlpackage found in \(modelDirectory.path)")
        }
        return pkg
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
            "Add https://github.com/huggingface/swift-transformers and link Transformers + Tokenizers."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("swift-transformers not linked")) }
    }
}

#endif
