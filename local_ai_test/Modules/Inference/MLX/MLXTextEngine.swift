//
//  MLXTextEngine.swift
//  local_ai_test
//
//  Text-only MLX inference. Renamed from MLXInferenceEngine; now adopts the
//  unified `InferenceEngine` (ChatTurn + ChatOutput) protocol. Ignores
//  attachments.
//
//  Generation is STATELESS: the full conversation history arrives in `turns`
//  and is converted to `Chat.Message`s each call, exactly like the llama.cpp
//  and Core ML backends. This keeps history consistent when the user switches
//  backends mid-conversation and means generation parameters always apply.
//
//  Requires SPM: https://github.com/ml-explore/mlx-swift-examples
//  Products: MLXLLM, MLXLMCommon
//

import Foundation
import MLX
import MLXLLM
import MLXLMCommon

final class MLXTextEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .textOnly

    private let modelDirectory: URL
    private var container: ModelContainer?

    var isLoaded: Bool { container != nil }

    init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    /// MLX's default buffer-cache limit is sized for Macs (~1.5 × the Metal
    /// working set — nearly 12 GB on a 12 GB iPhone), so on iOS the cache
    /// would never shrink before jetsam kills the process. Apple's own iOS
    /// samples pin it to 20 MB; shared by every MLX engine, applied before
    /// the first weight allocation. Idempotent.
    static func applyIOSCacheLimit() {
        MLX.GPU.set(cacheLimit: 20 * 1024 * 1024)
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        progress(.loadingWeights(0.0))
        Self.applyIOSCacheLimit()
        let config = ModelConfiguration(directory: modelDirectory)
        container = try await LLMModelFactory.shared.loadContainer(configuration: config) { p in
            progress(.loadingWeights(p.fractionCompleted))
        }
        progress(.ready)
    }

    func unload() async {
        container = nil
        MLX.GPU.clearCache()
    }

    /// Stateless engine — history lives with the caller.
    func resetConversation() async {}

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated; the compute
            // happens inside container.perform on the model actor.
            let task = Task { @MainActor [weak self] in
                guard let self, let container = self.container else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }

                // Full history -> Chat.Message. The model's own chat template
                // (from tokenizer_config.json) is applied by the processor.
                let chat: [Chat.Message] = turns.map { turn in
                    switch turn.role {
                    case .system:    return .system(turn.content)
                    case .user:      return .user(turn.content)
                    case .assistant: return .assistant(turn.content)
                    }
                }
                let userInput = UserInput(chat: chat)

                do {
                    let start = Date()

                    // Counters live inside the perform closure so no mutable
                    // state is captured across concurrency domains.
                    let tokensGenerated: Int = try await container.perform { context in
                        let input = try await context.processor.prepare(input: userInput)
                        let generateParams = GenerateParameters(
                            maxTokens: parameters.maxTokens,
                            temperature: parameters.temperature,
                            topP: parameters.topP,
                            repetitionPenalty: parameters.repetitionPenalty
                        )
                        let stream = try MLXLMCommon.generate(
                            input: input,
                            parameters: generateParams,
                            context: context
                        )
                        var firstTokenReported = false
                        var count = 0
                        for await event in stream {
                            if case .chunk(let chunk) = event {
                                if !firstTokenReported {
                                    let ms = Date().timeIntervalSince(start) * 1000
                                    continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                                    firstTokenReported = true
                                }
                                count += 1
                                continuation.yield(.textDelta(chunk))
                            }
                        }
                        return count
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
}
