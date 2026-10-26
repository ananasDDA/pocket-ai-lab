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
import Tokenizers

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
                let chat: [Chat.Message] = Self.chat(from: turns)
                // Some templates refuse a system role outright (Gemma 2's
                // raises "System role not supported"). The same instructions
                // folded into the first user turn are what those models were
                // trained to read instead.
                let foldedChat: [Chat.Message]? = turns.contains(where: { $0.role == .system })
                    ? Self.chat(from: SystemPrompt.foldedIntoFirstUser(turns))
                    : nil

                do {
                    let start = Date()

                    // Counters live inside the perform closure so no mutable
                    // state is captured across concurrency domains.
                    let tokensGenerated: Int = try await container.perform { context in
                        let input: LMInput
                        do {
                            input = try await context.processor.prepare(input: UserInput(chat: chat))
                        } catch {
                            guard let foldedChat else { throw error }
                            input = try await context.processor.prepare(input: UserInput(chat: foldedChat))
                        }
                        continuation.yield(.diagnostic(.promptRendered(
                            Self.renderedPrompt(input: input, tokenizer: context.tokenizer)
                        )))
                        // `parameters.topK`, `minP`, `presencePenalty` and
                        // `seed` are dropped (`MLXSampling`): MLX's
                        // `GenerateParameters` has no such fields, and its
                        // samplers build a private `MLXRandom.RandomState` from
                        // the clock rather than reading the global PRNG, so
                        // `MLXRandom.seed` would not reach them. Harmless for
                        // the benchmark, which runs at temperature 0 - MLX then
                        // samples with argmax and neither value applies.
                        let mapped = MLXSampling.map(parameters, passRepetitionPenalty: true)
                        let generateParams = GenerateParameters(
                            maxTokens: parameters.maxTokens,
                            temperature: mapped.temperature,
                            topP: mapped.topP,
                            repetitionPenalty: mapped.repetitionPenalty
                        )
                        let stream = try MLXLMCommon.generate(
                            input: input,
                            parameters: generateParams,
                            context: context
                        )
                        var firstTokenReported = false
                        var count = 0
                        var reported: Int?
                        for await event in stream {
                            switch event {
                            case .chunk(let chunk):
                                if !firstTokenReported {
                                    let ms = Date().timeIntervalSince(start) * 1000
                                    continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                                    firstTokenReported = true
                                }
                                count += 1
                                continuation.yield(.textDelta(chunk))
                            case .info(let info):
                                // Tokens, not chunks: a chunk is withheld
                                // while a multi-byte character is incomplete,
                                // and the reply-cap check needs the real count.
                                reported = info.generationTokenCount
                            default:
                                break
                            }
                        }
                        return reported ?? count
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

    static func chat(from turns: [ChatTurn]) -> [Chat.Message] {
        turns.map { turn in
            switch turn.role {
            case .system:    return .system(turn.content)
            case .user:      return .user(turn.content)
            case .assistant: return .assistant(turn.content)
            }
        }
    }

    /// The prepared prompt decoded back to text, special tokens included,
    /// for the prompt inspector. It is what the model reads, after the
    /// model's own template ran.
    nonisolated static func renderedPrompt(input: LMInput, tokenizer: any Tokenizer) -> RenderedPrompt {
        let ids = input.text.tokens.asArray(Int32.self).map(Int.init)
        return RenderedPrompt(
            text: tokenizer.decode(tokens: ids, skipSpecialTokens: false),
            source: .model,
            template: nil
        )
    }
}
