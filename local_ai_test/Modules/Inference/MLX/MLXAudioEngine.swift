//
//  MLXAudioEngine.swift
//  local_ai_test
//
//  Native audio-LLM inference for models like Qwen2-Audio. Accepts raw PCM
//  audio attachments and produces text.
//
//  Qwen2-Audio expects 16 kHz mono f32. The media pipeline (AudioRecorder)
//  already captures in that format. The engine passes audio bytes to the
//  VLM processor which internally computes mel-spectrogram features.
//
//  Requires SPM product `MLXVLM` (Qwen2-Audio is exposed via the same
//  factory as VLMs in mlx-swift-examples).
//

import Foundation

#if canImport(MLXVLM)

import MLX
import MLXVLM
import MLXLMCommon

final class MLXAudioEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .audioLLM

    private let modelDirectory: URL
    private var container: ModelContainer?

    var isLoaded: Bool { container != nil }

    init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        progress(.loadingWeights(0.0))
        MLXTextEngine.applyIOSCacheLimit()
        let config = ModelConfiguration(directory: modelDirectory)
        container = try await VLMModelFactory.shared.loadContainer(configuration: config) { p in
            progress(.loadingWeights(p.fractionCompleted))
        }
        progress(.ready)
    }

    func unload() async {
        container = nil
        MLX.GPU.clearCache()
    }

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
                do {
                    let lastUser = turns.last(where: { $0.role == .user })
                        ?? ChatTurn.user("")

                    // Feeding raw PCM into the Qwen2-Audio processor is not
                    // wired up yet (`UserInput` in the pinned MLXVLM release
                    // has no audio input). Fail loudly instead of silently
                    // answering from text only — the user believes the model
                    // heard the recording.
                    let audioAttachments = lastUser.attachments.contains { att in
                        if case .audio = att { return true }
                        return false
                    }
                    if audioAttachments {
                        throw InferenceError.unsupportedAttachment(
                            "Audio input for MLX audio models is not wired up yet. " +
                            "Use a Whisper model to transcribe speech instead."
                        )
                    }

                    let userInput = UserInput(prompt: .text(lastUser.content), images: [])

                    let start = Date()

                    // Counters live inside the perform closure so no mutable
                    // state is captured across concurrency domains.
                    let tokensGenerated: Int = try await container.perform { context in
                        let input = try await context.processor.prepare(input: userInput)
                        let generateParams = GenerateParameters(
                            maxTokens: parameters.maxTokens,
                            temperature: parameters.temperature,
                            topP: parameters.topP
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

#else

final class MLXAudioEngine: InferenceEngine, @unchecked Sendable {
    let capabilities: ModelCapabilities = .audioLLM
    var isLoaded: Bool { false }
    init(modelDirectory: URL) {}
    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        throw InferenceError.backendUnavailable(
            "MLXAudioEngine requires the MLXVLM SPM product for Qwen2-Audio support."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("MLXVLM not linked")) }
    }
}

#endif
