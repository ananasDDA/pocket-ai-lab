//
//  MLXVisionEngine.swift
//  local_ai_test
//
//  Vision-Language Model inference via MLXVLM (Gemma 3, Qwen2-VL, LLaVA,
//  MiniCPM-V). Accepts images as attachments; videos are converted to a
//  sequence of frames by the caller (VideoFrameExtractor).
//
//  Requires SPM product `MLXVLM` from:
//     https://github.com/ml-explore/mlx-swift-examples
//
//  When MLXVLM is not linked, the class still exists as a stub that throws
//  at load time (keeps the project compiling without the dependency).
//

import Foundation

#if canImport(MLXVLM)

import MLX
import MLXVLM
import MLXLMCommon
import UIKit

final class MLXVisionEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .videoVision

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

                    let images = self.collectImages(from: lastUser)
                    let ciImages: [CIImage] = images.compactMap { data in
                        guard let ui = UIImage(data: data), let cg = ui.cgImage else { return nil }
                        return CIImage(cgImage: cg)
                    }

                    // Build the full history as Chat.Messages with the images
                    // attached to the LAST user message. The legacy
                    // `UserInput(prompt:images:)` path silently drops images
                    // for some VLM processors (Gemma 3) in the pinned
                    // mlx-swift-examples release — the chat-based input is
                    // the path Apple's own VLMEval sample uses.
                    var chat: [Chat.Message] = []
                    for (index, turn) in turns.enumerated() {
                        let isLastUser = turn.role == .user
                            && index == turns.lastIndex(where: { $0.role == .user })
                        switch turn.role {
                        case .system:
                            chat.append(.system(turn.content))
                        case .assistant:
                            chat.append(.assistant(turn.content))
                        case .user:
                            chat.append(.user(
                                turn.content,
                                images: isLastUser ? ciImages.map { .ciImage($0) } : []
                            ))
                        }
                    }
                    let userInput = UserInput(chat: chat)

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

    private func collectImages(from turn: ChatTurn) -> [Data] {
        var images: [Data] = []
        for attachment in turn.attachments {
            switch attachment {
            case .image(let data):
                images.append(data)
            case .videoFrames(let frames, _):
                images.append(contentsOf: frames)
            case .audio:
                continue
            }
        }
        return images
    }
}

#else

final class MLXVisionEngine: InferenceEngine, @unchecked Sendable {
    let capabilities: ModelCapabilities = .videoVision
    var isLoaded: Bool { false }
    init(modelDirectory: URL) {}
    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        throw InferenceError.backendUnavailable(
            "MLXVisionEngine requires the MLXVLM SPM product. Add 'MLXVLM' to your target in Xcode."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("MLXVLM not linked")) }
    }
}

#endif
