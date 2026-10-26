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
import Tokenizers
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

                    let chat = Self.chat(from: turns, images: ciImages)
                    // Gemma-style templates that reject a system role get
                    // the instructions folded into the first user turn.
                    let foldedChat: [Chat.Message]? = turns.contains(where: { $0.role == .system })
                        ? Self.chat(from: SystemPrompt.foldedIntoFirstUser(turns), images: ciImages)
                        : nil

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
                            MLXTextEngine.renderedPrompt(input: input, tokenizer: context.tokenizer)
                        )))
                        // Temperature and topP only (`MLXSampling`).
                        let mapped = MLXSampling.map(parameters, passRepetitionPenalty: false)
                        let generateParams = GenerateParameters(
                            maxTokens: parameters.maxTokens,
                            temperature: mapped.temperature,
                            topP: mapped.topP
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

    /// Build the full history as Chat.Messages with the images attached to
    /// the LAST user message. The legacy `UserInput(prompt:images:)` path
    /// silently drops images for some VLM processors (Gemma 3) in the pinned
    /// mlx-swift-examples release - the chat-based input is the path Apple's
    /// own VLMEval sample uses.
    static func chat(from turns: [ChatTurn], images: [CIImage]) -> [Chat.Message] {
        let lastUser = turns.lastIndex(where: { $0.role == .user })
        return turns.enumerated().map { index, turn in
            switch turn.role {
            case .system:
                return .system(turn.content)
            case .assistant:
                return .assistant(turn.content)
            case .user:
                return .user(
                    turn.content,
                    images: index == lastUser ? images.map { .ciImage($0) } : []
                )
            }
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
