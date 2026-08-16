//
//  AppleIntelligenceEngine.swift
//  local_ai_test
//
//  Uses FoundationModels framework for Apple Intelligence on-device inference.
//  Available on iOS 26+ with Apple Intelligence enabled.
//

import Foundation
import FoundationModels

@available(iOS 26, *)
final class AppleIntelligenceEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .textOnly

    private var session: LanguageModelSession?

    var isLoaded: Bool { session != nil }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        guard SystemLanguageModel.default.availability == .available else {
            throw InferenceError.generationFailed("Apple Intelligence is not available on this device.")
        }
        progress(.loadingWeights(0.5))
        session = LanguageModelSession()
        progress(.ready)
    }

    func unload() async {
        session = nil
    }

    func resetConversation() async {
        guard session != nil else { return }
        session = LanguageModelSession()
    }

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated.
            let task = Task { @MainActor [weak self] in
                guard let self, let session = self.session else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }

                do {
                    let prompt = turns.last(where: { $0.role == .user })?.content ?? ""
                    let start = Date()
                    var previousText = ""
                    var firstTokenReported = false

                    for try await snapshot in session.streamResponse(to: prompt) {
                        let fullText = snapshot.content
                        let delta = String(fullText.dropFirst(previousText.count))
                        if !delta.isEmpty {
                            if !firstTokenReported {
                                let ms = Date().timeIntervalSince(start) * 1000
                                continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                                firstTokenReported = true
                            }
                            continuation.yield(.textDelta(delta))
                        }
                        previousText = fullText
                    }
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
