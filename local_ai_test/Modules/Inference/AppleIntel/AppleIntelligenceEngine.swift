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

    /// Instructions the live session was built with, so `generate` can tell
    /// whether the caller's system turn still matches it. FoundationModels
    /// takes instructions at session construction and offers no way to change
    /// them afterwards, so a different system prompt means a new session.
    private var sessionInstructions: String?

    var isLoaded: Bool { session != nil }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        guard SystemLanguageModel.default.availability == .available else {
            throw InferenceError.generationFailed("Apple Intelligence is not available on this device.")
        }
        progress(.loadingWeights(0.5))
        sessionInstructions = nil
        session = Self.makeSession(instructions: nil)
        progress(.ready)
    }

    func unload() async {
        session = nil
        sessionInstructions = nil
    }

    func resetConversation() async {
        guard session != nil else { return }
        // The instructions survive a reset - only the transcript is dropped.
        session = Self.makeSession(instructions: sessionInstructions)
    }

    /// `LanguageModelSession(instructions:)` is the framework's system-prompt
    /// channel; passing nil is the same as the plain initializer.
    private static func makeSession(instructions: String?) -> LanguageModelSession {
        LanguageModelSession(instructions: instructions)
    }

    /// Maps the app's unified sampling parameters onto FoundationModels'
    /// `GenerationOptions`.
    ///
    /// `GenerationOptions` exposes exactly three knobs: `sampling`,
    /// `temperature` and `maximumResponseTokens`. Temperature 0 is expressed
    /// as `.greedy` rather than as `temperature: 0` — greedy is the framework's
    /// own name for argmax decoding and is what makes the benchmark's
    /// deterministic profile mean the same thing here as on the other
    /// backends. Above 0, `topK` and `seed` are carried by
    /// `.random(top:seed:)`. `topP`, `repetitionPenalty`, `minP` and
    /// `presencePenalty` have no counterpart in the framework and are dropped.
    ///
    /// TODO: for chat, passing nil `sampling` and nil `temperature` lets the
    /// system choose its own defaults, which is what Apple recommends. Not
    /// changed in 1.2: this keeps today's options (top-k 40 at 0.7) unless
    /// the catalog entry carries `sampling`.
    private static func generationOptions(
        from parameters: GenerationParameters
    ) -> GenerationOptions {
        guard parameters.temperature > 0 else {
            return GenerationOptions(
                sampling: .greedy,
                maximumResponseTokens: parameters.maxTokens
            )
        }
        return GenerationOptions(
            sampling: .random(top: max(1, parameters.topK), seed: parameters.seed),
            temperature: Double(parameters.temperature),
            maximumResponseTokens: parameters.maxTokens
        )
    }

    /// What the inspector shows for this backend. There is no transcript
    /// string: the framework takes the instructions once per session, keeps
    /// the earlier turns itself, and receives only the new message.
    nonisolated static func renderedPrompt(instructions: String?, prompt: String) -> RenderedPrompt {
        RenderedPrompt(
            text: "[Session instructions]\n\(instructions ?? "(none)")\n\n[Message]\n\(prompt)",
            source: .session,
            template: nil
        )
    }

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated.
            let task = Task { @MainActor [weak self] in
                guard let self, self.session != nil else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }

                // The system turn is not part of the prompt: FoundationModels
                // carries it as session instructions, which is where the
                // framework applies its own weighting. A caller that changes
                // the system prompt mid-conversation therefore gets a fresh
                // session rather than a silently ignored instruction.
                let wanted = SystemPrompt.instructions(in: turns)
                if wanted != self.sessionInstructions {
                    self.sessionInstructions = wanted
                    self.session = Self.makeSession(instructions: wanted)
                }
                guard let session = self.session else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }

                do {
                    let prompt = turns.last(where: { $0.role == .user })?.content ?? ""
                    continuation.yield(.diagnostic(.promptRendered(
                        Self.renderedPrompt(instructions: wanted, prompt: prompt)
                    )))
                    let options = Self.generationOptions(from: parameters)
                    let start = Date()
                    var previousText = ""
                    var firstTokenReported = false

                    for try await snapshot in session.streamResponse(to: prompt, options: options) {
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
