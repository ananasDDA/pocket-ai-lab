//
//  InferenceEngine.swift
//  local_ai_test
//
//  Common protocol for all inference backends (MLX text / MLX vision /
//  MLX audio / Core ML / Core ML Whisper / Core ML Kokoro TTS / llama.cpp /
//  llama.cpp + mmproj / whisper.cpp / Apple Intelligence).
//

import Foundation

enum InferenceError: Error, LocalizedError {
    case modelNotLoaded
    case generationFailed(String)
    case unsupportedAttachment(String)
    case backendUnavailable(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "Model is not loaded into memory."
        case .generationFailed(let msg): return msg
        case .unsupportedAttachment(let msg): return "Attachment unsupported: \(msg)"
        case .backendUnavailable(let msg): return "Backend unavailable: \(msg)"
        case .cancelled: return "Generation was cancelled."
        }
    }
}

/// Produces a stream of `ChatOutput`s for a multimodal chat request.
protocol InferenceEngine: AnyObject, Sendable {
    var capabilities: ModelCapabilities { get }
    var isLoaded: Bool { get }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws

    /// Releases model weights and backend resources. `async` so callers can
    /// await actual deallocation before loading the next model — loading a
    /// new model while the old one is still resident doubles peak RAM and
    /// gets the app jetsam-killed on device.
    func unload() async

    /// Clears any engine-held conversation state (KV cache / session).
    /// `async` so callers can order it before the next `generate`.
    func resetConversation() async

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error>
}

extension InferenceEngine {
    /// Convenience — load without progress callback.
    func load() async throws {
        try await load { _ in }
    }

    /// Default text-only adapter for engines that do not natively handle
    /// attachments. Drops attachments and falls back to the main `generate`.
    /// Vision / audio engines override.
    func generateTextOnly(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        let textOnlyTurns = turns.map {
            ChatTurn(role: $0.role, content: $0.content, attachments: [])
        }
        return generate(turns: textOnlyTurns, parameters: parameters)
    }
}

/// Converts a stream of ChatOutputs into just the textDelta string chunks.
/// Used by legacy code paths that only care about assistant text.
extension AsyncThrowingStream where Element == ChatOutput, Failure == Error {
    func textDeltas() -> AsyncThrowingStream<String, Error> {
        AsyncThrowingStream<String, Error> { continuation in
            let task = Task {
                do {
                    for try await output in self {
                        if case .textDelta(let str) = output {
                            continuation.yield(str)
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
