//
//  ChatOutput.swift
//  local_ai_test
//
//  Output type streamed from an InferenceEngine. Unified so both text-LLMs
//  and TTS engines can be routed through the same pipeline.
//

import Foundation

enum ChatOutput: Sendable {
    /// Incremental text chunk (token or token group).
    case textDelta(String)

    /// Raw PCM audio chunk. `sampleRate` comes from the engine
    /// (Kokoro = 24 kHz, Whisper does not produce audio output).
    case audioChunk(Data, sampleRate: Int, channels: Int)

    /// Signals a meaningful pipeline milestone for diagnostics (e.g. prompt
    /// processing finished, first token emitted). Non-fatal, UI ignores by default.
    case diagnostic(Diagnostic)

    enum Diagnostic: Sendable, Equatable {
        case promptProcessed(tokenCount: Int, timeMs: Double)
        case firstTokenLatency(ms: Double)
        case finished(tokensGenerated: Int, tokensPerSecond: Double)
    }
}
