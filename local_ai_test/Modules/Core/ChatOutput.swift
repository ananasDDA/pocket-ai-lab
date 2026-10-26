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
        /// The exact string the engine is about to feed the model, and the
        /// template it came from. Emitted once, before the first token, so
        /// the UI can show what was really sent - history, markers, media
        /// placeholders and all.
        case promptRendered(RenderedPrompt)
        case promptProcessed(tokenCount: Int, timeMs: Double)
        case firstTokenLatency(ms: Double)
        case finished(tokensGenerated: Int, tokensPerSecond: Double)
        /// How the engine got the prompt into its KV cache: what it took
        /// from the previous turn and what it had to decode. Emitted once
        /// per generation, after the reply, by the llama.cpp engines.
        case promptCache(PromptCacheReport)
    }
}

/// What an engine with a KV prefix cache did with one prompt.
///
/// Counts are in tokens; an image counts as the embedding tokens it decodes
/// to. `reusedTokens + evaluatedTokens == promptTokens`.
nonisolated struct PromptCacheReport: Sendable, Equatable {
    /// The prompt as decoded, after anything the engine trimmed to fit.
    var promptTokens: Int
    /// Taken from the KV cache left by the previous turn.
    var reusedTokens: Int
    /// Decoded in this call.
    var evaluatedTokens: Int
    /// Images (media chunks) taken from the cache / encoded in this call.
    var reusedMedia: Int = 0
    var evaluatedMedia: Int = 0
    /// Tokens the engine cut from the front of the rendered prompt to fit
    /// its window. A last-resort safety net in the text engine, which
    /// trims by whole turns first; 0 when nothing was cut.
    var droppedFrontTokens: Int = 0
    /// Whole turns the engine left out of the prompt before rendering it
    /// (both llama.cpp engines drop older exchanges; 0 when none were).
    var droppedTurns: Int = 0
    /// Tokens removed from inside a single turn that was still too long
    /// once every older exchange was gone (its middle is cut, its role
    /// header kept); 0 when no turn was cut.
    var truncatedTokens: Int = 0
    /// The first user turn reached the model whole: not dropped (the
    /// engines keep a short first exchange while older ones go) and not
    /// cut inside.
    var firstUserTurnIntact: Bool = true
    /// How the cache was brought in line with the prompt.
    var rewind: Rewind

    nonisolated enum Rewind: String, Sendable, Equatable {
        /// Nothing was cached; the whole prompt was decoded.
        case empty
        /// The cache was cut back to the shared prefix.
        case prefix
        /// A recurrent-state snapshot was restored and the cache cut there.
        case checkpoint
        /// Nothing usable was shared, or the memory could not be cut back
        /// to it: the cache was cleared and the prompt decoded in full.
        case cleared
    }

    /// Whether the first user turn reached the model whole: kept by the
    /// turn-level fit, not cut inside, and not lost to a front token trim.
    var keptFirstUserTurn: Bool { firstUserTurnIntact && droppedFrontTokens == 0 }
}
