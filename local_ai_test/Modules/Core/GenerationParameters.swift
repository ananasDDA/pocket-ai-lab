//
//  GenerationParameters.swift
//  local_ai_test
//
//  Unified sampling / decoding parameters. Each backend maps these to its
//  native configuration object.
//

import Foundation

struct GenerationParameters: Sendable, Hashable {
    var temperature: Float
    var topP: Float
    var topK: Int
    var repetitionPenalty: Float
    var maxTokens: Int
    var seed: UInt64?
    /// Min-p cut-off; 0 is off (no min-p sampler in the chain). Only
    /// llama.cpp applies it.
    var minP: Float = 0
    /// Presence penalty; 0 is off. Only llama.cpp applies it.
    var presencePenalty: Float = 0

    // nonisolated: the project builds with MainActor default isolation;
    // these constants are read from engine tasks off the main actor.
    nonisolated static let `default` = GenerationParameters(
        temperature: 0.7,
        topP: 0.9,
        topK: 40,
        repetitionPenalty: 1.1,
        maxTokens: 1024,
        seed: nil
    )

    /// What the chat screen sends. The same sampling as `default` with a
    /// 512-token reply cap: graded daily-use answers never needed more than
    /// about 600 tokens, and the ones that ran to the old 1024 were rambling.
    /// In a 4096-token llama.cpp window the cap is also what is reserved for
    /// the reply (`ContextBudget`), so the 512 tokens it frees go to
    /// conversation history. Benchmarks never use it - they set their own
    /// caps on `deterministic`. A model's catalog `sampling` is merged over
    /// it per message (`applying(_:)`, `ChatViewModel.parameters(for:)`).
    nonisolated static let chat: GenerationParameters = {
        var p = GenerationParameters.default
        p.maxTokens = 512
        return p
    }()

    nonisolated static let deterministic = GenerationParameters(
        temperature: 0.0,
        topP: 1.0,
        topK: 1,
        repetitionPenalty: 1.0,
        maxTokens: 1024,
        seed: 42
    )
}
