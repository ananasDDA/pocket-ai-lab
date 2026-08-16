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

    nonisolated static let deterministic = GenerationParameters(
        temperature: 0.0,
        topP: 1.0,
        topK: 1,
        repetitionPenalty: 1.0,
        maxTokens: 1024,
        seed: 42
    )
}
