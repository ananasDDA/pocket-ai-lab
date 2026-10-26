//
//  MLXSampling.swift
//  local_ai_test
//
//  What of `GenerationParameters` reaches MLX. The pinned mlx-swift-examples
//  2.29.1 `GenerateParameters` has temperature, topP, repetitionPenalty and
//  repetitionContextSize, and nothing else: topK, minP, the presence
//  penalty and the seed are dropped (its samplers also draw from a private
//  clock-seeded random state). Kept free of MLX types so it can be tested.
//
//  The repetition penalty is passed by the text engine only, with MLX's
//  default 20-token context; the vision and audio engines never passed it
//  and still do not, so a model's `sampling.repetitionPenalty` does not
//  reach them either.
//

import Foundation

nonisolated struct MLXSampling: Equatable, Sendable {
    var temperature: Float
    var topP: Float
    /// nil: no repetition processor (MLX's own default).
    var repetitionPenalty: Float?

    static func map(_ parameters: GenerationParameters, passRepetitionPenalty: Bool) -> MLXSampling {
        MLXSampling(
            temperature: parameters.temperature,
            topP: parameters.topP,
            repetitionPenalty: passRepetitionPenalty ? parameters.repetitionPenalty : nil
        )
    }
}
