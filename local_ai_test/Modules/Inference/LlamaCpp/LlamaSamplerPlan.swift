//
//  LlamaSamplerPlan.swift
//  local_ai_test
//
//  The llama.cpp sampler chain as plain data, shared by the text engine
//  (`LlamaContext`) and the vision engine (`LlamaVisionContext`), so the
//  chain can be tested without llama.cpp. The engines turn each step into
//  the matching `llama_sampler_init_*` call (b8882 signatures).
//
//  Order: penalties -> top-k -> top-p -> min-p -> temperature -> dist, the
//  order llama.cpp's own common sampler uses. min-p is added only when it is
//  set (> 0), and the presence penalty rides in the penalties sampler, so for
//  `GenerationParameters` that leave both at 0 the chain is exactly the one
//  the engines built before per-model sampling existed.
//

import Foundation

nonisolated enum LlamaSamplerStep: Hashable, Sendable {
    /// `llama_sampler_init_penalties(lastN, repeat, frequency, presence)`.
    case penalties(lastN: Int32, repeat: Float, frequency: Float, presence: Float)
    /// `llama_sampler_init_top_k(k)`.
    case topK(Int32)
    /// `llama_sampler_init_top_p(p, minKeep)`.
    case topP(Float, minKeep: Int)
    /// `llama_sampler_init_min_p(p, minKeep)`.
    case minP(Float, minKeep: Int)
    /// `llama_sampler_init_temp(t)`.
    case temperature(Float)
    /// `llama_sampler_init_dist(seed)`; each engine supplies its own seed.
    case dist
}

nonisolated enum LlamaSamplerPlan {

    /// Tokens the repetition and presence penalties look back over.
    static let penaltyWindow: Int32 = 64

    static func steps(for parameters: GenerationParameters) -> [LlamaSamplerStep] {
        var steps: [LlamaSamplerStep] = [
            .penalties(
                lastN: penaltyWindow,
                repeat: parameters.repetitionPenalty,
                frequency: 0,
                presence: parameters.presencePenalty
            ),
            .topK(Int32(clamping: parameters.topK)),
            .topP(parameters.topP, minKeep: 1),
        ]
        if parameters.minP > 0 {
            steps.append(.minP(parameters.minP, minKeep: 1))
        }
        steps.append(.temperature(parameters.temperature))
        steps.append(.dist)
        return steps
    }

    /// One line per chain, in llama.cpp's argument order, for tests and logs.
    static func describe(_ steps: [LlamaSamplerStep]) -> String {
        steps.map { step -> String in
            switch step {
            case let .penalties(lastN, rep, freq, presence):
                return "penalties(\(lastN), \(rep), \(freq), \(presence))"
            case let .topK(k):
                return "top_k(\(k))"
            case let .topP(p, minKeep):
                return "top_p(\(p), \(minKeep))"
            case let .minP(p, minKeep):
                return "min_p(\(p), \(minKeep))"
            case let .temperature(t):
                return "temp(\(t))"
            case .dist:
                return "dist"
            }
        }.joined(separator: " -> ")
    }
}
