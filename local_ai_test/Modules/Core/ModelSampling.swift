//
//  ModelSampling.swift
//  local_ai_test
//
//  Per-model sampling from the catalog. Model authors recommend very
//  different settings (LFM2: temperature 0.3 with min-p 0.15; Qwen 3.5:
//  top-k 20 with a presence penalty; Gemma: top-k 64, no repetition
//  penalty), while the chat used to send one global set
//  (`GenerationParameters.chat`) to every model. A catalog entry may now
//  carry a `sampling` object; every member is optional and only the members
//  that are set replace the global value.
//
//  The object travels in `catalog.json` and is decoded with
//  `decodeIfPresent`, so catalogs without it decode exactly as before and
//  builds that do not know the key (1.1) ignore it. No schema bump.
//

import Foundation

nonisolated struct ModelSampling: Codable, Hashable, Sendable {
    var temperature: Float?
    var topP: Float?
    var topK: Int?
    /// llama.cpp only (min-p sampler). Ignored by MLX, Core ML and Apple.
    var minP: Float?
    var repetitionPenalty: Float?
    /// llama.cpp only (presence term of the penalties sampler, over the
    /// same 64-token window as the repetition penalty). Ignored elsewhere.
    var presencePenalty: Float?

    init(
        temperature: Float? = nil,
        topP: Float? = nil,
        topK: Int? = nil,
        minP: Float? = nil,
        repetitionPenalty: Float? = nil,
        presencePenalty: Float? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.repetitionPenalty = repetitionPenalty
        self.presencePenalty = presencePenalty
    }

    var isEmpty: Bool {
        temperature == nil && topP == nil && topK == nil && minP == nil
            && repetitionPenalty == nil && presencePenalty == nil
    }
}

nonisolated extension ModelSampling {

    /// The sampling that applies to `model` right now. An installed model is
    /// a snapshot taken when it was installed (the registry), so a curated
    /// model follows the catalog in effect - that is how values published
    /// later in the remote catalog reach models installed before them, and
    /// how removing them there takes them away again. An imported model, or
    /// a curated one the catalog no longer lists, uses its own descriptor.
    static func effective(for model: AIModel?, catalogEntry: AIModel?) -> ModelSampling? {
        guard let model else { return nil }
        if model.source == .curated, let catalogEntry, catalogEntry.id == model.id {
            return catalogEntry.sampling
        }
        return model.sampling
    }
}

nonisolated extension GenerationParameters {

    /// `self` with the members `sampling` sets. Values a remote catalog could
    /// get wrong are ignored rather than trusted: non-finite numbers,
    /// negative temperatures and penalties, a top-p outside (0, 1], a top-k
    /// below 1 and a min-p outside [0, 1). A repetition penalty of 0 would
    /// divide by zero, so it must be positive.
    func applying(_ sampling: ModelSampling?) -> GenerationParameters {
        guard let sampling else { return self }
        var p = self
        if let v = sampling.temperature, v.isFinite, v >= 0 { p.temperature = v }
        if let v = sampling.topP, v.isFinite, v > 0, v <= 1 { p.topP = v }
        if let v = sampling.topK, v >= 1 { p.topK = v }
        if let v = sampling.minP, v.isFinite, v >= 0, v < 1 { p.minP = v }
        if let v = sampling.repetitionPenalty, v.isFinite, v > 0 { p.repetitionPenalty = v }
        if let v = sampling.presencePenalty, v.isFinite, v >= 0 { p.presencePenalty = v }
        return p
    }
}
