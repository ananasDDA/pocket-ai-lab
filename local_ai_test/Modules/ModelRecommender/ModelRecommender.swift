//
//  ModelRecommender.swift
//  local_ai_test
//

import Foundation

struct RecommendationResult {
    /// The chat model to start with. nil when no catalog model that can hold
    /// a conversation fits this device (low RAM / full disk), even if a
    /// speech model does.
    let recommended: AIModel?
    /// Every model that fits, chat or not, best first.
    let allCompatible: [AIModel]
    let incompatible: [AIModel]
    let reason: String

    /// `allCompatible` minus the recommended model, which the Setup screen
    /// already shows on its own card. Matched by id: the recommendation is
    /// not necessarily the first compatible model (a better one may fit
    /// without clearing the headroom margin, or may not be a chat model).
    var otherCompatible: [AIModel] {
        guard let recommendedID = recommended?.id else { return allCompatible }
        return allCompatible.filter { $0.id != recommendedID }
    }
}

nonisolated extension ModelBackend {
    /// False for engines that only turn speech into text or text into
    /// speech. They are useful next to a chat model, but recommending one as
    /// "the" model leaves a new user with nothing to talk to.
    var canChat: Bool {
        switch self {
        case .coreMLWhisper, .whisperCpp, .coreMLKokoro:
            return false
        case .appleIntelligence, .mlx, .mlxVision, .mlxAudio, .coreML,
             .llamaCpp, .llamaCppVision:
            return true
        }
    }
}

final class ModelRecommender {

    static let shared = ModelRecommender()
    private init() {}

    /// Fraction of the memory budget a model must stay under to be offered as
    /// *the* recommendation. Everything up to 100% of the budget is still
    /// listed as compatible and installable — but a model that only barely
    /// fits loads and then dies as soon as the KV cache grows, which is a
    /// terrible first experience for someone who just tapped "recommended".
    static let recommendationMargin = 0.85

    /// `models` defaults to the catalog in effect; tests pass the bundled
    /// catalog instead of swapping the shared one.
    func recommend(for device: DeviceInfo, models: [AIModel] = ModelCatalog.all) -> RecommendationResult {
        let ramGB = Double(device.totalRAM) / 1_073_741_824
        let diskGB = Double(device.freeDiskSpace) / 1_073_741_824
        let aiAvailable = device.appleIntelligence == .available

        let (compatible, incompatible) = split(
            models: models,
            device: device,
            diskGB: diskGB,
            aiAvailable: aiAvailable
        )

        let sorted = Self.inListOrder(compatible)

        guard !sorted.isEmpty else {
            return RecommendationResult(
                recommended: nil,
                allCompatible: [],
                incompatible: Self.inListOrder(incompatible, byQuality: false),
                reason: "No model in the catalog fits this device "
                    + "(\(String(format: "%.1f", ramGB)) GB RAM, "
                    + "\(String(format: "%.1f", diskGB)) GB free disk). "
                    + "Free up storage and rescan."
            )
        }

        // Only a current model that can hold a conversation can be "the"
        // model. Speech-to-text entries fit every device, so without this
        // filter a small phone was told to start with Whisper; deprecated
        // entries are kept for people who want them, not offered. Both stay
        // in `allCompatible` and keep their cards.
        let candidates = sorted.filter(Self.isRecommendable)
        guard let bestFitting = candidates.first else {
            return RecommendationResult(
                recommended: nil,
                allCompatible: sorted,
                incompatible: Self.inListOrder(incompatible, byQuality: false),
                reason: noRecommendationReason(
                    compatible: sorted, incompatible: incompatible,
                    device: device, ramGB: ramGB, diskGB: diskGB
                )
            )
        }

        // Prefer the best model that leaves real headroom. If nothing clears
        // the margin (small-RAM devices, where every option is a squeeze),
        // fall back to the best fitting one and say so — recommending
        // nothing at all is worse than recommending a tight fit.
        let comfortable = candidates.first {
            $0.ramRequiredGB <= device.budgetGB(for: $0.backend) * Self.recommendationMargin
        }
        let best = comfortable ?? bestFitting
        let reason = makeReason(model: best, ramGB: ramGB, aiAvailable: aiAvailable)
            + (comfortable == nil
               ? " Memory on this device is tight — expect shorter conversations before it has to reload."
               : "")

        return RecommendationResult(
            recommended: best,
            allCompatible: sorted,
            incompatible: Self.inListOrder(incompatible, byQuality: false),
            reason: reason
        )
    }

    /// Whether a model may be offered as *the* recommendation.
    static func isRecommendable(_ model: AIModel) -> Bool {
        model.backend.canChat && !model.deprecated
    }

    /// Order of every model list on the Setup screen. Deprecated models go
    /// last whatever their quality: within one quality tier the larger model
    /// sorts first, which would otherwise put an old 3.8B above a current
    /// 2.6B. `byQuality: false` keeps catalog order apart from that (the
    /// incompatible lists).
    static func inListOrder(_ models: [AIModel], byQuality: Bool = true) -> [AIModel] {
        models.enumerated().sorted { a, b in
            let (l, r) = (a.element, b.element)
            if l.deprecated != r.deprecated { return !l.deprecated }
            if byQuality {
                if l.quality != r.quality { return l.quality > r.quality }
                if l.ramRequiredGB != r.ramRequiredGB { return l.ramRequiredGB > r.ramRequiredGB }
            }
            return a.offset < b.offset
        }.map(\.element)
    }

    /// Said when something fits but nothing can be recommended: only speech
    /// models, or only deprecated chat models. Points at storage only when a
    /// current chat model is held back by free disk space alone, because
    /// that is the one thing the user can change.
    private func noRecommendationReason(
        compatible: [AIModel],
        incompatible: [AIModel],
        device: DeviceInfo,
        ramGB: Double,
        diskGB: Double
    ) -> String {
        let blockedOnlyByDisk = incompatible.contains {
            Self.isRecommendable($0)
                && BackendAvailability.isLinked($0.backend)
                && !$0.requiresAppleIntelligence
                && $0.ramRequiredGB <= device.budgetGB(for: $0.backend)
                && $0.effectiveDiskSizeGB > diskGB
        }
        let figures = "(\(String(format: "%.1f", ramGB)) GB RAM, "
            + "\(String(format: "%.1f", diskGB)) GB free disk). "
        let fix = blockedOnlyByDisk
            ? "Free up storage and rescan to get one."
            : "This device does not have enough memory for one."

        if compatible.contains(where: { $0.backend.canChat }) {
            return "None of the current chat models fit this device " + figures
                + "The older models below still run here. " + fix
        }
        return "None of the chat models in the catalog fit this device " + figures
            + "The speech models below still work, but none of them can hold a conversation. "
            + fix
    }

    private func split(
        models: [AIModel],
        device: DeviceInfo,
        diskGB: Double,
        aiAvailable: Bool
    ) -> (compatible: [AIModel], incompatible: [AIModel]) {
        var compatible: [AIModel] = []
        var incompatible: [AIModel] = []

        for model in models {
            // A backend that is not linked into this binary can never run,
            // however much RAM the device has. Shipping a recommendation the
            // user cannot load ("Backend not available in this build") is the
            // worst possible first impression, so this check comes first.
            // Note this is a *binary-level* check — Apple Intelligence being
            // switched off by the user is handled separately below.
            if !BackendAvailability.isLinked(model.backend) {
                incompatible.append(model)
                continue
            }
            if model.requiresAppleIntelligence && !aiAvailable {
                incompatible.append(model)
                continue
            }
            // mmap-backed backends (llama.cpp GGUF) are judged against the
            // Metal working set; dirty-memory backends (MLX, Core ML)
            // against the jetsam allowance. See DeviceInfo.budgetGB.
            if model.ramRequiredGB > device.budgetGB(for: model.backend) {
                incompatible.append(model)
                continue
            }
            if model.effectiveDiskSizeGB > diskGB {
                incompatible.append(model)
                continue
            }
            compatible.append(model)
        }

        return (compatible, incompatible)
    }

    private func makeReason(model: AIModel, ramGB: Double, aiAvailable: Bool) -> String {
        switch model.backend {
        case .appleIntelligence:
            return "Apple Intelligence is available — the built-in model is fast and takes no disk space."
        case .mlx:
            return "\(model.name) (\(model.parameterSize), \(model.quantization)) is the best fit for \(Int(ramGB.rounded())) GB RAM. Runs via MLX on the Neural Engine."
        case .mlxVision:
            return "\(model.name) is a vision-language model. Runs via MLX-VLM — accepts photos alongside text."
        case .mlxAudio:
            return "\(model.name) is a native audio-LLM (Qwen2-Audio). Needs 12+ GB RAM to run comfortably."
        case .coreML:
            return "\(model.name) runs through Core ML + swift-transformers. First load compiles on-device (1-3 min)."
        case .coreMLWhisper, .whisperCpp:
            return "\(model.name) transcribes speech locally. Use as an input pipeline for chat."
        case .coreMLKokoro:
            return "\(model.name) synthesizes speech from text. Pairs with any text LLM for voice output."
        case .llamaCpp:
            return "\(model.name) runs via llama.cpp (GGUF). Works on older devices that MLX does not support."
        case .llamaCppVision:
            return "\(model.name) runs via llama.cpp + mmproj — vision-language model in GGUF format."
        }
    }
}

// MARK: - Equatable for AppleIntelligenceStatus

extension AppleIntelligenceStatus: Equatable {
    static func == (lhs: AppleIntelligenceStatus, rhs: AppleIntelligenceStatus) -> Bool {
        switch (lhs, rhs) {
        case (.available, .available): return true
        case (.notEnabled, .notEnabled): return true
        case (.notSupported, .notSupported): return true
        case (.modelNotReady, .modelNotReady): return true
        default: return false
        }
    }
}
