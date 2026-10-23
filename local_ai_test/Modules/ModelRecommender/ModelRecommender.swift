//
//  ModelRecommender.swift
//  local_ai_test
//

import Foundation

struct RecommendationResult {
    /// nil when no catalog model fits this device (low RAM / full disk).
    let recommended: AIModel?
    let allCompatible: [AIModel]
    let incompatible: [AIModel]
    let reason: String
}

final class ModelRecommender {

    static let shared = ModelRecommender()
    private init() {}

    /// Fraction of the memory budget a model must stay under to be offered as
    /// *the* recommendation. Everything up to 100% of the budget is still
    /// listed as compatible and installable — but a model that only barely
    /// fits loads and then dies as soon as the KV cache grows, which is a
    /// terrible first experience for someone who just tapped "recommended".
    private static let recommendationMargin = 0.85

    func recommend(for device: DeviceInfo) -> RecommendationResult {
        let ramGB = Double(device.totalRAM) / 1_073_741_824
        let diskGB = Double(device.freeDiskSpace) / 1_073_741_824
        let aiAvailable = device.appleIntelligence == .available

        let (compatible, incompatible) = split(
            models: ModelCatalog.all,
            device: device,
            diskGB: diskGB,
            aiAvailable: aiAvailable
        )

        let sorted = compatible.sorted {
            if $0.quality != $1.quality { return $0.quality > $1.quality }
            return $0.ramRequiredGB > $1.ramRequiredGB
        }

        guard !sorted.isEmpty else {
            return RecommendationResult(
                recommended: nil,
                allCompatible: [],
                incompatible: incompatible,
                reason: "No model in the catalog fits this device "
                    + "(\(String(format: "%.1f", ramGB)) GB RAM, "
                    + "\(String(format: "%.1f", diskGB)) GB free disk). "
                    + "Free up storage and rescan."
            )
        }

        // Prefer the best model that leaves real headroom. If nothing clears
        // the margin (small-RAM devices, where every option is a squeeze),
        // fall back to the best fitting one and say so — recommending
        // nothing at all is worse than recommending a tight fit.
        let comfortable = sorted.first {
            $0.ramRequiredGB <= device.budgetGB(for: $0.backend) * Self.recommendationMargin
        }
        let best = comfortable ?? sorted[0]
        let reason = makeReason(model: best, ramGB: ramGB, aiAvailable: aiAvailable)
            + (comfortable == nil
               ? " Memory on this device is tight — expect shorter conversations before it has to reload."
               : "")

        return RecommendationResult(
            recommended: best,
            allCompatible: sorted,
            incompatible: incompatible,
            reason: reason
        )
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
