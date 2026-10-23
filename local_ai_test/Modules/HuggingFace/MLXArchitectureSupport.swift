//
//  MLXArchitectureSupport.swift
//  local_ai_test
//
//  Which `model_type` strings the MLX engine can actually instantiate.
//
//  The two sets below mirror the model-type registries of the pinned
//  mlx-swift-examples 2.29.1:
//    Libraries/MLXLLM/LLMModelFactory.swift → LLMTypeRegistry.all()
//    Libraries/MLXVLM/VLMModelFactory.swift → VLMTypeRegistry.all()
//  They exist so an unsupported architecture fails in one small request
//  instead of after several gigabytes of download. Bumping the package means
//  re-reading both registries and updating these lists.
//

import Foundation

nonisolated enum MLXArchitectureSupport {

    /// The package release the lists below were transcribed from.
    static let packageVersion = "2.29.1"

    /// `LLMTypeRegistry` keys — text-only models.
    static let textArchitectures: Set<String> = [
        "mistral", "llama", "phi", "phi3", "phimoe", "gemma", "gemma2",
        "gemma3", "gemma3_text", "gemma3n", "qwen2", "qwen3", "qwen3_moe",
        "starcoder2", "cohere", "openelm", "internlm2", "deepseek_v3",
        "granite", "granitemoehybrid", "mimo", "glm4", "acereason",
        "falcon_h1", "bitnet", "smollm3", "ernie4_5", "lfm2", "baichuan_m1",
        "exaone4", "gpt_oss", "lille-130m", "olmoe", "olmo2", "bailing_moe",
        "lfm2_moe", "nanochat"
    ]

    /// `VLMTypeRegistry` keys — vision models.
    static let visionArchitectures: Set<String> = [
        "paligemma", "qwen2_vl", "qwen2_5_vl", "qwen3_vl", "idefics3",
        "gemma3", "smolvlm"
    ]

    static var allArchitectures: Set<String> {
        textArchitectures.union(visionArchitectures)
    }

    /// Matched case-insensitively: config.json is hand-edited often enough to
    /// carry "Qwen2" where the registry says "qwen2".
    static func isSupported(_ modelType: String) -> Bool {
        allArchitectures.contains(
            modelType.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        )
    }
}

// MARK: - Verdict

/// Outcome of the pre-download architecture check for an MLX repo.
nonisolated enum MLXArchitectureVerdict: Sendable, Equatable {
    /// config.json names an architecture the pinned engine can build.
    case supported(String)
    /// The engine has no implementation for it. Downloading would waste
    /// gigabytes on weights that could never load.
    case unsupported(String)
    /// No config.json, or no `model_type` in it. Not a reason to block —
    /// plenty of repos load fine without one, and the load attempt is the
    /// honest fallback.
    case unknown

    var blocksDownload: Bool {
        if case .unsupported = self { return true }
        return false
    }

    /// What to show the user. nil when there is nothing worth saying.
    var message: String? {
        switch self {
        case .supported, .unknown:
            return nil
        case .unsupported(let modelType):
            return "This model's architecture (\(modelType)) is not supported yet by "
                + "the app's MLX engine. A GGUF build of the same model may still "
                + "work, because llama.cpp handles architectures separately."
        }
    }
}
