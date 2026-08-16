//
//  BackendAvailability.swift
//  local_ai_test
//
//  Single source of truth for "is backend X actually linked into this
//  build?". Used by the UI to disable or explain unavailable models
//  without waiting for a runtime load failure.
//
//  Because `canImport` is a lexical check, this file intentionally lives
//  in one place that imports every optional dependency under `#if` so the
//  flags are coherent across the app.
//

import Foundation

enum BackendAvailability {

    // MARK: - Module flags (compile-time)

    static let hasMLXLLM: Bool = {
        #if canImport(MLXLLM)
        return true
        #else
        return false
        #endif
    }()

    static let hasMLXVLM: Bool = {
        #if canImport(MLXVLM)
        return true
        #else
        return false
        #endif
    }()

    static let hasLlama: Bool = {
        #if canImport(llama)
        return true
        #else
        return false
        #endif
    }()

    static let hasWhisper: Bool = {
        #if canImport(whisper)
        return true
        #else
        return false
        #endif
    }()

    static let hasTransformers: Bool = {
        #if canImport(Transformers)
        return true
        #else
        return false
        #endif
    }()

    // MARK: - Backend-level queries

    /// Is this backend fully wired (i.e. its `load()` has a real
    /// implementation, not a scaffolded stub)?
    static func isLinked(_ backend: ModelBackend) -> Bool {
        switch backend {
        case .appleIntelligence:
            return true
        case .mlx:
            return hasMLXLLM
        case .mlxVision, .mlxAudio:
            return hasMLXLLM && hasMLXVLM
        case .coreML:
            return hasTransformers
        case .coreMLWhisper:
            return hasWhisper
        case .coreMLKokoro:
            return true // uses Core ML directly, no external SPM required
        case .llamaCpp:
            return hasLlama
        case .llamaCppVision:
            // LlamaCppVisionEngine + LlamaVisionContext drive libmtmd
            // directly via the C API exposed in `mtmd.h` / `mtmd-helper.h`.
            // Both headers are listed in our vendored modulemap, so as
            // long as the `llama` module is linked, vision works.
            return hasLlama
        case .whisperCpp:
            return hasWhisper
        }
    }

    /// Human-readable one-liner describing what is missing, if anything.
    static func unavailableReason(_ backend: ModelBackend) -> String? {
        guard !isLinked(backend) else { return nil }
        switch backend {
        case .mlxVision, .mlxAudio:
            return "Requires the MLXVLM Swift package. Add https://github.com/ml-explore/mlx-swift-examples and link MLXVLM."
        case .coreML:
            return "Requires swift-transformers. Add https://github.com/huggingface/swift-transformers and link Transformers + Tokenizers."
        case .coreMLWhisper, .whisperCpp:
            return "Requires the whisper.cpp XCFramework. See Scripts/build_whisper_xcframework.sh or add a whisper Swift package."
        case .llamaCpp:
            return "Requires the llama.cpp Swift package. Add https://github.com/ggerganov/llama.cpp (or a built XCFramework exposing the `llama` module)."
        case .llamaCppVision:
            // At this point `isLinked` returned false, which (for this
            // backend) means the whole `llama` module is missing.
            return "Requires the llama.cpp XCFramework built with libmtmd. Run Scripts/build_llama_mtmd_xcframework.sh and link the produced Vendor/llama.xcframework. See MULTI_BACKEND_INTEGRATION.md §3.2b."
        case .mlx, .appleIntelligence, .coreMLKokoro:
            return nil
        }
    }

    /// Short badge text for list rows. Empty when the backend is fully linked.
    static func badgeText(_ backend: ModelBackend) -> String? {
        guard !isLinked(backend) else { return nil }
        switch backend {
        case .llamaCppVision: return "mtmd not linked"
        case .llamaCpp:        return "llama not linked"
        case .whisperCpp,
             .coreMLWhisper:   return "whisper not linked"
        case .mlxVision,
             .mlxAudio:        return "MLXVLM not linked"
        case .coreML:          return "Transformers not linked"
        default:               return "Not linked"
        }
    }
}

extension ModelBackend {
    /// Convenience forwarder so call sites can write `model.backend.isLinked`.
    var isLinked: Bool { BackendAvailability.isLinked(self) }
    var unavailableReason: String? { BackendAvailability.unavailableReason(self) }
    var unavailableBadge: String? { BackendAvailability.badgeText(self) }
}
