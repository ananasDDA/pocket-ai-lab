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

    /// swift-transformers ships a *product* named "Transformers", but that
    /// product vends three separate modules — `Tokenizers`, `Generation` and
    /// `Models` — and no module called `Transformers` exists. Checking for the
    /// product name here reported `false` in every build ever shipped, which
    /// silently compiled `CoreMLTextEngine` down to its unavailable stub.
    /// These are the three modules the engine actually imports.
    static let hasTransformers: Bool = {
        #if canImport(Models) && canImport(Generation) && canImport(Tokenizers)
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

    /// User-facing, plain-English explanation of why a model using this
    /// backend can't be used right now. Written for someone who has never
    /// heard of an XCFramework or a build script — no shell commands, no
    /// file names, no internal doc references. This is the string every
    /// existing call site (alerts, banners, error sheets) already reads
    /// through `ModelBackend.unavailableReason`, so keeping it friendly here
    /// keeps it friendly everywhere without touching those call sites.
    static func unavailableReason(_ backend: ModelBackend) -> String? {
        guard !isLinked(backend) else { return nil }
        switch backend {
        case .mlxVision, .mlxAudio:
            return "This model's vision and audio engine is not available in this version of the app. Please update the app or pick a model with a different engine."
        case .coreML:
            return "This model type is not available in this version of the app. Please update the app or pick a model with a different engine."
        case .coreMLWhisper, .whisperCpp:
            return "Speech transcription is not available in this version of the app. Please update the app or pick a different model."
        case .llamaCpp:
            return "This model's engine is not available in this version of the app. Please update the app or pick a model with a different engine."
        case .llamaCppVision:
            // At this point `isLinked` returned false, which (for this
            // backend) means the whole `llama` module is missing.
            return "This model's image understanding is not available in this version of the app. Please update the app or pick a model with a different engine."
        case .mlx:
            return "This model's engine is not available in this version of the app. Please update the app or pick a model with a different engine."
        case .appleIntelligence, .coreMLKokoro:
            return nil
        }
    }

    /// Developer-facing detail: the exact missing dependency, build script,
    /// or doc section to consult. Deliberately kept out of
    /// `unavailableReason` so it never reaches an end user by accident —
    /// surface this only in debugging/diagnostics UI (e.g. gated behind
    /// `#if DEBUG`, or in a copy-for-support report), never in a plain
    /// alert or banner.
    ///
    /// No call site wires this in yet — `EngineFailure` (InferenceManager.swift)
    /// only carries a display-name `String` for its backend, not a
    /// `ModelBackend`, so there is nowhere trivial to look this up from the
    /// error detail sheet today. Kept intentionally unreferenced for the
    /// diagnostics view and DEBUG logging call sites that will want it once
    /// that plumbing exists.
    static func developerDetail(_ backend: ModelBackend) -> String? {
        guard !isLinked(backend) else { return nil }
        switch backend {
        case .mlxVision, .mlxAudio:
            return "Requires the MLXVLM Swift package. Add https://github.com/ml-explore/mlx-swift-examples and link MLXVLM."
        case .coreML:
            return "Requires swift-transformers. Add https://github.com/huggingface/swift-transformers and link its Transformers product (modules Models + Generation + Tokenizers)."
        case .coreMLWhisper, .whisperCpp:
            return "Requires the whisper.cpp XCFramework. See Scripts/build_whisper_xcframework.sh or add a whisper Swift package."
        case .llamaCpp:
            return "Requires the llama.cpp Swift package. Add https://github.com/ggerganov/llama.cpp (or a built XCFramework exposing the `llama` module)."
        case .llamaCppVision:
            return "Requires the llama.cpp XCFramework built with libmtmd. Run Scripts/build_llama_mtmd_xcframework.sh and link the produced Vendor/llama.xcframework. See MULTI_BACKEND_INTEGRATION.md §3.2b."
        case .mlx, .appleIntelligence, .coreMLKokoro:
            return nil
        }
    }

    /// Short badge text for list rows. Empty when the backend is fully linked.
    /// Kept generic/user-safe (no internal module names like "mtmd") since
    /// it renders next to installed models in the model picker.
    static func badgeText(_ backend: ModelBackend) -> String? {
        guard !isLinked(backend) else { return nil }
        switch backend {
        case .llamaCppVision:     return "Vision engine unavailable"
        case .llamaCpp:           return "Engine unavailable"
        case .whisperCpp,
             .coreMLWhisper:      return "Speech engine unavailable"
        case .mlxVision,
             .mlxAudio:           return "Vision engine unavailable"
        case .coreML:             return "Engine unavailable"
        default:                  return "Not available"
        }
    }
}

extension ModelBackend {
    /// Convenience forwarder so call sites can write `model.backend.isLinked`.
    var isLinked: Bool { BackendAvailability.isLinked(self) }
    /// User-facing reason text. Safe to show directly in any alert, banner,
    /// or card caption.
    var unavailableReason: String? { BackendAvailability.unavailableReason(self) }
    /// Developer-facing detail (script names, doc references). Only use
    /// this in debugging/diagnostics contexts, never in ordinary UI.
    var developerDetail: String? { BackendAvailability.developerDetail(self) }
    var unavailableBadge: String? { BackendAvailability.badgeText(self) }
}
