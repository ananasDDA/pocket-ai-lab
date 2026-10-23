//
//  ModelCatalog.swift
//  local_ai_test
//
//  Curated multimodal catalog across all supported backends.
//  Sources:
//    - mlx-community  (MLX text + MLX-VLM + MLX-Audio weights)
//    - bartowski      (GGUF quantizations for llama.cpp)
//    - apple, coreml-projects, coreml-community  (Core ML bundles)
//    - ggerganov/whisper.cpp  (Whisper ggml weights)
//

import Foundation

enum ModelCatalog {

    /// Catalog in effect: cached remote `catalog.json` → bundled
    /// `catalog.json` → `builtIn`. See `RemoteCatalog`.
    static var all: [AIModel] { CatalogStorage.shared.current }

    /// Compiled-in copy of the shipped catalog. Only used when neither the
    /// bundled resource nor the cache can be read — a corrupt or missing
    /// resource must not leave the app with zero models.
    static let builtIn: [AIModel] = [

        // MARK: - Apple Intelligence (built-in)

        AIModel(
            id: "apple-intelligence",
            name: "Apple Intelligence",
            family: "Apple",
            parameterSize: "~3B",
            quantization: "system",
            backend: .appleIntelligence,
            capabilities: .textOnly,
            fileLayout: .huggingFaceMLX, // unused
            ramRequiredGB: 0,
            diskSizeGB: 0,
            huggingFaceRepo: "",
            requiresAppleIntelligence: true,
            contextLength: 4096,
            quality: .great
        ),

        // MARK: - MLX text (existing)

        AIModel(
            id: "mlx-community/gemma-2-2b-it-4bit",
            name: "Gemma 2 2B",
            family: "Gemma",
            parameterSize: "2B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 1.6, diskSizeGB: 1.5,
            huggingFaceRepo: "mlx-community/gemma-2-2b-it-4bit",
            contextLength: 8192,
            quality: .good,
            licenseName: "Gemma Terms of Use",
            licenseURL: "https://ai.google.dev/gemma/terms"
        ),

        AIModel(
            id: "mlx-community/Phi-3.5-mini-instruct-4bit",
            name: "Phi-3.5 Mini",
            family: "Phi",
            parameterSize: "3.8B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 2.4, diskSizeGB: 2.3,
            huggingFaceRepo: "mlx-community/Phi-3.5-mini-instruct-4bit",
            contextLength: 131072,
            quality: .good
        ),

        AIModel(
            id: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            name: "Llama 3.2 3B",
            family: "Llama",
            parameterSize: "3B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 2.2, diskSizeGB: 2.0,
            huggingFaceRepo: "mlx-community/Llama-3.2-3B-Instruct-4bit",
            contextLength: 131072,
            quality: .good,
            licenseName: "Llama 3.2 Community License",
            licenseURL: "https://www.llama.com/llama3_2/license/"
        ),

        AIModel(
            id: "mlx-community/Qwen2.5-3B-Instruct-4bit",
            name: "Qwen 2.5 3B",
            family: "Qwen",
            parameterSize: "3B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 2.2, diskSizeGB: 2.0,
            huggingFaceRepo: "mlx-community/Qwen2.5-3B-Instruct-4bit",
            contextLength: 32768,
            quality: .good
        ),

        AIModel(
            id: "mlx-community/Llama-3.1-8B-Instruct-4bit",
            name: "Llama 3.1 8B",
            family: "Llama",
            parameterSize: "8B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 4.7, diskSizeGB: 4.9,
            huggingFaceRepo: "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
            contextLength: 131072,
            quality: .great,
            licenseName: "Llama 3.1 Community License",
            licenseURL: "https://www.llama.com/llama3_1/license/"
        ),

        AIModel(
            id: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            name: "Mistral 7B v0.3",
            family: "Mistral",
            parameterSize: "7B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 4.5, diskSizeGB: 4.4,
            huggingFaceRepo: "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
            contextLength: 32768,
            quality: .great
        ),

        AIModel(
            id: "mlx-community/Qwen2.5-14B-Instruct-4bit",
            name: "Qwen 2.5 14B",
            family: "Qwen",
            parameterSize: "14B",
            quantization: "4bit",
            backend: .mlx,
            capabilities: .textOnly,
            ramRequiredGB: 8.2, diskSizeGB: 8.5,
            huggingFaceRepo: "mlx-community/Qwen2.5-14B-Instruct-4bit",
            contextLength: 32768,
            quality: .excellent
        ),

        // MARK: - MLX Vision (VLM)

        AIModel(
            id: "mlx-community/gemma-3-4b-it-4bit",
            name: "Gemma 3 4B (Vision)",
            family: "Gemma",
            parameterSize: "4B",
            quantization: "4bit",
            backend: .mlxVision,
            capabilities: .videoVision,
            ramRequiredGB: 4.8, diskSizeGB: 2.8,
            huggingFaceRepo: "mlx-community/gemma-3-4b-it-4bit",
            contextLength: 131072,
            quality: .great,
            licenseName: "Gemma Terms of Use",
            licenseURL: "https://ai.google.dev/gemma/terms"
        ),

        AIModel(
            id: "mlx-community/gemma-3-12b-it-4bit",
            name: "Gemma 3 12B (Vision)",
            family: "Gemma",
            parameterSize: "12B",
            quantization: "4bit",
            backend: .mlxVision,
            capabilities: .videoVision,
            ramRequiredGB: 8.4, diskSizeGB: 7.5,
            huggingFaceRepo: "mlx-community/gemma-3-12b-it-4bit",
            contextLength: 131072,
            quality: .excellent,
            licenseName: "Gemma Terms of Use",
            licenseURL: "https://ai.google.dev/gemma/terms"
        ),

        AIModel(
            id: "mlx-community/Qwen2-VL-7B-Instruct-4bit",
            name: "Qwen2-VL 7B (Vision)",
            family: "Qwen",
            parameterSize: "7B",
            quantization: "4bit",
            backend: .mlxVision,
            capabilities: .videoVision,
            ramRequiredGB: 5.2, diskSizeGB: 4.7,
            huggingFaceRepo: "mlx-community/Qwen2-VL-7B-Instruct-4bit",
            contextLength: 32768,
            quality: .great
        ),

        // MARK: - MLX Audio (native audio-LLM)

        // MARK: - llama.cpp (GGUF)

        AIModel(
            id: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            name: "Llama 3.2 3B (GGUF)",
            family: "Llama",
            parameterSize: "3B",
            quantization: "Q4_K_M",
            backend: .llamaCpp,
            capabilities: .textOnly,
            fileLayout: .singleGGUF,
            ramRequiredGB: 2.5, diskSizeGB: 2.1,
            huggingFaceRepo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            contextLength: 131072,
            quality: .good,
            preferredGGUFFilename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            licenseName: "Llama 3.2 Community License",
            licenseURL: "https://www.llama.com/llama3_2/license/"
        ),

        AIModel(
            id: "bartowski/Phi-3.5-mini-instruct-GGUF",
            name: "Phi-3.5 Mini (GGUF)",
            family: "Phi",
            parameterSize: "3.8B",
            quantization: "Q4_K_M",
            backend: .llamaCpp,
            capabilities: .textOnly,
            fileLayout: .singleGGUF,
            ramRequiredGB: 2.7, diskSizeGB: 2.4,
            huggingFaceRepo: "bartowski/Phi-3.5-mini-instruct-GGUF",
            contextLength: 131072,
            quality: .good,
            preferredGGUFFilename: "Phi-3.5-mini-instruct-Q4_K_M.gguf"
        ),

        AIModel(
            id: "unsloth/gemma-3-4b-it-GGUF",
            name: "Gemma 3 4B (GGUF, Vision)",
            family: "Gemma",
            parameterSize: "4B",
            quantization: "Q4_K_M",
            backend: .llamaCppVision,
            capabilities: .vision,
            fileLayout: .ggufWithMmproj,
            ramRequiredGB: 5.0, diskSizeGB: 3.4,
            huggingFaceRepo: "unsloth/gemma-3-4b-it-GGUF",
            contextLength: 131072,
            quality: .great,
            preferredGGUFFilename: "gemma-3-4b-it-Q4_K_M.gguf",
            mmprojFilename: "mmproj-F16.gguf",
            licenseName: "Gemma Terms of Use",
            licenseURL: "https://ai.google.dev/gemma/terms"
        ),

        // MARK: - llama.cpp (GGUF), 2026 generation
        //
        // diskSizeGB is the download (weights + projector) in decimal GB
        // rounded up to 0.1. ramRequiredGB is calibrated from measured peaks
        // on an iPhone Air (build-test harness, suite v3, 2026-10-21): the
        // mmap-ed weights plus the peak private footprint, plus 0.4 GB of
        // headroom, to the nearest 0.1 (all figures decimal GB). Those runs
        // are text-only, so for the vision entries an image encode adds on
        // top of the measured peak.
        //
        //   model            weights  peak    -> ramRequiredGB
        //   Qwen 3.5 0.8B    0.53     0.61       1.5
        //   Qwen 3.5 2B      1.28     1.09       2.8
        //   Qwen 3.5 4B      2.74     1.30       4.4
        //   Gemma 4 E2B      3.11     1.49       5.0
        //   LFM2 1.2B        0.73     0.22       1.4
        //   LFM2 2.6B        1.56     0.24       2.2
        //
        // contextLength is what the engines actually open (4096 text, 8192
        // vision), not the model's native window, which is far larger for
        // all of these.
        //
        // Gemma 4 E4B is deliberately absent: its weights alone (about 6 GB
        // with the projector) are over the GGUF budget of an 8 GB iPhone
        // (~5.1 GB).

        AIModel(
            id: "unsloth/gemma-4-E2B-it-GGUF",
            name: "Gemma 4 E2B (GGUF, Vision)",
            family: "Gemma",
            parameterSize: "E2B",
            quantization: "Q4_K_M",
            backend: .llamaCppVision,
            capabilities: .vision,
            fileLayout: .ggufWithMmproj,
            ramRequiredGB: 5.0, diskSizeGB: 4.1,
            huggingFaceRepo: "unsloth/gemma-4-E2B-it-GGUF",
            contextLength: 8192,
            quality: .great,
            preferredGGUFFilename: "gemma-4-E2B-it-Q4_K_M.gguf",
            mmprojFilename: "mmproj-F16.gguf"
        ),

        AIModel(
            id: "unsloth/Qwen3.5-0.8B-GGUF",
            name: "Qwen 3.5 0.8B (GGUF, Vision)",
            family: "Qwen",
            parameterSize: "0.8B",
            quantization: "Q4_K_M",
            backend: .llamaCppVision,
            capabilities: .vision,
            fileLayout: .ggufWithMmproj,
            ramRequiredGB: 1.5, diskSizeGB: 0.8,
            huggingFaceRepo: "unsloth/Qwen3.5-0.8B-GGUF",
            contextLength: 8192,
            quality: .good,
            preferredGGUFFilename: "Qwen3.5-0.8B-Q4_K_M.gguf",
            mmprojFilename: "mmproj-F16.gguf"
        ),

        AIModel(
            id: "unsloth/Qwen3.5-2B-GGUF",
            name: "Qwen 3.5 2B (GGUF, Vision)",
            family: "Qwen",
            parameterSize: "2B",
            quantization: "Q4_K_M",
            backend: .llamaCppVision,
            capabilities: .vision,
            fileLayout: .ggufWithMmproj,
            ramRequiredGB: 2.8, diskSizeGB: 2.0,
            huggingFaceRepo: "unsloth/Qwen3.5-2B-GGUF",
            contextLength: 8192,
            quality: .great,
            preferredGGUFFilename: "Qwen3.5-2B-Q4_K_M.gguf",
            mmprojFilename: "mmproj-F16.gguf"
        ),

        AIModel(
            id: "unsloth/Qwen3.5-4B-GGUF",
            name: "Qwen 3.5 4B (GGUF, Vision)",
            family: "Qwen",
            parameterSize: "4B",
            quantization: "Q4_K_M",
            backend: .llamaCppVision,
            capabilities: .vision,
            fileLayout: .ggufWithMmproj,
            ramRequiredGB: 4.4, diskSizeGB: 3.5,
            huggingFaceRepo: "unsloth/Qwen3.5-4B-GGUF",
            contextLength: 8192,
            quality: .great,
            preferredGGUFFilename: "Qwen3.5-4B-Q4_K_M.gguf",
            mmprojFilename: "mmproj-F16.gguf"
        ),

        AIModel(
            id: "LiquidAI/LFM2-1.2B-GGUF",
            name: "LFM2 1.2B (GGUF)",
            family: "LFM",
            parameterSize: "1.2B",
            quantization: "Q4_K_M",
            backend: .llamaCpp,
            capabilities: .textOnly,
            fileLayout: .singleGGUF,
            ramRequiredGB: 1.4, diskSizeGB: 0.8,
            huggingFaceRepo: "LiquidAI/LFM2-1.2B-GGUF",
            contextLength: 4096,
            quality: .good,
            // The plain Q4_K_M, not the "-hip-optimized" sibling built for
            // AMD GPUs.
            preferredGGUFFilename: "LFM2-1.2B-Q4_K_M.gguf",
            licenseName: "LFM Open License v1.0",
            licenseURL: "https://huggingface.co/LiquidAI/LFM2-1.2B-GGUF/blob/main/LICENSE"
        ),

        AIModel(
            id: "LiquidAI/LFM2-2.6B-GGUF",
            name: "LFM2 2.6B (GGUF)",
            family: "LFM",
            parameterSize: "2.6B",
            quantization: "Q4_K_M",
            backend: .llamaCpp,
            capabilities: .textOnly,
            fileLayout: .singleGGUF,
            ramRequiredGB: 2.2, diskSizeGB: 1.6,
            huggingFaceRepo: "LiquidAI/LFM2-2.6B-GGUF",
            contextLength: 4096,
            quality: .good,
            preferredGGUFFilename: "LFM2-2.6B-Q4_K_M.gguf",
            licenseName: "LFM Open License v1.0",
            licenseURL: "https://huggingface.co/LiquidAI/LFM2-2.6B-GGUF/blob/main/LICENSE"
        ),

        // MARK: - whisper.cpp (ASR)

        AIModel(
            id: "ggerganov/whisper.cpp/tiny.en",
            name: "Whisper tiny (English)",
            family: "Whisper",
            parameterSize: "39M",
            quantization: "ggml",
            backend: .whisperCpp,
            capabilities: .speechToText,
            fileLayout: .whisperGGML,
            ramRequiredGB: 0.5, diskSizeGB: 0.08,
            huggingFaceRepo: "ggerganov/whisper.cpp",
            contextLength: 0,
            quality: .good,
            preferredGGUFFilename: "ggml-tiny.en.bin"
        ),

        AIModel(
            id: "ggerganov/whisper.cpp/base",
            name: "Whisper base",
            family: "Whisper",
            parameterSize: "74M",
            quantization: "ggml",
            backend: .whisperCpp,
            capabilities: .speechToText,
            fileLayout: .whisperGGML,
            ramRequiredGB: 0.8, diskSizeGB: 0.15,
            huggingFaceRepo: "ggerganov/whisper.cpp",
            contextLength: 0,
            quality: .great,
            preferredGGUFFilename: "ggml-base.bin"
        ),

        // MARK: - Core ML (HF apple/ + coreml-community/)

        // `apple/mistral-coreml` used to sit here and cannot come back as-is.
        // It is a WWDC24 Mac demo: the repo holds two `.mlpackage`s and
        // NOTHING else - no tokenizer.json, no tokenizer_config.json, no
        // config.json - so `AutoTokenizer.from(modelFolder:)` fails with
        // "Required configuration file missing: tokenizer.json" no matter how
        // well the Core ML side works. (Measured on an iPhone Air, iOS 26.6.1:
        // the 3.8 GB Int4 package downloads and compiles in 0.3 s, then the
        // load dies on the tokenizer.) Apple's own swift-chat demo borrows the
        // tokenizer from `mistralai/Mistral-7B-Instruct-v0.3`, which is gated
        // and cannot be fetched unauthenticated. Apple's blog for this very
        // model also says outright that they would not run Mistral 7B on an
        // iPhone.
        //
        // The replacement is a stateful export whose Core ML feature names
        // match the ones swift-transformers hard-codes (`inputIds`,
        // `causalMask`, states `keyCache`/`valueCache`, output `logits`) and
        // which ships its own tokenizer. Most Core ML LLM exports on the Hub
        // use snake_case names instead and would trip `LanguageModel`'s
        // force-unwrap of `inputDescriptionsByName["inputIds"]`.
        AIModel(
            id: "anziank/grio-qwen2.5-1.5b-instruct-coreml-stateful-int8",
            name: "Qwen2.5 1.5B (Core ML)",
            family: "Qwen",
            parameterSize: "1.5B",
            quantization: "int8-W8A16",
            backend: .coreML,
            capabilities: .textOnly,
            fileLayout: .coreMLPackage,
            ramRequiredGB: 2.5, diskSizeGB: 2.3,
            huggingFaceRepo: "anziank/grio-qwen2.5-1.5b-instruct-coreml-stateful-int8",
            // The export is built for a 512-token window: `inputIds` is
            // constrained to [1, 1...512]. Claiming more would let the prompt
            // builder overrun the model's own shape constraint.
            contextLength: 512,
            quality: .good,
            // The repo publishes the COMPILED `.mlmodelc` next to the
            // `.mlpackage`, so there is nothing to compile on the phone. That
            // is Apple's own advice (compile off-device, ship the bundle) and
            // it is also the smaller install: the package would land at
            // 1.65 GB and then leave a 2.27 GB compiled copy beside it.
            requiresCompilation: false,
            // Third-party repo: pin the commit so a re-upload cannot change
            // what a half-finished download is resuming into.
            revision: "a9442469f80fc807d5f2e3a138a69fb023f9f7d7",
            // Must be pinned: the repo holds both containers, and the
            // smallest-wins fallback would pick the `.mlpackage` and put the
            // compile back.
            preferredPackage: "Qwen2.5-1.5B-Instruct-stateful-int8.mlmodelc"
        ),
    ]

    static func model(id: String) -> AIModel? {
        all.first { $0.id == id }
    }

    static func models(forBackendFamily family: BackendFamily) -> [AIModel] {
        all.filter { $0.backend.family == family }
    }

    static func models(matching caps: ModelCapabilities) -> [AIModel] {
        all.filter { !$0.capabilities.intersection(caps).isEmpty }
    }
}
