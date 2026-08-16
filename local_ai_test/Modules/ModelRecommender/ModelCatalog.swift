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

        AIModel(
            id: "apple/mistral-coreml",
            name: "Mistral 7B (Core ML)",
            family: "Mistral",
            parameterSize: "7B",
            quantization: "palettized-4bit",
            backend: .coreML,
            capabilities: .textOnly,
            fileLayout: .coreMLPackage,
            ramRequiredGB: 4.2, diskSizeGB: 4.3,
            huggingFaceRepo: "apple/mistral-coreml",
            contextLength: 4096,
            quality: .great,
            requiresCompilation: true
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
