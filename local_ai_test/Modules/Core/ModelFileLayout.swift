//
//  ModelFileLayout.swift
//  local_ai_test
//
//  Describes how a model is laid out on disk after download. The downloader
//  uses this to decide which files to fetch and which to skip; the engine
//  uses it to locate weights / tokenizer / auxiliary assets.
//

import Foundation

nonisolated enum ModelFileLayout: String, Sendable, Codable {
    /// Classic HuggingFace MLX layout: safetensors + config.json + tokenizer files.
    /// Downloader grabs everything except .gguf and well-known junk.
    case huggingFaceMLX

    /// One .gguf file + (optional) tokenizer / chat template JSON.
    /// Downloader selects a single quantization file by name.
    case singleGGUF

    /// Multimodal GGUF: main .gguf + mmproj-*.gguf (vision projector).
    case ggufWithMmproj

    /// Apple Core ML: .mlpackage (directory) + tokenizer files. Requires
    /// on-device compilation via `MLModel.compileModel(at:)`.
    case coreMLPackage

    /// Bundled whisper.cpp ggml binary.
    case whisperGGML
}
