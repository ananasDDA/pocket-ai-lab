//
//  LlamaArchitectureSupport.swift
//  local_ai_test
//
//  Which `general.architecture` values the bundled llama.cpp can load.
//
//  The set below is llama.cpp's own LLM_ARCH_NAMES table, read out of the
//  pinned b8882 binary (Vendor/llama.xcframework, `strings` on the framework
//  shows the table as one contiguous block that ends at "(unknown)"). A GGUF
//  whose architecture is not in it fails to load with "unknown model
//  architecture", after the whole file has been downloaded. Bumping the
//  llama.cpp pin means regenerating this list from the new binary.
//

import Foundation

nonisolated enum LlamaArchitectureSupport {

    /// The llama.cpp release the list was read from.
    static let llamaRevision = "b8882"

    /// The architecture a vision/audio projector (mmproj) declares. It is in
    /// llama.cpp's table, but only `mtmd` loads it, never as a chat model.
    static let projectorArchitecture = "clip"

    static let architectures: Set<String> = [
        "clip", "llama", "llama4", "deci", "falcon", "grok", "gpt2", "gptj", "gptneox",
        "baichuan", "starcoder", "refact", "bert", "modern-bert", "nomic-bert",
        "nomic-bert-moe", "neo-bert", "jina-bert-v2", "jina-bert-v3", "eurobert", "bloom",
        "stablelm", "qwen", "qwen2", "qwen2moe", "qwen2vl", "qwen3", "qwen3moe",
        "qwen3next", "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "phi2", "phi3",
        "phimoe", "plamo", "plamo2", "plamo3", "codeshell", "orion", "internlm2",
        "minicpm", "minicpm3", "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
        "gemma-embedding", "starcoder2", "mamba", "mamba2", "jamba", "falcon-h1",
        "xverse", "command-r", "cohere2", "dbrx", "olmo", "olmo2", "olmoe", "openelm",
        "arctic", "deepseek", "deepseek2", "deepseek2-ocr", "chatglm", "glm4", "glm4moe",
        "glm-dsa", "bitnet", "t5encoder", "jais", "jais2", "nemotron", "nemotron_h",
        "nemotron_h_moe", "exaone", "exaone4", "exaone-moe", "rwkv6", "rwkv6qwen2",
        "rwkv7", "arwkv7", "granite", "granitemoe", "granitehybrid", "chameleon",
        "wavtokenizer-dec", "bailingmoe", "bailingmoe2", "dots1", "arcee", "afmoe",
        "ernie4_5", "ernie4_5-moe", "hunyuan-moe", "hunyuan-dense", "smollm3", "gpt-oss",
        "lfm2", "lfm2moe", "dream", "smallthinker", "llada", "llada-moe", "seed_oss",
        "grovemoe", "apertus", "minimax-m2", "cogvlm", "rnd1", "pangu-embedded",
        "mistral3", "mistral4", "paddleocr", "mimo2", "step35", "llama-embed",
        "maincoder", "kimi-linear",
    ]

    /// What a GGUF's architecture means for loading it as a chat model.
    enum Verdict: Sendable, Equatable {
        case supported
        /// A projector: adds image input to a model, is not one.
        case projector
        case unsupported(String)
    }

    /// nil architecture (header unreadable, HF did not say) is not a reason
    /// to refuse: the load attempt is the honest fallback.
    static func verdict(for architecture: String?) -> Verdict? {
        guard let raw = architecture?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty
        else { return nil }
        let arch = raw.lowercased()
        if arch == projectorArchitecture { return .projector }
        return architectures.contains(arch) ? .supported : .unsupported(raw)
    }
}
