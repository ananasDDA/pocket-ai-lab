//
//  ImportedModelBuilder.swift
//  local_ai_test
//
//  Builds the `AIModel` descriptor for a user-imported repo. Curated entries
//  are hand-tuned; these have to be guessed from the repo name and file sizes,
//  so every heuristic here degrades to something harmless rather than nil.
//

import Foundation

enum ImportedModelBuilder {

    /// Families the app has prompt formatting for, longest name first so that
    /// "smollm" is not matched as "llm" by a shorter entry.
    private static let knownFamilies: [(needle: String, name: String)] = [
        ("deepseek", "DeepSeek"), ("smollm", "SmolLM"), ("stablelm", "StableLM"),
        ("tinyllama", "TinyLlama"), ("codellama", "CodeLlama"),
        ("llama", "Llama"), ("qwen", "Qwen"), ("gemma", "Gemma"),
        ("mistral", "Mistral"), ("mixtral", "Mixtral"), ("phi", "Phi"),
        ("whisper", "Whisper"), ("kokoro", "Kokoro"), ("falcon", "Falcon"),
        ("yi-", "Yi"), ("olmo", "OLMo"), ("granite", "Granite"),
        ("internlm", "InternLM"), ("minicpm", "MiniCPM"), ("openelm", "OpenELM")
    ]

    private static let parameterPattern = try? NSRegularExpression(
        pattern: "([0-9]+(?:\\.[0-9]+)?)\\s*[bB](?![a-zA-Z0-9])"
    )

    static func makeModel(
        analysis: ImportAnalysis,
        format: ImportFormat,
        quant: GGUFQuantOption?,
        estimate: ResourceEstimate,
        contextLength: Int?
    ) -> AIModel {
        let repo = analysis.repo
        let quantLabel = quant?.label ?? defaultQuantizationLabel(for: format.layout)

        // Two quantizations of the same repo must be installable side by side,
        // so the id — which also names the download directory — carries it.
        let id = quant.map { "\(repo)#\($0.label)" } ?? repo

        return AIModel(
            id: id,
            name: displayName(repo: repo, quant: quant),
            family: family(from: repo),
            parameterSize: parameterSize(from: repo),
            quantization: quantLabel,
            backend: format.backend,
            capabilities: capabilities(for: format.backend),
            fileLayout: format.layout,
            ramRequiredGB: (estimate.ramGB * 10).rounded() / 10,
            diskSizeGB: (estimate.diskGB * 10).rounded() / 10,
            huggingFaceRepo: repo,
            contextLength: contextLength ?? 4096,
            quality: .good,
            requiresCompilation: format.layout == .coreMLPackage,
            preferredGGUFFilename: preferredFilename(
                format: format, quant: quant, analysis: analysis
            ),
            mmprojFilename: format.layout == .ggufWithMmproj ? analysis.mmprojFilename : nil,
            source: .imported,
            revision: analysis.revision
        )
    }

    // MARK: - Heuristics

    static func displayName(repo: String, quant: GGUFQuantOption?) -> String {
        var stem = repo.split(separator: "/").last.map(String.init) ?? repo
        for suffix in ["-GGUF", "-gguf", "_GGUF", "-MLX", "-mlx", "-CoreML", "-coreml"]
        where stem.hasSuffix(suffix) {
            stem = String(stem.dropLast(suffix.count))
            break
        }
        let pretty = stem
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .trimmingCharacters(in: .whitespaces)
        let base = pretty.isEmpty ? repo : pretty
        guard let quant else { return base }
        return "\(base) (\(quant.label))"
    }

    static func family(from repo: String) -> String {
        let haystack = repo.lowercased()
        for candidate in knownFamilies where haystack.contains(candidate.needle) {
            return candidate.name
        }
        // Fall back to the owner, which is at least a real grouping.
        return repo.split(separator: "/").first.map(String.init) ?? "Imported"
    }

    static func parameterSize(from repo: String) -> String {
        guard let regex = parameterPattern else { return "?" }
        let name = repo.split(separator: "/").last.map(String.init) ?? repo
        let range = NSRange(name.startIndex..<name.endIndex, in: name)
        guard let match = regex.matches(in: name, range: range).first,
              let digits = Range(match.range(at: 1), in: name)
        else { return "?" }
        return String(name[digits]) + "B"
    }

    private static func capabilities(for backend: ModelBackend) -> ModelCapabilities {
        switch backend {
        case .mlxVision, .llamaCppVision:            return .vision
        case .mlxAudio:                              return .audioLLM
        case .whisperCpp, .coreMLWhisper:            return .speechToText
        case .coreMLKokoro:                          return .textToSpeech
        case .mlx, .llamaCpp, .coreML, .appleIntelligence: return .textOnly
        }
    }

    private static func preferredFilename(
        format: ImportFormat,
        quant: GGUFQuantOption?,
        analysis: ImportAnalysis
    ) -> String? {
        switch format.layout {
        // `ModelDownloader.filterFiles` keys off this for both GGUF layouts
        // and for whisper's single-file repos.
        case .singleGGUF, .ggufWithMmproj: return quant?.filename
        case .whisperGGML: return analysis.whisperFilename
        case .huggingFaceMLX, .coreMLPackage: return nil
        }
    }

    private static func defaultQuantizationLabel(for layout: ModelFileLayout) -> String {
        switch layout {
        case .huggingFaceMLX:  return "as-published"
        case .coreMLPackage:   return "coreml"
        case .whisperGGML:     return "ggml"
        case .singleGGUF, .ggufWithMmproj: return "gguf"
        }
    }
}
