//
//  AIModel.swift
//  local_ai_test
//

import Foundation

// MARK: - AI Model descriptor

nonisolated struct AIModel: Identifiable, Hashable, Sendable, Codable {
    let id: String
    let name: String
    let family: String
    let parameterSize: String
    let quantization: String
    let backend: ModelBackend
    let capabilities: ModelCapabilities
    let fileLayout: ModelFileLayout
    let ramRequiredGB: Double
    let diskSizeGB: Double
    let huggingFaceRepo: String
    let requiresAppleIntelligence: Bool
    let contextLength: Int
    let quality: ModelQuality
    let requiresCompilation: Bool
    let preferredGGUFFilename: String?
    let mmprojFilename: String?
    /// Where this entry came from. Curated entries are replaceable by a
    /// catalog refresh; imported ones live only in the on-disk registry.
    let source: ModelSource
    /// Pinned Hugging Face commit sha. Downloading through `resolve/{sha}`
    /// instead of `resolve/main` keeps a half-finished download consistent
    /// even if the repo is updated mid-flight.
    let revision: String?
    /// Display name of the model's license when it requires explicit
    /// acceptance before download (Llama Community License, Gemma Terms).
    /// nil for permissively-licensed models — no gate is shown.
    let licenseName: String?
    /// Where the full license text lives; opened from the acceptance alert.
    let licenseURL: String?
    /// Name of the Core ML container directory to download — either an
    /// already-compiled `.mlmodelc` or a `.mlpackage` that still needs
    /// compiling. Repos routinely ship several: two quantizations of the same
    /// model, or a `.mlpackage` next to the `.mlmodelc` built from it. nil
    /// lets the downloader pick the smallest on its own, which is wrong
    /// whenever both forms of the SAME model are present — the `.mlpackage` is
    /// smaller and would put an on-device compile back. Pin it, and keep
    /// `requiresCompilation` in step with the suffix.
    let preferredPackage: String?
    /// An older model kept installable but no longer promoted: never
    /// recommended, and listed after every current model. Optional in
    /// `catalog.json` (absent means false) and written back only when true,
    /// so catalogs without it decode and re-encode unchanged.
    let deprecated: Bool
    /// The authors' recommended sampling, merged over the chat's global
    /// parameters (`ModelSampling`). Optional in `catalog.json` and written
    /// back only when present, like `deprecated`; nil keeps the global set.
    let sampling: ModelSampling?

    init(
        id: String,
        name: String,
        family: String,
        parameterSize: String,
        quantization: String,
        backend: ModelBackend,
        capabilities: ModelCapabilities = .textOnly,
        fileLayout: ModelFileLayout = .huggingFaceMLX,
        ramRequiredGB: Double,
        diskSizeGB: Double,
        huggingFaceRepo: String,
        requiresAppleIntelligence: Bool = false,
        contextLength: Int,
        quality: ModelQuality,
        requiresCompilation: Bool = false,
        preferredGGUFFilename: String? = nil,
        mmprojFilename: String? = nil,
        source: ModelSource = .curated,
        revision: String? = nil,
        licenseName: String? = nil,
        licenseURL: String? = nil,
        preferredPackage: String? = nil,
        deprecated: Bool = false,
        sampling: ModelSampling? = nil
    ) {
        self.id = id
        self.name = name
        self.family = family
        self.parameterSize = parameterSize
        self.quantization = quantization
        self.backend = backend
        self.capabilities = capabilities
        self.fileLayout = fileLayout
        self.ramRequiredGB = ramRequiredGB
        self.diskSizeGB = diskSizeGB
        self.huggingFaceRepo = huggingFaceRepo
        self.requiresAppleIntelligence = requiresAppleIntelligence
        self.contextLength = contextLength
        self.quality = quality
        self.requiresCompilation = requiresCompilation
        self.preferredGGUFFilename = preferredGGUFFilename
        self.mmprojFilename = mmprojFilename
        self.source = source
        self.revision = revision
        self.licenseName = licenseName
        self.licenseURL = licenseURL
        self.preferredPackage = preferredPackage
        self.deprecated = deprecated
        self.sampling = sampling
    }

    /// Effective disk footprint. Core ML models that require on-device
    /// compilation carry roughly 2x their download size once the
    /// `.mlmodelc` cache is written. Recommender/downloader use this
    /// for space checks.
    var effectiveDiskSizeGB: Double {
        requiresCompilation ? diskSizeGB * 2.0 : diskSizeGB
    }
}

// MARK: - Codable

nonisolated extension AIModel {

    private enum CodingKeys: String, CodingKey {
        case id, name, family, parameterSize, quantization, backend
        case capabilities, fileLayout, ramRequiredGB, diskSizeGB
        case huggingFaceRepo, requiresAppleIntelligence, contextLength
        case quality, requiresCompilation, preferredGGUFFilename
        case mmprojFilename, source, revision, licenseName, licenseURL
        case preferredPackage, deprecated, sampling
    }

    /// Hand-written so that hand-edited `catalog.json` entries may omit
    /// everything that has a memberwise default, and so that registry files
    /// written by older builds keep decoding after a field is added.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            id: try c.decode(String.self, forKey: .id),
            name: try c.decode(String.self, forKey: .name),
            family: try c.decode(String.self, forKey: .family),
            parameterSize: try c.decodeIfPresent(String.self, forKey: .parameterSize) ?? "",
            quantization: try c.decodeIfPresent(String.self, forKey: .quantization) ?? "",
            backend: try c.decode(ModelBackend.self, forKey: .backend),
            capabilities: try c.decodeIfPresent(ModelCapabilities.self, forKey: .capabilities) ?? .textOnly,
            fileLayout: try c.decodeIfPresent(ModelFileLayout.self, forKey: .fileLayout) ?? .huggingFaceMLX,
            ramRequiredGB: try c.decodeIfPresent(Double.self, forKey: .ramRequiredGB) ?? 0,
            diskSizeGB: try c.decodeIfPresent(Double.self, forKey: .diskSizeGB) ?? 0,
            huggingFaceRepo: try c.decodeIfPresent(String.self, forKey: .huggingFaceRepo) ?? "",
            requiresAppleIntelligence: try c.decodeIfPresent(Bool.self, forKey: .requiresAppleIntelligence) ?? false,
            contextLength: try c.decodeIfPresent(Int.self, forKey: .contextLength) ?? 4096,
            quality: try c.decodeIfPresent(ModelQuality.self, forKey: .quality) ?? .good,
            requiresCompilation: try c.decodeIfPresent(Bool.self, forKey: .requiresCompilation) ?? false,
            preferredGGUFFilename: try c.decodeIfPresent(String.self, forKey: .preferredGGUFFilename),
            mmprojFilename: try c.decodeIfPresent(String.self, forKey: .mmprojFilename),
            source: try c.decodeIfPresent(ModelSource.self, forKey: .source) ?? .curated,
            revision: try c.decodeIfPresent(String.self, forKey: .revision),
            licenseName: try c.decodeIfPresent(String.self, forKey: .licenseName),
            licenseURL: try c.decodeIfPresent(String.self, forKey: .licenseURL),
            preferredPackage: try c.decodeIfPresent(String.self, forKey: .preferredPackage),
            deprecated: try c.decodeIfPresent(Bool.self, forKey: .deprecated) ?? false,
            // Lenient: a malformed `sampling` object drops the sampling, not
            // the entry - and with it the whole remote catalog.
            sampling: (try? c.decodeIfPresent(ModelSampling.self, forKey: .sampling)) ?? nil
        )
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(family, forKey: .family)
        try c.encode(parameterSize, forKey: .parameterSize)
        try c.encode(quantization, forKey: .quantization)
        try c.encode(backend, forKey: .backend)
        try c.encode(capabilities, forKey: .capabilities)
        try c.encode(fileLayout, forKey: .fileLayout)
        try c.encode(ramRequiredGB, forKey: .ramRequiredGB)
        try c.encode(diskSizeGB, forKey: .diskSizeGB)
        try c.encode(huggingFaceRepo, forKey: .huggingFaceRepo)
        try c.encode(requiresAppleIntelligence, forKey: .requiresAppleIntelligence)
        try c.encode(contextLength, forKey: .contextLength)
        try c.encode(quality, forKey: .quality)
        try c.encode(requiresCompilation, forKey: .requiresCompilation)
        try c.encodeIfPresent(preferredGGUFFilename, forKey: .preferredGGUFFilename)
        try c.encodeIfPresent(mmprojFilename, forKey: .mmprojFilename)
        try c.encode(source, forKey: .source)
        try c.encodeIfPresent(revision, forKey: .revision)
        try c.encodeIfPresent(licenseName, forKey: .licenseName)
        try c.encodeIfPresent(licenseURL, forKey: .licenseURL)
        try c.encodeIfPresent(preferredPackage, forKey: .preferredPackage)
        if deprecated {
            try c.encode(true, forKey: .deprecated)
        }
        try c.encodeIfPresent(sampling, forKey: .sampling)
    }
}

// MARK: - Source

nonisolated enum ModelSource: String, Sendable, Codable, CaseIterable {
    /// Shipped in the app catalog (bundled JSON or the remote refresh of it).
    case curated
    /// Added by the user from a Hugging Face link.
    case imported
}

// MARK: - Backend

nonisolated enum ModelBackend: String, Sendable, Codable, CaseIterable, Identifiable {
    var id: String { rawValue }

    case appleIntelligence
    case mlx
    case mlxVision
    case mlxAudio
    case coreML
    case coreMLWhisper
    case coreMLKokoro
    case llamaCpp
    case llamaCppVision
    case whisperCpp

    var displayName: String {
        switch self {
        case .appleIntelligence: return "Apple Intelligence"
        case .mlx:               return "MLX"
        case .mlxVision:         return "MLX Vision"
        case .mlxAudio:          return "MLX Audio"
        case .coreML:            return "Core ML"
        case .coreMLWhisper:     return "Core ML Whisper"
        case .coreMLKokoro:      return "Core ML Kokoro TTS"
        case .llamaCpp:          return "llama.cpp"
        case .llamaCppVision:    return "llama.cpp Vision"
        case .whisperCpp:        return "whisper.cpp"
        }
    }

    var family: BackendFamily {
        switch self {
        case .appleIntelligence: return .apple
        case .mlx, .mlxVision, .mlxAudio: return .mlx
        case .coreML, .coreMLWhisper, .coreMLKokoro: return .coreML
        case .llamaCpp, .llamaCppVision, .whisperCpp: return .llamaCpp
        }
    }
}

// MARK: - Chat model eligibility

nonisolated extension ModelBackend {

    /// Whether a model on this backend can be the chat model. Speech
    /// backends have their own roles instead: whisper.cpp and Core ML Whisper
    /// transcribe voice messages, Kokoro reads replies aloud. In 1.1 a user
    /// could pick Whisper as the chat model, and the first image they sent
    /// failed with `unsupported_attachment`.
    var canRunChat: Bool {
        switch self {
        case .whisperCpp, .coreMLWhisper, .coreMLKokoro:
            return false
        case .appleIntelligence, .mlx, .mlxVision, .mlxAudio, .coreML, .llamaCpp, .llamaCppVision:
            return true
        }
    }
}

nonisolated extension AIModel {

    /// Selectable as the chat model: a chat backend, and text in, text out
    /// (an imported entry's capabilities are guessed, so both are checked).
    var canBeChatModel: Bool {
        backend.canRunChat && capabilities.contains(.textIn) && capabilities.contains(.textOut)
    }
}

nonisolated enum BackendFamily: String, Sendable, CaseIterable {
    case apple
    case mlx
    case coreML
    case llamaCpp

    var displayName: String {
        switch self {
        case .apple: return "Apple"
        case .mlx: return "MLX"
        case .coreML: return "Core ML"
        case .llamaCpp: return "llama.cpp"
        }
    }
}

nonisolated enum ModelQuality: Int, Comparable, Sendable, Codable {
    case good = 1
    case great = 2
    case excellent = 3

    static func < (lhs: ModelQuality, rhs: ModelQuality) -> Bool {
        lhs.rawValue < rhs.rawValue
    }
}
