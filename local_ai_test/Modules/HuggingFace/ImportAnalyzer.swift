//
//  ImportAnalyzer.swift
//  local_ai_test
//
//  Turns a Hugging Face file listing into "which of our backends can run
//  this, and will it fit on this phone". Pure functions over the file list so
//  the rules can be tested without touching the network.
//

import Foundation

// MARK: - Detected format

struct ImportFormat: Identifiable, Hashable, Sendable {
    let backend: ModelBackend
    let layout: ModelFileLayout

    var id: String { backend.rawValue + "|" + layout.rawValue }
    var displayName: String { backend.displayName }
}

struct GGUFQuantOption: Identifiable, Hashable, Sendable {
    let filename: String
    let size: Int64

    var id: String { filename }

    /// "Q4_K_M" pulled out of "Model-Name-Q4_K_M.gguf". Falls back to the file
    /// stem for repos that do not follow the convention.
    var label: String { ImportAnalyzer.quantLabel(for: filename) }
}

struct ImportAnalysis: Sendable {
    let repo: String
    let revision: String?
    let downloads: Int
    let likes: Int
    let tags: [String]
    let gated: HFGatedStatus

    /// Best guess, listed first in `formats`.
    let detected: ImportFormat
    /// Every backend this repo could plausibly be run through. A repo with
    /// both GGUF and safetensors offers a real choice.
    let formats: [ImportFormat]

    let ggufQuants: [GGUFQuantOption]
    let defaultQuant: GGUFQuantOption?
    let mmprojFilename: String?
    /// Smallest `ggml-*.bin`. whisper.cpp repos host every model size, so the
    /// downloader must be pinned to one file or it fetches tens of gigabytes.
    let whisperFilename: String?

    /// Raw listing, kept so sizing can be recomputed when the user switches
    /// backend or quantization without another network round trip.
    let files: [HFSibling]

    /// Bytes the download would move, and the subset of those that are model
    /// weights (what the RAM estimate scales from), for a given choice.
    func sizing(layout: ModelFileLayout, quant: GGUFQuantOption?) -> (weights: Int64, download: Int64) {
        switch layout {
        case .singleGGUF:
            let weights = quant?.size ?? 0
            return (weights, weights)

        case .ggufWithMmproj:
            let weights = quant?.size ?? 0
            let projector = files.first { $0.rfilename == mmprojFilename }?.size ?? 0
            return (weights, weights + projector)

        case .huggingFaceMLX:
            let weights = ImportAnalyzer.totalBytes(
                of: files.filter { $0.rfilename.hasSuffix(".safetensors") }
            )
            let download = ImportAnalyzer.totalBytes(of: files.filter {
                !$0.rfilename.hasSuffix(".gguf") && !$0.rfilename.contains(".mlpackage")
            })
            return (weights, download)

        case .coreMLPackage:
            let download = ImportAnalyzer.totalBytes(
                of: files.filter { $0.rfilename.contains(".mlpackage") }
            )
            return (download, download)

        case .whisperGGML:
            let weights = files.first { $0.rfilename == whisperFilename }?.size ?? 0
            return (weights, weights)
        }
    }
}

// MARK: - Resource verdict

enum ResourceVerdict: Sendable, Equatable {
    case fits
    case tight
    case wontFit
}

struct ResourceEstimate: Sendable, Equatable {
    let diskGB: Double
    let ramGB: Double
    let usableRAMGB: Double
    let freeDiskGB: Double
    let ramVerdict: ResourceVerdict

    var diskFits: Bool { diskGB <= freeDiskGB }
}

// MARK: - Analyzer

enum ImportAnalyzer {

    static let bytesPerGB = 1_073_741_824.0

    /// Headroom the recommender reserves for iOS itself. Kept in sync with
    /// `ModelRecommender.split` — both must agree on what "usable" means.
    static let reservedSystemRAMGB = 3.0

    // MARK: Format detection

    static func analyze(_ info: HFRepoInfo) throws -> ImportAnalysis {
        let files = info.siblings
        let formats = try detectFormats(files: files, tags: info.tags)

        let quants = ggufQuants(in: files)

        return ImportAnalysis(
            repo: info.id,
            revision: info.sha,
            downloads: info.downloads,
            likes: info.likes,
            tags: info.tags,
            gated: info.gated,
            detected: formats[0],
            formats: formats,
            ggufQuants: quants,
            defaultQuant: defaultQuant(from: quants),
            mmprojFilename: mmprojFilename(in: files),
            whisperFilename: whisperFilename(in: files),
            files: files
        )
    }

    /// All backends that could run the repo, best guess first.
    /// Throws `unsupportedRepo` when nothing matches.
    static func detectFormats(files: [HFSibling], tags: [String]) throws -> [ImportFormat] {
        let names = files.map(\.rfilename)
        var result: [ImportFormat] = []

        let ggufs = names.filter { $0.hasSuffix(".gguf") }
        if !ggufs.isEmpty {
            let hasMmproj = ggufs.contains { basename($0).lowercased().hasPrefix("mmproj") }
            result.append(hasMmproj
                ? ImportFormat(backend: .llamaCppVision, layout: .ggufWithMmproj)
                : ImportFormat(backend: .llamaCpp, layout: .singleGGUF))
        }

        let hasSafetensors = names.contains { $0.hasSuffix(".safetensors") }
        let hasConfig = names.contains { basename($0) == "config.json" }
        if hasSafetensors && hasConfig {
            result.append(ImportFormat(backend: isVision(tags: tags) ? .mlxVision : .mlx,
                                       layout: .huggingFaceMLX))
        }

        if names.contains(where: { $0.contains(".mlpackage") }) {
            result.append(ImportFormat(backend: .coreML, layout: .coreMLPackage))
        }

        if names.contains(where: {
            let base = basename($0)
            return base.hasPrefix("ggml-") && base.hasSuffix(".bin")
        }) {
            result.append(ImportFormat(backend: .whisperCpp, layout: .whisperGGML))
        }

        guard !result.isEmpty else {
            throw HuggingFaceError.unsupportedRepo(
                "This repo has no weights this app can run. Looking for .gguf (llama.cpp), "
                + ".safetensors + config.json (MLX), .mlpackage (Core ML) or ggml-*.bin (whisper.cpp)."
            )
        }
        return result
    }

    private static func isVision(tags: [String]) -> Bool {
        let visionTags: Set<String> = [
            "image-text-to-text", "vision", "visual-question-answering",
            "image-to-text", "multimodal"
        ]
        return tags.contains { visionTags.contains($0.lowercased()) }
    }

    // MARK: GGUF helpers

    /// Selectable quantizations, smallest first. Vision projectors are not
    /// quantization choices, so they are excluded.
    static func ggufQuants(in files: [HFSibling]) -> [GGUFQuantOption] {
        files
            .filter {
                $0.rfilename.hasSuffix(".gguf")
                    && !basename($0.rfilename).lowercased().hasPrefix("mmproj")
            }
            .map { GGUFQuantOption(filename: $0.rfilename, size: $0.size ?? 0) }
            .sorted { $0.size < $1.size }
    }

    /// Q4_K_M is the usual sweet spot; without it, take the middle of the
    /// size-sorted list rather than the smallest or largest.
    static func defaultQuant(from quants: [GGUFQuantOption]) -> GGUFQuantOption? {
        guard !quants.isEmpty else { return nil }
        if let preferred = quants.first(where: { $0.label.uppercased() == "Q4_K_M" }) {
            return preferred
        }
        return quants[quants.count / 2]
    }

    static func mmprojFilename(in files: [HFSibling]) -> String? {
        let projectors = files
            .map(\.rfilename)
            .filter { basename($0).lowercased().hasPrefix("mmproj") }
        // F16 projectors are the safe default when a repo ships several.
        return projectors.first { $0.uppercased().contains("F16") } ?? projectors.first
    }

    static func whisperFilename(in files: [HFSibling]) -> String? {
        files
            .filter {
                let base = basename($0.rfilename)
                return base.hasPrefix("ggml-") && base.hasSuffix(".bin")
                    && !$0.rfilename.contains("/")
            }
            .min { ($0.size ?? .max) < ($1.size ?? .max) }?
            .rfilename
    }

    private static let quantPattern = try? NSRegularExpression(
        pattern: "(IQ[0-9]+(?:_[A-Za-z0-9]+)*|Q[0-9]+(?:_[A-Za-z0-9]+)*|BF16|F16|F32)",
        options: [.caseInsensitive]
    )

    static func quantLabel(for filename: String) -> String {
        var stem = basename(filename)
        if stem.hasSuffix(".gguf") { stem = String(stem.dropLast(5)) }
        guard let regex = quantPattern else { return stem }
        let range = NSRange(stem.startIndex..<stem.endIndex, in: stem)
        // Last match wins: the quant tag is a suffix, and model names can
        // contain things like "Q" or numbers earlier on.
        guard let match = regex.matches(in: stem, range: range).last,
              let matched = Range(match.range, in: stem)
        else { return stem }
        return String(stem[matched]).uppercased()
    }

    // MARK: Sizing

    static func totalBytes(of files: [HFSibling]) -> Int64 {
        files.reduce(0) { $0 + ($1.size ?? 0) }
    }

    /// Peak RAM for a loaded model. Weights dominate; the multiplier covers
    /// the KV cache and compute buffers, the constant covers the runtime.
    static func ramEstimateGB(layout: ModelFileLayout, weightBytes: Int64) -> Double {
        let weightsGB = Double(weightBytes) / bytesPerGB
        switch layout {
        case .singleGGUF, .ggufWithMmproj:
            return weightsGB * 1.15 + 0.8
        case .huggingFaceMLX:
            return weightsGB * 1.2 + 0.8
        case .coreMLPackage:
            return weightsGB * 1.1 + 0.5
        case .whisperGGML:
            return weightsGB * 1.5 + 0.2
        }
    }

    static func usableRAMGB(totalRAMBytes: UInt64) -> Double {
        Double(totalRAMBytes) / bytesPerGB - reservedSystemRAMGB
    }

    static func estimate(
        layout: ModelFileLayout,
        weightBytes: Int64,
        downloadBytes: Int64,
        device: DeviceInfo,
        freeDiskBytes: Int64
    ) -> ResourceEstimate {
        let ramGB = ramEstimateGB(layout: layout, weightBytes: weightBytes)
        // GGUF layouts mmap their weights (clean pages, not charged by
        // jetsam) and are judged against the Metal working set; MLX and
        // Core ML copy into dirty Metal buffers and get the strict jetsam
        // allowance. See DeviceInfo.
        let usable: Double
        switch layout {
        case .singleGGUF, .ggufWithMmproj, .whisperGGML:
            usable = device.mmapCeilingGB
        case .huggingFaceMLX, .coreMLPackage:
            usable = device.usableRAMGB
        }

        let verdict: ResourceVerdict
        if ramGB > usable {
            verdict = .wontFit
        } else if ramGB > usable - 1.0 {
            // Within 1 GB of the ceiling: it loads, but a background app or a
            // long context will push it over.
            verdict = .tight
        } else {
            verdict = .fits
        }

        return ResourceEstimate(
            diskGB: Double(downloadBytes) / bytesPerGB,
            ramGB: ramGB,
            usableRAMGB: usable,
            freeDiskGB: Double(freeDiskBytes) / bytesPerGB,
            ramVerdict: verdict
        )
    }

    // MARK: Utilities

    static func basename(_ path: String) -> String {
        path.split(separator: "/").last.map(String.init) ?? path
    }
}
