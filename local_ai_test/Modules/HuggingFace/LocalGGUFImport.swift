//
//  LocalGGUFImport.swift
//  local_ai_test
//
//  "Import from Files" for GGUF weights the user already has on the device:
//  inspect the picked file, copy it into the app container, and describe it
//  with the same kind of `AIModel` the Hugging Face import produces so that
//  nothing downstream has to special-case it.
//

import Foundation

// MARK: - Errors

nonisolated enum LocalImportError: Error, LocalizedError, Equatable {
    case unreadable(String)
    case notGGUF(String)
    /// The header could not even be opened — as opposed to `.notGGUF`, where
    /// it opened fine but the magic bytes were wrong. An undownloaded iCloud
    /// placeholder or a sandbox permissions error hits this path, and neither
    /// one means the file is corrupt.
    case fileNotAccessible(String)
    case projectorOnly(String)
    case notEnoughSpace(neededGB: Double, freeGB: Double)
    case copyFailed(String)

    var errorDescription: String? {
        switch self {
        case .unreadable(let filename):
            return "\(filename) could not be opened. Copy it into the Files app first, "
                + "then pick it from there."
        case .notGGUF(let filename):
            return "\(filename) is not a GGUF file. Its header is missing the GGUF marker, "
                + "so llama.cpp would not be able to load it."
        case .fileNotAccessible(let filename):
            return "Could not read \(filename). If it is stored in iCloud, download it in "
                + "the Files app first, then try again."
        case .projectorOnly(let filename):
            return "\(filename) is a vision projector (mmproj), not a model. "
                + "Projector files cannot be imported on their own."
        case .notEnoughSpace(let neededGB, let freeGB):
            return String(
                format: "Not enough storage: the file needs %.1f GB and only %.1f GB is free.",
                neededGB, freeGB
            )
        case .copyFailed(let reason):
            return "Could not copy the file into the app: \(reason)"
        }
    }
}

// MARK: - Picked file

/// What a file picked in the Files app turned out to be, established before
/// a single byte is copied.
nonisolated struct LocalGGUFCandidate: Sendable, Equatable {

    /// Registry ids for local imports. A Hugging Face id is always
    /// "owner/name", so this prefix can never collide with one.
    static let idPrefix = "local/"

    /// Name of the picked file, e.g. "my-model-Q4_K_M.gguf".
    let filename: String
    let fileSizeBytes: Int64
    let header: GGUFHeader

    /// Filename without its ".gguf" extension. Derived exactly the way
    /// `ImportAnalyzer.quantLabel` derives its own stem, so the two can be
    /// compared to tell a real quantization tag from an echoed filename.
    var stem: String {
        filename.hasSuffix(".gguf") ? String(filename.dropLast(5)) : filename
    }

    var modelID: String { Self.idPrefix + stem }
}

// MARK: - File operations

/// Everything that touches the picked file. `nonisolated` throughout: these
/// open and stream multi-gigabyte files and must never run on the main actor.
nonisolated enum LocalGGUFFile {

    private static let copyChunkSize = 8 * 1024 * 1024

    /// Establishes what the picked file is without copying it. The caller
    /// owns the security-scoped access to `fileURL` and must hold it across
    /// this call.
    static func inspect(fileURL: URL) throws -> LocalGGUFCandidate {
        let filename = fileURL.lastPathComponent

        // Pairing a projector with its model is a v2 problem; on its own an
        // mmproj file is useless, so say so instead of importing a stub.
        guard !filename.lowercased().contains("mmproj") else {
            throw LocalImportError.projectorOnly(filename)
        }

        guard let size = try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size > 0
        else { throw LocalImportError.unreadable(filename) }

        let header: GGUFHeader?
        do {
            header = try GGUFHeaderReader.read(fileURL: fileURL)
        } catch {
            // The file itself could not be opened (undownloaded iCloud
            // placeholder, sandbox permissions, …) — distinct from a file
            // that opened fine but was not GGUF at all.
            throw LocalImportError.fileNotAccessible(filename)
        }
        guard let header else {
            throw LocalImportError.notGGUF(filename)
        }

        return LocalGGUFCandidate(
            filename: filename,
            fileSizeBytes: Int64(size),
            header: header
        )
    }

    /// Free space on the volume that holds the app container, read fresh
    /// rather than taken from the device scan — a scan can be minutes old and
    /// the file being imported is measured in gigabytes.
    static func freeDiskBytes() -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfFileSystem(
            forPath: NSHomeDirectory()
        ) else { return 0 }
        return (attrs[.systemFreeSize] as? Int64) ?? 0
    }

    /// Streams `source` into `destination` in chunks, reporting a 0…1
    /// fraction whenever the whole-percent figure changes.
    ///
    /// `FileManager.copyItem` would do the same move in one call but reports
    /// no progress, and a silent five-minute freeze on a 6 GB file reads as a
    /// hang. The stream writes into a sibling `.part` file rather than
    /// `destination` directly: on "Replace" this can be overwriting the only
    /// copy of a model the user already has, and a mid-copy failure or
    /// cancellation must not delete that existing file until the new one has
    /// fully landed. The `.part` file is removed on failure.
    static func copy(
        from source: URL,
        to destination: URL,
        totalBytes: Int64,
        onProgress: @Sendable (Double) -> Void
    ) throws {
        let manager = FileManager.default
        do {
            try manager.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
        } catch {
            throw LocalImportError.copyFailed(error.localizedDescription)
        }

        let staging = destination.appendingPathExtension("part")
        try? manager.removeItem(at: staging)
        guard manager.createFile(atPath: staging.path, contents: nil) else {
            throw LocalImportError.copyFailed("could not create \(staging.lastPathComponent)")
        }

        do {
            try streamCopy(
                from: source, into: staging, totalBytes: totalBytes, onProgress: onProgress
            )

            // Only now touch the pre-existing destination (if any): the copy
            // above fully succeeded and is safely on disk, so replacing the
            // old file can no longer lose data.
            if manager.fileExists(atPath: destination.path) {
                _ = try manager.replaceItemAt(destination, withItemAt: staging)
            } else {
                try manager.moveItem(at: staging, to: destination)
            }
            onProgress(1.0)
        } catch {
            try? manager.removeItem(at: staging)
            if error is CancellationError { throw error }
            throw LocalImportError.copyFailed(error.localizedDescription)
        }
    }

    /// Reads `source` and writes it into `staging`, closing both handles
    /// (synchronized) before returning — on any path, including an early
    /// throw, so the caller can safely rename or remove `staging` next.
    private static func streamCopy(
        from source: URL,
        into staging: URL,
        totalBytes: Int64,
        onProgress: @Sendable (Double) -> Void
    ) throws {
        let input = try FileHandle(forReadingFrom: source)
        let output = try FileHandle(forWritingTo: staging)
        defer {
            try? input.close()
            try? output.close()
        }

        var copied: Int64 = 0
        var lastPercent = -1

        while true {
            try Task.checkCancellation()
            guard let chunk = try input.read(upToCount: copyChunkSize),
                  !chunk.isEmpty
            else { break }

            try output.write(contentsOf: chunk)
            copied += Int64(chunk.count)

            let fraction = totalBytes > 0
                ? min(Double(copied) / Double(totalBytes), 1.0)
                : 0
            let percent = Int(fraction * 100)
            if percent != lastPercent {
                lastPercent = percent
                onProgress(fraction)
            }
        }

        try output.synchronize()
    }
}

// MARK: - Descriptor

/// Builds the registry entry for a picked file. Main-actor because it reuses
/// the naming heuristics the Hugging Face import already carries.
enum LocalGGUFModelBuilder {

    /// A descriptor shaped exactly like the one `ImportedModelBuilder`
    /// produces for a single-file GGUF repo: same backend, same layout, same
    /// `preferredGGUFFilename` contract, so `LlamaCppTextEngine` finds the
    /// weights through its usual path.
    static func makeModel(
        candidate: LocalGGUFCandidate, estimate: ResourceEstimate
    ) -> AIModel {
        let quantization = quantization(for: candidate)

        return AIModel(
            id: candidate.modelID,
            name: displayName(for: candidate, quantization: quantization),
            family: family(for: candidate),
            parameterSize: ImportedModelBuilder.parameterSize(from: candidate.stem),
            quantization: quantization,
            backend: .llamaCpp,
            // Vision GGUFs need a companion projector this flow does not
            // import, so every local file is run as a text model.
            capabilities: .textOnly,
            fileLayout: .singleGGUF,
            ramRequiredGB: (estimate.ramGB * 10).rounded() / 10,
            diskSizeGB: (estimate.diskGB * 10).rounded() / 10,
            // No repo: the weights came off the user's own device, and the
            // card degrades to a plain title when this is empty.
            huggingFaceRepo: "",
            contextLength: 4096,
            quality: .good,
            preferredGGUFFilename: candidate.filename,
            source: .imported
        )
    }

    // MARK: - Heuristics

    /// The filename tag is what users recognise ("Q4_K_M"); the header's
    /// `general.file_type` covers files named without one.
    static func quantization(for candidate: LocalGGUFCandidate) -> String {
        quantizationFromFilename(candidate) ?? candidate.header.fileTypeLabel ?? "gguf"
    }

    /// `ImportAnalyzer.quantLabel` echoes the file stem back when the name
    /// carries no known tag, so anything that still looks like a filename is
    /// discarded rather than shown as a quantization.
    private static func quantizationFromFilename(_ candidate: LocalGGUFCandidate) -> String? {
        let label = ImportAnalyzer.quantLabel(for: candidate.filename)
        guard label.lowercased() != candidate.stem.lowercased(),
              label.count <= 12,
              !label.contains("."),
              !label.contains(" ")
        else { return nil }
        return label
    }

    /// The card renders the quantization separately, so it is trimmed off the
    /// title rather than repeated in it.
    static func displayName(for candidate: LocalGGUFCandidate, quantization: String) -> String {
        var stem = candidate.stem
        for separator in ["-", "_", "."] {
            let tail = (separator + quantization).lowercased()
            if stem.lowercased().hasSuffix(tail) {
                stem = String(stem.dropLast(tail.count))
                break
            }
        }
        // `displayName` expects "owner/repo"; a bare stem plays the repo part
        // and gets the same underscore and hyphen cleanup.
        let pretty = ImportedModelBuilder.displayName(repo: stem, quant: nil)
        if !pretty.isEmpty { return pretty }
        return candidate.header.name ?? candidate.filename
    }

    /// Family from the filename when the app recognises one, otherwise the
    /// architecture the GGUF header declares.
    static func family(for candidate: LocalGGUFCandidate) -> String {
        // `family(from:)` falls back to the part before the first "/", which
        // for a bare stem is the stem itself — that is the "no match" signal.
        let fromFilename = ImportedModelBuilder.family(from: candidate.stem)
        if fromFilename != candidate.stem { return fromFilename }
        if let architecture = candidate.header.architecture, !architecture.isEmpty {
            return architecture.capitalized
        }
        return "Imported"
    }
}
