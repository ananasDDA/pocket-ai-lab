//
//  LlamaLoadDiagnostics.swift
//  local_ai_test
//
//  Why a llama.cpp load failed, in words a user can act on.
//
//  `llama_model_load_from_file` and `llama_init_from_model` return NULL and
//  nothing else; the reason only ever reaches llama.cpp's log. 1.1 turned
//  every such failure into "llama.cpp: failed to load model at <path>", so
//  seven failed loads of oversized and incomplete 27B files in the first
//  field data all read the same. The engine now records the log around each
//  load (`LlamaLogCapture`), and `LlamaLoadFailure.classify` names the likely
//  cause from it: out of memory, an architecture this llama.cpp cannot run,
//  missing parts of a split model, or a damaged / incompatible file.
//
//  The classifier is pure string matching against messages taken from the
//  pinned b8882 binary, so it is unit-tested without a model or a device.
//

import Foundation

// MARK: - Cause

nonisolated enum LlamaLoadCause: Sendable, Equatable {
    case outOfMemory
    case unsupportedArchitecture(String)
    /// A split model whose other parts are not on disk.
    case missingSplitParts
    /// A part other than the first of a split model was handed to llama.cpp.
    case notFirstSplitPart
    /// Opens as GGUF, but the tensors do not match what this llama.cpp
    /// expects (made by a newer converter, an unsupported variant).
    case incompatibleFile
    /// Not a complete GGUF: truncated, corrupted, not GGUF at all.
    case damagedFile
    case unknown

    /// One-line headline.
    var summary: String {
        switch self {
        case .outOfMemory:              return "Not enough memory to load this model"
        case .unsupportedArchitecture:  return "This model's architecture is not supported"
        case .missingSplitParts, .notFirstSplitPart: return "Part of this model is missing"
        case .incompatibleFile:         return "This model file is not compatible"
        case .damagedFile:              return "This model file is damaged or incomplete"
        case .unknown:                  return "Failed to load model"
        }
    }

    /// What happened and what to do, in plain words.
    var explanation: String {
        let rev = LlamaArchitectureSupport.llamaRevision
        switch self {
        case .outOfMemory:
            return "The phone could not give llama.cpp enough memory for this model. Close "
                + "other apps and try again, or use a smaller model or a lower quantization."
        case .unsupportedArchitecture(let arch):
            return "This model is built on the \(arch) architecture, which the llama.cpp "
                + "version in this app (\(rev)) cannot run. Pick a different model."
        case .missingSplitParts:
            return "This model is split into several files and not all of them are on the "
                + "phone. Remove it and import it again from its Hugging Face link: the app "
                + "now downloads every part."
        case .notFirstSplitPart:
            return "This file is one part, not the first, of a model split into several "
                + "files, and it cannot load on its own. Remove it and import the model from "
                + "its Hugging Face link: the app now downloads every part."
        case .incompatibleFile:
            return "The file opened, but its contents do not match what llama.cpp \(rev) "
                + "expects for this architecture. It was probably made with a newer "
                + "llama.cpp, or it is a variant this version cannot run."
        case .damagedFile:
            return "The file is not a complete GGUF model. The download or copy may have "
                + "been cut short. Remove the model and download it again."
        case .unknown:
            return "llama.cpp could not load this model and gave no clear reason. The log "
                + "lines below may help."
        }
    }

    /// Machine kind for the measurement record (`error_kind`).
    var errorKind: String {
        switch self {
        case .outOfMemory:             return "out_of_memory"
        case .unsupportedArchitecture: return "unsupported_architecture"
        case .missingSplitParts, .notFirstSplitPart: return "split_incomplete"
        case .incompatibleFile:        return "incompatible_file"
        case .damagedFile:             return "bad_file"
        case .unknown:                 return "engine_load_failed"
        }
    }
}

// MARK: - Error

nonisolated struct LlamaLoadError: Error, LocalizedError, Sendable {
    let cause: LlamaLoadCause
    /// The llama.cpp log lines that decided the cause, for the copyable
    /// error detail. Never sent anywhere: they contain file paths.
    let logExcerpt: [String]

    var errorDescription: String? { cause.explanation }

    /// Explanation plus the log excerpt, for `EngineFailure.detail`.
    var detail: String {
        guard !logExcerpt.isEmpty else { return cause.explanation }
        return cause.explanation + "\n\nllama.cpp log:\n" + logExcerpt.joined(separator: "\n")
    }
}

// MARK: - Classifier

nonisolated enum LlamaLoadFailure {

    enum Stage: Sendable {
        /// `llama_model_load_from_file` returned NULL.
        case model
        /// The model loaded, `llama_init_from_model` returned NULL.
        case context
    }

    static func classify(log: [String], stage: Stage) -> LlamaLoadCause {
        let lines = log.map { $0.lowercased() }
        func any(_ needles: String...) -> Bool {
            lines.contains { line in needles.contains { line.contains($0) } }
        }

        if let arch = unknownArchitecture(in: log) {
            return .unsupportedArchitecture(arch)
        }
        if any("illegal split file idx", "must be loaded with the first split") {
            return .notFirstSplitPart
        }
        if any("failed to load gguf split", "invalid split count", "invalid split file",
               "missing key split") {
            return .missingSplitParts
        }
        // A missing part surfaces as a failed open of "<prefix>-0000N-of-0000M.gguf".
        if lines.contains(where: { $0.contains("failed to open") && $0.contains("-of-0") }) {
            return .missingSplitParts
        }
        if any("out of memory", "failed to allocate", "unable to allocate", "insufficient memory",
               "cannot allocate memory", "not enough memory", "outofmemory") {
            return .outOfMemory
        }
        if any("wrong number of tensors", "missing tensor", "has wrong shape", "unknown type",
               "invalid ggml type", "key not found in model", "error loading model hyperparameters",
               "error loading model vocabulary", "corrupted model:") {
            return .incompatibleFile
        }
        if any("not within the file bounds", "invalid magic", "unexpectedly reached end of file",
               "failed to read", "failed to open gguf file", "gguf_init_from_file") {
            return .damagedFile
        }
        // With the weights in and nothing else said, a context that cannot be
        // created is the KV cache or compute buffers not fitting.
        return stage == .context ? .outOfMemory : .unknown
    }

    /// The name in "unknown model architecture: 'xyz'".
    static func unknownArchitecture(in log: [String]) -> String? {
        let marker = "unknown model architecture: '"
        for line in log {
            guard let start = line.range(of: marker, options: .caseInsensitive) else { continue }
            let rest = line[start.upperBound...]
            let name = rest.prefix { $0 != "'" }
            if !name.isEmpty { return String(name) }
        }
        return nil
    }

    /// The lines worth showing: warnings and errors llama.cpp printed, last
    /// ones first in importance, capped.
    static func excerpt(from log: [String], limit: Int = 8) -> [String] {
        let keywords = ["error", "failed", "unknown", "illegal", "invalid", "missing",
                        "not within", "wrong", "memory", "allocate"]
        let relevant = log.filter { line in
            let lower = line.lowercased()
            return keywords.contains { lower.contains($0) }
        }
        return Array(relevant.suffix(limit))
    }
}

// MARK: - Log capture

/// Keeps the recent llama.cpp / ggml log lines so a failed load can be
/// explained. Installed once as llama.cpp's log callback (see
/// `LlamaRuntime.bootstrap`), which still forwards every line to stderr as
/// llama.cpp's default logger does.
nonisolated final class LlamaLogCapture: @unchecked Sendable {

    static let shared = LlamaLogCapture()

    private let lock = NSLock()
    private var lines: [String] = []
    private var pending = ""
    /// Total lines ever appended, so `mark()` survives the ring dropping
    /// old lines.
    private var appended = 0
    private let capacity = 400

    /// Level values of `ggml_log_level`.
    enum Level: Int {
        case none = 0, debug, info, warn, error, cont
    }

    func append(level: Int, text: String) {
        // Debug output is per-tensor noise; everything else is kept.
        guard level != Level.debug.rawValue else { return }
        lock.lock()
        defer { lock.unlock() }
        pending += text
        while let newline = pending.firstIndex(of: "\n") {
            let line = String(pending[..<newline])
            pending = String(pending[pending.index(after: newline)...])
            guard !line.isEmpty else { continue }
            lines.append(String(line.prefix(400)))
            appended += 1
            if lines.count > capacity { lines.removeFirst(lines.count - capacity) }
        }
    }

    /// A position to read the log from, taken right before a load.
    func mark() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return appended
    }

    /// Lines appended since `mark`, including a trailing unterminated one.
    func lines(since mark: Int) -> [String] {
        lock.lock()
        defer { lock.unlock() }
        let count = max(appended - mark, 0)
        var result = Array(lines.suffix(min(count, lines.count)))
        if !pending.isEmpty { result.append(pending) }
        return result
    }
}
