//
//  CoreMLCompiler.swift
//  local_ai_test
//
//  Compiles a downloaded `.mlpackage` into an on-device `.mlmodelc`.
//  Apple does not expose compilation progress through `MLModel.compileModel`,
//  so we emit a smoothly-interpolated progress value via a background timer
//  while the compile runs. This is informational only; if the caller cancels,
//  we abort the compile at the next Task-cancellation check point.
//
//  `.mlmodelc` is cached next to `.mlpackage` under `compiled.mlmodelc`, so
//  subsequent loads are instant.
//

import Foundation
import CoreML

enum CoreMLCompileError: Error, LocalizedError {
    case mlpackageNotFound(URL)
    case compileFailed(String)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .mlpackageNotFound(let url): return "No .mlpackage found at \(url.path)"
        case .compileFailed(let msg): return "Core ML compile failed: \(msg)"
        case .cancelled: return "Core ML compile was cancelled"
        }
    }
}

enum CoreMLCompiler {

    /// Returns the cached `.mlmodelc` URL, compiling on-device if needed.
    /// `progress` is called ~10x/sec with a monotonically-increasing
    /// best-guess value in [0, 1]. It reaches 1.0 exactly at completion.
    static func compileIfNeeded(
        mlpackageURL: URL,
        cacheDirectory: URL,
        estimatedDurationSeconds: TimeInterval = 60,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> URL {
        let cached = cacheDirectory.appendingPathComponent("compiled.mlmodelc")
        if FileManager.default.fileExists(atPath: cached.path) {
            progress(1.0)
            return cached
        }

        guard FileManager.default.fileExists(atPath: mlpackageURL.path) else {
            throw CoreMLCompileError.mlpackageNotFound(mlpackageURL)
        }

        let progressTask = Task<Void, Never> {
            let start = Date()
            while !Task.isCancelled {
                let elapsed = Date().timeIntervalSince(start)
                let fraction = min(0.95, elapsed / max(estimatedDurationSeconds, 1))
                progress(fraction)
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }

        defer { progressTask.cancel() }

        do {
            let compiledTemp = try await MLModel.compileModel(at: mlpackageURL)
            try? FileManager.default.removeItem(at: cached)
            try FileManager.default.createDirectory(
                at: cached.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try FileManager.default.moveItem(at: compiledTemp, to: cached)
            progress(1.0)
            return cached
        } catch {
            if Task.isCancelled { throw CoreMLCompileError.cancelled }
            throw CoreMLCompileError.compileFailed(error.localizedDescription)
        }
    }

    /// Deletes the cached compiled model without touching the `.mlpackage`.
    static func clearCache(in cacheDirectory: URL) {
        let cached = cacheDirectory.appendingPathComponent("compiled.mlmodelc")
        try? FileManager.default.removeItem(at: cached)
    }
}
