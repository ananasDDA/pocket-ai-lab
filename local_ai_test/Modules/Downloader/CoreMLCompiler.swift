//
//  CoreMLCompiler.swift
//  local_ai_test
//
//  Compiles a `.mlpackage` into an on-device `.mlmodelc`.
//
//  THIS IS THE IMPORT PATH, NOT THE CATALOG PATH. Catalog entries point at
//  repos that publish an already-compiled `.mlmodelc` and are downloaded as
//  such, so nothing in the catalog reaches `compileIfNeeded` at all - that is
//  Apple's own recommendation (compile off-device, ship the `.mlmodelc`) and
//  it also saves the duplicate on-disk copy a device compile leaves behind.
//  What still needs compiling is a bare `.mlpackage` a user imports from
//  Files, and that is what this file is for.
//
//  Apple does not expose compilation progress through `MLModel.compileModel`,
//  so we emit a smoothly-interpolated progress value via a background timer
//  while the compile runs. This is informational only; if the caller cancels,
//  we abort the compile at the next Task-cancellation check point.
//
//  The result is cached next to the `.mlpackage` as `compiled.mlmodelc`, so
//  subsequent loads are instant.
//
//  TWO RULES THIS FILE EXISTS TO ENFORCE, both learned the hard way when a
//  compile that actually takes seconds looked like a 60-minute hang:
//
//  1. `progress` is never called again once `compileIfNeeded` has returned or
//     thrown. The ticker is cancelled AND awaited before the terminal value
//     goes out, because callers publish progress by hopping to the main actor
//     and a tick still in flight would land after — and overwrite — the
//     terminal state they set on return.
//  2. The ticker runs on a DETACHED task so it can neither be starved by nor
//     block the caller's actor. The compile itself is awaited directly: the
//     async `MLModel.compileModel(at:)` is `nonisolated`, so it never runs on
//     the caller's actor anyway, and awaiting it in-place is what keeps the
//     caller's cancellation reaching it. Wrapping it in `Task.detached` (as
//     this file used to) starts an unstructured task that inherits no
//     cancellation, so cancelling the caller left the compile running and made
//     the `Task.isCancelled` check on the failure path unreachable.
//     Do NOT reintroduce the deprecated synchronous overload here; under
//     `-default-isolation=MainActor` it would compile a multi-gigabyte package
//     on the main thread.
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

    /// Name of the on-disk compile cache, inside the model's directory.
    static let cacheName = "compiled.mlmodelc"

    /// Returns the cached `.mlmodelc` URL, compiling on-device if needed.
    /// `progress` is called ~10x/sec with a monotonically-increasing
    /// best-guess value in [0, 1]. It reaches 1.0 exactly at completion, and
    /// is guaranteed not to be called after this function returns or throws.
    static func compileIfNeeded(
        mlpackageURL: URL,
        cacheDirectory: URL,
        estimatedDurationSeconds: TimeInterval = 60,
        progress: @Sendable @escaping (Double) -> Void
    ) async throws -> URL {
        let cached = cacheDirectory.appendingPathComponent(cacheName)
        if FileManager.default.fileExists(atPath: cached.path) {
            log("cache hit at \(cached.path) - skipping compile")
            progress(1.0)
            return cached
        }

        guard FileManager.default.fileExists(atPath: mlpackageURL.path) else {
            throw CoreMLCompileError.mlpackageNotFound(mlpackageURL)
        }

        // Detached: the ticker must keep running even while the caller's actor
        // is busy, and must not itself contend for the main thread.
        let progressTask = Task.detached(priority: .utility) {
            let start = Date()
            while !Task.isCancelled {
                let elapsed = Date().timeIntervalSince(start)
                let fraction = min(0.95, elapsed / max(estimatedDurationSeconds, 1))
                progress(fraction)
                try? await Task.sleep(nanoseconds: 100_000_000)
            }
        }

        /// Stops the ticker and WAITS for it, so no further `progress` call
        /// can be in flight once this returns. Every exit path goes through
        /// it; a `defer` cannot, because it may not await.
        func stopTicker() async {
            progressTask.cancel()
            _ = await progressTask.value
        }

        let packageBytes = directorySize(of: mlpackageURL)
        let started = Date()
        log(String(
            format: "compile start %@ (%.2f GB) footprint %.0f MB",
            mlpackageURL.lastPathComponent,
            Double(packageBytes) / 1_073_741_824,
            footprintMB()
        ))

        do {
            // Awaited in place: `MLModel.compileModel(at:)` is a nonisolated
            // async function, so it runs off the caller's actor by itself and
            // stays inside the caller's cancellation scope.
            let compiledTemp = try await MLModel.compileModel(at: mlpackageURL)

            try? FileManager.default.removeItem(at: cached)
            try FileManager.default.createDirectory(
                at: cached.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // Same volume (both live in the app's data container), so this is
            // a rename, not a 4 GB copy.
            try FileManager.default.moveItem(at: compiledTemp, to: cached)

            let elapsed = Date().timeIntervalSince(started)
            log(String(
                format: "compile finished in %.1f s (%.1f min) footprint %.0f MB -> %@",
                elapsed, elapsed / 60, footprintMB(), cached.lastPathComponent
            ))

            await stopTicker()
            progress(1.0)
            return cached
        } catch {
            await stopTicker()
            let elapsed = Date().timeIntervalSince(started)
            if Task.isCancelled {
                log(String(format: "compile cancelled after %.1f s", elapsed))
                throw CoreMLCompileError.cancelled
            }
            log(String(format: "compile FAILED after %.1f s: %@", elapsed, error.localizedDescription))
            throw CoreMLCompileError.compileFailed(error.localizedDescription)
        }
    }

    /// Deletes the cached compiled model without touching the `.mlpackage`.
    static func clearCache(in cacheDirectory: URL) {
        let cached = cacheDirectory.appendingPathComponent(cacheName)
        try? FileManager.default.removeItem(at: cached)
    }

    // MARK: - Package selection

    /// The `.mlpackage` to compile and to load, chosen the SAME way in both
    /// places. `contentsOfDirectory` has no defined order, so picking
    /// `.first` let the downloader compile one variant while the engine
    /// loaded another — and the single `compiled.mlmodelc` cache made that
    /// mismatch invisible until inference produced nonsense. Repos that ship
    /// several variants (apple/mistral-coreml had an FP16 and an Int4
    /// package) are filtered down to one at download time now, but installs
    /// made before that rule landed still hold both.
    ///
    /// Smallest wins; ties break on name so the result is deterministic.
    nonisolated static func locatePackage(in directory: URL) -> URL? {
        locateContainer(in: directory, extension: "mlpackage")
    }

    /// An ALREADY-COMPILED `.mlmodelc` in the model's directory, if there is
    /// one — either downloaded that way (most Core ML LLM repos publish the
    /// compiled bundle, and Apple recommends shipping it rather than compiling
    /// multi-gigabyte packages on the phone) or left by an earlier compile of
    /// an imported `.mlpackage`, which lands as `compiled.mlmodelc`.
    ///
    /// When both exist they are the same model, so either is correct; the
    /// smallest-wins rule below just makes the choice deterministic.
    nonisolated static func locateCompiledModel(in directory: URL) -> URL? {
        locateContainer(in: directory, extension: "mlmodelc")
    }

    nonisolated private static func locateContainer(
        in directory: URL, extension pathExtension: String
    ) -> URL? {
        let items = (try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )) ?? []
        let packages = items.filter { $0.pathExtension == pathExtension }
        guard packages.count > 1 else { return packages.first }

        var best = packages[0]
        var bestSize = directorySize(of: best)
        for candidate in packages.dropFirst() {
            let size = directorySize(of: candidate)
            if size < bestSize || (size == bestSize && candidate.lastPathComponent < best.lastPathComponent) {
                best = candidate
                bestSize = size
            }
        }
        return best
    }

    /// Recursive on-disk size of a directory, 0 when it cannot be read.
    nonisolated static func directorySize(of url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    // MARK: - Diagnostics

    /// Timestamped, greppable, and on stdout so it lands in the xcodebuild
    /// test log next to the harness's own `[compile]` lines. Compile duration
    /// is the one number nobody could read off the old logs.
    nonisolated private static func log(_ message: String) {
        let stamp = ISO8601DateFormatter().string(from: Date())
        print("[CoreMLCompiler] \(stamp) \(message)")
    }

    /// `phys_footprint` in MB — the figure jetsam charges the process.
    nonisolated private static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size
        )
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}
