//
//  CoreMLTextDeviceTests.swift
//
//  The Core ML text path end to end on a real device: download -> on-device
//  compile -> `MLModel` load -> tokenizer -> generation.
//
//  WHY THE TIMINGS ARE PRINTED: the 1.0 release sweep reported the Core ML
//  model as "compiling" for sixty minutes and then deleted it. The compile had
//  in fact finished in 0.3 s - a progress callback landing after the terminal
//  state pinned `ModelDownloader.states` at `.compiling` forever. No log
//  anywhere said how long the compile actually took, so nobody could tell a
//  stuck state machine from a slow compiler. Now `CoreMLCompiler` stamps
//  compile start and end, and this test stamps download, load and generation.
//
//  WHY THIS EXISTS ALONGSIDE THE BUILD-TEST SWEEP: the sweep resolves models
//  through `ModelCatalog.all`, which prefers the catalog the app last
//  downloaded from the public repo over the one compiled into the build. That
//  is right for shipping and wrong for verifying a catalog change - a Core ML
//  entry added or swapped in THIS build is invisible to the sweep until the
//  public catalog.json catches up. This test falls back to
//  `ModelCatalog.builtIn`, so it can answer "does the Core ML model this build
//  ships actually load and answer on device?" today.
//

import Foundation
import Testing
@testable import local_ai_test

@Suite(.tags(.device, .coreML))
struct CoreMLTextDeviceTests {

    /// Budget for the transfer. A ~2 GB bundle over Wi-Fi is minutes.
    private static let downloadTimeout: TimeInterval = 30 * 60

    private func environmentFlag(_ key: String) -> Bool {
        ["1", "true", "yes"].contains(
            (ProcessInfo.processInfo.environment[key] ?? "").lowercased()
        )
    }

    /// The Core ML text entry THIS BUILD ships.
    ///
    /// The identity comes from `ModelCatalog.builtIn`, never from
    /// `ModelCatalog.all`: `all` prefers the catalog the device last fetched
    /// from the public repo, which on any device that has run an older build
    /// still names the previous Core ML model. Resolving through it would
    /// quietly test the model we just replaced. The live catalog is still
    /// consulted for the same id, so a remote entry that corrects a size or a
    /// pin is the one that gets exercised.
    @MainActor
    private func coreMLTextModel() -> AIModel? {
        guard let shipped = ModelCatalog.builtIn.first(
            where: { $0.backend == .coreML && $0.fileLayout == .coreMLPackage }
        ) else { return nil }
        return ModelCatalog.all.first { $0.id == shipped.id } ?? shipped
    }

    @Test @MainActor func generatesResponseFromCoreML() async throws {
        // `COREML_DEVICE_TEST_REQUIRED=1` turns a clean skip into a failure -
        // release verification sets it; a developer running the suite on a
        // phone with no Core ML model downloaded does not.
        let required = environmentFlag("COREML_DEVICE_TEST_REQUIRED")
        // `COREML_DEVICE_TEST_DOWNLOAD=1` lets the test fetch the package
        // itself (and delete it again). Off by default: a unit test must not
        // spend gigabytes of someone's data plan without being asked.
        let mayDownload = environmentFlag("COREML_DEVICE_TEST_DOWNLOAD")

        guard let model = coreMLTextModel() else {
            // Legitimately absent: a release that ships no Core ML text model
            // still ships the engine, for imported packages.
            print("[coreml-test] this build has no Core ML text model in its catalog")
            #expect(!required, "a Core ML text model is required but none is in the catalog")
            return
        }
        print("[coreml-test] model: \(model.id) (\(model.name))")

        let downloader = ModelDownloader.shared
        var installedByTest = false
        if !InstalledModelsStore.shared.installedModels.contains(where: { $0.id == model.id }) {
            guard mayDownload else {
                print("[coreml-test] \(model.id) is not installed; set COREML_DEVICE_TEST_DOWNLOAD=1 to fetch it")
                #expect(!required, "\(model.id) is required but not installed on this device")
                return
            }
            installedByTest = true
            try await download(model, using: downloader)
        }
        // Leave the device as we found it when we were the ones who filled it.
        defer {
            if installedByTest {
                print("[coreml-test] removing \(model.id)")
                downloader.delete(model)
            }
        }

        let directory = downloader.modelDirectory(for: model)
        // A catalog entry downloads its `.mlmodelc` already compiled, so this
        // is normally present before the first load ever runs; only an
        // imported `.mlpackage` arrives without one and has to compile.
        let prebuilt = CoreMLCompiler.locateCompiledModel(in: directory)
        print("[coreml-test] compiled model \(prebuilt.map { "present: \($0.lastPathComponent)" } ?? "absent (this load compiles)")")

        let loadStart = Date()
        await InferenceManager.shared.prepare(model: model)
        let loadSeconds = Date().timeIntervalSince(loadStart)
        print(String(
            format: "[coreml-test] prepare() finished in %.1f s (%.1f min), state: %@",
            loadSeconds, loadSeconds / 60, String(describing: InferenceManager.shared.state)
        ))

        guard case .ready = InferenceManager.shared.state else {
            Issue.record("Core ML load failed: \(InferenceManager.shared.state)")
            return
        }
        // Either the download brought a `.mlmodelc` or the compile wrote one.
        // If neither is true, every later load pays the compile again.
        #expect(
            CoreMLCompiler.locateCompiledModel(in: directory) != nil,
            "load succeeded but no .mlmodelc is on disk - the next load would recompile"
        )

        let generateStart = Date()
        var out = ""
        var tokensPerSecond: Double?
        for try await output in InferenceManager.shared.generate(
            turns: [.user("What is the capital of France? Answer in one short sentence.")],
            parameters: .deterministic
        ) {
            switch output {
            case .textDelta(let s): out += s
            case .diagnostic(.finished(_, let rate)): tokensPerSecond = rate
            default: break
            }
            if out.count > 200 { break }
        }
        print(String(
            format: "[coreml-test] generated %d chars in %.1f s at %@ tok/s: %@",
            out.count,
            Date().timeIntervalSince(generateStart),
            tokensPerSecond.map { String(format: "%.2f", $0) } ?? "n/a",
            out
        ))
        #expect(!out.isEmpty)

        // Cancellation must not take the process down. swift-transformers
        // wraps the Core ML prediction in `try!`, and a cancelled task made
        // that prediction throw CancellationError - a fatal error that
        // killed two overnight sweeps (a benchmark task timing out) and the
        // chat's Stop button. Leaving the stream early cancels the task.
        var firstPiece = ""
        for try await output in InferenceManager.shared.generate(
            turns: [.user("Count from one to fifty, one number per line.")],
            parameters: .deterministic
        ) {
            if case .textDelta(let s) = output {
                firstPiece = s
                break
            }
        }
        #expect(!firstPiece.isEmpty, "the cancelled generation produced no first token")
        print("[coreml-test] cancelled a generation after its first piece; the process survived")

        // And the engine is still usable afterwards.
        var afterCancel = ""
        for try await output in InferenceManager.shared.generate(
            turns: [.user("Name one colour. One word.")],
            parameters: .deterministic
        ) {
            if case .textDelta(let s) = output { afterCancel += s }
            if afterCancel.count > 60 { break }
        }
        #expect(!afterCancel.isEmpty, "no output after a cancelled generation")
        print("[coreml-test] generation after cancel: \(afterCancel)")

        await InferenceManager.shared.unloadCurrent()
    }

    /// Drives `ModelDownloader` to `.installed`, printing the same phases the
    /// build-test harness prints so the two logs read alike.
    @MainActor
    private func download(_ model: AIModel, using downloader: ModelDownloader) async throws {
        print(String(format: "[coreml-test] downloading %@ (~%.1f GB)", model.id, model.diskSizeGB))
        downloader.download(model)
        let deadline = Date().addingTimeInterval(Self.downloadTimeout)
        var lastPrint = Date.distantPast

        while Date() < deadline {
            switch downloader.state(for: model) {
            case .installed:
                print("[coreml-test] installed")
                return
            case .error(let message):
                throw CoreMLDeviceTestError.download(message)
            case .downloading(let fraction, let done, let total):
                if Date().timeIntervalSince(lastPrint) >= 30 {
                    lastPrint = Date()
                    print(String(
                        format: "[coreml-test] download %.0f%% (%.2f / %.2f GB)",
                        fraction * 100,
                        Double(done) / 1_073_741_824, Double(total) / 1_073_741_824
                    ))
                }
            case .compiling, .fetchingManifest, .idle:
                break
            }
            try await Task.sleep(nanoseconds: 2_000_000_000)
        }
        throw CoreMLDeviceTestError.download("timed out after \(Int(Self.downloadTimeout / 60)) min")
    }
}

private enum CoreMLDeviceTestError: Error, CustomStringConvertible {
    case download(String)
    var description: String {
        switch self {
        case .download(let message): return "download failed: \(message)"
        }
    }
}
