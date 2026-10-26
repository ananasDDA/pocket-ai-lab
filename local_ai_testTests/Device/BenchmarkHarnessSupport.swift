//
//  BenchmarkHarnessSupport.swift
//
//  Environment plumbing for `CatalogBenchmarkTests`, and nothing else.
//
//  The sweep itself - model selection, downloads, the ownership marker,
//  cooldowns, cleanup, the artefact - lives in the app module now, in
//  `BuildTestRunner`. It moved there so the same run can be started from the
//  app's Testing screen and watched while it happens, instead of only from
//  Xcode with a blank screen on the device. This file is what is left: turn
//  environment variables into `BuildTestOptions`.
//
//  Anything you are looking for that used to be here -
//  `BenchmarkHarnessSelection`, `BenchmarkHarnessDownloader`,
//  `BenchmarkHarnessOwnership`, `BenchmarkHarnessThermal`,
//  `BenchmarkSweepSummary` - is in
//  `Modules/Benchmark/BuildTest/BuildTestRunner.swift` and
//  `BuildTestReport.swift`. The ownership marker filename in particular is
//  defined once, as `BuildTestRunner.ownershipMarkerName`.
//

import Foundation
import Testing
@testable import local_ai_test

// MARK: - Tag

extension Tag {
    /// Long-running release sweep. Declared here rather than in
    /// `DeviceTestTags.swift` so the benchmark harness stays self-contained;
    /// `@Tag` works from any extension of `Tag` in the test target.
    @Tag static var benchmark: Self
}

// MARK: - Configuration

/// Sweep scope, read from the process environment.
///
/// Under `xcodebuild` the runner's environment is populated from build
/// settings prefixed `TEST_RUNNER_`, so `TEST_RUNNER_BENCH_STAGES=perf`
/// on the command line arrives here as `BENCH_STAGES=perf`.
struct BenchmarkHarnessConfig {

    /// `BENCH_ENABLED=1` - the gate. Absent means "skip cleanly".
    let enabled: Bool
    /// `BENCH_LOG_JSON=1` - also dump the whole report JSON to the test log.
    /// Harness-only: it changes what the test prints, not what it runs, so it
    /// is not part of `BuildTestOptions`.
    let logJSON: Bool
    /// Everything the runner needs, parsed from the remaining variables.
    let options: BuildTestOptions

    /// Upper clamp for `BENCH_REPS`. Ten repetitions already give a stable
    /// spread, and a stray `BENCH_REPS=1000` must not turn one model into an
    /// hour of decode.
    static let maxDecodeRepetitions = 10

    static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> BenchmarkHarnessConfig {

        func value(_ key: String) -> String? {
            guard let raw = environment[key] else { return nil }
            let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }

        func flag(_ key: String) -> Bool {
            guard let raw = value(key)?.lowercased() else { return false }
            return raw == "1" || raw == "true" || raw == "yes"
        }

        // Anything other than an explicit "perf" means the full suite: a typo
        // in the scope must not silently produce an unscored release report
        // that looks like a complete one.
        let stages: BenchmarkOptions.Stages = {
            switch value("BENCH_STAGES")?.lowercased() {
            case "perf", "performance", "performanceonly": return .performanceOnly
            default: return .full
            }
        }()

        let maxModels = value("BENCH_MAX_MODELS").flatMap(Int.init).flatMap { $0 > 0 ? $0 : nil }

        // 0 or garbage falls back to the engine default rather than becoming
        // "no repetitions": the sweep must always measure something.
        let decodeRepetitions: Int? = value("BENCH_REPS").flatMap(Int.init).flatMap {
            guard $0 > 0 else { return nil }
            if $0 > maxDecodeRepetitions {
                print("[benchmark] BENCH_REPS=\($0) clamped to \(maxDecodeRepetitions)")
                return maxDecodeRepetitions
            }
            return $0
        }

        let modelIDs = (value("BENCH_MODEL_IDS") ?? "")
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        // `BENCH_NO_THERMAL_GATE=1` turns off BOTH the engine's own gate and
        // the between-model cooldown, exactly as it did before the split into
        // two flags. Nothing in the environment sets `cooldownBetweenModels`
        // on its own - that knob exists for in-app runs.
        let thermalGating = !flag("BENCH_NO_THERMAL_GATE")

        // `BENCH_QUALITY_BUDGET_SEC=600` stops each model's quality stage
        // after ten minutes (a partial, non-comparable score). Absent, zero
        // or garbage keeps the release default: no budget, every task runs.
        let qualityBudget = value("BENCH_QUALITY_BUDGET_SEC")
            .flatMap(TimeInterval.init)
            .flatMap(BenchmarkOptions.sanitizedBudget)

        // The conditions gate of an unattended run: cool to nominal plus a
        // settle period, power gate, download stall detection. Tuned with
        // BENCH_COOLDOWN_TARGET / _MAX_SEC / _SETTLE_SEC,
        // BENCH_STOP_ON_BAD_POWER and BENCH_DOWNLOAD_STALL_SEC.
        let conditions = RunConditionsPolicy.fromEnvironment(environment, prefix: "BENCH_")

        return BenchmarkHarnessConfig(
            enabled: flag("BENCH_ENABLED"),
            logJSON: flag("BENCH_LOG_JSON"),
            options: BuildTestOptions(
                stages: stages,
                modelIDs: modelIDs,
                maxModels: maxModels,
                decodeRepetitions: decodeRepetitions,
                runStandardProtocol: !flag("BENCH_NO_PROTOCOL"),
                thermalGating: thermalGating,
                cooldownBetweenModels: thermalGating,
                qualityBudget: qualityBudget,
                conditions: conditions,
                interleaveByWeight: flag("BENCH_INTERLEAVE")
            )
        )
    }
}
