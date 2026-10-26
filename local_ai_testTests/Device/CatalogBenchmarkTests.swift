//
//  CatalogBenchmarkTests.swift
//
//  The Xcode entry point to the build test: one sweep over every catalog
//  model this build can actually run on this device, producing the JSON that
//  `Scripts/generate_benchmark_report.py` turns into the release report.
//
//  This test is a WRAPPER. The sweep itself lives in the app, in
//  `BuildTestRunner` - the same code the app's Testing screen runs, so a
//  release sweep started from Xcode and one started by tapping "Test this
//  build" produce byte-identical artefacts. All this file does is parse
//  environment variables into `BuildTestOptions`, call the runner, attach the
//  report to the result bundle and decide the pass/fail verdict.
//
//  It is NOT part of the normal suite. It downloads multiple gigabytes,
//  takes hours, and heats the phone up - so it is tagged `.benchmark` and
//  gated behind `BENCH_ENABLED`, and stays skipped until that is set.
//
//  ---------------------------------------------------------------------
//  Running it
//  ---------------------------------------------------------------------
//
//      xcodebuild test \
//        -project local_ai_test.xcodeproj -scheme local_ai_test \
//        -destination 'platform=iOS,id=<device-udid>' \
//        -only-testing:local_ai_testTests/CatalogBenchmarkTests \
//        TEST_RUNNER_BENCH_ENABLED=1
//
//  Xcode copies every build setting named `TEST_RUNNER_<NAME>` into the test
//  runner's environment as `<NAME>`, which is how these reach
//  `ProcessInfo.processInfo.environment` on the device.
//
//  ---------------------------------------------------------------------
//  Environment variables (all optional except the gate)
//  ---------------------------------------------------------------------
//
//    BENCH_ENABLED=1        Required. Without it the test skips cleanly.
//    BENCH_STAGES=full|perf Scope. `full` (default) runs load + throughput +
//                           the quality suite and produces a score; `perf`
//                           runs load + throughput only and produces none.
//    BENCH_MAX_MODELS=N     Stop after N models. Default: no cap.
//    BENCH_MODEL_IDS=a,b,c  Comma-separated allowlist of model ids. Default:
//                           every eligible model.
//    BENCH_LOG_JSON=1       Also dump the whole report JSON to the test log
//                           (useful when the result bundle is not available).
//    BENCH_REPS=N           Measured decode repetitions per model, clamped
//                           to 1...10. Default: the engine's
//                           (BenchmarkDefaults.decodeRepetitions).
//    BENCH_NO_PROTOCOL=1    Skip the pp512/tg128 standard-protocol probe.
//    BENCH_NO_THERMAL_GATE=1  Do not wait for a hot device to cool down -
//                           neither the engine's pre-measurement gate nor the
//                           between-model cooldown. Runs taken hot are still
//                           marked `thermallyConstrained`.
//    BENCH_QUALITY_BUDGET_SEC=N  Stop each model's quality stage after N
//                           seconds; the score is then partial and not
//                           comparable. Default: no budget, every task runs.
//    BENCH_INTERLEAVE=1     Run heaviest, lightest, second heaviest, ... by
//                           catalog ramRequiredGB, so a light model gives the
//                           phone a break. Default: catalog order.
//
//  Conditions gate, before every model (`RunConditionsGate`):
//
//    BENCH_COOLDOWN_TARGET=nominal|fair|off  Thermal state to wait for.
//                           Default nominal (fair was not enough: the phone
//                           went back to serious within minutes).
//    BENCH_COOLDOWN_SETTLE_SEC=N  Extra idle time once the target is reached
//                           after a wait. Default 60.
//    BENCH_COOLDOWN_MAX_SEC=N  Cap of one wait (thermal, power, network).
//                           Default 1200. A run measured after the cap ran
//                           out carries the reason in measurementIssues.
//    Power: Low Power Mode must be off and the phone charging/full or at
//    40% battery or more; the gate waits for that like for the temperature.
//    BENCH_STOP_ON_BAD_POWER=0|1  Power still bad at the cap: 1 (default)
//                           skips this and every later model and ends the
//                           sweep; 0 measures and flags.
//    BENCH_DOWNLOAD_STALL_SEC=N  Fail a download that moves no bytes for N
//                           seconds (default 300; 0 = only the 30-min
//                           budget). A lost network path is waited for
//                           (up to the cap) before each download.
//
//  There is deliberately NO timeout around the benchmark itself: the v2 suite
//  takes as long as it takes, and only the download step has a budget (30
//  min, in `BuildTestRunner`). The engine bounds each quality task with its
//  own per-task timeout.
//
//  ---------------------------------------------------------------------
//  Output
//  ---------------------------------------------------------------------
//
//  The runner prints every log line as it produces it, and it runs inside the
//  test host, so the sweep's progress appears in the xcodebuild log live -
//  there is nothing for this wrapper to mirror. The finished report is
//  written to `Documents/BuildTestReports/build-test-<timestamp>.json` on the
//  device by `BuildTestReportStore`, and attached to the test result here.
//
//  ---------------------------------------------------------------------
//  Verdict
//  ---------------------------------------------------------------------
//
//  A model that fails is recorded and the sweep moves on. The test itself
//  only fails when nothing was eligible, or when every eligible model failed.
//

import Foundation
import Testing
@testable import local_ai_test

@Suite(.tags(.device, .benchmark, .performance))
struct CatalogBenchmarkTests {

    @Test(
        "Release sweep: benchmark every eligible catalog model",
        .enabled(
            if: BenchmarkHarnessConfig.fromEnvironment().enabled,
            "set TEST_RUNNER_BENCH_ENABLED=1 to run the release benchmark sweep"
        )
    )
    @MainActor
    func benchmarkWholeCatalog() async throws {

        let config = BenchmarkHarnessConfig.fromEnvironment()
        let runner = BuildTestRunner.shared

        // Read in the morning: first, whether the phone was set up for the
        // night at all.
        NightChecklist.lines.forEach { print("[benchmark] " + $0) }

        let report = try await runner.run(options: config.options)

        // How many models the sweep considered runnable, as opposed to how
        // many produced an entry: a model skipped for disk space was eligible
        // but never attempted, and the "nothing was eligible" verdict below
        // has to tell those apart. The runner keeps this until the next run.
        let eligibleCount = runner.totalCount

        let json = BuildTestReportStore.exportJSON(report)

        if config.logJSON {
            print("[benchmark] ----- BEGIN BUILD TEST JSON -----")
            print(json)
            print("[benchmark] ----- END BUILD TEST JSON -----")
        }

        // Attached to the result bundle so a CI run does not need the
        // device's container to be pulled.
        Attachment.record(json, named: BuildTestReportStore.filename(for: report))

        // MARK: Verdict

        #expect(
            eligibleCount > 0,
            """
            No catalog model is eligible on this device. Either the catalog is \
            empty, no text backend is linked into this build, or BENCH_MODEL_IDS \
            named nothing runnable.
            """
        )
        #expect(
            !(report.passed == 0 && report.failed > 0),
            """
            Every eligible model failed to benchmark - see the attached \
            \(BuildTestReportStore.filename(for: report)).
            """
        )
        #expect(!report.cancelled, "the sweep was stopped before it finished")
        if let stopReason = runner.stopReason {
            Issue.record("the conditions gate stopped the sweep: \(stopReason)")
        }
    }
}
