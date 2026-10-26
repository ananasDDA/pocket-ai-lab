//
//  BuildTestRunner.swift
//  local_ai_test
//
//  The build test: one sweep over every catalog model this build can
//  actually run on this device. Download it, benchmark it, delete it again,
//  move on - and leave a `BuildTestReport` behind.
//
//  This used to live in the test target, which meant it could only be
//  started from Xcode and showed the tester a blank screen while it worked.
//  It lives in the app now, so:
//
//    * an in-app "Test this build" control can start it and watch it,
//    * `CatalogBenchmarkTests` is a thin wrapper that parses environment
//      variables and calls `run(options:)`,
//    * both frontends produce the SAME artefact, from the same code.
//
//  ---------------------------------------------------------------------
//  THE CLEANUP CONTRACT - read this before changing anything below
//  ---------------------------------------------------------------------
//
//  A build test downloads multiple gigabytes. It must leave the device
//  exactly as it found it, and it must NEVER delete a model the user
//  installed themselves.
//
//    * A model the runner downloaded is deleted again - on success, on
//      failure, on a thrown error, and on cancellation. The delete is in a
//      `defer` so an escaping throw cannot skip it.
//    * Ownership is recorded ON DISK (a `.harness-installed` marker inside
//      the model's own directory, written BEFORE the transfer starts), not
//      just in a local variable: a hard crash - a llama.cpp `GGML_ASSERT`
//      abort, say - kills the process without running any `defer`, and the
//      next run has to be able to tell "we downloaded this and died" from
//      "the user installed this". A model found installed WITH the marker is
//      reclaimed: benchmarked without re-downloading, then deleted.
//    * A model already installed and NOT marked is left in place, untouched,
//      forever. This is a hard product requirement, not an optimisation. The
//      user's models are the user's.
//    * A transfer that is ALREADY IN FLIGHT belongs to the user. There is one
//      `ModelDownloader` state per model, so "the download the user started"
//      and "the download the runner would start" are the same transfer: the
//      runner's `download(_:)` would be a no-op, it would wait for the user's
//      bytes, benchmark them and then delete them. So a model whose state is
//      active (`fetchingManifest` / `downloading` / `compiling`) when the
//      sweep reaches it is never marked and never touched - it is recorded as
//      a skip. That also stops a stale non-terminal state from burning the
//      30-minute download deadline on a call that does nothing.
//    * "Not installed" is NOT proof of ownership. A complete model whose
//      `.complete` marker went missing reads back as uninstalled, and
//      `ModelDownloader.startDownload` then short-circuits it straight to
//      `.installed` without moving a byte. Ownership is therefore claimed
//      only once this run has been SEEN to move bytes: the download waiter
//      observed `.downloading` or `.compiling`, or the model's directory was
//      empty before the transfer started (so whatever ends up installed can
//      only be ours). `.fetchingManifest` does not count - it is set before
//      the manifest says whether there is anything to fetch. A first observed
//      state of `.installed` over a directory that already held files means
//      the user had it: not owned, not deleted.
//    * Orphans from a crashed run are reclaimed at the START of the sweep,
//      before a partial's own bytes can make the free-disk check skip the
//      model and strand it forever. See `reclaimOrphanedMarkers`.
//
//  ---------------------------------------------------------------------
//  Per model
//  ---------------------------------------------------------------------
//
//    1. Skip if a transfer is already in flight, reclaim if marked,
//       otherwise download if needed (free-disk check first; too little
//       space is a skip, not a failure). Per-model download deadline: 30
//       minutes.
//    2. Cool down if the previous model left the device at `serious` or
//       hotter - 10 s polls, up to 5 minutes. Deliberately AFTER the
//       download: a multi-gigabyte transfer re-warms the device, so waiting
//       before it would be wasted. Skipped for the first model.
//    3. Benchmark it with `BenchmarkEngine.runAndStore`, so the per-model
//       report is in the normal history too.
//    4. Delete it if the runner owns it, and unload the engine so the next
//       model's load time is measured from cold.
//
//  A model that fails is recorded and the sweep moves on; only the whole run
//  throws, and only for reasons that make the run itself meaningless.
//

import Foundation
import Observation

// MARK: - Options

/// Scope of one build test. `.default` is the release sweep: every eligible
/// model, full stages, engine defaults.
///
/// Codable because it is embedded verbatim in the report - a stored artefact
/// should say what produced it.
nonisolated struct BuildTestOptions: Codable, Sendable, Hashable {

    /// `.full` runs load + throughput + the quality suite and produces a
    /// score; `.performanceOnly` measures speed and produces none.
    var stages: BenchmarkOptions.Stages

    /// Allowlist of model ids. Empty means "every eligible model". Ids that
    /// name nothing eligible are recorded as skips rather than dropped - a
    /// typo has to be visible in the artefact.
    var modelIDs: [String]

    /// Stop after this many models. `nil` means no cap.
    var maxModels: Int?

    /// Measured decode repetitions per model. `nil` keeps the engine default
    /// (`BenchmarkDefaults.decodeRepetitions`).
    var decodeRepetitions: Int?

    /// Run the pp512 / tg128 standard-protocol probe.
    var runStandardProtocol: Bool

    /// Let the engine wait for a hot device before measuring. Turning it off
    /// does not turn off the `thermallyConstrained` flag on the results.
    var thermalGating: Bool

    /// Run the thermal half of the conditions gate before each model (cool
    /// to the target of `conditionsPolicy`). Separate from `thermalGating`
    /// so a quick smoke sweep can keep the engine's own gate and drop only
    /// the long between-model wait. The power half does not depend on it.
    var cooldownBetweenModels: Bool

    /// Quality-stage time budget per model, in seconds. `nil` (the default)
    /// runs every task: the build test and the release sweep publish full
    /// scores, never partial ones. The harness can set it with
    /// `TEST_RUNNER_BENCH_QUALITY_BUDGET_SEC` for a quicker look. Optional,
    /// so journals and reports written before it existed still decode.
    var qualityBudget: TimeInterval?

    /// What the run waits for before each model: temperature, power,
    /// network (`RunConditionsGate`). `nil` - the in-app run - means
    /// `RunConditionsPolicy.interactive`; the Xcode harness sets the night
    /// policy. Optional, like `qualityBudget`, so older journals decode.
    var conditions: RunConditionsPolicy?

    /// Run the models heaviest, lightest, second heaviest, ... by catalog
    /// `ramRequiredGB`, so a light model gives the phone a break after a
    /// heavy one. `nil` or false keeps catalog order.
    var interleaveByWeight: Bool?

    /// The policy in force.
    var conditionsPolicy: RunConditionsPolicy { conditions ?? .interactive }

    init(
        stages: BenchmarkOptions.Stages = .full,
        modelIDs: [String] = [],
        maxModels: Int? = nil,
        decodeRepetitions: Int? = nil,
        runStandardProtocol: Bool = true,
        thermalGating: Bool = true,
        cooldownBetweenModels: Bool = true,
        qualityBudget: TimeInterval? = nil,
        conditions: RunConditionsPolicy? = nil,
        interleaveByWeight: Bool? = nil
    ) {
        self.stages = stages
        self.modelIDs = modelIDs
        self.maxModels = maxModels
        self.decodeRepetitions = decodeRepetitions
        self.runStandardProtocol = runStandardProtocol
        self.thermalGating = thermalGating
        self.cooldownBetweenModels = cooldownBetweenModels
        self.qualityBudget = BenchmarkOptions.sanitizedBudget(qualityBudget)
        self.conditions = conditions
        self.interleaveByWeight = interleaveByWeight
    }

    /// The release sweep.
    static let `default` = BuildTestOptions()

    /// Engine options for this scope. `unloadWhenFinished` is always true:
    /// the next model in the sweep has to load from cold or its load time and
    /// peak memory measure the previous model's leftovers.
    var benchmarkOptions: BenchmarkOptions {
        var options = BenchmarkOptions(stages: stages, unloadWhenFinished: true)
        if let decodeRepetitions {
            options.decodeRepetitions = decodeRepetitions
        }
        options.runStandardProtocol = runStandardProtocol
        options.thermalGating = thermalGating
        options.qualityBudget = qualityBudget
        return options
    }

    /// One line for a log header or a UI subtitle.
    var summaryLine: String {
        var parts = ["stages: \(stages.rawValue)"]
        if !modelIDs.isEmpty { parts.append("models: \(modelIDs.joined(separator: ", "))") }
        if let maxModels { parts.append("cap: \(maxModels)") }
        if let decodeRepetitions { parts.append("reps: \(decodeRepetitions)") }
        if !runStandardProtocol { parts.append("no pp512/tg128") }
        if !thermalGating { parts.append("no thermal gate") }
        if !cooldownBetweenModels { parts.append("no cooldown") }
        if let qualityBudget { parts.append("quality budget: \(Int(qualityBudget)) s") }
        if let conditions { parts.append("conditions: \(conditions.summaryLine)") }
        if interleaveByWeight == true { parts.append("heavy/light interleaved") }
        return parts.joined(separator: ", ")
    }
}

// MARK: - Errors

/// Failures the runner itself produces. A benchmark failure is a
/// `BenchmarkError` and is recorded per model, not thrown out of the sweep.
nonisolated enum BuildTestError: Error, LocalizedError, Sendable {

    /// `run` was called while a sweep was already in flight.
    case alreadyRunning
    /// `ModelDownloader` reported `.error(...)` for this model.
    case downloadFailed(String)
    /// The download did not reach `.installed` within the per-model budget.
    case downloadTimedOut(seconds: TimeInterval, lastState: String)
    /// No byte arrived for `seconds` (`RunConditionsPolicy.downloadStallTimeout`).
    case downloadStalled(seconds: TimeInterval, network: String)
    /// The conditions gate ran out with the power still bad and
    /// `RunConditionsPolicy.stopOnBadPower` set: this model and every later
    /// one are skipped.
    case runStopped(String)

    var errorDescription: String? {
        switch self {
        case .alreadyRunning:
            return "A build test is already running."
        case .downloadFailed(let detail):
            return "download failed: \(detail)"
        case .downloadTimedOut(let seconds, let lastState):
            return "download did not finish within \(Int(seconds)) s "
                + "(last state: \(lastState))"
        case .downloadStalled(let seconds, let network):
            return "download stalled: no bytes for \(Int(seconds)) s (network \(network))"
        case .runStopped(let detail):
            return "run stopped: \(detail)"
        }
    }
}

// MARK: - Download progress

/// Bytes and fraction of the model currently being fetched, for a progress
/// bar. `nil` on the runner while nothing is downloading.
nonisolated struct BuildTestDownloadProgress: Sendable, Hashable {
    /// 0…1.
    let fraction: Double
    let downloadedBytes: Int64
    let totalBytes: Int64

    /// "1.24 / 4.10 GB", or just the fraction when the total is unknown.
    var byteLine: String {
        guard totalBytes > 0 else { return String(format: "%.0f%%", fraction * 100) }
        return String(
            format: "%.2f / %.2f GB",
            Double(downloadedBytes) / 1_073_741_824,
            Double(totalBytes) / 1_073_741_824
        )
    }
}

// MARK: - Runner

/// Runs a build test and publishes its progress. `@Observable`, so a SwiftUI
/// screen can bind straight to `phase`, `log`, `completedEntries` and watch a
/// multi-hour sweep happen.
@MainActor
@Observable
final class BuildTestRunner {

    static let shared = BuildTestRunner()

    // MARK: Phase

    /// Where the sweep is right now. Raw values are stable enough to log.
    enum Phase: String, Sendable, Hashable, CaseIterable {
        /// Nothing is running. The starting state and the state after a
        /// finished run is read.
        case idle
        /// Deciding which models to run.
        case selecting
        /// Fetching the current model.
        case downloading
        /// Measuring the current model.
        case benchmarking
        /// Waiting for the device to cool between models.
        case coolingDown
        /// Deleting what the runner downloaded.
        case cleaningUp
        /// The sweep completed. `lastReport` holds the artefact.
        case finished
        /// The sweep was stopped early. `lastReport` holds what it got to.
        case cancelled

        var displayName: String {
            switch self {
            case .idle:         return "Idle"
            case .selecting:    return "Selecting models"
            case .downloading:  return "Downloading"
            case .benchmarking: return "Benchmarking"
            case .coolingDown:  return "Cooling down"
            case .cleaningUp:   return "Cleaning up"
            case .finished:     return "Finished"
            case .cancelled:    return "Cancelled"
            }
        }

        /// True while a sweep is in flight - the phases a Stop button applies
        /// to.
        var isActive: Bool {
            switch self {
            case .idle, .finished, .cancelled: return false
            default: return true
            }
        }
    }

    // MARK: Observable state

    /// True from the moment `run` is entered until its report is stored.
    private(set) var isRunning = false

    private(set) var phase: Phase = .idle {
        didSet { journalPhaseChange() }
    }

    /// 1-based position of the model being worked on, 0 before the first.
    private(set) var currentIndex = 0

    /// How many models the sweep will attempt (skips decided during
    /// selection are NOT counted here - they are in `completedEntries`).
    private(set) var totalCount = 0

    /// The model being downloaded or benchmarked right now.
    private(set) var currentModel: AIModel?

    /// Progress of the current download, `nil` when nothing is downloading.
    private(set) var downloadProgress: BuildTestDownloadProgress?

    /// Progress of the current benchmark, `nil` outside the benchmark stage.
    private(set) var benchmarkProgress: BenchmarkProgress?

    /// Finished models, in sweep order, growing as the run goes. This is
    /// exactly what ends up in the report's `entries`.
    private(set) var completedEntries: [BuildTestEntry] = []

    /// Latest thermal reading, as `ProcessInfo.ThermalState.reportName`
    /// ("nominal", "fair", "serious", "critical").
    private(set) var thermalState: String = ProcessInfo.processInfo.thermalState.reportName

    /// When the current (or last) run started.
    private(set) var startedAt: Date?

    /// Set when the conditions gate stopped the run (power still bad at the
    /// cap). Every model from that one on is recorded as skipped with it.
    private(set) var stopReason: String?

    /// Human-readable transcript, newest last. Every line here was also
    /// printed, which is how the Xcode wrapper gets its console output.
    private(set) var log: [String] = []

    /// The artefact of the last finished or cancelled run.
    private(set) var lastReport: BuildTestReport?

    /// The report reconstructed at launch from a journal the previous
    /// process left behind, until the user has seen it (a new run clears
    /// it). `nil` when the last run ended on its own terms.
    private(set) var interruptedReport: BuildTestReport?

    /// Options that run only the models the interrupted run never reached.
    /// `nil` when there is nothing to resume.
    private(set) var resumableOptions: BuildTestOptions?

    /// The crash insurance: rewritten at every step worth surviving (see
    /// BuildTestJournal.swift). `nil` outside a run.
    private var journal: BuildTestJournal?

    // MARK: Derived

    /// 0…1 across the whole sweep: finished models plus the current model's
    /// own progress. Rough by construction - models are not equal in cost -
    /// but honest enough for a progress bar.
    ///
    /// Each phase owns a slice of one model and starts where the previous
    /// phase ended - download 0…0.5, cooldown parked at 0.5, benchmark
    /// 0.5…1, cleanup 1 - so the bar cannot walk backwards when the phase
    /// changes. The clamps keep it monotonic within a phase too, whatever a
    /// progress callback reports.
    var overallFraction: Double {
        guard totalCount > 0 else { return 0 }
        let finished = Double(max(currentIndex - 1, 0))
        let inner: Double
        switch phase {
        case .downloading:  inner = clampedFraction(downloadProgress?.fraction) * 0.5
        case .coolingDown:  inner = 0.5
        case .benchmarking: inner = 0.5 + clampedFraction(benchmarkProgress?.fraction) * 0.5
        case .cleaningUp:   inner = 1
        default:            inner = 0
        }
        return min((finished + inner) / Double(totalCount), 1)
    }

    private func clampedFraction(_ value: Double?) -> Double {
        min(max(value ?? 0, 0), 1)
    }

    /// "Benchmarking - Qwen3 4B (2 of 7)" and friends: one line a screen can
    /// show without knowing the phase enum.
    var statusLine: String {
        guard phase.isActive else { return phase.displayName }
        var line = phase.displayName
        if let currentModel { line += " - \(currentModel.name)" }
        if totalCount > 0, currentIndex > 0 { line += " (\(currentIndex) of \(totalCount))" }
        return line
    }

    /// Seconds since the run started, or `nil` when it never did.
    var elapsedSeconds: Double? {
        guard let startedAt else { return nil }
        return Date().timeIntervalSince(startedAt)
    }

    // MARK: Private

    @ObservationIgnored
    private var runningTask: Task<BuildTestReport, Never>?

    @ObservationIgnored
    private let store: BuildTestReportStore

    /// Oldest lines are dropped past this - a multi-hour sweep must not grow
    /// an unbounded array in a foreground app.
    @ObservationIgnored
    private let maxLogLines = 2_000

    init(store: BuildTestReportStore? = nil) {
        // Not a default argument: a default expression is evaluated in a
        // nonisolated context and `BuildTestReportStore.shared` is
        // main-actor bound.
        self.store = store ?? .shared
    }

    // MARK: - API

    /// Runs a build test to completion and returns its report.
    ///
    /// Never throws for a per-model problem: a model that fails to download
    /// or benchmark becomes a `.failed` entry and the sweep continues. It
    /// throws `BuildTestError.alreadyRunning` when a sweep is already in
    /// flight, and that is the only error it produces.
    ///
    /// Cancellation - `cancel()`, or the calling task being torn down - is
    /// not an error either: the run stops after the current model, cleans up
    /// anything it downloaded, and returns a report with `cancelled == true`
    /// and the entries it managed to collect. That report is stored like any
    /// other; a sweep that was stopped four models in is still four models
    /// of data.
    @discardableResult
    func run(options: BuildTestOptions = .default) async throws -> BuildTestReport {
        guard !isRunning else { throw BuildTestError.alreadyRunning }
        isRunning = true
        interruptedReport = nil
        resumableOptions = nil

        let task = Task { @MainActor in
            await self.sweep(options: options)
        }
        runningTask = task

        // Forward cancellation of the CALLING task into the sweep. `Task {}`
        // is unstructured, so without this a caller that goes away would
        // leave a multi-hour sweep running with nobody watching it.
        let report = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }

        runningTask = nil
        isRunning = false
        return report
    }

    /// Stops the run after the current model. The in-flight download is
    /// cancelled and its partial files removed, anything the runner owns is
    /// deleted, and the partial report is still produced and stored.
    ///
    /// Safe to call when nothing is running.
    func cancel() {
        guard let runningTask else { return }
        appendLog("[build-test] cancelling - finishing the current model's cleanup")
        runningTask.cancel()
    }

    // MARK: - The sweep

    private func sweep(options: BuildTestOptions) async -> BuildTestReport {

        // MARK: Reset

        let started = Date()
        startedAt = started
        phase = .selecting
        currentIndex = 0
        totalCount = 0
        currentModel = nil
        downloadProgress = nil
        benchmarkProgress = nil
        completedEntries = []
        log = []
        lastReport = nil
        stopReason = nil
        refreshThermalState()

        // MARK: Selection

        let device = DeviceScanner.shared.scan()
        let deviceSnapshot = DeviceSnapshot(device)
        var eligible = Self.eligibleModels(device: device)

        // Allowlist. Ids that name nothing eligible are recorded rather than
        // silently dropped - a typo in the release command must be visible in
        // the artefact, not just absent from it. Deduped as we go: an id
        // repeated in the allowlist must not produce two entries, because
        // `BuildTestEntry.id` is the model id and a duplicate id breaks every
        // `ForEach` and every by-id lookup over the report.
        if !options.modelIDs.isEmpty {
            let allowed = Set(options.modelIDs)
            let matched = eligible.filter { allowed.contains($0.id) }
            let matchedIDs = Set(matched.map(\.id))
            var reportedIDs: Set<String> = []
            for id in options.modelIDs {
                guard !matchedIDs.contains(id), reportedIDs.insert(id).inserted else { continue }
                let reason = "skipped: named in the model allowlist but not eligible "
                    + "on this device (or not in the catalog)"
                if let model = ModelCatalog.model(id: id) {
                    record(.skipped(model, reason: reason))
                } else {
                    record(.skipped(unknownModelId: id, reason: reason))
                }
            }
            eligible = matched
        }

        if let cap = options.maxModels, eligible.count > cap {
            for model in eligible.dropFirst(cap) {
                record(.skipped(
                    model, reason: "skipped: past the cap of \(cap) models"
                ))
            }
            eligible = Array(eligible.prefix(cap))
        }

        // After the cap, so the cap still picks the same models; only their
        // order changes.
        if options.interleaveByWeight == true, eligible.count > 2 {
            eligible = RunOrdering.interleaveHeavyLight(eligible) { $0.ramRequiredGB }
        }

        totalCount = eligible.count

        journal = BuildTestJournal(
            runID: UUID().uuidString,
            startedAt: started,
            appVersion: BenchmarkAppInfo.version,
            appBuild: BenchmarkAppInfo.build,
            device: deviceSnapshot,
            options: options,
            plannedModelIDs: eligible.map(\.id),
            entries: completedEntries,
            logTail: Array(log.suffix(BuildTestJournal.logTailLines)),
            updatedAt: Date()
        )
        writeJournal()

        appendLog("[build-test] device: \(deviceSnapshot.marketingName) "
            + "(\(deviceSnapshot.chip), \(deviceSnapshot.ramGB) GB, iOS \(deviceSnapshot.iOSVersion))")
        appendLog("[build-test] app: \(BenchmarkAppInfo.version) (\(BenchmarkAppInfo.build))")
        appendLog("[build-test] options: \(options.summaryLine)")
        appendLog("[build-test] eligible models: \(eligible.count), "
            + "pre-skipped: \(completedEntries.count)")
        if options.interleaveByWeight == true {
            appendLog("[build-test] order (heavy/light): " + eligible
                .map { String(format: "%@ %.1f GB", $0.id, $0.ramRequiredGB) }
                .joined(separator: ", "))
        }

        // MARK: Reclaim

        // Before anything else, and before any free-disk arithmetic: a
        // previous run's orphans are this run's bytes to give back.
        await reclaimOrphanedMarkers(eligibleIDs: Set(eligible.map(\.id)))

        // MARK: Loop

        var cancelled = false

        for (index, model) in eligible.enumerated() {
            // Cancellation between models: nothing is in flight and nothing
            // was downloaded, so there is nothing to clean up - just stop.
            if Task.isCancelled {
                cancelled = true
                appendLog("[build-test] cancelled before \(model.name)")
                break
            }

            currentIndex = index + 1
            currentModel = model
            downloadProgress = nil
            benchmarkProgress = nil
            refreshThermalState()
            journalPickup(model)

            appendLog("[build-test] (\(index + 1)/\(eligible.count)) "
                + "\(model.name) [\(model.backend.displayName)] \(model.id)")

            // A transfer already in flight is the user's - there is one
            // downloader state per model, so joining it would mean waiting
            // for the user's bytes and then deleting them. Skip without
            // marking anything. This also covers a stale non-terminal state,
            // which would otherwise make `download(_:)` a no-op and spend the
            // whole 30-minute deadline watching nothing happen.
            if ModelDownloader.shared.state(for: model).isActive {
                let reason = "skipped: this model is already being downloaded"
                appendLog("  \(reason)")
                record(.skipped(model, reason: reason))
                continue
            }

            // Reclaim BEFORE the free-disk check, not after: a partial left
            // by a crashed run occupies the very space the check is about, so
            // checking first would skip the model for disk and leave the
            // partial - and its marker - on the device forever.
            if Self.isOwned(model), !Self.isInstalled(model) {
                phase = .cleaningUp
                await Self.deleteAndWait(model)
                appendLog("  [cleanup] reclaimed partial from a previous run")
            }

            let preInstalled = Self.isInstalled(model)
            let needsDownload = !preInstalled && Self.requiresDownload(model)

            if needsDownload, let reason = Self.insufficientSpaceReason(for: model) {
                appendLog("  \(reason)")
                record(.skipped(model, reason: reason))
                continue
            }

            // The cleanup contract, in one flag: true means "these files
            // arrived because of this run and must not outlive it". It is set
            // by `downloadAndWait` the moment this run is seen to move bytes -
            // not before, because "not installed" is not proof of ownership -
            // and from that moment on a transfer that fails half way still
            // gets its partial files removed.
            var ownedByRunner = false
            defer {
                // Belt and braces for the paths that leave this iteration
                // without reaching the awaited delete below - a throw that
                // escapes the sweep, or a torn-down task. The awaited version
                // is preferred because it also waits for the files to go
                // away, but this one cannot be skipped.
                if ownedByRunner {
                    ModelDownloader.shared.delete(model)
                }
            }

            let policy = options.conditionsPolicy

            do {
                // A lost network path is waited out before the transfer
                // starts rather than discovered by a stalled download.
                if needsDownload, policy.downloadStallTimeout != nil {
                    _ = try await RunConditionsWaiter.waitForNetwork(
                        maxWait: policy.maxWait, log: { self.appendLog("  " + $0) }
                    )
                }

                if needsDownload {
                    phase = .downloading
                    // `downloadAndWait` flips the flag and writes the on-disk
                    // marker itself, at the first moment this run is seen to
                    // be moving bytes - so a crash from there on still leaves
                    // proof that these files are ours, and a model that was
                    // simply already complete on disk is left alone.
                    try await downloadAndWait(
                        model,
                        ownedByRunner: &ownedByRunner,
                        stallTimeout: policy.downloadStallTimeout
                    )
                } else if preInstalled {
                    if Self.isOwned(model) {
                        // Already installed AND still marked: a previous run
                        // downloaded this and died before its cleanup. Finish
                        // that run's job - benchmark it (the files are
                        // complete) then delete it below.
                        ownedByRunner = true
                        appendLog("  [download] reclaimed from a previous run")
                    } else {
                        appendLog("  [download] already installed by the user - left in place")
                    }
                }

                // Conditions gate (cool-down + power), immediately before the
                // measurement so a long download cannot re-warm the device
                // after the wait. The in-app run skips the thermal half for
                // the first model - the engine's own gate covers a run started
                // hot; an unattended run (explicit policy) gates every model.
                let checkThermal = options.thermalGating && options.cooldownBetweenModels
                    && (index > 0 || options.conditions != nil)
                var gate = RunConditionsOutcome.clear
                if checkThermal || policy.powerGate {
                    phase = .coolingDown
                    gate = try await RunConditionsWaiter.wait(
                        policy: policy,
                        checkThermal: checkThermal,
                        log: { self.appendLog("  " + $0) },
                        onPoll: { self.refreshThermalState() }
                    )
                }

                // The engine marks the run invalid if one of these holds while
                // it measures; logging the state here puts the cause next to
                // the number in the console.
                appendLog("  [conditions] " + MeasurementConditions.logLine())

                if gate.stop {
                    throw BuildTestError.runStopped(gate.issues.joined(separator: "; "))
                }
                var benchmarkOptions = options.benchmarkOptions
                benchmarkOptions.preRunIssues = gate.issues

                phase = .benchmarking
                var lastStage: BenchmarkProgress.Stage?
                var lastLabel: String?
                let report = try await BenchmarkEngine.shared.runAndStore(
                    model: model,
                    options: benchmarkOptions
                ) { [weak self] progress in
                    guard let self else { return }
                    self.benchmarkProgress = progress
                    // The benchmark is where the device actually heats up, and
                    // it is the longest stretch of the run: without this the
                    // thermal chip would sit on the reading taken before the
                    // download and only move between models.
                    self.refreshThermalState()
                    // Log on a stage OR label change (never the same line
                    // twice): the engine re-labels within a stage - e.g.
                    // "Waiting for the device to cool down" - and a log that
                    // hides those reads like an unexplained stall.
                    guard progress.stage != lastStage || progress.label != lastLabel
                    else { return }
                    lastStage = progress.stage
                    lastLabel = progress.label
                    self.appendLog("  [\(progress.stage.rawValue)] \(progress.label)")
                    self.journalProgress(stage: progress.stage.rawValue, label: progress.label)
                }

                appendLog("  " + report.summaryLine)
                record(.completed(report))

            } catch is CancellationError {
                // A cancelled model is a lost model; unwind rather than
                // recording a bogus failure. Delete and WAIT here rather than
                // leaning on the synchronous `defer`: `ModelDownloader.delete`
                // unlinks the directory inside a background-session
                // `getAllTasks` completion that may never fire while the task
                // is being torn down. Flip the flag first so the `defer` does
                // not fire a second delete.
                appendLog("  cancelled")
                cancelled = true
                if ownedByRunner {
                    ownedByRunner = false
                    phase = .cleaningUp
                    ModelDownloader.shared.cancel(model)
                    await Self.deleteAndWait(model)
                    appendLog("  [cleanup] removed \(model.id) (cancelled)")
                }
                break

            } catch BuildTestError.runStopped(let detail) {
                // Not a failure of the model: the phone is in no state to
                // measure anything. Recorded as a skip, cleaned up below, and
                // the loop ends after the cleanup.
                stopReason = detail
                let reason = "skipped: run stopped - \(detail)"
                appendLog("  " + reason)
                record(.skipped(model, reason: reason))

            } catch {
                let detail = (error as? LocalizedError)?.errorDescription
                    ?? error.localizedDescription
                appendLog("  FAILED: \(detail)")
                record(.failed(model, error: detail))
            }

            if ownedByRunner {
                ownedByRunner = false
                phase = .cleaningUp
                await Self.deleteAndWait(model)
                appendLog("  [cleanup] removed \(model.id)")
            }

            // The engine unloads itself on success (`unloadWhenFinished`) and
            // on failure, but an aborted load can leave a half-warm engine
            // behind. The next model's load time is measured from cold or it
            // is not worth measuring.
            await InferenceManager.shared.unloadCurrent()

            if let stopReason {
                for later in eligible.dropFirst(index + 1) {
                    record(.skipped(later, reason: "skipped: run stopped before this model - \(stopReason)"))
                }
                appendLog("[build-test] STOPPED after \(index + 1) of \(eligible.count) models: \(stopReason). "
                    + "Check the phone: Low Power Mode off, cable in and charging.")
                break
            }
        }

        // MARK: Artefact

        currentModel = nil
        downloadProgress = nil
        benchmarkProgress = nil
        refreshThermalState()

        let report = BuildTestReport(
            date: Date(),
            appVersion: BenchmarkAppInfo.version,
            appBuild: BenchmarkAppInfo.build,
            stages: options.stages,
            device: deviceSnapshot,
            options: options,
            entries: completedEntries,
            durationSeconds: Date().timeIntervalSince(started),
            cancelled: cancelled
        )

        let url = try? store.save(report)
        lastReport = report
        // Only once the report is on disk: a death between the two writes
        // must still leave a journal to recover from, never neither.
        journal = nil
        if url != nil { BuildTestJournalFile.remove() }
        phase = cancelled ? .cancelled : .finished

        appendLog("[build-test] \(report.summaryLine)")
        appendLog("[build-test] report: \(url?.path ?? "(could not be written to Documents)")")

        return report
    }

    // MARK: - Reclaim

    /// Gives back everything a previous run left marked, before the sweep
    /// starts measuring anything.
    ///
    /// A crash - a llama.cpp `GGML_ASSERT` abort, the watchdog, a force quit -
    /// skips every `defer`, so it can leave two kinds of orphan behind:
    ///
    ///   * a marked directory that is NOT installed: a partial from a
    ///     mid-download death. It has to go here rather than in the loop,
    ///     because its bytes are exactly the free space
    ///     `insufficientSpaceReason` is about - check disk first and the model
    ///     is skipped for space and its partial stranded, run after run.
    ///   * a marked directory that IS installed but is not in this sweep's
    ///     eligible list - a narrower allowlist, a lower cap, or a device that
    ///     no longer finds it compatible. Nothing later in this run will ever
    ///     look at it, so nothing later would delete it either.
    ///
    /// Marked models that ARE eligible are deliberately left: the loop
    /// reclaims them properly, by benchmarking the complete files first and
    /// deleting them afterwards.
    ///
    /// Swept over the whole catalog, not the eligible list, for the same
    /// reason: eligibility is a property of this run, ownership is a property
    /// of the disk.
    private func reclaimOrphanedMarkers(eligibleIDs: Set<String>) async {
        for model in ModelCatalog.all where Self.isOwned(model) {
            // Files that are moving right now are not ours to remove, marker
            // or no marker; a later run will find them at rest.
            if ModelDownloader.shared.state(for: model).isActive { continue }

            let installed = Self.isInstalled(model)
            if installed, eligibleIDs.contains(model.id) { continue }

            phase = .cleaningUp
            await Self.deleteAndWait(model)
            appendLog(installed
                ? "[build-test] [cleanup] \(model.id): reclaimed from a previous run"
                : "[build-test] [cleanup] \(model.id): reclaimed partial from a previous run")
        }
        phase = .selecting
    }

    // MARK: - Logging

    /// Appends one line and prints it. Printing is not a debug leftover: it
    /// is how the Xcode wrapper's console output happens, since the runner
    /// executes inside the test host.
    private func appendLog(_ line: String) {
        print(line)
        log.append(line)
        if log.count > maxLogLines {
            log.removeFirst(log.count - maxLogLines)
        }
        // Not written here - the next journal write picks it up. A log line
        // per file write would be the wrong trade for a tail nobody reads
        // until something has gone wrong.
        journal?.logTail.append(line)
        if let count = journal?.logTail.count, count > BuildTestJournal.logTailLines {
            journal?.logTail.removeFirst(count - BuildTestJournal.logTailLines)
        }
    }

    // MARK: - Journal

    /// Appends to `completedEntries` and journals it: an entry is final the
    /// moment it exists, so it must be on disk before the next model can
    /// take the process down.
    private func record(_ entry: BuildTestEntry) {
        completedEntries.append(entry)
        journal?.entries = completedEntries
        journal?.inFlight = nil
        writeJournal()
    }

    private func journalPickup(_ model: AIModel) {
        guard journal != nil else { return }
        journal?.inFlight = BuildTestJournal.InFlight(
            modelId: model.id,
            modelName: model.name,
            backend: model.backend.displayName,
            phase: phase.displayName,
            stage: nil,
            label: nil,
            since: Date(),
            availableMemoryMB: InferenceManager.availableMemoryMB(),
            thermalState: thermalState
        )
        writeJournal()
    }

    private func journalPhaseChange() {
        guard journal?.inFlight != nil else { return }
        journal?.inFlight?.phase = phase.displayName
        journal?.inFlight?.availableMemoryMB = InferenceManager.availableMemoryMB()
        journal?.inFlight?.thermalState = thermalState
        writeJournal()
    }

    private func journalProgress(stage: String, label: String) {
        guard journal?.inFlight != nil else { return }
        journal?.inFlight?.stage = stage
        journal?.inFlight?.label = label
        journal?.inFlight?.availableMemoryMB = InferenceManager.availableMemoryMB()
        journal?.inFlight?.thermalState = thermalState
        writeJournal()
    }

    private func writeJournal() {
        guard var journal else { return }
        journal.updatedAt = Date()
        self.journal = journal
        do {
            try BuildTestJournalFile.write(journal)
        } catch {
            // Said once per run, not once per step: the run itself is fine,
            // only its insurance is not.
            if !journalWriteFailureLogged {
                journalWriteFailureLogged = true
                appendLog("[build-test] journal could not be written: \(error.localizedDescription)")
            }
        }
    }

    private var journalWriteFailureLogged = false

    // MARK: - Recovery

    /// Turns a journal the previous process left behind into a stored
    /// report, frees whatever that run had downloaded, and remembers what
    /// is left to resume. Call once, at launch. No-op when there is no
    /// journal or a run is in progress.
    @discardableResult
    func recoverInterruptedRun() -> BuildTestReport? {
        guard !isRunning, let journal = BuildTestJournalFile.read() else { return nil }
        journalWriteFailureLogged = false

        let report = BuildTestRecovery.report(
            from: journal,
            catalog: ModelCatalog.all,
            diagnostics: CrashDiagnosticsCollector.summaries(),
            now: Date()
        )
        do {
            try store.save(report)
        } catch {
            // Nothing to do but keep the journal: a launch that cannot write
            // to Documents will try again next time.
            appendLog("[build-test] could not store the recovered report: \(error.localizedDescription)")
            return nil
        }
        BuildTestJournalFile.remove()

        lastReport = report
        interruptedReport = report
        let remaining = BuildTestRecovery.remainingModelIDs(in: journal)
        if !remaining.isEmpty {
            var options = journal.options
            options.modelIDs = remaining
            options.maxModels = nil
            resumableOptions = options
        }

        appendLog("[build-test] recovered an interrupted run: \(report.summaryLine)")

        // The crashed model's files are this runner's to give back, exactly
        // as they would have been at the end of the run - but only when the
        // marker says the run downloaded them. A model the user installed
        // stays.
        if let inFlight = journal.inFlight,
           let model = ModelCatalog.model(id: inFlight.modelId),
           Self.isOwned(model),
           !ModelDownloader.shared.state(for: model).isActive {
            Task { @MainActor in
                await Self.deleteAndWait(model)
                self.appendLog("[build-test] [cleanup] removed \(model.id) (left by the interrupted run)")
            }
        }
        return report
    }

    /// MetricKit usually delivers a crash a few seconds after the launch
    /// that recovered it. If the latest interrupted report's crashed entry
    /// has no diagnostic yet and one of these fits its window, the report
    /// is rewritten with it.
    func attachCrashDiagnostics(_ summaries: [CrashSummary]) {
        guard let report = store.reports.first(where: { $0.interrupted }),
              let crashed = report.crashedEntry,
              let error = crashed.error,
              !error.contains("MetricKit:") else { return }
        // The run ended at `report.date` minus nothing useful; the crashed
        // model started at most a few hours before that. Accept a diagnostic
        // from the day of the report.
        let window = report.date.addingTimeInterval(-24 * 3600)...report.date.addingTimeInterval(30 * 60)
        guard let match = summaries.filter({ window.contains($0.timeStampEnd) })
            .max(by: { $0.timeStampEnd < $1.timeStampEnd }) else { return }

        let entries = report.entries.map { entry -> BuildTestEntry in
            guard entry.id == crashed.id else { return entry }
            return BuildTestEntry(
                modelId: entry.modelId,
                modelName: entry.modelName,
                backend: entry.backend,
                status: .crashed,
                error: error + " MetricKit: \(match.text)",
                log: entry.log
            )
        }
        let updated = BuildTestReport(
            id: report.id,
            date: report.date,
            appVersion: report.appVersion,
            appBuild: report.appBuild,
            stages: BenchmarkOptions.Stages(rawValue: report.stages) ?? .full,
            device: report.device,
            options: report.options,
            entries: entries,
            durationSeconds: report.durationSeconds,
            cancelled: report.cancelled,
            interrupted: true
        )
        _ = try? store.save(updated)
        if lastReport?.id == report.id { lastReport = updated }
        if interruptedReport?.id == report.id { interruptedReport = updated }
    }

    private func refreshThermalState() {
        thermalState = ProcessInfo.processInfo.thermalState.reportName
    }
}

// MARK: - Selection

extension BuildTestRunner {

    /// Models this build can actually benchmark on this device, in catalog
    /// order.
    ///
    /// Compatibility is not re-derived here: `ModelRecommender.recommend`
    /// already applies the split predicate (backend linked into the binary,
    /// Apple Intelligence actually available, RAM within
    /// `DeviceInfo.budgetGB(for:)`, `effectiveDiskSizeGB` within free disk),
    /// and a second implementation of it would drift.
    ///
    /// On top of that, the text-in/text-out filter mirrors the guard in
    /// `BenchmarkEngine.run`, which throws `.unsupportedModel` for anything
    /// else - that is what drops Whisper (audio in only) and Kokoro TTS
    /// (audio out only). Apple Intelligence and the vision-language models
    /// take text in and give text out, so they stay in.
    static func eligibleModels(device: DeviceInfo) -> [AIModel] {
        let compatibleIDs = Set(
            ModelRecommender.shared.recommend(for: device).allCompatible.map(\.id)
        )
        return ModelCatalog.all.filter { model in
            compatibleIDs.contains(model.id)
                && model.capabilities.contains(.textIn)
                && model.capabilities.contains(.textOut)
        }
    }

    /// True when the model has files to fetch. Apple Intelligence is part of
    /// the OS: it has no repo, occupies no disk, and must never be "deleted"
    /// by the cleanup step.
    static func requiresDownload(_ model: AIModel) -> Bool {
        !model.huggingFaceRepo.isEmpty
    }

    /// Already on disk according to the registry the downloader maintains.
    static func isInstalled(_ model: AIModel) -> Bool {
        if case .installed = ModelDownloader.shared.state(for: model) { return true }
        return InstalledModelsStore.shared.isInstalled(model)
    }
}

// MARK: - Disk

extension BuildTestRunner {

    /// Free space kept in reserve on top of the model's own footprint. The
    /// recommender allows a model that exactly fills the disk; a run that
    /// does that leaves no room for the report, the system, or the next
    /// download's temporary file.
    nonisolated static let diskReserveGB = 2.0

    static func freeDiskGB() -> Double {
        Double(DeviceScanner.shared.scan().freeDiskSpace) / 1_073_741_824
    }

    /// `nil` when there is room, otherwise the reason to record as a skip.
    static func insufficientSpaceReason(for model: AIModel) -> String? {
        let free = freeDiskGB()
        let needed = model.effectiveDiskSizeGB + diskReserveGB
        guard free < needed else { return nil }
        return String(
            format: "skipped: insufficient disk (needs %.1f GB + %.1f GB reserve, %.1f GB free)",
            model.effectiveDiskSizeGB, diskReserveGB, free
        )
    }
}

// MARK: - Ownership marker

extension BuildTestRunner {

    /// Filename of the on-disk ownership marker, defined ONCE here and used
    /// by every frontend. Spelled `.harness-installed` for continuity: an
    /// orphan left by an older build of the Xcode harness has to be
    /// recognised and cleaned up by this runner, and renaming the marker
    /// would strand those files on testers' devices forever.
    nonisolated static let ownershipMarkerName = ".harness-installed"

    static func ownershipMarkerURL(for model: AIModel) -> URL {
        ModelDownloader.shared.modelDirectory(for: model)
            .appendingPathComponent(ownershipMarkerName)
    }

    /// Marks `model` as runner-owned. Call BEFORE the transfer starts, so the
    /// marker is on disk even if the process dies mid-download.
    ///
    /// Creates the model's directory first rather than trusting it to exist:
    /// `ModelDownloader.startDownload` only creates it after its manifest
    /// fetch returns, which is later than this call. Not a race -
    /// `createDirectory(withIntermediateDirectories: true)` is a no-op when
    /// the directory is already there.
    static func markOwned(_ model: AIModel) {
        let dir = ModelDownloader.shared.modelDirectory(for: model)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try? Data().write(to: ownershipMarkerURL(for: model))
    }

    /// True when a previous run downloaded this model, marked it, and died
    /// before cleaning up - as opposed to the user having installed it.
    ///
    /// `ModelDownloader.delete` removes the model's whole directory, marker
    /// included, so a model cleaned up normally reads back false here.
    static func isOwned(_ model: AIModel) -> Bool {
        FileManager.default.fileExists(atPath: ownershipMarkerURL(for: model).path)
    }

    /// True when the model's directory already holds something other than our
    /// own marker.
    ///
    /// Read BEFORE a transfer starts, it is the second half of the ownership
    /// proof: a directory that was empty and comes back `.installed` can only
    /// hold this run's bytes, even if the transfer finished between two polls
    /// and no `.downloading` state was ever observed. A directory that already
    /// had files in it gets no such benefit of the doubt.
    static func hasFilesOnDisk(_ model: AIModel) -> Bool {
        let dir = ModelDownloader.shared.modelDirectory(for: model)
        guard let names = try? FileManager.default.contentsOfDirectory(atPath: dir.path)
        else { return false }
        return names.contains { $0 != ownershipMarkerName }
    }
}

// MARK: - Download

extension BuildTestRunner {

    /// Per-model ceiling. An 8 GB model over a slow connection genuinely
    /// takes this long; anything past it is a stall, not a download.
    nonisolated static let downloadTimeout: TimeInterval = 30 * 60
    /// Budget for on-device Core ML compilation, counted from the moment the
    /// downloader reports `.compiling`; separate from the transfer budget.
    ///
    /// `BENCH_COMPILE_MINUTES=<n>` overrides it. Read straight from the
    /// environment rather than routed through `BuildTestOptions` because it
    /// exists to investigate the compile step, not to shape a release sweep:
    /// a run that needs hours of budget to answer "does this finish at all?"
    /// should not have to become a supported in-app option to do it.
    nonisolated static let compileTimeout: TimeInterval = {
        let fallback: TimeInterval = 60 * 60
        guard let raw = ProcessInfo.processInfo.environment["BENCH_COMPILE_MINUTES"]?
            .trimmingCharacters(in: .whitespacesAndNewlines),
              let minutes = Double(raw), minutes > 0
        else { return fallback }
        print("[build-test] BENCH_COMPILE_MINUTES=\(raw): Core ML compile budget set to \(minutes) min")
        return minutes * 60
    }()

    /// How often `ModelDownloader.states` is sampled. It publishes through an
    /// `@Observable` dictionary rather than a continuation, so the runner
    /// polls it: two seconds is invisible next to a multi-gigabyte transfer
    /// and needs no observation bookkeeping that a mid-run cancel could leak.
    private nonisolated static let downloadPollInterval: TimeInterval = 2

    /// Log at most one progress line per this many seconds - a 40-minute
    /// download must not bury the rest of the run's output.
    private nonisolated static let downloadLogInterval: TimeInterval = 30

    /// Starts the download and returns once the model is installed, keeping
    /// `downloadProgress` current while it waits.
    ///
    /// Also decides ownership, because this is the only place that can: the
    /// caller cannot tell "the runner fetched these gigabytes" from "the files
    /// were already there and the downloader just noticed" - both start from
    /// `isInstalled == false`. `ownedByRunner` is set, and the on-disk marker
    /// written, at the first moment this run is SEEN to be moving bytes:
    ///
    ///   * a `.downloading` or `.compiling` state (real work, on our behalf),
    ///     or
    ///   * an `.installed` state over a directory that held no files when this
    ///     call began - a transfer small or cached enough to land between two
    ///     polls, whose bytes can still only be ours.
    ///
    /// `.fetchingManifest` deliberately does not count: `startDownload`
    /// publishes it before the manifest can say whether there is anything to
    /// fetch, so a complete model that merely lost its `.complete` marker
    /// shows it too, on its way to `.installed` without a byte transferred.
    /// That model is the user's and is left exactly where it is.
    ///
    /// `ownedByRunner` is `inout` rather than a return value so that ownership
    /// claimed before a failure survives the throw - the caller's cleanup has
    /// to remove a partial left by a timeout or a downloader error just as
    /// surely as one left by a successful download.
    ///
    /// - Throws: `BuildTestError` on a downloader error or a timeout,
    ///   `CancellationError` when the run is cancelled.
    private func downloadAndWait(
        _ model: AIModel,
        ownedByRunner: inout Bool,
        stallTimeout: TimeInterval? = nil,
        timeout: TimeInterval = BuildTestRunner.downloadTimeout
    ) async throws {

        let downloader = ModelDownloader.shared
        // Sampled before the transfer, while the answer still means something.
        let startedEmpty = !Self.hasFilesOnDisk(model)
        downloader.download(model)

        // `download(model)` hands the actual work to a child Task rather than
        // doing it inline, so the very next `state(for:)` read can still
        // observe whatever a PREVIOUS attempt left behind (e.g. a stale
        // `.error`) before this download's own task has run at all. Give it a
        // moment to actually start before trusting what we read.
        await Task.yield()
        try? await Task.sleep(nanoseconds: 250_000_000)

        var deadline = Date().addingTimeInterval(timeout)
        // On-device Core ML compilation of a multi-GB package can take far
        // longer than the transfer; it gets its own budget once it starts.
        // `effectiveTimeout` tracks which budget the deadline came from — the
        // failure message used to quote the transfer budget for a compile
        // timeout, which made a 60-minute compile read as a 30-minute one.
        var compileDeadlineGranted = false
        var effectiveTimeout = timeout
        var compileStarted: Date?
        var lastLog = Date.distantPast
        var lastState = "idle"
        // The FIRST state read after the yield above gets one extra poll
        // cycle before an `.error` is trusted, in case it is still stale.
        var isFirstRead = true
        // Fails the model after `stallTimeout` seconds without a new byte
        // instead of watching a dead transfer for the whole budget.
        var stallWatch = stallTimeout.map { DownloadStallWatch(timeout: $0) }
        let transferStart = Date()
        var lastBytes: Int64 = 0

        while Date() < deadline {
            try Task.checkCancellation()
            var observedBytes: Int64?
            var compiling = false

            switch downloader.state(for: model) {
            case .installed:
                if let compileStarted {
                    appendLog(String(
                        format: "  [compile] %@: finished in %.1f s (%.1f min)",
                        model.id,
                        Date().timeIntervalSince(compileStarted),
                        Date().timeIntervalSince(compileStarted) / 60
                    ))
                }
                // Installed without this run ever seeing a transfer state.
                // Ours only if the directory was empty when we started;
                // otherwise the files predate the run and stay where they are.
                if !ownedByRunner, startedEmpty {
                    ownedByRunner = true
                    Self.markOwned(model)
                }
                downloadProgress = BuildTestDownloadProgress(
                    fraction: 1, downloadedBytes: 0, totalBytes: 0
                )
                appendLog(ownedByRunner
                    ? "  [download] \(model.id): installed"
                    : "  [download] \(model.id): already complete on disk - left in place")
                return

            case .error(let message):
                if isFirstRead {
                    isFirstRead = false
                    try await Task.sleep(
                        nanoseconds: UInt64(Self.downloadPollInterval * 1_000_000_000)
                    )
                    continue
                }
                throw BuildTestError.downloadFailed(message)

            case .downloading(let fraction, let downloaded, let total):
                lastState = "downloading"
                observedBytes = downloaded
                lastBytes = downloaded
                // Bytes are moving because this run asked for them: claim
                // ownership now, so a crash from here on still finds proof on
                // disk that this directory is ours to remove.
                if !ownedByRunner {
                    ownedByRunner = true
                    Self.markOwned(model)
                }
                downloadProgress = BuildTestDownloadProgress(
                    fraction: fraction, downloadedBytes: downloaded, totalBytes: total
                )
                if Date().timeIntervalSince(lastLog) >= Self.downloadLogInterval {
                    lastLog = Date()
                    appendLog(String(
                        format: "  [download] %@: %.0f%% (%.2f / %.2f GB)",
                        model.id, fraction * 100,
                        Double(downloaded) / 1_073_741_824,
                        Double(total) / 1_073_741_824
                    ))
                }

            case .compiling(let fraction):
                lastState = "compiling"
                compiling = true
                if !ownedByRunner {
                    ownedByRunner = true
                    Self.markOwned(model)
                }
                if !compileDeadlineGranted {
                    compileDeadlineGranted = true
                    effectiveTimeout = Self.compileTimeout
                    deadline = Date().addingTimeInterval(Self.compileTimeout)
                    compileStarted = Date()
                    appendLog("  [compile] \(model.id): compiling on device (up to \(Int(Self.compileTimeout / 60)) min) - see [CoreMLCompiler] lines for exact timings")
                }
                downloadProgress = BuildTestDownloadProgress(
                    fraction: fraction, downloadedBytes: 0, totalBytes: 0
                )
                if Date().timeIntervalSince(lastLog) >= Self.downloadLogInterval {
                    lastLog = Date()
                    appendLog(String(format: "  [compile] %@: %.0f%%", model.id, fraction * 100))
                }

            case .fetchingManifest:
                // The manifest fetch is the first thing `download` does, but
                // it happens on a child task, so `.idle` is also normal for
                // the first fraction of a second. Neither is terminal, and
                // neither claims ownership: the manifest has not yet said
                // whether there is anything at all to fetch.
                lastState = "fetchingManifest"

            case .idle:
                lastState = "idle"
            }

            isFirstRead = false

            if stallWatch != nil {
                let elapsed = Date().timeIntervalSince(transferStart)
                if compiling {
                    stallWatch?.reset(at: elapsed)
                } else if stallWatch?.observe(bytes: observedBytes, elapsed: elapsed) == true {
                    let network = NetworkPathWatcher.shared.stateDescription
                    appendLog(String(
                        format: "  [download] %@: no bytes for %.0f s (at %.2f GB, state %@), network %@ "
                            + "- failing this model",
                        model.id, stallWatch?.idle(at: elapsed) ?? 0,
                        Double(lastBytes) / 1_073_741_824, lastState, network
                    ))
                    appendLog("  [conditions] " + MeasurementConditions.logLine())
                    throw BuildTestError.downloadStalled(
                        seconds: stallWatch?.timeout ?? 0, network: network
                    )
                }
            }

            try await Task.sleep(nanoseconds: UInt64(Self.downloadPollInterval * 1_000_000_000))
        }

        // A background URLSession treats every transfer as discretionary
        // while the app is in the background (and Low Power Mode defers
        // them), so a stall at 0% is usually the phone's state, not the host.
        appendLog("  [conditions] " + MeasurementConditions.logLine())
        throw BuildTestError.downloadTimedOut(seconds: effectiveTimeout, lastState: lastState)
    }

    /// Removes a model the runner installed and gives the unlink a moment to
    /// land: `ModelDownloader.delete` cancels the session's tasks on a
    /// background callback and only THEN unlinks the directory.
    ///
    /// The wait runs on a detached task on purpose. `Task.sleep` throws
    /// immediately inside a cancelled task, and cleanup after a cancellation
    /// is exactly when the wait matters most.
    static func deleteAndWait(_ model: AIModel) async {
        ModelDownloader.shared.delete(model)
        await Task.detached { try? await Task.sleep(nanoseconds: 3_000_000_000) }.value
    }
}
