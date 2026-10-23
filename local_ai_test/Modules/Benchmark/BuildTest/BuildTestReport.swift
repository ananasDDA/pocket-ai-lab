//
//  BuildTestReport.swift
//  local_ai_test
//
//  The artefact one "build test" leaves behind: a whole sweep over every
//  catalog model this build can run on this device, in one JSON file.
//
//  ---------------------------------------------------------------------
//  Why this shape
//  ---------------------------------------------------------------------
//
//  A sweep produces three kinds of per-model outcome - it ran, it failed, it
//  was never attempted - and the previous release harness kept them in three
//  parallel arrays (`reports`, `failures`, `skips`). That made the ORDER of
//  the sweep unrecoverable: which model was benchmarked third, and whether
//  the failure between two reports happened before or after them, could not
//  be read back. `entries` is one ordered list instead, so the file reads as
//  the timeline it is.
//
//  `BuildTestEntry` is a STRUCT with optional payload fields rather than an
//  enum with associated values, on purpose. A Swift enum with payloads
//  encodes as a nested single-key object (`{"completed": {...}}`), which
//  every non-Swift reader - the Python report generator above all - then has
//  to special-case. A flat `{"status": "completed", "report": {...}}` is
//  readable from any language and diffs cleanly between builds.
//
//  ---------------------------------------------------------------------
//  Compatibility with the old summary
//  ---------------------------------------------------------------------
//
//  Field names that existed in the old `BenchmarkSweepSummary` keep their
//  spelling here - `generatedAt`, `appVersion`, `appBuild`, `stages`,
//  `device` - so a consumer that only reads the envelope needs no changes.
//  The per-model arrays moved into `entries`; `Scripts/generate_benchmark_
//  report.py` reads BOTH shapes (it detects this one by the `entries` key)
//  and old summaries from earlier releases keep working unchanged.
//
//  Counts (`passed`/`failed`/`skipped`) are COMPUTED, not stored: `entries`
//  is the single source of truth, and a stored count that disagrees with the
//  list it summarises is a bug waiting to be shipped.
//

import Foundation

// MARK: - Entry

/// One model's outcome inside a build test.
nonisolated struct BuildTestEntry: Codable, Sendable, Hashable, Identifiable {

    /// Which of the three outcomes this entry records. Raw values are part
    /// of the JSON contract - do not rename them.
    enum Status: String, Codable, Sendable, Hashable {
        /// The model was benchmarked and produced a report.
        case completed
        /// The download or the benchmark threw. `error` says what.
        case failed
        /// The sweep deliberately never ran it. `reason` says why.
        case skipped
        /// The process died while this model was in flight - the entry was
        /// written by `BuildTestRecovery` from the journal on the next
        /// launch, not by the run itself. `error` says what the sweep was
        /// doing and what the device looked like; `log` is the tail of the
        /// run's transcript.
        case crashed
    }

    let modelId: String
    let modelName: String
    /// `ModelBackend.displayName` ("MLX", "llama.cpp", …), matching
    /// `BenchmarkReport.backend`.
    let backend: String
    let status: Status

    /// Set only for `.completed`.
    let report: BenchmarkReport?
    /// Set only for `.failed` - the localized description of what threw.
    let error: String?
    /// Set only for `.skipped`.
    let reason: String?
    /// Set only for `.crashed`: the last lines the run logged before the
    /// process died, newest last.
    let log: [String]?

    /// Stable across a run: one entry per model, and a sweep never
    /// benchmarks the same model twice.
    var id: String { modelId }

    init(
        modelId: String,
        modelName: String,
        backend: String,
        status: Status,
        report: BenchmarkReport? = nil,
        error: String? = nil,
        reason: String? = nil,
        log: [String]? = nil
    ) {
        self.modelId = modelId
        self.modelName = modelName
        self.backend = backend
        self.status = status
        self.report = report
        self.error = error
        self.reason = reason
        self.log = log
    }

    // MARK: Constructors

    static func completed(_ report: BenchmarkReport) -> BuildTestEntry {
        BuildTestEntry(
            modelId: report.modelId,
            modelName: report.modelName,
            backend: report.backend,
            status: .completed,
            report: report
        )
    }

    static func failed(_ model: AIModel, error: String) -> BuildTestEntry {
        BuildTestEntry(
            modelId: model.id,
            modelName: model.name,
            backend: model.backend.displayName,
            status: .failed,
            error: error
        )
    }

    static func skipped(_ model: AIModel, reason: String) -> BuildTestEntry {
        BuildTestEntry(
            modelId: model.id,
            modelName: model.name,
            backend: model.backend.displayName,
            status: .skipped,
            reason: reason
        )
    }

    static func crashed(
        modelId: String,
        modelName: String,
        backend: String,
        error: String,
        log: [String]
    ) -> BuildTestEntry {
        BuildTestEntry(
            modelId: modelId,
            modelName: modelName,
            backend: backend,
            status: .crashed,
            error: error,
            log: log.isEmpty ? nil : log
        )
    }

    /// For an id that names nothing in the catalog - a typo in the allowlist
    /// has to show up in the artefact rather than vanish from it.
    static func skipped(unknownModelId modelId: String, reason: String) -> BuildTestEntry {
        BuildTestEntry(
            modelId: modelId,
            modelName: modelId,
            backend: "unknown",
            status: .skipped,
            reason: reason
        )
    }

    /// One line for a list row or a log: what happened to this model.
    var summaryLine: String {
        switch status {
        case .completed:
            return report?.summaryLine ?? "\(modelName) [\(backend)] - completed"
        case .failed:
            return "\(modelName) [\(backend)] - failed: \(error ?? "unknown error")"
        case .skipped:
            return "\(modelName) [\(backend)] - \(reason ?? "skipped")"
        case .crashed:
            return "\(modelName) [\(backend)] - CRASHED: \(error ?? "the app was terminated")"
        }
    }
}

// MARK: - Report

/// One build test: every eligible model, in sweep order, with the device and
/// the build that produced it.
nonisolated struct BuildTestReport: Codable, Sendable, Hashable, Identifiable {

    /// Bumped when this envelope changes shape. 1 was the old three-array
    /// `BenchmarkSweepSummary`; 2 is this ordered-`entries` schema.
    static let currentSchemaVersion = 2

    let schemaVersion: Int

    /// UUID string, as in `BenchmarkReport` - readable in a hand-written
    /// fixture, stable as a SwiftUI identity.
    let id: String

    /// When the run finished. Encoded as `generatedAt`, the name the old
    /// summary used, so the report generator's envelope handling is unchanged.
    let date: Date

    let appVersion: String
    let appBuild: String

    /// "full" or "performanceOnly" - `BenchmarkOptions.Stages.rawValue`.
    /// Duplicated out of `options` on purpose: it is the one scope knob every
    /// reader needs, and it was a top-level field in the old schema.
    let stages: String

    let device: DeviceSnapshot

    /// The full scope the sweep ran with, so a report can be reproduced from
    /// itself.
    let options: BuildTestOptions

    /// Every model the sweep touched, in the order it touched them.
    let entries: [BuildTestEntry]

    /// Wall clock from the first selection to the last cleanup.
    let durationSeconds: Double

    /// True when the run was stopped early - by the user, or by the host
    /// tearing the task down. The entries present are real measurements; the
    /// models that never ran are simply absent, NOT recorded as skipped.
    let cancelled: Bool

    /// True when this report was reconstructed from the journal after the
    /// process died mid-run (see `BuildTestRecovery`). Exactly one entry is
    /// `.crashed`; the models never reached are recorded as skipped with a
    /// reason, unlike a cancellation, because here nobody chose to stop.
    let interrupted: Bool

    init(
        id: String = UUID().uuidString,
        date: Date = Date(),
        appVersion: String,
        appBuild: String,
        stages: BenchmarkOptions.Stages,
        device: DeviceSnapshot,
        options: BuildTestOptions,
        entries: [BuildTestEntry],
        durationSeconds: Double,
        cancelled: Bool = false,
        interrupted: Bool = false
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.id = id
        self.date = date
        self.appVersion = appVersion
        self.appBuild = appBuild
        self.stages = stages.rawValue
        self.device = device
        self.options = options
        self.entries = entries
        self.durationSeconds = durationSeconds
        self.cancelled = cancelled
        self.interrupted = interrupted
    }

    /// Hand-written so `interrupted` (added after 1.0.1's reports were
    /// archived) decodes as false when absent instead of failing the file.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        id = try c.decode(String.self, forKey: .id)
        date = try c.decode(Date.self, forKey: .date)
        appVersion = try c.decode(String.self, forKey: .appVersion)
        appBuild = try c.decode(String.self, forKey: .appBuild)
        stages = try c.decode(String.self, forKey: .stages)
        device = try c.decode(DeviceSnapshot.self, forKey: .device)
        options = try c.decode(BuildTestOptions.self, forKey: .options)
        entries = try c.decode([BuildTestEntry].self, forKey: .entries)
        durationSeconds = try c.decode(Double.self, forKey: .durationSeconds)
        cancelled = try c.decodeIfPresent(Bool.self, forKey: .cancelled) ?? false
        interrupted = try c.decodeIfPresent(Bool.self, forKey: .interrupted) ?? false
    }

    private enum CodingKeys: String, CodingKey {
        case schemaVersion
        case id
        // The old summary called this `generatedAt`; keeping the wire name
        // costs one line here and keeps every existing envelope reader.
        case date = "generatedAt"
        case appVersion
        case appBuild
        case stages
        case device
        case options
        case entries
        case durationSeconds
        case cancelled
        case interrupted
    }

    // MARK: Derived

    /// Models that produced a report.
    var passed: Int { entries.filter { $0.status == .completed }.count }
    /// Models whose download or benchmark threw.
    var failed: Int { entries.filter { $0.status == .failed }.count }
    /// Models the sweep deliberately did not run.
    var skipped: Int { entries.filter { $0.status == .skipped }.count }
    /// Models that took the process down with them - 0 or 1 per report.
    var crashed: Int { entries.filter { $0.status == .crashed }.count }
    /// The model that took the process down, when this report has one.
    var crashedEntry: BuildTestEntry? { entries.first { $0.status == .crashed } }
    /// Models actually attempted - passed, failed and crashed. A skip was
    /// never an attempt, so it does not count as tested.
    var tested: Int { passed + failed + crashed }

    /// The reports, in sweep order, for a consumer that only wants the
    /// measurements.
    var reports: [BenchmarkReport] { entries.compactMap(\.report) }

    // MARK: Aggregates

    /// Completed entries whose report covers the whole quality suite - the
    /// only ones that may be averaged or ranked. A partial run (quality
    /// stage stopped at its time budget) is scored over a different set of
    /// tasks, so averaging it with full runs, or ranking it against them,
    /// would compare two different exams.
    var aggregatableEntries: [BuildTestEntry] {
        entries.filter { entry in
            guard entry.status == .completed, let report = entry.report else { return false }
            return !report.isPartial
        }
    }

    /// Completed entries left out of the aggregates because their quality
    /// run was cut short.
    var partialEntryCount: Int {
        entries.filter { $0.status == .completed && $0.report?.isPartial == true }.count
    }

    /// Mean score of the full graded runs. `nil` when none was graded: a
    /// performance-only sweep has no score, and "0/100" would read as a
    /// total failure.
    var averageScore: Int? {
        let scores = aggregatableEntries.compactMap { $0.report?.overallScore }
        guard !scores.isEmpty else { return nil }
        return Int((scores.reduce(0, +) / Double(scores.count)).rounded())
    }

    /// Highest generation throughput among the aggregatable entries.
    var fastestEntry: BuildTestEntry? {
        aggregatableEntries.max {
            ($0.report?.performance.generationTokensPerSecond ?? 0)
                < ($1.report?.performance.generationTokensPerSecond ?? 0)
        }
    }

    /// Aggregatable entries, best first: by score when the sweep graded
    /// quality, then by generation throughput.
    ///
    /// A single tuple key, not a two-branch comparator: mixing "compare by
    /// score" and "compare by speed" per pair is not a strict weak ordering
    /// (A can beat B, B beat C, and C beat A) and `sorted` can trap on it.
    /// `-1` sorts every ungraded (performance-only) entry behind every graded
    /// one, then ties break on speed.
    var rankedEntries: [BuildTestEntry] {
        aggregatableEntries.sorted { lhs, rhs in
            let lhsKey = (lhs.report?.overallScore ?? -1, lhs.report?.performance.generationTokensPerSecond ?? 0)
            let rhsKey = (rhs.report?.overallScore ?? -1, rhs.report?.performance.generationTokensPerSecond ?? 0)
            return lhsKey > rhsKey
        }
    }

    /// "1 partial run excluded" for the note under an aggregate, `nil` when
    /// nothing was left out.
    var partialExclusionNote: String? {
        let count = partialEntryCount
        guard count > 0 else { return nil }
        return count == 1 ? "1 partial run excluded" : "\(count) partial runs excluded"
    }

    /// One line for a log or a list row.
    var summaryLine: String {
        let verb = interrupted ? "interrupted" : (cancelled ? "cancelled" : "finished")
        var line = String(
            format: "%@ - %@ in %.0f s: %d passed, %d failed, %d skipped",
            device.marketingName, verb, durationSeconds, passed, failed, skipped
        )
        if let crashedEntry {
            line += ", crashed on \(crashedEntry.modelName)"
        }
        return line
    }

    /// True when nothing ran and nothing could have: an empty verdict is a
    /// harness problem, not a build problem, and the caller has to say so.
    var isEmpty: Bool { entries.isEmpty }
}
