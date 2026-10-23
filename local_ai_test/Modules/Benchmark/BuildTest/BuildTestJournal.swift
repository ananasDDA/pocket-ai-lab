//
//  BuildTestJournal.swift
//  local_ai_test
//
//  The build test's crash insurance.
//
//  A sweep used to exist only in memory until its last model was cleaned
//  up, and then it was written to disk in one go. A process death anywhere
//  in the preceding hours - a GGML_ASSERT abort inside llama.cpp, a jetsam
//  kill, the watchdog - therefore lost EVERYTHING: not just the model that
//  died, but every measurement before it, and with it the one fact the run
//  existed to establish, namely which model brought the app down. The
//  2026-09-05 overnight run ended exactly like that: no report, no log, and
//  a guess.
//
//  The journal is a small JSON file rewritten (atomically) at every step
//  worth surviving: when a model is picked up, when its phase or benchmark
//  stage changes, and when its entry is final. It carries the entries so
//  far, the model in flight with what it was doing and how much memory it
//  had left, and the tail of the log. On the next launch
//  `BuildTestRecovery` turns a journal that is still there into a report
//  with the dead model recorded as `crashed`, and the file is removed. A run
//  that finishes removes it itself.
//

import Foundation

nonisolated struct BuildTestJournal: Codable, Sendable, Hashable {

    static let currentSchemaVersion = 1

    /// What the sweep was doing to the in-flight model when the journal was
    /// last written. Everything a post-mortem needs and nothing it does not.
    struct InFlight: Codable, Sendable, Hashable {
        let modelId: String
        let modelName: String
        /// `ModelBackend.displayName`, as in `BuildTestEntry`.
        let backend: String
        /// `BuildTestRunner.Phase.displayName` ("Downloading", "Benchmarking", ...).
        var phase: String
        /// `BenchmarkProgress.Stage.rawValue` during the benchmark, else nil.
        var stage: String?
        /// The engine's last progress label ("Decode run 2 of 3", ...).
        var label: String?
        /// When this model was picked up.
        let since: Date
        /// `os_proc_available_memory()` in MB at the last write, -1 if
        /// unknown. The single most useful number after a silent death:
        /// a few hundred MB here means jetsam, not a bug in the model.
        var availableMemoryMB: Int64
        /// `ProcessInfo.ThermalState.reportName` at the last write.
        var thermalState: String
    }

    let schemaVersion: Int
    /// Becomes the recovered report's id, so a journal cannot produce two
    /// reports.
    let runID: String
    let startedAt: Date
    let appVersion: String
    let appBuild: String
    let device: DeviceSnapshot
    let options: BuildTestOptions
    /// Every model the sweep set out to attempt, in sweep order - the
    /// entries that never happened are derived from this.
    let plannedModelIDs: [String]

    var entries: [BuildTestEntry]
    var inFlight: InFlight?
    /// Newest last. Bounded by `BuildTestJournal.logTailLines`.
    var logTail: [String]
    var updatedAt: Date

    static let logTailLines = 80

    init(
        runID: String,
        startedAt: Date,
        appVersion: String,
        appBuild: String,
        device: DeviceSnapshot,
        options: BuildTestOptions,
        plannedModelIDs: [String],
        entries: [BuildTestEntry] = [],
        inFlight: InFlight? = nil,
        logTail: [String] = [],
        updatedAt: Date
    ) {
        self.schemaVersion = Self.currentSchemaVersion
        self.runID = runID
        self.startedAt = startedAt
        self.appVersion = appVersion
        self.appBuild = appBuild
        self.device = device
        self.options = options
        self.plannedModelIDs = plannedModelIDs
        self.entries = entries
        self.inFlight = inFlight
        self.logTail = logTail
        self.updatedAt = updatedAt
    }
}

// MARK: - On disk

/// One file, next to the reports, hidden so `BuildTestReportStore.reload`
/// (which skips hidden files) never mistakes it for a report.
nonisolated enum BuildTestJournalFile {

    static let filename = ".in-progress.json"

    static var url: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("BuildTestReports", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(filename)
    }

    /// Atomic, so a death mid-write leaves the previous journal rather than
    /// half of the new one.
    static func write(_ journal: BuildTestJournal) throws {
        let data = try BenchmarkReportStore.makeEncoder().encode(journal)
        try data.write(to: url, options: .atomic)
    }

    /// `nil` when there is no journal or it cannot be read. An unreadable
    /// journal is removed: it would otherwise resurrect the same broken
    /// recovery on every launch.
    static func read() -> BuildTestJournal? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        guard let journal = try? BenchmarkReportStore.makeDecoder().decode(BuildTestJournal.self, from: data)
        else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        return journal
    }

    static func remove() {
        try? FileManager.default.removeItem(at: url)
    }

    static var exists: Bool {
        FileManager.default.fileExists(atPath: url.path)
    }
}
