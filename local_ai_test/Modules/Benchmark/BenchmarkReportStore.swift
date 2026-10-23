//
//  BenchmarkReportStore.swift
//  local_ai_test
//
//  Persistence for benchmark reports: one JSON file per report under
//  Documents/BenchmarkReports/.
//
//  One file per report rather than a single registry (the shape
//  `InstalledModelsStore` uses) because reports are append-only, independent,
//  and occasionally exported one at a time. A corrupted or half-written file
//  costs exactly one report instead of the whole history, and a user can drop
//  a report they were sent into the folder via Files.app and have it appear.
//
//  Documents (not Application Support) on purpose: the reports ARE user
//  output — they are meant to be visible, shareable and deletable.
//
//  Every `save` prunes: at most `maxReportsPerModel` reports survive per
//  model, newest first, so a folder that only ever grew now has a ceiling.
//  Files this store cannot decode are never touched by the prune — the
//  release harness drops its own summaries into the same directory.
//
//  COMPARING STORED REPORTS. The folder mixes builds, suite versions and
//  measurement methodologies, and two of those three make reports
//  incomparable:
//
//    • QUALITY SCORES compare only within one `quality.suiteVersion`. Adding
//      one hard task moves every model's pass-rate without any model
//      changing, so a UI that shows "+8 since last run" must first check
//      `BenchmarkReport.isScoreComparable(with:)` and otherwise say which
//      suite each score came from.
//    • THROUGHPUT compares only between reports of the same
//      `performance.metricsVersion` (v1 was single-shot and unwarmed; v2
//      warms up and averages three repetitions), and never at all when
//      `performance.thermallyConstrained` is set.
//
//  Neither rule is enforced here — this store keeps files, it does not
//  interpret them — but every consumer that ranks or diffs reports owes the
//  reader both checks.
//

import Foundation
import Observation

@MainActor
@Observable
final class BenchmarkReportStore {

    static let shared = BenchmarkReportStore()

    /// Every stored report, newest first. Loaded once at init and kept in
    /// sync by `save`/`delete` — the folder is only re-read on `reload()`.
    private(set) var reports: [BenchmarkReport] = []

    /// How many reports are kept per model. Every `save` prunes the surplus,
    /// so re-benchmarking one model twenty times leaves ten files, not twenty.
    /// A `static let` rather than a literal so the release harness — which
    /// sweeps the whole catalog and wants the deeper history — can raise it in
    /// one place.
    static let maxReportsPerModel = 10

    init(directoryName: String = "BenchmarkReports") {
        self.directoryName = directoryName
        reload()
    }

    @ObservationIgnored
    private let directoryName: String

    // MARK: - Location

    @ObservationIgnored
    private lazy var directory: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent(directoryName, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    /// `2026-09-02T141233Z-mlx-community-qwen3-4b-4bit.json`. Date first so a
    /// plain alphabetical listing in Files.app is already chronological, and
    /// the model slug after it so a folder full of reports is readable.
    nonisolated static func filename(for report: BenchmarkReport) -> String {
        "\(timestampSlug(report.date))-\(slug(report.modelId)).json"
    }

    /// Lowercased, `[a-z0-9-]` only, collapsed and clipped. Model ids are
    /// Hugging Face repo paths with slashes and dots in them, none of which
    /// belong in a filename.
    nonisolated static func slug(_ text: String) -> String {
        let mapped = text.lowercased().map { character -> Character in
            character.isLetter || character.isNumber ? character : "-"
        }
        let collapsed = String(mapped)
            .split(separator: "-", omittingEmptySubsequences: true)
            .joined(separator: "-")
        return String(collapsed.prefix(80))
    }

    /// UTC, second resolution, filename-safe: two runs of the same model in
    /// the same second would collide, which no human can produce.
    nonisolated static func timestampSlug(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd'T'HHmmss'Z'"
        return formatter.string(from: date)
    }

    // MARK: - Coders

    nonisolated static func makeEncoder() -> JSONEncoder {
        let encoder = JSONEncoder()
        // Pretty + sorted so a report diffs cleanly between builds and reads
        // sensibly when a tester opens it in Files.app.
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }

    nonisolated static func makeDecoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }

    // MARK: - Reading

    /// Re-reads the folder. Unreadable files are skipped rather than fatal —
    /// one bad report must not hide the rest.
    func reload() {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []

        let decoder = Self.makeDecoder()
        reports = urls
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> BenchmarkReport? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(BenchmarkReport.self, from: data)
            }
            .sorted { $0.date > $1.date }
    }

    /// Newest report for a model, or `nil` when it has never been benchmarked.
    /// The in-app flow uses this to show "last run" without re-running.
    func latest(for modelId: String) -> BenchmarkReport? {
        reports.first { $0.modelId == modelId }
    }

    /// Every report for one model, newest first.
    func reports(for modelId: String) -> [BenchmarkReport] {
        reports.filter { $0.modelId == modelId }
    }

    // MARK: - Writing

    /// Writes a report, inserts it into `reports`, then applies the retention
    /// policy for that model (see `pruneReports(for:)`). Throws only when the
    /// write itself fails — the caller has a run's worth of work to lose, so
    /// this failure is not swallowed the way cache writes elsewhere are.
    /// Pruning happens after the write and never throws: losing an old report
    /// is not a reason to fail a successful save.
    @discardableResult
    func save(_ report: BenchmarkReport) throws -> URL {
        let url = directory.appendingPathComponent(Self.filename(for: report))
        let data = try Self.makeEncoder().encode(report)
        // Atomic: a crash mid-write must not leave a truncated file that
        // `reload()` then silently drops.
        try data.write(to: url, options: .atomic)

        reports.removeAll { $0.id == report.id }
        reports.append(report)
        reports.sort { $0.date > $1.date }
        pruneReports(for: report.modelId)
        PALAnalytics.shared.benchmarkSaved(report)
        return url
    }

    /// Keeps at most `maxReportsPerModel` reports for one model, newest by
    /// date, and deletes the rest from disk and from `reports`.
    ///
    /// Benchmarking is cheap to repeat and each report is a standalone file,
    /// so without a cap the folder grows for as long as the app is installed.
    /// Ten runs is enough to see a trend across builds and small enough that
    /// the whole history stays under a megabyte.
    ///
    /// Scope is deliberately narrow: only reports this store has decoded are
    /// candidates, and `reload()` skips anything that does not decode as a
    /// `BenchmarkReport`. Such a file never enters `reports`, so it can never
    /// be pruned — an undecodable file (an import, or an older build's
    /// `release-summary-*.json` left in this folder) is left alone rather
    /// than treated as junk. A build test's own artefact lives in
    /// `Documents/BuildTestReports/` and never lands here at all; the
    /// per-model reports it produces do, through `save` like any other.
    private func pruneReports(for modelId: String) {
        let forModel = reports
            .filter { $0.modelId == modelId }
            .sorted { $0.date > $1.date }
        guard forModel.count > Self.maxReportsPerModel else { return }

        for stale in forModel.dropFirst(Self.maxReportsPerModel) {
            delete(stale)
        }
    }

    /// Removes one report from disk and from `reports`.
    func delete(_ report: BenchmarkReport) {
        if let url = fileURL(for: report) {
            try? FileManager.default.removeItem(at: url)
        }
        reports.removeAll { $0.id == report.id }
    }

    /// Removes everything. Used by the harness between sweeps and by a
    /// "clear history" control in the UI.
    func deleteAll() {
        for report in reports { delete(report) }
        reports = []
    }

    // MARK: - Export

    /// On-disk URL of a stored report, for a share sheet — `nil` when the
    /// report is not on disk.
    ///
    /// The derived filename is tried first, then the folder is scanned by
    /// report id. The scan matters for files this store did not write: a
    /// report dropped in via Files.app, or one written by an older build
    /// whose naming differed, would otherwise be impossible to delete and
    /// would reappear on the next `reload()`.
    func fileURL(for report: BenchmarkReport) -> URL? {
        let derived = directory.appendingPathComponent(Self.filename(for: report))
        if FileManager.default.fileExists(atPath: derived.path) { return derived }

        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        let decoder = Self.makeDecoder()
        return urls.first { url in
            guard url.pathExtension == "json",
                  let data = try? Data(contentsOf: url),
                  let stored = try? decoder.decode(BenchmarkReport.self, from: data)
            else { return false }
            return stored.id == report.id
        }
    }

    /// Pretty-printed JSON of one report — what the share sheet and the
    /// "copy" button hand over.
    nonisolated static func exportJSON(_ report: BenchmarkReport) -> String {
        guard let data = try? makeEncoder().encode(report),
              let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }

    /// Pretty-printed JSON of a whole sweep — the release harness dumps this
    /// as one artefact per run.
    nonisolated static func exportJSON(_ reports: [BenchmarkReport]) -> String {
        guard let data = try? makeEncoder().encode(reports),
              let text = String(data: data, encoding: .utf8)
        else { return "[]" }
        return text
    }
}
