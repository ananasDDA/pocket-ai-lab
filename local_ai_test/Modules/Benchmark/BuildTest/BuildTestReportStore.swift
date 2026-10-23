//
//  BuildTestReportStore.swift
//  local_ai_test
//
//  Persistence for build test reports: one JSON file per run under
//  Documents/BuildTestReports/.
//
//  Deliberately the same shape as `BenchmarkReportStore` - one file per
//  artefact, Documents rather than Application Support, undecodable files
//  left alone - for the same reasons:
//
//    * one file per run keeps a corrupted write to one run instead of the
//      whole history, and lets a tester drop a colleague's report into the
//      folder via Files.app and have it appear;
//    * Documents because these reports ARE user output - visible, shareable,
//      deletable, and pullable with Xcode's Download Container;
//    * a separate folder from BenchmarkReports because the two are read by
//      different things: per-model reports are the app's benchmark history,
//      build tests are release artefacts. `Scripts/generate_benchmark_
//      report.py` can be pointed at either.
//
//  Retention is a flat "keep the newest 10 runs". Unlike per-model reports
//  there is no per-model dimension to keep: a build test is a whole-device
//  event, and ten of them is more history than any release needs.
//

import Foundation
import Observation

@MainActor
@Observable
final class BuildTestReportStore {

    static let shared = BuildTestReportStore()

    /// Every stored report, NEWEST FIRST. Loaded at init and kept in sync by
    /// `save`/`delete`; the folder is only re-read on `reload()`.
    private(set) var reports: [BuildTestReport] = []

    /// How many runs survive. Every `save` prunes the surplus.
    static let maxReports = 10

    init(directoryName: String = "BuildTestReports") {
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

    /// `build-test-2026-09-03T141233Z.json`. Date first so a plain
    /// alphabetical listing in Files.app is already chronological.
    nonisolated static func filename(for report: BuildTestReport) -> String {
        "build-test-\(BenchmarkReportStore.timestampSlug(report.date)).json"
    }

    // MARK: - Reading

    /// Re-reads the folder. Unreadable files are skipped rather than fatal -
    /// one bad report must not hide the rest.
    func reload() {
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []

        let decoder = BenchmarkReportStore.makeDecoder()
        reports = urls
            .filter { $0.pathExtension == "json" }
            .compactMap { url -> BuildTestReport? in
                guard let data = try? Data(contentsOf: url) else { return nil }
                return try? decoder.decode(BuildTestReport.self, from: data)
            }
            .sorted { $0.date > $1.date }
    }

    /// The most recent run, or `nil` when this build has never been tested.
    var latest: BuildTestReport? { reports.first }

    /// The run immediately before `report`, for a "compared with last time"
    /// view. `nil` when `report` is the oldest one kept, or is not in the
    /// store at all.
    ///
    /// Ordered by date rather than by list position so a report loaded from
    /// elsewhere (an import, a hand-edited fixture) still finds its
    /// predecessor.
    func previous(before report: BuildTestReport) -> BuildTestReport? {
        reports.first { $0.date < report.date && $0.id != report.id }
    }

    // MARK: - Writing

    /// Writes a report, inserts it into `reports`, then prunes to
    /// `maxReports`. Throws only when the write itself fails - the caller has
    /// a multi-hour run to lose, so this is not swallowed. Pruning never
    /// throws: losing an old report is not a reason to fail a good save.
    @discardableResult
    func save(_ report: BuildTestReport) throws -> URL {
        let url = directory.appendingPathComponent(Self.filename(for: report))
        let data = try BenchmarkReportStore.makeEncoder().encode(report)
        // Atomic: a crash mid-write must not leave a truncated file that
        // `reload()` then silently drops.
        try data.write(to: url, options: .atomic)

        reports.removeAll { $0.id == report.id }
        reports.append(report)
        reports.sort { $0.date > $1.date }
        prune()
        return url
    }

    /// Keeps the newest `maxReports` runs and deletes the rest.
    ///
    /// Scope is deliberately narrow: only reports this store decoded are
    /// candidates, so an unrelated file a tester dropped into the folder is
    /// never treated as junk.
    private func prune() {
        guard reports.count > Self.maxReports else { return }
        for stale in reports.dropFirst(Self.maxReports) {
            delete(stale)
        }
    }

    /// Removes one report from disk and from `reports`.
    func delete(_ report: BuildTestReport) {
        if let url = fileURL(for: report) {
            try? FileManager.default.removeItem(at: url)
        }
        reports.removeAll { $0.id == report.id }
    }

    /// Removes everything.
    func deleteAll() {
        for report in reports { delete(report) }
        reports = []
    }

    // MARK: - Export

    /// On-disk URL of a stored report, for a share sheet - `nil` when the
    /// report is not on disk.
    ///
    /// The derived filename is tried first, then the folder is scanned by
    /// report id. The scan matters for files this store did not write: an
    /// imported report, or one from a build whose naming differed, would
    /// otherwise be impossible to delete and would reappear on the next
    /// `reload()`.
    func fileURL(for report: BuildTestReport) -> URL? {
        let derived = directory.appendingPathComponent(Self.filename(for: report))
        if FileManager.default.fileExists(atPath: derived.path) { return derived }

        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? []
        let decoder = BenchmarkReportStore.makeDecoder()
        return urls.first { url in
            guard url.pathExtension == "json",
                  let data = try? Data(contentsOf: url),
                  let stored = try? decoder.decode(BuildTestReport.self, from: data)
            else { return false }
            return stored.id == report.id
        }
    }

    /// Pretty-printed JSON of one run - what the share sheet, the "copy"
    /// button and the Xcode wrapper's result attachment all hand over.
    nonisolated static func exportJSON(_ report: BuildTestReport) -> String {
        guard let data = try? BenchmarkReportStore.makeEncoder().encode(report),
              let text = String(data: data, encoding: .utf8)
        else { return "{}" }
        return text
    }
}
