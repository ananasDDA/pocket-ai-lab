//
//  BuildTestReportView.swift
//  local_ai_test
//
//  The artefact a build test leaves behind, read the way a release engineer
//  reads it: the envelope (which build, which device, how long, was it
//  stopped early), four counts, then every model the sweep touched IN SWEEP
//  ORDER - which is the one ordering `BuildTestReport.entries` preserves and
//  the reason it is a single list rather than three arrays.
//
//  Rows for models that completed push the ordinary `BenchmarkReportView`
//  for that model's report. The push is destination-based rather than
//  value-based on purpose: this view is reached both from `DiagnosticsView`
//  (which registers a `navigationDestination(for: BenchmarkReport.self)`)
//  and from a bare `NavigationStack` inside the post-run sheet, and a
//  value-based link would only work in the first.
//
//  Deltas against the previous build test reuse `BenchmarkRunComparison`, so
//  a "no delta" here means exactly what it means on a per-model report:
//  different suite version, different metrics version, or a thermally
//  constrained run on either side.
//

import SwiftUI
import UIKit

struct BuildTestReportView: View {
    let report: BuildTestReport

    @State private var showDeleteConfirm = false
    @State private var shareCardImage: UIImage?
    @State private var exportFileURL: URL?
    @Environment(\.dismiss) private var dismiss

    private var store: BuildTestReportStore { .shared }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                headerSection
                Divider()
                summarySection
                Divider()
                entriesSection
            }
            .padding(20)
        }
        .navigationTitle("Build Test Report")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { toolbarMenu }
        .task(id: report.id) {
            // Yield once before the synchronous `ImageRenderer` work below:
            // a `.task(id:)` with no `await` in front of it runs synchronously
            // on the main actor and hitches the sheet's first frame.
            await Task.yield()
            prepareExports()
        }
        .confirmationDialog(
            "Delete this build test?",
            isPresented: $showDeleteConfirm,
            titleVisibility: .visible
        ) {
            Button("Delete", role: .destructive) {
                store.delete(report)
                dismiss()
            }
            Button("Cancel", role: .cancel) {}
        }
    }

    // MARK: - Toolbar

    private var toolbarMenu: some ToolbarContent {
        ToolbarItem(placement: .primaryAction) {
            Menu {
                if let shareCardImage {
                    ShareLink(
                        item: Image(uiImage: shareCardImage),
                        preview: SharePreview(
                            "Build test v\(report.appVersion) (\(report.appBuild))",
                            image: Image(uiImage: shareCardImage)
                        )
                    ) {
                        Label("Share card", systemImage: "square.and.arrow.up")
                    }
                }
                if let exportFileURL {
                    ShareLink(item: exportFileURL) {
                        Label("Export JSON", systemImage: "doc.text")
                    }
                }
                Button(role: .destructive) {
                    showDeleteConfirm = true
                } label: {
                    Label("Delete report", systemImage: "trash")
                }
            } label: {
                Image(systemName: "ellipsis.circle")
            }
        }
    }

    // MARK: - Header

    private var headerSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                VStack(alignment: .leading, spacing: 2) {
                    Text("v\(report.appVersion) (\(report.appBuild))")
                        .font(.title2).bold()
                    Text(BuildTestFormat.stagesLabel(report.stages) + " sweep")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if report.interrupted {
                    Text("Crashed")
                        .font(.caption2).bold()
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.red.opacity(0.18), in: Capsule())
                        .foregroundStyle(.red)
                } else if report.cancelled {
                    Text("Stopped early")
                        .font(.caption2).bold()
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(Color.orange.opacity(0.18), in: Capsule())
                        .foregroundStyle(.orange)
                }
            }

            HStack(spacing: 14) {
                Label(report.device.marketingName, systemImage: "iphone")
                Label(report.date.formatted(date: .abbreviated, time: .shortened), systemImage: "calendar")
                Label(BuildTestFormat.duration(report.durationSeconds), systemImage: "clock")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)

            Text(report.options.summaryLine)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if report.interrupted {
                Text("The app was terminated during this run. This report was rebuilt "
                    + "from the run's journal on the next launch: the models that finished "
                    + "are recorded as measured, the one in flight is marked as crashed, and "
                    + "the rest are skipped.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else if report.cancelled {
                Text("The run was stopped before every model was reached. "
                    + "Models it never got to are absent, not recorded as skipped.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Summary

    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 16) {
                BenchmarkStatTile(
                    icon: "square.stack.3d.up",
                    value: "\(report.tested)",
                    label: "Tested"
                )
                BenchmarkStatTile(
                    icon: "checkmark.circle",
                    value: "\(report.passed)",
                    label: "Passed"
                )
                BenchmarkStatTile(
                    icon: "xmark.circle",
                    value: "\(report.failed)",
                    label: "Failed"
                )
                BenchmarkStatTile(
                    icon: "minus.circle",
                    value: "\(report.skipped)",
                    label: "Skipped"
                )
            }

            HStack(spacing: 16) {
                BenchmarkStatTile(
                    icon: "gauge.with.dots.needle.67percent",
                    value: report.averageScore.map { "\($0)/100" } ?? "n/a",
                    label: "Average score"
                )
                BenchmarkStatTile(
                    icon: "hare",
                    value: report.fastestEntry?.report.map {
                        String(format: "%.1f tok/s", $0.performance.generationTokensPerSecond)
                    } ?? "n/a",
                    label: report.fastestEntry?.modelName ?? "Fastest model"
                )
            }

            // Partial runs are scored over fewer tasks; they stay in the
            // model list below but out of the two aggregates above.
            if let note = report.partialExclusionNote {
                Text(note + " from the average and the fastest model.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if report.isEmpty {
                Text("This run recorded no models at all. That is a harness problem, "
                    + "not a verdict on the build.")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: - Entries

    private var entriesSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Models").font(.headline)
                Spacer()
                Text("in sweep order")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if report.entries.isEmpty {
                Text("No entries.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(report.entries) { entry in
                    BuildTestEntryRow(
                        entry: entry,
                        previousReport: previousReport(for: entry)
                    )
                    Divider()
                }
            }

            if let previous {
                Text("Deltas compared with the build test from "
                    + previous.date.formatted(date: .abbreviated, time: .shortened)
                    + " (v\(previous.appVersion) (\(previous.appBuild))).")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// The build test immediately preceding this one, if the store still has
    /// it.
    private var previous: BuildTestReport? {
        store.previous(before: report)
    }

    /// The same model's report in the previous build test - `nil` when that
    /// run did not reach the model or did not complete it.
    private func previousReport(for entry: BuildTestEntry) -> BenchmarkReport? {
        previous?.entries.first { $0.modelId == entry.modelId }?.report
    }

    // MARK: - Exports

    /// Renders the share card and resolves the JSON URL once per report, so
    /// opening the toolbar menu never has to wait on either.
    ///
    /// The store's own file is preferred over a fresh temporary copy: it is
    /// the artefact, named the way the release scripts expect. A temporary
    /// re-encode is the fallback for a report that is not (or no longer) on
    /// disk.
    private func prepareExports() {
        let renderer = ImageRenderer(content: BuildTestShareCard(report: report))
        renderer.scale = 3
        shareCardImage = renderer.uiImage

        if let stored = store.fileURL(for: report) {
            exportFileURL = stored
            return
        }
        let json = BuildTestReportStore.exportJSON(report)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(BuildTestReportStore.filename(for: report))
        do {
            try json.write(to: url, atomically: true, encoding: .utf8)
            exportFileURL = url
        } catch {
            exportFileURL = nil
        }
    }
}

// MARK: - Entry row

/// One model in the sweep. A completed row pushes the model's own benchmark
/// report; a failed row is red and taps to reveal the full error text (they
/// run to several lines, and a truncated stack of them is unreadable); a
/// skipped row is gray and states its reason.
private struct BuildTestEntryRow: View {
    let entry: BuildTestEntry
    let previousReport: BenchmarkReport?

    @State private var errorExpanded = false

    var body: some View {
        if let report = entry.report {
            NavigationLink {
                BenchmarkReportView(report: report)
            } label: {
                HStack(spacing: 8) {
                    content
                    Image(systemName: "chevron.right")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .buttonStyle(.plain)
        } else {
            content
        }
    }

    private var content: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 8) {
                Image(systemName: BuildTestFormat.icon(for: entry.status))
                    .font(.caption)
                    .foregroundStyle(BuildTestFormat.color(for: entry.status))
                Text(entry.modelName)
                    .font(.subheadline).bold()
                    .foregroundStyle(entry.status == .skipped ? Color.secondary : Color.primary)
                    .lineLimit(1)
                Spacer(minLength: 6)
                BuildTestBackendChip(backend: entry.backend)
            }

            switch entry.status {
            case .completed:
                if let report = entry.report {
                    metrics(report)
                    if let previousReport {
                        BenchmarkDeltaLine(
                            comparison: BenchmarkRunComparison(
                                current: report, previous: previousReport
                            ),
                            style: .compact
                        )
                    }
                }

            case .failed:
                Text(entry.error ?? "unknown error")
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(errorExpanded ? nil : 2)
                    .fixedSize(horizontal: false, vertical: errorExpanded)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .contentShape(Rectangle())
                    .onTapGesture { errorExpanded.toggle() }

            case .skipped:
                Text(entry.reason ?? "skipped")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)

            case .crashed:
                VStack(alignment: .leading, spacing: 4) {
                    Text("This model took the app down. " + (entry.error ?? ""))
                        .font(.caption2)
                        .foregroundStyle(.red)
                        .fixedSize(horizontal: false, vertical: true)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    if let log = entry.log, !log.isEmpty {
                        DisclosureGroup("Last \(log.count) log lines before the crash") {
                            Text(log.joined(separator: "\n"))
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .font(.caption2)
                    }
                }
            }
        }
        .padding(.vertical, 4)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func metrics(_ report: BenchmarkReport) -> some View {
        HStack(spacing: 12) {
            Text(BuildTestFormat.score(report))
                .font(.caption).bold()
            Text(BenchmarkFormat.tokPerSec(
                report.performance.generationTokensPerSecond,
                stdDev: report.performance.generationTokensPerSecondStdDev
            ))
            Text(BenchmarkFormat.seconds(report.performance.loadSeconds))
            Text(BenchmarkFormat.mb(report.performance.peakMemoryMB))
            if report.performance.thermallyConstrained {
                ThrottledChip(dropPercent: report.performance.throttleDropPercent)
            }
            Spacer(minLength: 0)
            // Informational: the thermal state a row ended in. Whether the
            // row's tok/s is comparable is the throttled chip's job.
            thermalBadge(report.performance.thermalStateAfter)
        }
        .font(.caption2)
        .foregroundStyle(.secondary)
        .lineLimit(1)
        .minimumScaleFactor(0.75)
    }

    private func thermalBadge(_ state: String) -> some View {
        Text(state.capitalized)
            .font(.caption2).bold()
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(BenchmarkFormat.thermalColor(state).opacity(0.18), in: Capsule())
            .foregroundStyle(BenchmarkFormat.thermalColor(state))
    }
}
