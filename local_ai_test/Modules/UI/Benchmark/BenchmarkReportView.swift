//
//  BenchmarkReportView.swift
//  local_ai_test
//
//  The full report for one benchmark run: header, score (or a
//  "performance only" badge), the performance grid, and -- when the run
//  graded quality -- a per-category breakdown that expands into per-task
//  results. Reachable by pushing a `BenchmarkReport` value (see the
//  `navigationDestination(for:)` registered on `DiagnosticsView`) or by
//  presenting it modally right after a run finishes.
//

import SwiftUI
import UIKit

struct BenchmarkReportView: View {
    let report: BenchmarkReport

    @State private var showDeleteConfirm = false
    @State private var shareCardImage: UIImage?
    @State private var exportFileURL: URL?
    @Environment(\.dismiss) private var dismiss

    private var store: BenchmarkReportStore { .shared }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                headerSection
                scoreSection
                Divider()
                performanceSection
                if report.quality != nil {
                    Divider()
                    qualitySection
                }
            }
            .padding(20)
        }
        .navigationTitle("Benchmark Report")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { toolbarMenu }
        .task(id: report.id) { prepareExports() }
        .confirmationDialog(
            "Delete this report?",
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
                            "\(report.modelName) benchmark",
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
                    Text(report.modelName)
                        .font(.title2).bold()
                    Text("\(report.quantization) - \(report.parameterSize)")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Text(report.backend)
                    .font(.caption2).bold()
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.accentColor.opacity(0.15), in: Capsule())
                    .foregroundStyle(Color.accentColor)
            }

            HStack(spacing: 14) {
                Label(report.device.marketingName, systemImage: "iphone")
                Label(report.date.formatted(date: .abbreviated, time: .shortened), systemImage: "calendar")
                Label("v\(report.appVersion) (\(report.appBuild))", systemImage: "number")
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
    }

    // MARK: - Score

    private var scoreSection: some View {
        VStack(spacing: 12) {
            if let score = report.overallScoreRounded {
                ScoreRingView(score: score, partialLabel: report.partialTasksLabel)
                if report.isPartial {
                    Text("The task suite stopped at its time budget. This score covers "
                        + "only the tasks that ran and is not compared with other runs. "
                        + "Run Full to grade every task.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else {
                VStack(spacing: 6) {
                    Text("Performance only")
                        .font(.headline)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(Color.secondary.opacity(0.12), in: Capsule())
                    Text("No quality score - run Standard or Full to grade this model.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 8)
            }

            if let previousReport {
                BenchmarkDeltaLine(
                    comparison: BenchmarkRunComparison(current: report, previous: previousReport)
                )
            }
        }
        .frame(maxWidth: .infinity)
    }

    /// Newest report for the same model that predates this one -- `nil` for
    /// a model's first run. Looked up on demand rather than cached: the
    /// history changes underneath this view (a run, a delete) far less
    /// often than the view itself is recreated.
    private var previousReport: BenchmarkReport? {
        store.reports(for: report.modelId)
            .filter { $0.id != report.id && $0.date < report.date }
            .max { $0.date < $1.date }
    }

    // MARK: - Performance

    private var performanceSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Performance").font(.headline)

            if report.performance.thermallyConstrained {
                ThrottledRunBanner(dropPercent: report.performance.throttleDropPercent)
            }

            HStack(spacing: 24) {
                heroStat(
                    label: "Generation",
                    value: BenchmarkFormat.tokPerSec(
                        report.performance.generationTokensPerSecond,
                        stdDev: report.performance.generationTokensPerSecondStdDev
                    ),
                    caption: generationHeroCaption
                )
                heroStat(
                    label: "Prefill (est.)",
                    value: BenchmarkFormat.tokPerSec(report.performance.prefillTokensPerSecond),
                    caption: prefillHeroCaption
                )
            }

            HStack(spacing: 16) {
                BenchmarkStatTile(
                    icon: "clock",
                    value: BenchmarkFormat.seconds(report.performance.loadSeconds),
                    label: "Load time"
                )
                BenchmarkStatTile(
                    icon: "memorychip",
                    value: BenchmarkFormat.mb(report.performance.peakMemoryMB),
                    label: "Peak RAM"
                )
                BenchmarkStatTile(
                    icon: "number",
                    value: "\(report.performance.tokensGenerated)",
                    label: tokensTileLabel
                )
            }

            MemoryUsageBar(
                peakMB: report.performance.peakMemoryMB,
                budgetMB: report.performance.memoryBudgetMB
            )

            HStack {
                Text("Thermal").font(.caption).foregroundStyle(.secondary)
                Spacer()
                ThermalTransitionView(
                    before: report.performance.thermalStateBefore,
                    after: report.performance.thermalStateAfter
                )
            }

            // Thermal readings are context, not a verdict: the throttle
            // banner above says whether the numbers can be trusted.
            if let peak = peakThermalStateBeyondEndpoints {
                Text("Peaked at \(peak.capitalized) during the run")
                    .font(.caption2)
                    .foregroundStyle(BenchmarkFormat.thermalColor(peak))
            }

            if let sp = report.performance.standardProtocol {
                standardProtocolRow(sp)
            }

            if report.performance.warmupPerformed {
                Label("Warmed up before measuring", systemImage: "flame")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// "n=3 - spread 4.1%" under the generation hero. `nil` on a v1 report
    /// and on single-repetition runs, so old reports render exactly as before.
    private var generationHeroCaption: String? {
        guard let runs = report.performance.generationRunsTokensPerSecond,
              runs.count > 1 else { return nil }
        if let spread = report.performance.generationSpreadPercent {
            return String(format: "n=%d - spread %.1f%%", runs.count, spread)
        }
        return "n=\(runs.count)"
    }

    /// "n=2" under the prefill hero when the probe was repeated.
    private var prefillHeroCaption: String? {
        guard let runs = report.performance.prefillRunsTokensPerSecond,
              runs.count > 1 else { return nil }
        return "n=\(runs.count)"
    }

    /// v2 changed `tokensGenerated` to the SUM across every decode
    /// repetition, so the tile says how many runs the count covers. A v1
    /// report (one repetition) keeps the plain "Tokens" label.
    private var tokensTileLabel: String {
        let reps = report.performance.generationRepetitions
        return reps > 1 ? "Tokens (\(reps) runs)" : "Tokens"
    }

    /// The hottest state seen mid-run, but only when it exceeds both
    /// endpoint chips — otherwise the transition row already tells the story.
    private var peakThermalStateBeyondEndpoints: String? {
        guard let peak = report.performance.peakThermalState else { return nil }
        let rank = BenchmarkFormat.thermalRank(peak)
        guard rank > BenchmarkFormat.thermalRank(report.performance.thermalStateBefore),
              rank > BenchmarkFormat.thermalRank(report.performance.thermalStateAfter)
        else { return nil }
        return peak
    }

    private func standardProtocolRow(_ sp: StandardProtocolMetrics) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Standard protocol")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(sp.summaryLine)
                    .font(.system(.caption, design: .monospaced)).bold()
            }
            Text("pp512 / tg128 follow the community llama-bench convention - tg128 averaged over \(sp.repetitions) run\(sp.repetitions == 1 ? "" : "s").")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }

    private func heroStat(label: String, value: String, caption: String? = nil) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(.title, design: .rounded)).bold()
                .lineLimit(1)
                .minimumScaleFactor(0.6)
            Text(label)
                .font(.caption)
                .foregroundStyle(.secondary)
            if let caption {
                Text(caption)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    // MARK: - Quality

    private var qualitySection: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Quality breakdown").font(.headline)
                Spacer()
                if let quality = report.quality {
                    // The suite NAME matters as much as its version now that
                    // a score can come from an imported suite: "84 / 100" is
                    // only meaningful next to the exam that produced it.
                    Text("\(report.suiteDisplayName ?? "Built-in") · suite v\(quality.suiteVersion)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.trailing)
                }
            }

            VStack(spacing: 10) {
                ForEach(report.quality?.categories ?? []) { category in
                    BenchmarkCategoryRow(category: category)
                }
            }
        }
    }

    // MARK: - Exports

    /// Renders the share card and writes the JSON export once per report,
    /// so opening the toolbar menu never has to wait on either.
    private func prepareExports() {
        let renderer = ImageRenderer(content: BenchmarkShareCard(report: report))
        renderer.scale = 3
        shareCardImage = renderer.uiImage

        let json = BenchmarkReportStore.exportJSON(report)
        let filename = "benchmark-\(BenchmarkReportStore.slug(report.modelId))"
            + "-\(BenchmarkReportStore.timestampSlug(report.date)).json"
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(filename)
        do {
            try json.write(to: url, atomically: true, encoding: .utf8)
            exportFileURL = url
        } catch {
            exportFileURL = nil
        }
    }
}

// MARK: - Category row

/// One quality category: pass fraction and a thin bar, expanding into the
/// individual task results.
private struct BenchmarkCategoryRow: View {
    let category: CategoryScore

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(category.tasks) { task in
                    BenchmarkTaskRow(task: task)
                }
            }
            .padding(.top, 8)
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(category.category.capitalized)
                        .font(.subheadline).bold()
                    Spacer()
                    Text("\(category.passed)/\(category.total)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                // The chance-normalised category score, the same formula as
                // the overall one. Equal to passed/total for free-answer
                // tasks, which is every built-in task.
                ThinProgressBar(fraction: category.normalizedScore, tint: barColor)
            }
        }
    }

    private var barColor: Color {
        switch category.normalizedScore {
        case 0.8...:    return .green
        case 0.5..<0.8: return .yellow
        default:        return .red
        }
    }
}

// MARK: - Task row

/// Pass/fail icon, task id, latency; a failed task also reveals its
/// (already truncated) raw answer in a monospaced block.
private struct BenchmarkTaskRow: View {
    let task: TaskResult

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: task.passed ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(task.passed ? .green : .red)
                    .font(.caption)
                Text("Task: \(task.taskId)")
                    .font(.caption).bold()
                Spacer()
                Text(BenchmarkFormat.seconds(task.latencySeconds))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            if !task.passed {
                if let error = task.error {
                    Text(error)
                        .font(.caption2)
                        .foregroundStyle(.red)
                } else if !task.answer.isEmpty {
                    Text(task.answer)
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.secondary)
                        .lineLimit(4)
                        .padding(8)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 6))
                }
            }
        }
    }
}
