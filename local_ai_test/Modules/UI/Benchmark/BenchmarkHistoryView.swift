//
//  BenchmarkHistoryView.swift
//  local_ai_test
//
//  Flat, newest-first list of every stored benchmark report. Pushed from
//  the "Model Benchmark" section; rows push into `BenchmarkReportView` via
//  the `navigationDestination(for: BenchmarkReport.self)` registered on
//  `DiagnosticsView`, so this view does not need its own.
//

import SwiftUI

struct BenchmarkHistoryView: View {

    private var store: BenchmarkReportStore { .shared }

    var body: some View {
        List {
            if store.reports.isEmpty {
                Text("No benchmark runs yet.")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            } else {
                ForEach(store.reports) { report in
                    NavigationLink(value: report) {
                        BenchmarkHistoryRow(
                            report: report,
                            previous: previousReport(for: report)
                        )
                    }
                }
                .onDelete { offsets in
                    // Resolve to values BEFORE deleting: `store.delete`
                    // mutates `store.reports`, so indexing into it again
                    // partway through the loop would read stale positions.
                    let reportsToDelete = offsets.map { store.reports[$0] }
                    for report in reportsToDelete {
                        store.delete(report)
                    }
                }
            }
        }
        .navigationTitle("Past runs")
        .navigationBarTitleDisplayMode(.inline)
    }

    /// Newest report for the same model that predates this one -- the same
    /// lookup `BenchmarkReportView` uses, so a row's delta chips always match
    /// what the pushed report shows.
    private func previousReport(for report: BenchmarkReport) -> BenchmarkReport? {
        store.reports(for: report.modelId)
            .filter { $0.id != report.id && $0.date < report.date }
            .max { $0.date < $1.date }
    }
}

private struct BenchmarkHistoryRow: View {
    let report: BenchmarkReport
    let previous: BenchmarkReport?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(report.modelName)
                    .font(.subheadline).bold()
                    .lineLimit(1)
                Spacer()
                if let score = report.scoreLabel {
                    Text(score)
                        .font(.caption).bold()
                        .foregroundStyle(.secondary)
                } else {
                    Text("Perf only")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            HStack(spacing: 10) {
                Text(BenchmarkFormat.tokPerSec(report.performance.generationTokensPerSecond))
                if report.performance.thermallyConstrained {
                    ThrottledChip(dropPercent: report.performance.throttleDropPercent)
                }
                Text(report.date.formatted(date: .abbreviated, time: .shortened))
            }
            .font(.caption2)
            .foregroundStyle(.secondary)

            if let previous {
                BenchmarkDeltaLine(
                    comparison: BenchmarkRunComparison(current: report, previous: previous),
                    style: .compact
                )
            }
        }
        .padding(.vertical, 2)
    }
}
