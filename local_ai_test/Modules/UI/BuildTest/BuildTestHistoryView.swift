//
//  BuildTestHistoryView.swift
//  local_ai_test
//
//  Newest-first list of the stored build tests (the store keeps ten). Rows
//  push `BuildTestReportView` directly rather than through a
//  `navigationDestination(for: BuildTestReport.self)`, so this list works in
//  any stack it is pushed into - including the bare one inside the post-run
//  sheet.
//

import SwiftUI

struct BuildTestHistoryView: View {

    private var store: BuildTestReportStore { .shared }

    var body: some View {
        List {
            if store.reports.isEmpty {
                Text("This build has not been tested yet.")
                    .foregroundStyle(.secondary)
                    .font(.caption)
            } else {
                ForEach(store.reports) { report in
                    NavigationLink {
                        BuildTestReportView(report: report)
                    } label: {
                        BuildTestHistoryRow(report: report)
                    }
                }
                .onDelete { offsets in
                    // Resolve to values BEFORE deleting: `store.delete`
                    // mutates `store.reports`, so indexing into it again
                    // partway through the loop would read stale positions.
                    let doomed = offsets.map { store.reports[$0] }
                    for report in doomed { store.delete(report) }
                }
            }
        }
        .navigationTitle("Past build tests")
        .navigationBarTitleDisplayMode(.inline)
    }
}

// MARK: - Row

struct BuildTestHistoryRow: View {
    let report: BuildTestReport

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("v\(report.appVersion) (\(report.appBuild))")
                    .font(.subheadline).bold()
                    .lineLimit(1)
                if report.interrupted {
                    Text("crashed")
                        .font(.caption2).bold()
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.red.opacity(0.18), in: Capsule())
                        .foregroundStyle(.red)
                } else if report.cancelled {
                    Text("stopped")
                        .font(.caption2).bold()
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.orange.opacity(0.18), in: Capsule())
                        .foregroundStyle(.orange)
                }
                Spacer()
                Text(BuildTestFormat.stagesLabel(report.stages))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 10) {
                Text("\(report.passed) passed")
                    .foregroundStyle(report.passed > 0 ? Color.green : Color.secondary)
                Text("\(report.failed) failed")
                    .foregroundStyle(report.failed > 0 ? Color.red : Color.secondary)
                Text("\(report.skipped) skipped")
                    .foregroundStyle(.secondary)
            }
            .font(.caption2).bold()

            HStack(spacing: 10) {
                Text(report.date.formatted(date: .abbreviated, time: .shortened))
                Text(BuildTestFormat.duration(report.durationSeconds))
                Text(report.device.marketingName)
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
        .padding(.vertical, 2)
    }
}
