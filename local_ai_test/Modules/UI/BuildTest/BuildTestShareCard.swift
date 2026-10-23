//
//  BuildTestShareCard.swift
//  local_ai_test
//
//  Compact card for a whole build test, snapshotted with `ImageRenderer` by
//  `BuildTestReportView`. Same fixed dark treatment as
//  `BenchmarkShareCard` and for the same reason: a card whose look followed
//  the viewer's system theme renders differently for whoever it is shared
//  with, which makes it useless as a screenshot.
//

import SwiftUI

struct BuildTestShareCard: View {
    let report: BuildTestReport

    static let size = CGSize(width: 400, height: 500)

    private static let background = Color(red: 0.07, green: 0.08, blue: 0.12)
    private static let accent = Color(red: 0.4, green: 0.65, blue: 1.0)

    /// The three best models, ranked by score when the sweep graded quality
    /// and by generation throughput when it did not (see
    /// `BuildTestReport.rankedEntries`). Partial runs are not ranked.
    private var topEntries: [BuildTestEntry] {
        Array(report.rankedEntries.prefix(3))
    }

    var body: some View {
        VStack(spacing: 0) {
            header

            Spacer(minLength: 14)

            countsRow

            Spacer(minLength: 18)

            VStack(alignment: .leading, spacing: 10) {
                Text("TOP MODELS")
                    .font(.system(size: 9, weight: .bold))
                    .tracking(1.6)
                    .foregroundStyle(.white.opacity(0.35))

                if topEntries.isEmpty {
                    Text("No model completed this sweep.")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.5))
                } else {
                    ForEach(topEntries) { entry in
                        topRow(entry)
                    }
                }

                if let note = report.partialExclusionNote {
                    Text(note)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.4))
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            Spacer(minLength: 14)

            footer
        }
        .padding(28)
        .frame(width: Self.size.width, height: Self.size.height)
        .background(Self.background)
        .colorScheme(.dark)
    }

    // MARK: - Header

    private var header: some View {
        VStack(spacing: 5) {
            Text("POCKET AI LAB")
                .font(.caption2).bold()
                .tracking(2.5)
                .foregroundStyle(Self.accent)

            Text("Build test")
                .font(.title3).bold()
                .foregroundStyle(.white)

            Text("v\(report.appVersion) (\(report.appBuild)) - \(BuildTestFormat.stagesLabel(report.stages))")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.55))

            Text(report.device.marketingName)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.4))

            Text(report.date.formatted(date: .abbreviated, time: .omitted)
                + " - " + BuildTestFormat.duration(report.durationSeconds))
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.35))

            if report.cancelled {
                Text("STOPPED EARLY")
                    .font(.system(size: 8, weight: .bold))
                    .tracking(1.2)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.2), in: Capsule())
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: - Counts

    private var countsRow: some View {
        HStack(spacing: 0) {
            countTile("tested", "\(report.tested)", .white)
            countTile("passed", "\(report.passed)", .green)
            countTile("failed", "\(report.failed + report.crashed)", report.failed + report.crashed > 0 ? .red : .white)
            countTile("skipped", "\(report.skipped)", .white)
        }
    }

    private func countTile(_ label: String, _ value: String, _ tint: Color) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.system(size: 26, weight: .bold, design: .rounded))
                .foregroundStyle(tint)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.45))
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Top models

    private func topRow(_ entry: BuildTestEntry) -> some View {
        HStack(spacing: 8) {
            Text(entry.modelName)
                .font(.caption).bold()
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Spacer(minLength: 6)
            if let model = entry.report {
                Text(BuildTestFormat.score(model))
                    .font(.caption2).bold()
                    .foregroundStyle(Self.accent)
                Text(String(format: "%.1f tok/s", model.performance.generationTokensPerSecond))
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6))
            }
        }
    }

    // MARK: - Footer

    private var footer: some View {
        Text("pocket-ai-lab - free & open source")
            .font(.caption2)
            .foregroundStyle(.white.opacity(0.32))
    }
}
