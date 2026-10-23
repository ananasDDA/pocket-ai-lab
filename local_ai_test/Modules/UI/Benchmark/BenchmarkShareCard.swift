//
//  BenchmarkShareCard.swift
//  local_ai_test
//
//  Compact branded card rendered to an image (see `BenchmarkReportView`,
//  which snapshots this with `ImageRenderer` at 3x) for the report's share
//  sheet. Pinned to one dark background and `.colorScheme(.dark)` on
//  purpose: a card whose look depended on the *viewer's* system theme would
//  render differently for whoever the user shares it with, which makes it
//  useless as a fixed screenshot for Reddit/X.
//

import SwiftUI

struct BenchmarkShareCard: View {
    let report: BenchmarkReport

    static let size = CGSize(width: 400, height: 500)

    private static let background = Color(red: 0.07, green: 0.08, blue: 0.12)
    private static let accent = Color(red: 0.4, green: 0.65, blue: 1.0)

    var body: some View {
        VStack(spacing: 0) {
            header

            Spacer(minLength: 8)

            if let score = report.overallScoreRounded {
                ScoreRingView(score: score, size: 148, partialLabel: report.partialTasksLabel)
            } else {
                heroTokensPerSecond
            }

            Spacer(minLength: 20)

            statsRow

            if let sp = report.performance.standardProtocol {
                Spacer(minLength: 10)
                // The line Reddit / llama-bench readers recognize on sight.
                Text(sp.summaryLine)
                    .font(.system(.caption, design: .monospaced))
                    .foregroundStyle(.white.opacity(0.6))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }

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

            Text(report.modelName)
                .font(.title3).bold()
                .foregroundStyle(.white)
                .lineLimit(2)
                .multilineTextAlignment(.center)

            Text("\(report.quantization) - \(report.backend)")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.55))

            Text(report.device.marketingName)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.4))

            if report.performance.thermallyConstrained {
                Text(ThrottleWording.badge(report.performance.throttleDropPercent))
                    .font(.system(size: 8, weight: .bold))
                    .tracking(1.2)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.2), in: Capsule())
                    .foregroundStyle(.orange)
            }
        }
    }

    // MARK: - Hero (performance-only runs)

    private var heroTokensPerSecond: some View {
        VStack(spacing: 2) {
            Text(String(format: "%.1f", report.performance.generationTokensPerSecond))
                .font(.system(size: 58, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
            Text("tokens / sec")
                .font(.caption)
                .foregroundStyle(.white.opacity(0.55))
            if let sd = report.performance.generationTokensPerSecondStdDev {
                Text(String(format: "± %.1f over %d runs", sd, report.performance.generationRepetitions))
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.45))
            }
        }
    }

    // MARK: - Stats

    private var statsRow: some View {
        HStack(spacing: 0) {
            statTile("tok/s", generationStatValue)
            statTile("prefill", String(format: "%.0f est.", report.performance.prefillTokensPerSecond))
            statTile("load", String(format: "%.1fs", report.performance.loadSeconds))
            statTile("RAM", BenchmarkFormat.mb(report.performance.peakMemoryMB))
        }
    }

    /// "24.6±1.0" when the spread was measured; the tile's
    /// `minimumScaleFactor` absorbs the extra width. Plain "24.6" on v1.
    private var generationStatValue: String {
        if let sd = report.performance.generationTokensPerSecondStdDev {
            return String(format: "%.1f±%.1f", report.performance.generationTokensPerSecond, sd)
        }
        return String(format: "%.1f", report.performance.generationTokensPerSecond)
    }

    private func statTile(_ label: String, _ value: String) -> some View {
        VStack(spacing: 3) {
            Text(value)
                .font(.subheadline).bold()
                .foregroundStyle(.white)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.45))
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Footer

    private var footer: some View {
        Text("pocket-ai-lab - free & open source")
            .font(.caption2)
            .foregroundStyle(.white.opacity(0.32))
    }
}
