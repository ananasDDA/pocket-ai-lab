//
//  BuildTestUIComponents.swift
//  local_ai_test
//
//  Small pieces shared by the build test's live screen, its report and its
//  share card. Everything numeric and every colour ramp defers to
//  `BenchmarkFormat` so a build test row and a per-model benchmark row can
//  never disagree about what "serious" looks like or how a tok/s figure is
//  spelled.
//

import SwiftUI

// MARK: - Formatting

enum BuildTestFormat {

    /// "1:04:12" for a long sweep, "4:12" for a short one. Wall clock, not a
    /// measurement: seconds are rounded, never padded with decimals.
    static func duration(_ seconds: Double) -> String {
        let total = Int(max(seconds, 0).rounded())
        let hours = total / 3600
        let minutes = (total % 3600) / 60
        let secs = total % 60
        return hours > 0
            ? String(format: "%d:%02d:%02d", hours, minutes, secs)
            : String(format: "%d:%02d", minutes, secs)
    }

    /// "72/100" for a graded run, "61/100, 73 of 118 tasks" for a partial
    /// one, "Perf only" for a performance-only one - the same distinctions
    /// `BenchmarkReport.scoreLabel` makes, so a performance sweep never
    /// renders as a score of zero and a cut-short one never as a full score.
    static func score(_ report: BenchmarkReport) -> String {
        report.scoreLabel ?? "Perf only"
    }

    static func icon(for status: BuildTestEntry.Status) -> String {
        switch status {
        case .completed: return "checkmark.circle.fill"
        case .failed:    return "xmark.circle.fill"
        case .skipped:   return "minus.circle.fill"
        case .crashed:   return "bolt.trianglebadge.exclamationmark.fill"
        }
    }

    static func color(for status: BuildTestEntry.Status) -> Color {
        switch status {
        case .completed: return .green
        case .failed:    return .red
        case .skipped:   return .secondary
        case .crashed:   return .red
        }
    }

    /// "Full" / "Performance only" from the report's stored `stages` string.
    static func stagesLabel(_ raw: String) -> String {
        BenchmarkOptions.Stages(rawValue: raw) == .performanceOnly ? "Performance only" : "Full"
    }
}

// MARK: - Chips

/// Backend name in the accent capsule the benchmark report header uses.
struct BuildTestBackendChip: View {
    let backend: String

    var body: some View {
        Text(backend)
            .font(.caption2).bold()
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Color.accentColor.opacity(0.15), in: Capsule())
            .foregroundStyle(Color.accentColor)
    }
}

/// Live thermal state, on the shared nominal/fair/serious/critical scale.
struct BuildTestThermalChip: View {
    let state: String

    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: "thermometer.medium")
            Text(state.capitalized)
        }
        .font(.caption2).bold()
        .padding(.horizontal, 8)
        .padding(.vertical, 3)
        .background(BenchmarkFormat.thermalColor(state).opacity(0.18), in: Capsule())
        .foregroundStyle(BenchmarkFormat.thermalColor(state))
    }
}

// MARK: - Outcome list row

/// One finished model, in the shape both the live screen's "done so far"
/// list and the report's sweep list use: status dot, name, backend, and
/// either the numbers or the reason it has none.
struct BuildTestOutcomeSummary: View {
    let entry: BuildTestEntry

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Image(systemName: BuildTestFormat.icon(for: entry.status))
                    .font(.caption)
                    .foregroundStyle(BuildTestFormat.color(for: entry.status))
                Text(entry.modelName)
                    .font(.subheadline).bold()
                    .lineLimit(1)
                Spacer(minLength: 6)
                BuildTestBackendChip(backend: entry.backend)
            }

            switch entry.status {
            case .completed:
                if let report = entry.report {
                    HStack(spacing: 10) {
                        Text(BuildTestFormat.score(report))
                            .font(.caption).bold()
                        Text(BenchmarkFormat.tokPerSec(
                            report.performance.generationTokensPerSecond,
                            stdDev: report.performance.generationTokensPerSecondStdDev
                        ))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        if report.performance.thermallyConstrained {
                            ThrottledChip(dropPercent: report.performance.throttleDropPercent)
                        }
                    }
                }
            case .failed:
                Text(entry.error ?? "unknown error")
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(3)
            case .crashed:
                Text("Crashed the app - " + (entry.error ?? "the process was terminated"))
                    .font(.caption2)
                    .foregroundStyle(.red)
                    .lineLimit(3)
            case .skipped:
                Text(entry.reason ?? "skipped")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
        }
    }
}
