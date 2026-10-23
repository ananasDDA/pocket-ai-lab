//
//  BenchmarkUIComponents.swift
//  local_ai_test
//
//  Small, reusable pieces shared by the benchmark section, the full report
//  view and the share card: number formatting, the thermal color scale
//  (kept in sync with `LiveMetricsPanel`'s nominal/fair/serious/critical
//  palette), the score ring, and a couple of thin progress bars.
//

import SwiftUI

// MARK: - Formatting

enum BenchmarkFormat {

    static func tokPerSec(_ value: Double) -> String {
        String(format: "%.1f tok/s", value)
    }

    /// "24.6 ± 1.0 tok/s" when a spread was measured, plain "24.6 tok/s"
    /// otherwise — so a v1 report (no repetitions) renders exactly as before.
    static func tokPerSec(_ value: Double, stdDev: Double?) -> String {
        guard let stdDev else { return tokPerSec(value) }
        return String(format: "%.1f ± %.1f tok/s", value, stdDev)
    }

    static func seconds(_ value: Double) -> String {
        String(format: "%.1f s", value)
    }

    static func mb(_ value: Double) -> String {
        value >= 1024 ? String(format: "%.2f GB", value / 1024) : String(format: "%.0f MB", value)
    }

    /// Same palette as `LiveMetricsPanel.thermalColor`, driven off the
    /// report's stored `reportName` strings rather than `ThermalState`.
    static func thermalColor(_ state: String) -> Color {
        switch state {
        case "nominal":  return .green
        case "fair":     return .yellow
        case "serious":  return .orange
        case "critical": return .red
        default:         return .gray
        }
    }

    static func thermalRank(_ state: String) -> Int {
        switch state {
        case "nominal":  return 0
        case "fair":     return 1
        case "serious":  return 2
        case "critical": return 3
        default:         return -1
        }
    }
}

// MARK: - Score ring

/// 0-100 gauge for the overall quality score. Sized to work both inline in
/// the full report and shrunk down inside the share card.
struct ScoreRingView: View {
    let score: Int
    var size: CGFloat = 150
    /// "73 of 118 tasks" when the run stopped at its time budget. Printed
    /// inside the ring so a partial score cannot be cropped away from its
    /// caveat, in the report view and on the share card alike.
    var partialLabel: String? = nil

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.18), lineWidth: size * 0.09)
            Circle()
                .trim(from: 0, to: CGFloat(max(0, min(score, 100))) / 100)
                .stroke(ringColor, style: StrokeStyle(lineWidth: size * 0.09, lineCap: .round))
                .rotationEffect(.degrees(-90))
            VStack(spacing: 0) {
                Text("\(score)")
                    .font(.system(size: size * 0.3, weight: .bold, design: .rounded))
                Text("/ 100")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                if let partialLabel {
                    Text("partial, \(partialLabel)")
                        .font(.caption2)
                        .foregroundStyle(.orange)
                        .multilineTextAlignment(.center)
                        .lineLimit(2)
                        .minimumScaleFactor(0.8)
                        .frame(maxWidth: size * 0.62)
                }
            }
        }
        .frame(width: size, height: size)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(partialLabel.map { "Partial score \(score) of 100, \($0)" }
            ?? "Score \(score) of 100")
    }

    private var ringColor: Color {
        switch score {
        case 80...:   return .green
        case 60..<80: return .yellow
        case 40..<60: return .orange
        default:      return .red
        }
    }
}

// MARK: - Thin progress bar

/// A hairline fraction bar used under category rows and the memory gauge.
struct ThinProgressBar: View {
    let fraction: Double
    var tint: Color = .accentColor

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(Color.secondary.opacity(0.15))
                Capsule()
                    .fill(tint)
                    .frame(width: geo.size.width * min(max(fraction, 0), 1))
            }
        }
        .frame(height: 5)
    }
}

// MARK: - Memory usage bar

/// Peak memory against the backend's budget for this device, with the same
/// bar shape as `ThinProgressBar` but its own color ramp (headroom matters
/// more than a flat 0-1 fraction here).
struct MemoryUsageBar: View {
    let peakMB: Double
    let budgetMB: Int64

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Peak memory").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Text(label).font(.caption2).foregroundStyle(.secondary)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.15))
                    RoundedRectangle(cornerRadius: 3)
                        .fill(barColor)
                        .frame(width: geo.size.width * min(max(fraction, 0), 1))
                }
            }
            .frame(height: 6)
        }
    }

    private var fraction: Double {
        budgetMB > 0 ? peakMB / Double(budgetMB) : 0
    }

    private var barColor: Color {
        if fraction > 0.9 { return .red }
        if fraction > 0.7 { return .orange }
        return .accentColor
    }

    private var label: String {
        budgetMB > 0
            ? "\(BenchmarkFormat.mb(peakMB)) / \(BenchmarkFormat.mb(Double(budgetMB))) budget"
            : BenchmarkFormat.mb(peakMB)
    }
}

// MARK: - Thermal transition

/// Before -> after thermal chips, color-coded on the shared scale, with a
/// small flame marker when the run made things measurably hotter.
struct ThermalTransitionView: View {
    let before: String
    let after: String

    var body: some View {
        HStack(spacing: 8) {
            chip(before)
            Image(systemName: "arrow.right")
                .font(.caption2)
                .foregroundStyle(.secondary)
            chip(after)
            if degraded {
                Image(systemName: "flame.fill")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
    }

    private var degraded: Bool {
        BenchmarkFormat.thermalRank(after) > BenchmarkFormat.thermalRank(before)
    }

    private func chip(_ state: String) -> some View {
        Text(state.capitalized)
            .font(.caption2).bold()
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(BenchmarkFormat.thermalColor(state).opacity(0.18), in: Capsule())
            .foregroundStyle(BenchmarkFormat.thermalColor(state))
    }
}

// MARK: - Delta vs previous run

/// The rules for comparing two reports of the same model, in one place so
/// the report view and the history list can never disagree:
///
///   - score deltas only when `BenchmarkReport.isScoreComparable(with:)` is
///     true (same suite AND same suite version — a v1 score against a v2
///     score is two different exams, not a regression, and a score from an
///     imported suite shares nothing with a built-in one);
///   - tok/s deltas additionally suppressed when either run was throttled
///     (`PerformanceMetrics.throughputIsComparable` is false): a run whose
///     own repetitions disagree measures the phone ramping down, not the
///     model;
///   - tok/s deltas also suppressed when the two runs were measured under
///     different `PerformanceMetrics.metricsVersion`s: v1 throughput was
///     single-shot and unwarmed, so a v2-vs-v1 delta shows the methodology
///     change, not the model.
struct BenchmarkRunComparison {
    let current: BenchmarkReport
    let previous: BenchmarkReport

    /// Generation tok/s delta, or `nil` when the two throughputs are not
    /// comparable (either run throttled, or the two were measured under
    /// different metrics versions).
    var throughputDelta: Double? {
        guard current.performance.throughputIsComparable,
              previous.performance.throughputIsComparable,
              current.performance.metricsVersion == previous.performance.metricsVersion
        else { return nil }
        return current.performance.generationTokensPerSecond
            - previous.performance.generationTokensPerSecond
    }

    /// True when a tok/s delta was withheld because a run was throttled.
    var throughputSuppressedByThrottling: Bool {
        current.performance.thermallyConstrained
            || previous.performance.thermallyConstrained
    }

    /// True when a tok/s delta was withheld because the two runs measured
    /// throughput differently (e.g. v1 single-shot vs v2 warmed-up mean).
    var throughputSuppressedByMetricsVersion: Bool {
        current.performance.metricsVersion != previous.performance.metricsVersion
    }

    /// Rounded score delta, or `nil` when either run has no score or the two
    /// runs were graded by different suite versions.
    var scoreDelta: Int? {
        guard current.isScoreComparable(with: previous),
              let cur = current.overallScoreRounded,
              let prev = previous.overallScoreRounded else { return nil }
        return cur - prev
    }

    /// True when a score delta was withheld because either run stopped at
    /// its time budget: a partial score is over a subset of the suite.
    var scoreSuppressedByPartialRun: Bool {
        current.overallScoreRounded != nil
            && previous.overallScoreRounded != nil
            && (current.isPartial || previous.isPartial)
    }

    /// True when both runs carry a score but they are not comparable because
    /// of the suite (version or identity), i.e. a score delta exists in
    /// principle but must not be shown. A partial run is reported by
    /// `scoreSuppressedByPartialRun` instead.
    var scoreSuppressedBySuiteVersion: Bool {
        current.overallScoreRounded != nil
            && previous.overallScoreRounded != nil
            && !scoreSuppressedByPartialRun
            && !current.isScoreComparable(with: previous)
    }

    /// True when the two runs were graded by DIFFERENT suites rather than by
    /// two versions of the same one — worth saying separately, because
    /// "different versions" reads like "upgrade and re-run" while this one
    /// never becomes comparable.
    var scoreSuppressedByDifferentSuites: Bool {
        guard scoreSuppressedBySuiteVersion,
              let mine = current.suiteIdentity,
              let theirs = previous.suiteIdentity else { return false }
        return mine != theirs
    }

    /// Why one or both deltas are hidden — `nil` when nothing was suppressed.
    var suppressionNote: String? {
        var notes: [String] = []
        if throughputSuppressedByThrottling {
            notes.append("tok/s delta hidden - a throttled run is not comparable")
        }
        if throughputSuppressedByMetricsVersion {
            notes.append("tok/s delta hidden - measurement methodology changed between the runs")
        }
        if scoreSuppressedByPartialRun {
            notes.append("score delta hidden - a partial run (time budget reached) "
                + "is not comparable")
        } else if scoreSuppressedByDifferentSuites {
            notes.append("score delta hidden - the two runs were graded by different "
                + "task suites")
        } else if scoreSuppressedBySuiteVersion {
            notes.append("score delta hidden - different test suite versions")
        }
        return notes.isEmpty ? nil : notes.joined(separator: ". ")
    }
}

/// Green/red arrows comparing a run against the previous one for the same
/// model. `.full` (report view) adds the "vs <date>" trailer and spells out
/// why a delta is hidden; `.compact` (history rows) shows just the chips
/// with a short note.
struct BenchmarkDeltaLine: View {

    enum Style {
        case full
        case compact
    }

    let comparison: BenchmarkRunComparison
    var style: Style = .full

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 14) {
                if let delta = comparison.throughputDelta {
                    deltaChip(suffix: "tok/s", delta: delta)
                }
                if let delta = comparison.scoreDelta {
                    deltaChip(suffix: "score", delta: Double(delta))
                }
                if style == .compact, let shortNote {
                    Text(shortNote)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if style == .full {
                    Spacer()
                    Text("vs \(comparison.previous.date.formatted(date: .abbreviated, time: .omitted))")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            if style == .full, let note = comparison.suppressionNote {
                Text(note)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }

    private var shortNote: String? {
        var notes: [String] = []
        if comparison.throughputSuppressedByThrottling { notes.append("tok/s n/a (throttled)") }
        if comparison.throughputSuppressedByMetricsVersion { notes.append("tok/s n/a (methodology changed)") }
        if comparison.scoreSuppressedByPartialRun { notes.append("score n/a (partial run)") }
        if comparison.scoreSuppressedBySuiteVersion { notes.append("score n/a (suite changed)") }
        return notes.isEmpty ? nil : notes.joined(separator: ", ")
    }

    /// Any delta that would render as "0.0" at one decimal place is treated
    /// as neutral: gray, no arrow, no sign. "+0.0" in green with an up-arrow
    /// would claim an improvement the numbers do not show.
    private static let neutralEpsilon = 0.05

    private func deltaChip(suffix: String, delta: Double) -> some View {
        let neutral = abs(delta) < Self.neutralEpsilon
        return HStack(spacing: 3) {
            if !neutral {
                Image(systemName: delta > 0 ? "arrow.up.right" : "arrow.down.right")
            }
            Text(neutral
                ? String(format: "0.0 %@", suffix)
                : String(format: "%@%.1f %@", delta > 0 ? "+" : "", delta, suffix))
        }
        .font(.caption2).bold()
        .foregroundStyle(neutral ? Color.secondary : (delta > 0 ? Color.green : Color.red))
    }
}

// MARK: - Throttling wording

/// Every user-facing string for `PerformanceMetrics.thermallyConstrained`,
/// in one place so the banner, the badge and the list chips cannot drift
/// apart. The word is "throttled", not "thermally constrained": the flag now
/// reports a measured slowdown, and the phone's thermal state is shown
/// separately as information.
nonisolated enum ThrottleWording {

    /// "throttled -46%", or plain "throttled" when the drop is unknown (an
    /// older report, or a single-repetition run flagged by the tg128 check).
    static func chip(_ dropPercent: Double?) -> String {
        guard let drop = dropPercent, drop.isFinite, drop > 0 else { return "throttled" }
        return String(format: "throttled -%.0f%%", drop)
    }

    /// Same, upper-cased for the share-card badge.
    static func badge(_ dropPercent: Double?) -> String {
        chip(dropPercent).uppercased()
    }

    /// The full sentence for the report banner.
    static func banner(_ dropPercent: Double?) -> String {
        guard let drop = dropPercent, drop.isFinite, drop > 0 else {
            return "Run was throttled - numbers are not comparable"
        }
        return String(
            format: "Run was throttled - speed fell %.0f%% across runs, numbers are not comparable",
            drop
        )
    }
}

/// Inline "throttled" marker for a list row, next to the tok/s figure.
struct ThrottledChip: View {
    let dropPercent: Double?

    var body: some View {
        Label(ThrottleWording.chip(dropPercent), systemImage: "speedometer")
            .font(.caption2)
            .foregroundStyle(BenchmarkFormat.thermalColor("serious"))
            .lineLimit(1)
    }
}

// MARK: - Throttled banner

/// Subtle full-width warning shown on a report whose throughput numbers were
/// taken while the device was ramping down. Uses the "serious" color from the
/// shared thermal scale — the run is not comparable either way, and the
/// reader already reads that color as "careful".
struct ThrottledRunBanner: View {

    let dropPercent: Double?

    private var tint: Color { BenchmarkFormat.thermalColor("serious") }

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "speedometer")
                .font(.caption)
            Text(ThrottleWording.banner(dropPercent))
                .font(.caption)
        }
        .foregroundStyle(tint)
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 8))
    }
}

// MARK: - Labeled stat

/// Icon + value + caption, the same shape `ModelCardView.stat` uses, kept
/// here so the report view and the share card can share it.
struct BenchmarkStatTile: View {
    let icon: String
    let value: String
    let label: String

    var body: some View {
        VStack(spacing: 2) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.subheadline).bold()
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }
}
