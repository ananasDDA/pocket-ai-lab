//
//  LiveMetricsPanel.swift
//  local_ai_test
//
//  Drop-down system-metrics panel docked to the very top of the screen.
//  It is a *layout* sibling of the chat's NavigationStack (see `ChatView`),
//  not an overlay: while open it occupies real vertical space and pushes the
//  whole chat screen — nav bar included — downwards. The background material
//  extends through the status bar / Dynamic Island area so the card visually
//  "hangs" from the top edge of the display, while interactive content stays
//  safely below the Dynamic Island.
//
//  Driven by `LiveDeviceMetrics.shared`, which auto-starts / auto-stops
//  1 Hz sampling via the caller's `.onChange(of:)` in `ChatView`. Zero
//  background cost when the overlay is hidden.
//
//  Dismissal:
//    • Drag translation.y < -40 → close with spring animation.
//    • Tap on the grabber (placed at the *bottom* edge — this drawer
//      slides down from the top, so the swipe-up affordance belongs
//      where the thumb is, not under the Dynamic Island).
//

import SwiftUI
import Charts

struct LiveMetricsPanel: View {
    @Bindable var metrics: LiveDeviceMetrics
    let onDismiss: () -> Void

    @State private var dragOffset: CGFloat = 0

    /// Corner radius applied only to the bottom corners — the panel docks
    /// flush with the top of the screen, so rounding the top would reveal
    /// the content behind the status bar.
    private let cornerRadius: CGFloat = 28

    var body: some View {
        VStack(spacing: 0) {
            content
                .padding(.horizontal, 18)
                .padding(.top, 6)
                .padding(.bottom, 8)
            // Grabber lives at the bottom edge — this panel slides down
            // from above and is dismissed with an upward swipe, so the
            // affordance belongs where the user's thumb will actually be.
            grabber
                .padding(.bottom, 6)
        }
        .frame(maxWidth: .infinity)
        // Only the background ignores the top safe area: the material
        // extends up through the status bar / Dynamic Island, while the
        // interactive content stays below them without any extra padding.
        .background {
            UnevenRoundedRectangle(
                topLeadingRadius: 0,
                bottomLeadingRadius: cornerRadius,
                bottomTrailingRadius: cornerRadius,
                topTrailingRadius: 0,
                style: .continuous
            )
            .fill(.ultraThinMaterial)
            .shadow(color: .black.opacity(0.22), radius: 22, y: 10)
            .ignoresSafeArea(edges: .top)
        }
        .offset(y: min(0, dragOffset))
        .gesture(dismissGesture)
    }

    // MARK: - Grabber + close

    private var grabber: some View {
        Capsule()
            .fill(Color.secondary.opacity(0.35))
            .frame(width: 38, height: 5)
            .padding(.top, 4)
            .frame(maxWidth: .infinity)
            .contentShape(Rectangle())
            .onTapGesture(perform: onDismiss)
    }

    // MARK: - Content

    private var content: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 6) {
                Image(systemName: "waveform.path.ecg")
                    .foregroundStyle(.blue)
                Text("Live metrics")
                    .font(.subheadline).bold()
                Spacer()
                if let last = metrics.history.last {
                    Text(last.timestamp, format: .dateTime.hour().minute().second())
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            MetricRow(
                title: "CPU",
                value: String(format: "%.0f%%", metrics.history.last?.cpuPercent ?? 0),
                color: .orange,
                data: metrics.history.map { $0.cpuPercent },
                range: 0...100
            )

            MetricRow(
                title: "App RAM",
                value: formatMB(metrics.history.last?.appMemoryMB ?? 0),
                color: .blue,
                data: metrics.history.map { $0.appMemoryMB },
                range: nil
            )

            systemRAMRow

            HStack(spacing: 8) {
                thermalBadge
                thermalScale
                Spacer()
            }
        }
    }

    // MARK: - System RAM bar

    private var systemRAMRow: some View {
        let last = metrics.history.last
        let used = last?.systemUsedMemoryMB ?? 0
        let total = last?.systemTotalMemoryMB ?? 1
        let ratio = min(1, max(0, used / total))
        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("System RAM")
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text("\(formatMB(used)) / \(formatMB(total))")
                    .font(.caption2).monospacedDigit()
                    .foregroundStyle(.secondary)
            }
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.secondary.opacity(0.15))
                    Capsule()
                        .fill(LinearGradient(
                            colors: [.green, .yellow, .red],
                            startPoint: .leading, endPoint: .trailing
                        ))
                        .frame(width: geo.size.width * ratio)
                }
            }
            .frame(height: 6)
        }
    }

    // MARK: - Badges

    /// Four-step pressure gauge sitting next to the thermal badge. iOS gives
    /// third-party apps no temperature reading at all — `thermalState` is the
    /// finest signal available, so the gauge visualises its level rather than
    /// pretending to show degrees.
    private var thermalScale: some View {
        let thermal = metrics.history.last?.thermal ?? .nominal
        let step = thermal.severityStep
        let color = thermalColor(thermal)
        return HStack(spacing: 3) {
            ForEach(1...4, id: \.self) { index in
                Capsule()
                    .fill(index <= step ? color : Color.secondary.opacity(0.2))
                    .frame(width: 10, height: 5)
            }
        }
        .animation(.easeInOut(duration: 0.25), value: step)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Thermal pressure \(step) of 4")
    }

    private var thermalBadge: some View {
        let thermal = metrics.history.last?.thermal ?? .nominal
        return Label(thermal.displayName, systemImage: "thermometer.medium")
            .font(.caption2).bold()
            .padding(.horizontal, 8).padding(.vertical, 4)
            .background(
                Capsule().fill(thermalColor(thermal).opacity(0.18))
            )
            .foregroundStyle(thermalColor(thermal))
    }

    // MARK: - Gestures

    private var dismissGesture: some Gesture {
        DragGesture(minimumDistance: 5)
            .onChanged { value in
                // Only respond to upward drags (negative y). Downward is a no-op.
                if value.translation.height < 0 {
                    dragOffset = value.translation.height
                }
            }
            .onEnded { value in
                if value.translation.height < -40 {
                    onDismiss()
                }
                withAnimation(.interactiveSpring()) { dragOffset = 0 }
            }
    }

    // MARK: - Helpers

    private func formatMB(_ value: Double) -> String {
        if value >= 1024 { return String(format: "%.1f GB", value / 1024) }
        return String(format: "%.0f MB", value)
    }

    private func thermalColor(_ state: ProcessInfo.ThermalState) -> Color {
        switch state {
        case .nominal:  return .green
        case .fair:     return .yellow
        case .serious:  return .orange
        case .critical: return .red
        @unknown default: return .gray
        }
    }
}

// MARK: - Metric row with sparkline

private struct MetricRow: View {
    let title: String
    let value: String
    let color: Color
    let data: [Double]
    /// Optional fixed y-axis range. Pass `nil` to let Charts autoscale.
    let range: ClosedRange<Double>?

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(title)
                    .font(.caption).foregroundStyle(.secondary)
                Spacer()
                Text(value)
                    .font(.caption).bold().monospacedDigit()
                    .foregroundStyle(color)
            }
            chart
                .frame(height: 34)
        }
    }

    private var chart: some View {
        Chart {
            ForEach(Array(data.enumerated()), id: \.offset) { idx, v in
                LineMark(x: .value("t", idx), y: .value(title, v))
                    .foregroundStyle(color)
                    .interpolationMethod(.monotone)
                AreaMark(x: .value("t", idx), y: .value(title, v))
                    .foregroundStyle(
                        LinearGradient(
                            colors: [color.opacity(0.35), color.opacity(0.02)],
                            startPoint: .top, endPoint: .bottom
                        )
                    )
                    .interpolationMethod(.monotone)
            }
        }
        .chartXAxis(.hidden)
        .chartYAxis(.hidden)
        .chartYScale(domain: yDomain)
    }

    private var yDomain: ClosedRange<Double> {
        if let range { return range }
        guard let maxV = data.max(), maxV > 0 else { return 0...1 }
        return 0...(maxV * 1.15)
    }
}
