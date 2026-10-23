//
//  BuildTestLiveView.swift
//  local_ai_test
//
//  The live screen of a build test: what is happening right now, what has
//  already been measured, and what is left. Presented full screen by the app
//  root (`ContentView`) for the whole life of a sweep - including a sweep an
//  Xcode test run started, which is the case this screen exists for: a
//  multi-hour run driven from a Mac used to show a blank app.
//
//  This view does NOT own the run. It never calls `run(options:)` and its
//  disappearance does not stop anything; it reads `BuildTestRunner.shared`
//  and calls `cancel()`. That is deliberate: a sweep started by the test
//  harness has no view behind it, so tying the run's lifetime to a `.task`
//  here (the way `BenchmarkRunView` does for a single model) would make the
//  harness's run die the moment the cover was dismissed.
//
//  It reports the end of the run through `onFinished` rather than dismissing
//  itself, so the root view can take the cover down and put the report up in
//  one place instead of two.
//

import SwiftUI

struct BuildTestLiveView: View {

    /// Called once when the sweep is over, with the artefact it produced
    /// (`nil` only if the runner somehow finished without one). The caller
    /// owns dismissal.
    var onFinished: (BuildTestReport?) -> Void

    @State private var showCancelConfirm = false
    @State private var showLog = false
    @State private var deviceName = ""
    /// `onFinished` is a one-shot: two of the three end-of-run signals below
    /// can land in the same tick, and handing the caller the same report
    /// twice would present the report sheet twice.
    @State private var finishSignalled = false

    /// How many of the newest log lines the collapsible section shows. The
    /// runner keeps up to 2000; rendering all of them in a `VStack` inside a
    /// `ScrollView` would build 2000 `Text` views on every progress tick.
    private static let visibleLogLines = 50

    private var runner: BuildTestRunner { .shared }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                header
                overallSection
                currentSection
                finishedSection
                logSection
            }
            .padding(20)
            .padding(.bottom, 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
        .safeAreaInset(edge: .bottom) { cancelBar }
        .onAppear {
            if deviceName.isEmpty {
                deviceName = DeviceSnapshot.current().marketingName
            }
            // A sweep short enough to finish before this cover finished
            // appearing would otherwise never fire either `onChange` below
            // and leave the screen up forever.
            if !runner.isRunning, !runner.phase.isActive {
                finish()
            }
        }
        // The normal end: `run` clears `isRunning` once the report is stored.
        .onChange(of: runner.isRunning) { _, isRunning in
            guard !isRunning else { return }
            finish()
        }
        // The belt-and-braces end. The sweep sets its terminal phase and
        // `lastReport` BEFORE `run` clears `isRunning`, so this fires first;
        // it is also the only signal left if the caller that owns `run` -
        // the Xcode test wrapper, say - goes away without unwinding it. The
        // `lastReport` guard rules out the tick between `isRunning` going
        // true and the sweep task actually starting, where `phase` is still
        // the previous run's terminal value.
        .onChange(of: runner.phase) { _, phase in
            guard !phase.isActive, runner.lastReport != nil else { return }
            finish()
        }
        .confirmationDialog(
            "Stop the build test?",
            isPresented: $showCancelConfirm,
            titleVisibility: .visible
        ) {
            Button("Stop the test", role: .destructive) { runner.cancel() }
            Button("Keep testing", role: .cancel) {}
        } message: {
            Text("Models downloaded by the test are removed. "
                + "Everything measured so far is kept in the report.")
        }
    }

    private func finish() {
        guard !finishSignalled else { return }
        finishSignalled = true
        onFinished(runner.lastReport)
    }

    // MARK: - Header

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text("Testing this build")
                    .font(.title2).bold()
                Spacer()
                BuildTestThermalChip(state: runner.thermalState)
            }

            Text("v\(BenchmarkAppInfo.version) (\(BenchmarkAppInfo.build))"
                + (deviceName.isEmpty ? "" : " - \(deviceName)"))
                .font(.caption)
                .foregroundStyle(.secondary)

            Text("Every model is downloaded, measured, then deleted again. "
                + "Models you installed yourself are left alone.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: - Overall progress

    private var overallSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            ProgressView(value: runner.overallFraction)
                .progressViewStyle(.linear)
                .tint(.accentColor)

            HStack {
                Text(modelCountLine)
                    .font(.caption).bold()
                Spacer()
                TimelineView(.periodic(from: .now, by: 1)) { _ in
                    Text(elapsedLine)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            HStack(spacing: 10) {
                counterChip(count: passedCount, label: "passed", tint: .green)
                counterChip(count: failedCount, label: "failed", tint: .red)
                counterChip(count: skippedCount, label: "skipped", tint: .secondary)
                Spacer()
                Text(String(format: "%.0f%%", runner.overallFraction * 100))
                    .font(.caption2).bold()
                    .foregroundStyle(.secondary)
                    .monospacedDigit()
            }
        }
    }

    private var modelCountLine: String {
        guard runner.totalCount > 0 else { return runner.phase.displayName }
        let index = max(runner.currentIndex, 1)
        return "Model \(min(index, runner.totalCount)) of \(runner.totalCount)"
    }

    private var elapsedLine: String {
        guard let elapsed = runner.elapsedSeconds else { return "" }
        return "Elapsed \(BuildTestFormat.duration(elapsed))"
    }

    private func counterChip(count: Int, label: String, tint: Color) -> some View {
        HStack(spacing: 3) {
            Text("\(count)").bold()
            Text(label)
        }
        .font(.caption2)
        .foregroundStyle(count == 0 ? Color.secondary : tint)
    }

    private var passedCount: Int {
        runner.completedEntries.filter { $0.status == .completed }.count
    }
    private var failedCount: Int {
        runner.completedEntries.filter { $0.status == .failed }.count
    }
    private var skippedCount: Int {
        runner.completedEntries.filter { $0.status == .skipped }.count
    }

    // MARK: - Current model

    private var currentSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                if runner.phase.isActive {
                    ProgressView().controlSize(.small)
                }
                Text(runner.currentModel?.name ?? runner.phase.displayName)
                    .font(.headline)
                    .lineLimit(1)
                Spacer(minLength: 6)
                if let backend = runner.currentModel?.backend.displayName {
                    BuildTestBackendChip(backend: backend)
                }
            }

            phaseDetail
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private var phaseDetail: some View {
        switch runner.phase {
        case .idle:
            phaseLine(icon: "hourglass", text: "Waiting to start")

        case .selecting:
            phaseLine(
                icon: "list.bullet.rectangle",
                text: "Working out which catalog models this device can run"
            )

        case .downloading:
            VStack(alignment: .leading, spacing: 8) {
                phaseLine(
                    icon: "arrow.down.circle",
                    text: runner.downloadProgress?.byteLine ?? "Starting the download"
                )
                ProgressView(value: runner.downloadProgress?.fraction ?? 0)
                    .progressViewStyle(.linear)
                    .tint(.accentColor)
            }

        case .benchmarking:
            BuildTestBenchmarkStageView(progress: runner.benchmarkProgress)

        case .coolingDown:
            HStack(spacing: 8) {
                Image(systemName: "thermometer.high")
                    .foregroundStyle(.orange)
                    .symbolEffect(.pulse)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Cooling down")
                        .font(.subheadline).bold()
                    Text("Waiting for the device to drop back to fair before measuring. "
                        + "Now at \(runner.thermalState).")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

        case .cleaningUp:
            phaseLine(
                icon: "trash",
                text: "Removing the files this test downloaded"
            )

        case .finished, .cancelled:
            phaseLine(icon: "checkmark.circle", text: "Writing the report")
        }
    }

    private func phaseLine(icon: String, text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 16)
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    // MARK: - Finished so far

    @ViewBuilder
    private var finishedSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Done so far")
                .font(.headline)

            if runner.completedEntries.isEmpty {
                Text("Nothing measured yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                // Newest on top: during a multi-hour sweep the interesting
                // row is the one that just landed, not the first one.
                ForEach(Array(runner.completedEntries.reversed())) { entry in
                    BuildTestOutcomeSummary(entry: entry)
                        .padding(.vertical, 6)
                    Divider()
                }
            }
        }
    }

    // MARK: - Log

    private var logSection: some View {
        DisclosureGroup(isExpanded: $showLog) {
            // Collapsed, this content is never on screen - skip building the
            // (up to 50) `Text` rows for it on every progress tick.
            if showLog {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(Array(recentLog.enumerated()), id: \.offset) { _, line in
                        Text(line)
                            .font(.system(.caption2, design: .monospaced))
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
                .padding(.top, 8)
            }
        } label: {
            HStack {
                Text("Log").font(.headline)
                Spacer()
                Text("\(runner.log.count) lines")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var recentLog: [String] {
        runner.log.suffix(Self.visibleLogLines).map { $0 }
    }

    // MARK: - Cancel

    private var cancelBar: some View {
        VStack(spacing: 0) {
            Divider()
            Button(role: .destructive) {
                showCancelConfirm = true
            } label: {
                Text(runner.phase.isActive ? "Stop build test" : "Finishing...")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .disabled(!runner.phase.isActive)
            .padding(.horizontal, 20)
            .padding(.top, 10)
            .padding(.bottom, 8)
        }
        .background(.bar)
    }
}

// MARK: - Benchmark stage

/// The benchmark stage of one model, rendered the way `BenchmarkRunView`
/// renders a standalone run: stage title, the engine's own label on up to
/// two lines, the task counter during the quality stage, and the pulsing
/// thermometer that tells a minute of apparent idleness apart from a hang.
///
/// The cool-down look is driven off `BenchmarkProgress.isThermalWait` rather
/// than by comparing the label against `BenchmarkDefaults.thermalWaitLabel`,
/// which is what that flag exists for.
struct BuildTestBenchmarkStageView: View {
    let progress: BenchmarkProgress?

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                if isCoolingDown {
                    Image(systemName: "thermometer.high")
                        .foregroundStyle(.orange)
                        .symbolEffect(.pulse)
                }
                Text(stageTitle)
                    .font(.subheadline).bold()
                Spacer(minLength: 0)
                if let progress, progress.totalTasks > 0 {
                    Text("task \(progress.completedTasks) of \(progress.totalTasks)")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
            }

            ProgressView(value: progress?.fraction ?? 0)
                .progressViewStyle(.linear)
                .tint(isCoolingDown ? .orange : .accentColor)

            Text(progress?.label ?? "Starting...")
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var stageTitle: String {
        isCoolingDown ? "Cooling down" : (progress?.stage ?? .preparing).displayName
    }

    private var isCoolingDown: Bool {
        progress?.isThermalWait ?? false
    }
}
