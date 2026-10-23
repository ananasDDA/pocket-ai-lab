//
//  BuildTestSectionView.swift
//  local_ai_test
//
//  "Build test" section at the top of `DiagnosticsView`: what the sweep will
//  do to the device, the two knobs worth exposing, the start button, and the
//  way back into the artefacts it has already produced.
//
//  Follows the same split as `BenchmarkSectionView`: all state and every
//  alert live in `BuildTestSectionModel`, owned by `DiagnosticsView` and
//  attached to its `List`, while `BuildTestSection` stays a dumb row. A
//  `.alert` anchored to a `Section` inside a `List` is fragile - scrolling
//  can recycle the row it hangs off.
//
//  The section does NOT present the live screen. `ContentView` owns the
//  full-screen cover and binds it to `BuildTestRunner.shared.isRunning`, so
//  a sweep started from Xcode's test target is watchable too and there is
//  exactly one place that can put that cover up. This button only starts the
//  runner and closes Diagnostics so the cover has a clear field: a
//  `fullScreenCover` on the root cannot appear while this view's own sheet
//  chain is still up.
//

import SwiftUI
import Observation

// MARK: - State

@Observable
final class BuildTestSectionModel {

    var stages: BenchmarkOptions.Stages = .full

    /// Off means "every eligible model" (`BuildTestOptions.maxModels == nil`).
    var limitEnabled = false
    var modelLimit = 5

    /// Drives the confirmation alert hosted on `DiagnosticsView`'s `List`.
    var showStartConfirm = false

    var store: BuildTestReportStore { .shared }
    var isRunning: Bool { BuildTestRunner.shared.isRunning }
    var interruptedReport: BuildTestReport? { BuildTestRunner.shared.interruptedReport }
    var resumableOptions: BuildTestOptions? { BuildTestRunner.shared.resumableOptions }

    var options: BuildTestOptions {
        BuildTestOptions(stages: stages, maxModels: limitEnabled ? modelLimit : nil)
    }

    /// Starts the sweep. Deliberately fire-and-forget in an unstructured
    /// `Task`: the sweep must outlive this view, which is dismissed
    /// immediately afterwards so the root's cover can come up.
    ///
    /// `run` throws exactly one error - `alreadyRunning` - and the guard
    /// above already rules it out, so there is nothing left to surface.
    func start() {
        guard !BuildTestRunner.shared.isRunning else { return }
        let options = options
        Task { @MainActor in
            _ = try? await BuildTestRunner.shared.run(options: options)
        }
    }

    /// Runs only the models the interrupted run never reached. The model
    /// that crashed is not among them - see `BuildTestRecovery`.
    func resume() {
        guard !BuildTestRunner.shared.isRunning, let options = resumableOptions else { return }
        Task { @MainActor in
            _ = try? await BuildTestRunner.shared.run(options: options)
        }
    }
}

// MARK: - Section

struct BuildTestSection: View {

    @Bindable var model: BuildTestSectionModel

    var body: some View {
        Section("Build test") {
            Text("Downloads every catalog model this device can run, benchmarks it, "
                + "and deletes it again. Models you installed yourself are kept. "
                + "Takes 1-3 hours and several GB of transfer - keep the phone "
                + "plugged in and on Wi-Fi.")
                .font(.caption)
                .foregroundStyle(.secondary)

            Picker("Stage", selection: $model.stages) {
                Text("Full").tag(BenchmarkOptions.Stages.full)
                Text("Performance only").tag(BenchmarkOptions.Stages.performanceOnly)
            }
            .pickerStyle(.segmented)

            Toggle("Limit the number of models", isOn: $model.limitEnabled)

            if model.limitEnabled {
                Stepper(
                    "Test \(model.modelLimit) model\(model.modelLimit == 1 ? "" : "s")",
                    value: $model.modelLimit,
                    in: 1...50
                )
            }

            Button {
                model.showStartConfirm = true
            } label: {
                Label(
                    model.isRunning ? "Build test running" : "Start build test",
                    systemImage: "shippingbox"
                )
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .disabled(model.isRunning)

            if let interrupted = model.interruptedReport {
                VStack(alignment: .leading, spacing: 6) {
                    Label {
                        Text("The last run crashed"
                            + (interrupted.crashedEntry.map { " on \($0.modelName)" } ?? "")
                            + ". Its report was recovered from the journal.")
                            .font(.caption)
                    } icon: {
                        Image(systemName: "bolt.trianglebadge.exclamationmark.fill")
                            .foregroundStyle(.red)
                    }
                    if let resumable = model.resumableOptions {
                        Button {
                            model.resume()
                        } label: {
                            Label(
                                "Resume the remaining \(resumable.modelIDs.count) "
                                    + (resumable.modelIDs.count == 1 ? "model" : "models"),
                                systemImage: "arrow.clockwise"
                            )
                            .frame(maxWidth: .infinity)
                        }
                        .buttonStyle(.bordered)
                        .disabled(model.isRunning)
                    }
                }
            }

            if let latest = model.store.latest {
                NavigationLink {
                    BuildTestReportView(report: latest)
                } label: {
                    LatestBuildTestRow(report: latest)
                }
            }

            NavigationLink {
                BuildTestHistoryView()
            } label: {
                Label("Past build tests", systemImage: "clock.arrow.circlepath")
            }
        }
    }
}

// MARK: - Latest row

private struct LatestBuildTestRow: View {
    let report: BuildTestReport

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Latest build test")
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                Text("v\(report.appVersion) (\(report.appBuild))")
                    .font(.subheadline).bold()
                Text("\(report.passed) passed")
                    .font(.caption)
                    .foregroundStyle(report.passed > 0 ? Color.green : Color.secondary)
                Text("\(report.failed) failed")
                    .font(.caption)
                    .foregroundStyle(report.failed > 0 ? Color.red : Color.secondary)
            }
            Text(report.date.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}
