//
//  BenchmarkRunView.swift
//  local_ai_test
//
//  Full-screen progress UI for one benchmark run. Owns the run's lifetime
//  through its `.task`: SwiftUI cancels that task automatically when the
//  view disappears, which is exactly what "Cancel" needs to do -- the
//  engine's own cancellation handling (unload, throw `CancellationError`)
//  takes it from there.
//

import SwiftUI

struct BenchmarkRunView: View {
    let model: AIModel
    let options: BenchmarkOptions
    let onFinished: (Result<BenchmarkReport, Error>) -> Void

    @State private var progress: BenchmarkProgress?
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 28) {
            Spacer()

            VStack(spacing: 8) {
                ProgressView()
                    .controlSize(.large)
                HStack(spacing: 6) {
                    if isCoolingDown {
                        Image(systemName: "thermometer.high")
                            .foregroundStyle(.orange)
                            .symbolEffect(.pulse)
                    }
                    Text(stageTitle)
                        .font(.title3).bold()
                }
                Text(model.name)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            VStack(spacing: 10) {
                ProgressView(value: progress?.fraction ?? 0)
                    .progressViewStyle(.linear)
                    .tint(.accentColor)

                // v2 labels run longer than v1's ("Waiting for the device to
                // cool down", "Prompt processing - run 1 of 2", "Standard
                // protocol - pp512"), so reserve two lines instead of
                // truncating mid-word on narrow devices.
                Text(progress?.label ?? "Starting...")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(minHeight: 40)

                if let progress, progress.totalTasks > 0 {
                    Text("\(progress.completedTasks) / \(progress.totalTasks) tasks")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(.horizontal, 32)

            Spacer()

            Button(role: .destructive) {
                dismiss()
            } label: {
                Text("Cancel")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.bordered)
            .padding(.horizontal, 32)
            .padding(.bottom, 24)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
        .task {
            do {
                let report = try await BenchmarkEngine.shared.runAndStore(
                    model: model,
                    options: options
                ) { update in
                    progress = update
                }
                onFinished(.success(report))
            } catch is CancellationError {
                // The user hit Cancel (or the view was otherwise torn down);
                // the engine has already unwound and unloaded. Nothing to
                // report -- the sheet is on its way out already.
            } catch {
                onFinished(.failure(error))
            }
        }
    }

    private var stageTitle: String {
        isCoolingDown ? "Cooling down" : (progress?.stage ?? .preparing).displayName
    }

    /// The engine's thermal gate is running: stage `.preparing` with the
    /// exact label `BenchmarkDefaults.thermalWaitLabel`. Shown with its own
    /// title and a pulsing thermometer so a minute of apparent idleness
    /// reads as deliberate waiting, not a hang.
    private var isCoolingDown: Bool {
        progress?.stage == .preparing
            && progress?.label == BenchmarkDefaults.thermalWaitLabel
    }
}
