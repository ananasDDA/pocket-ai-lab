//
//  BenchmarkSectionView.swift
//  local_ai_test
//
//  "Model Benchmark" section at the top of `DiagnosticsView`: pick an
//  installed, text-capable model, choose Standard (task suite capped at 20
//  minutes), Full (every task, no cap) or Performance only, run it, and see
//  the latest report for whatever is selected.
//
//  All state and run lifecycle live in `BenchmarkSectionModel`, owned by
//  `DiagnosticsView` and attached to its `List`. `BenchmarkSection` itself
//  stays a dumb row: no `.task`, `.onAppear`, `.fullScreenCover`, `.sheet` or
//  `.alert` here. Those modifiers used to sit on this `Section` inside a
//  `List`, where SwiftUI can recycle/redraw row content on scroll -- a
//  `.task` there refires more than intended, and a `.fullScreenCover`/
//  `.sheet` anchored to a recyclable row is fragile. Presentation belongs at
//  the `List` level, next to the `navigationDestination` `DiagnosticsView`
//  already registers there.
//

import SwiftUI
import Observation

// MARK: - State + run lifecycle

/// Everything the "Model Benchmark" section needs: selection, the in-flight
/// run, and the sheets/alerts a run can produce. Owned by `DiagnosticsView`
/// so every presentation modifier it drives can live on the `List`, not on
/// a `Section` row.
@Observable
final class BenchmarkSectionModel {

    var selectedModelID: String?
    var mode: BenchmarkRunMode = .standard

    /// Which suite grades the run. `BenchmarkSuiteDocument.builtInID` is the
    /// suite that ships with the app; anything else is an imported one.
    var selectedSuiteID: String = BenchmarkSuiteDocument.builtInID

    /// Set while a run's full-screen progress cover is up.
    var runningJob: RunningJob?
    /// Set right after a run finishes successfully; drives the post-run
    /// report sheet. Kept separate from the push-based navigation the
    /// "latest run" row and history use, so the two never fight over the
    /// same destination.
    var justFinishedReport: BenchmarkReport?
    /// A model is loaded and the user has not yet confirmed the run will
    /// unload it.
    var pendingRunModel: AIModel?
    var runError: BenchmarkRunError?

    private var installedStore: InstalledModelsStore { .shared }
    private var reportStore: BenchmarkReportStore { .shared }
    private var suiteStore: CustomBenchmarkSuiteStore { .shared }

    var customSuites: [StoredBenchmarkSuite] { suiteStore.suites }

    /// Tasks the selected suite would run, for the row under the picker.
    var selectedSuiteTaskCount: Int {
        guard selectedSuiteID != BenchmarkSuiteDocument.builtInID else {
            return RemoteBenchmarkSuite.shared.suite.tasks.count
        }
        return suiteStore.suite(id: selectedSuiteID)?.taskCount ?? 0
    }

    /// Installed models the engine can actually benchmark: text in, text
    /// out. Excludes Whisper (audio-in only) and Kokoro/other TTS (no
    /// text-out) the same way `BenchmarkEngine.run` itself would reject
    /// them, so the picker never offers a model that's just going to throw
    /// `unsupportedModel`.
    var eligibleModels: [AIModel] {
        installedStore.installedModels.filter {
            $0.capabilities.contains(.textIn) && $0.capabilities.contains(.textOut)
        }
    }

    var selectedModel: AIModel? {
        eligibleModels.first { $0.id == selectedModelID } ?? eligibleModels.first
    }

    func latestReport(for model: AIModel) -> BenchmarkReport? {
        reportStore.latest(for: model.id)
    }

    /// Falls back to the built-in suite when the selected one was deleted
    /// from the suites screen while this section was on screen.
    func ensureSuiteSelection() {
        guard selectedSuiteID != BenchmarkSuiteDocument.builtInID else { return }
        if suiteStore.suite(id: selectedSuiteID) == nil {
            selectedSuiteID = BenchmarkSuiteDocument.builtInID
        }
    }

    // MARK: Selection

    func ensureSelection() {
        guard !eligibleModels.isEmpty else { return }
        if let selectedModelID, eligibleModels.contains(where: { $0.id == selectedModelID }) {
            return
        }
        let chatSelection = installedStore.selectedModel
        if let chatSelection, eligibleModels.contains(where: { $0.id == chatSelection.id }) {
            selectedModelID = chatSelection.id
        } else {
            selectedModelID = eligibleModels.first?.id
        }
    }

    // MARK: Run lifecycle

    func requestRun() {
        guard let model = selectedModel else { return }
        if InferenceManager.shared.currentModelId != nil {
            pendingRunModel = model
        } else {
            startRun(model: model)
        }
    }

    func startRun(model: AIModel) {
        ensureSuiteSelection()
        var options = mode.options
        // Only the quality stage reads a suite, but carrying the id either
        // way keeps the option struct honest about what was selected.
        options.suiteID = selectedSuiteID
        runningJob = RunningJob(model: model, options: options)
    }

    func handleRunFinished(_ result: Result<BenchmarkReport, Error>) {
        runningJob = nil
        switch result {
        case .success(let report):
            // A brief delay lets the full-screen cover's dismissal
            // animation finish before the report sheet comes up; presenting
            // both in the same tick occasionally drops the second one.
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 250_000_000)
                justFinishedReport = report
            }
        case .failure(let error):
            Task { @MainActor in
                try? await Task.sleep(nanoseconds: 250_000_000)
                runError = BenchmarkRunError(message: error.localizedDescription)
            }
        }
    }
}

// MARK: - Support types

/// The three ways to start a benchmark from this section.
enum BenchmarkRunMode: String, CaseIterable, Identifiable {
    /// Speed plus the task suite, with the suite capped at
    /// `BenchmarkDefaults.standardQualityBudget`. The default: no model
    /// keeps the phone busy for an hour unless somebody asked for that.
    case standard
    /// Speed plus every task, however long it takes.
    case full
    /// Speed only, no score.
    case performanceOnly

    var id: String { rawValue }

    var title: String {
        switch self {
        case .standard:        return "Standard"
        case .full:            return "Full"
        case .performanceOnly: return "Perf only"
        }
    }

    var options: BenchmarkOptions {
        switch self {
        case .standard:        return .standard
        case .full:            return .full
        case .performanceOnly: return .performanceOnly
        }
    }

    var runsSuite: Bool { self != .performanceOnly }

    /// One line under the picker saying what the choice costs.
    func caption(taskCount: Int) -> String {
        switch self {
        case .standard:
            let minutes = Int(BenchmarkDefaults.standardQualityBudget / 60)
            return "\(taskCount) tasks, stopped after \(minutes) min. A run that stops "
                + "early gets a partial score."
        case .full:
            return "All \(taskCount) tasks, however long it takes. Large models can "
                + "need an hour."
        case .performanceOnly:
            return "Speed and memory only, no score."
        }
    }
}

/// Not `private`: `DiagnosticsView` binds `$model.runningJob` for its
/// `fullScreenCover(item:)`, so this needs module-wide visibility.
struct RunningJob: Identifiable {
    let id = UUID()
    let model: AIModel
    let options: BenchmarkOptions
}

/// Not `private`: `DiagnosticsView` binds `$model.runError` for its `.alert`.
struct BenchmarkRunError: Identifiable {
    let id = UUID()
    let message: String
}

// MARK: - Section (dumb row content)

struct BenchmarkSection: View {

    @Bindable var model: BenchmarkSectionModel

    var body: some View {
        Section("Model Benchmark") {
            if model.eligibleModels.isEmpty {
                Text("Install a text model to run a benchmark.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Picker("Model", selection: $model.selectedModelID) {
                    ForEach(model.eligibleModels) { entry in
                        Text(entry.name).tag(Optional(entry.id))
                    }
                }

                Picker("Mode", selection: $model.mode) {
                    ForEach(BenchmarkRunMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .pickerStyle(.segmented)

                // Only shown when the suite runs: a performance-only run
                // never opens a suite, so offering a choice would imply it did.
                if model.mode.runsSuite {
                    Picker("Suite", selection: $model.selectedSuiteID) {
                        Text("Built-in").tag(BenchmarkSuiteDocument.builtInID)
                        ForEach(model.customSuites) { suite in
                            Text(suite.name).tag(suite.id)
                        }
                    }
                }
                Text(model.mode.caption(taskCount: model.selectedSuiteTaskCount))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Button {
                    model.requestRun()
                } label: {
                    Label("Run Benchmark", systemImage: "gauge.with.dots.needle.67percent")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(model.selectedModel == nil)

                if let selectedModel = model.selectedModel,
                   let latest = model.latestReport(for: selectedModel) {
                    NavigationLink(value: latest) {
                        LatestReportRow(report: latest)
                    }
                }
            }

            NavigationLink {
                BenchmarkSuiteListView()
            } label: {
                Label("Task suites", systemImage: "list.bullet.rectangle")
            }

            NavigationLink {
                BenchmarkHistoryView()
            } label: {
                Label("Past runs", systemImage: "clock.arrow.circlepath")
            }
        }
    }
}

// MARK: - Latest report row

private struct LatestReportRow: View {
    let report: BenchmarkReport

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text("Latest run")
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(spacing: 10) {
                if let score = report.scoreLabel {
                    Text(score)
                        .font(.subheadline).bold()
                } else {
                    Text("Performance only")
                        .font(.subheadline).bold()
                }
                Text(BenchmarkFormat.tokPerSec(report.performance.generationTokensPerSecond))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(report.date.formatted(date: .abbreviated, time: .shortened))
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 2)
    }
}
