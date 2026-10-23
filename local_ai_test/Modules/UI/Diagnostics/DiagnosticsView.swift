//
//  DiagnosticsView.swift
//  local_ai_test
//
//  In-app smoke test runner for each backend. Runs a short generation on
//  the currently loaded engine and on any installed companion engines
//  (whisper/kokoro), records tok/s, peak RAM, load time, and whether the
//  run succeeded. Useful when diagnosing issues on real user devices where
//  it is impractical to attach a debugger.
//

import SwiftUI

struct DiagnosticsView: View {

    @State private var runs: [DiagnosticRun] = []
    @State private var running = false
    @State private var exportJSON = ""
    @State private var showingExport = false
    @State private var showErrorDetail = false
    /// Owns the "Model Benchmark" section's selection, run lifecycle, and
    /// every sheet/cover/alert it can present. Kept here (rather than on
    /// `BenchmarkSection` itself) so all of that presentation attaches to
    /// this `List`, not to a `Section` row that scrolling can recycle.
    @State private var benchmarkModel = BenchmarkSectionModel()
    /// Owns the "Build test" section's options and its confirmation alert.
    /// Same reasoning as `benchmarkModel`: presentation belongs on the
    /// `List`, not on a recyclable `Section` row. It does NOT own the live
    /// screen - `ContentView` presents that, bound to
    /// `BuildTestRunner.shared.isRunning`, so an Xcode-started sweep is
    /// watchable too.
    @State private var buildTestModel = BuildTestSectionModel()
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        @Bindable var benchmarkModel = benchmarkModel
        @Bindable var buildTestModel = buildTestModel
        NavigationStack {
            List {
                BenchmarkSection(model: benchmarkModel)

                BuildTestSection(model: buildTestModel)

                Section("Available memory") {
                    Text("\(InferenceManager.availableMemoryMB()) MB")
                        .font(.system(.body, design: .monospaced))
                }

                Section("Loaded engine") {
                    let manager = InferenceManager.shared
                    if let id = manager.currentModelId {
                        Text(id)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                    } else {
                        Text("No engine loaded")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    }
                    if let failure = manager.lastError {
                        Button {
                            showErrorDetail = true
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "exclamationmark.triangle.fill")
                                    .foregroundStyle(.orange)
                                VStack(alignment: .leading) {
                                    Text(failure.summary).font(.caption).bold()
                                    Text(failure.detail)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }

                Section("Smoke tests") {
                    Button("Run text-only smoke (current model)", systemImage: "text.bubble") {
                        Task { await runTextSmoke() }
                    }
                    .disabled(running)

                    Button("Run all installed models", systemImage: "play.rectangle.on.rectangle") {
                        Task { await runAllInstalled() }
                    }
                    .disabled(running)
                }

                Section("Results") {
                    if runs.isEmpty {
                        Text("No runs yet.")
                            .foregroundStyle(.secondary)
                            .font(.caption)
                    } else {
                        ForEach(runs) { run in
                            DiagnosticRunView(run: run)
                        }
                    }
                }

                if !runs.isEmpty {
                    Section {
                        Button("Export JSON", systemImage: "square.and.arrow.up") { showExport() }
                        Button("Clear results", role: .destructive) { runs.removeAll() }
                    }
                }
            }
            .navigationTitle("Testing")
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: BenchmarkReport.self) { report in
                BenchmarkReportView(report: report)
            }
            // MARK: Build test section presentation
            //
            // Starting the sweep also closes Diagnostics: the live screen is
            // a `fullScreenCover` on the app root, and the root cannot
            // present it while this sheet is still up.
            .alert(
                "Start the build test?",
                isPresented: $buildTestModel.showStartConfirm
            ) {
                Button("Cancel", role: .cancel) {}
                Button("Start") {
                    buildTestModel.start()
                    dismiss()
                }
            } message: {
                Text("Every catalog model this device can run will be downloaded, "
                    + "benchmarked, and then deleted. Models you installed yourself "
                    + "are kept.\n\nSet Auto-Lock to Never in Settings for the "
                    + "duration, keep the phone plugged in, and stay on Wi-Fi.")
            }
            // MARK: Model Benchmark section lifecycle + presentation
            //
            // Moved up from `BenchmarkSection` (a `Section` inside this
            // `List`): a `.task`/`.onAppear` on a row can refire on scroll,
            // and a `.fullScreenCover`/`.sheet`/`.alert` anchored to a
            // recyclable row is fragile. All of it lives here instead, next
            // to the `navigationDestination` above.
            .task {
                // Best-effort refresh of the task suite. Never blocks a run
                // -- a run started before this lands just uses whatever was
                // cached or bundled already.
                await RemoteBenchmarkSuite.shared.refresh()
            }
            .onAppear {
                benchmarkModel.ensureSelection()
                // A suite deleted on the suites screen must not stay selected
                // here; the run would fail with `customSuiteMissing`.
                benchmarkModel.ensureSuiteSelection()
            }
            .onChange(of: InstalledModelsStore.shared.installedModels) { _, _ in
                benchmarkModel.ensureSelection()
            }
            .fullScreenCover(item: $benchmarkModel.runningJob) { job in
                BenchmarkRunView(model: job.model, options: job.options) { result in
                    benchmarkModel.handleRunFinished(result)
                }
            }
            .sheet(item: $benchmarkModel.justFinishedReport) { report in
                NavigationStack {
                    BenchmarkReportView(report: report)
                        .toolbar {
                            ToolbarItem(placement: .cancellationAction) {
                                Button("Done") { benchmarkModel.justFinishedReport = nil }
                            }
                        }
                }
            }
            .alert(
                "Unload the current model?",
                isPresented: Binding(
                    get: { benchmarkModel.pendingRunModel != nil },
                    set: { if !$0 { benchmarkModel.pendingRunModel = nil } }
                ),
                presenting: benchmarkModel.pendingRunModel
            ) { model in
                Button("Cancel", role: .cancel) { benchmarkModel.pendingRunModel = nil }
                Button("Run Benchmark") {
                    benchmarkModel.pendingRunModel = nil
                    benchmarkModel.startRun(model: model)
                }
            } message: { model in
                Text("A model is currently loaded in chat. Benchmarking \(model.name) will unload it first.")
            }
            .alert(
                "Benchmark Failed",
                isPresented: Binding(
                    get: { benchmarkModel.runError != nil },
                    set: { if !$0 { benchmarkModel.runError = nil } }
                ),
                presenting: benchmarkModel.runError
            ) { _ in
                Button("OK", role: .cancel) { benchmarkModel.runError = nil }
            } message: { error in
                Text(error.message)
            }
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .sheet(isPresented: $showingExport) {
                NavigationStack {
                    ScrollView {
                        Text(exportJSON)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .padding()
                    }
                    .navigationTitle("Export")
                    .navigationBarTitleDisplayMode(.inline)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { showingExport = false }
                        }
                        ToolbarItem(placement: .primaryAction) {
                            Button {
                                UIPasteboard.general.string = exportJSON
                            } label: {
                                Image(systemName: "doc.on.doc")
                            }
                        }
                    }
                }
            }
            .sheet(isPresented: $showErrorDetail) {
                if let failure = InferenceManager.shared.lastError {
                    ErrorDetailSheet(failure: failure)
                }
            }
        }
    }

    // MARK: - Actions

    private func runTextSmoke() async {
        running = true
        defer { running = false }
        let runId = UUID()
        let prompt = "What is 2+2? Answer with one sentence."
        let start = Date()

        var tokensGenerated = 0
        var firstTokenMs: Double = -1
        var errorMessage: String?

        let stream = InferenceManager.shared.generate(
            turns: [.user(prompt)],
            parameters: GenerationParameters(
                temperature: 0.0,
                topP: 1.0,
                topK: 1,
                repetitionPenalty: 1.0,
                maxTokens: 48,
                seed: nil
            )
        )

        do {
            for try await output in stream {
                switch output {
                case .textDelta: tokensGenerated += 1
                case .diagnostic(.firstTokenLatency(let ms)): firstTokenMs = ms
                default: break
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }

        let elapsed = Date().timeIntervalSince(start)
        // Report against the *actually loaded* engine, not the currently
        // selected catalog entry (the two can diverge when a prepare()
        // call fails and leaves the previous engine resident).
        let loadedId = InferenceManager.shared.currentModelId
        let catalogEntry = loadedId.flatMap { id in
            InstalledModelsStore.shared.installedModels.first(where: { $0.id == id })
        }
        let modelName = catalogEntry?.name
            ?? loadedId
            ?? "(no engine loaded)"
        let backend = catalogEntry?.backend.displayName ?? "(none)"

        runs.insert(DiagnosticRun(
            id: runId,
            modelName: modelName,
            backend: backend,
            durationSeconds: elapsed,
            tokensGenerated: tokensGenerated,
            firstTokenLatencyMs: firstTokenMs,
            peakMemoryMB: InferenceManager.availableMemoryMB(),
            success: errorMessage == nil,
            errorMessage: errorMessage
        ), at: 0)
    }

    private func runAllInstalled() async {
        running = true
        defer { running = false }
        for model in InstalledModelsStore.shared.installedModels {
            await InferenceManager.shared.prepare(model: model)
            guard case .ready = InferenceManager.shared.state else { continue }
            await runTextSmoke()
        }
    }

    private func showExport() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(runs),
           let str = String(data: data, encoding: .utf8) {
            exportJSON = str
            showingExport = true
        }
    }
}

struct DiagnosticRun: Identifiable, Codable {
    let id: UUID
    let modelName: String
    let backend: String
    let durationSeconds: Double
    let tokensGenerated: Int
    let firstTokenLatencyMs: Double
    let peakMemoryMB: Int64
    let success: Bool
    let errorMessage: String?

    var tokensPerSecond: Double {
        durationSeconds > 0 ? Double(tokensGenerated) / durationSeconds : 0
    }
}

private struct DiagnosticRunView: View {
    let run: DiagnosticRun

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Image(systemName: run.success ? "checkmark.circle.fill" : "xmark.circle.fill")
                    .foregroundStyle(run.success ? .green : .red)
                Text(run.modelName).font(.subheadline).bold()
                Spacer()
                Text(run.backend).font(.caption2).foregroundStyle(.secondary)
            }
            if run.success {
                HStack(spacing: 12) {
                    stat("tok/s", String(format: "%.1f", run.tokensPerSecond))
                    stat("TTFT", String(format: "%.0f ms", run.firstTokenLatencyMs))
                    stat("total", String(format: "%.1f s", run.durationSeconds))
                    stat("avail RAM", "\(run.peakMemoryMB) MB")
                }
                .font(.caption)
            } else if let msg = run.errorMessage {
                Text(msg).font(.caption).foregroundStyle(.red).lineLimit(3)
            }
        }
        .padding(.vertical, 2)
    }

    private func stat(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Text(value).bold()
            Text(label).foregroundStyle(.secondary).font(.caption2)
        }
    }
}
