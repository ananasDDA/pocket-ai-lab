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
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
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
