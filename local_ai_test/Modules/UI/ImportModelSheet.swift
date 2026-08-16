//
//  ImportModelSheet.swift
//  local_ai_test
//
//  "Paste a Hugging Face link, get a model card" pipeline. Three steps in one
//  sheet: enter a link → inspect what the repo actually contains → download.
//

import SwiftUI
import UIKit

struct ImportModelSheet: View {

    /// Device the estimates are judged against. Passed in so the sheet does
    /// not re-scan and possibly disagree with the card list behind it.
    let device: DeviceInfo

    @Environment(\.dismiss) private var dismiss

    @State private var link = ""
    @State private var step: Step = .input
    @State private var errorMessage: String?
    @State private var selectedFormat: ImportFormat?
    @State private var selectedQuant: GGUFQuantOption?
    @State private var spaces: [String] = []
    @State private var webLink: WebLink?
    @State private var isStartingDownload = false

    private enum Step {
        case input
        case checking
        case analyzed(ImportAnalysis)
    }

    var body: some View {
        NavigationStack {
            Group {
                switch step {
                case .input:               inputStep
                case .checking:            checkingStep
                case .analyzed(let result): analyzedStep(result)
                }
            }
            .navigationTitle("Import model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .sheet(item: $webLink) { SafariView(url: $0.url) }
        }
    }

    // MARK: - Step 1: link

    private var inputStep: some View {
        Form {
            Section {
                TextField("huggingface.co/owner/repo", text: $link, axis: .vertical)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
                    .lineLimit(1...3)

                Button {
                    if let pasted = UIPasteboard.general.string {
                        link = pasted.trimmingCharacters(in: .whitespacesAndNewlines)
                        errorMessage = nil
                    }
                } label: {
                    Label("Paste from clipboard", systemImage: "doc.on.clipboard")
                }
            } header: {
                Text("Repo link")
            } footer: {
                Text("A full link or a bare owner/repo. GGUF, MLX safetensors, "
                     + "Core ML packages and whisper.cpp weights are supported.")
            }

            if let errorMessage {
                Section {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }

            Section {
                Button {
                    Task { await check() }
                } label: {
                    Label("Check repo", systemImage: "magnifyingglass")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(link.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
    }

    private var checkingStep: some View {
        VStack(spacing: 14) {
            ProgressView().scaleEffect(1.3)
            Text("Reading the repo…")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Step 2: analysis

    @ViewBuilder
    private func analyzedStep(_ result: ImportAnalysis) -> some View {
        let format = selectedFormat ?? result.detected
        let estimate = estimate(for: result, format: format)

        Form {
            Section {
                // Counters below come from the analysis we already fetched, so
                // the shared header only contributes the avatar and the name.
                HFModelHeader(repo: result.repo, displayName: result.repo, showsMetrics: false)
                HStack(spacing: 18) {
                    Label(HFMetadataStore.compactCount(result.downloads),
                          systemImage: "arrow.down.to.line")
                    Label(HFMetadataStore.compactCount(result.likes), systemImage: "heart")
                    if result.gated.requiresLicenseAcceptance {
                        Label("gated", systemImage: "lock")
                            .foregroundStyle(.orange)
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }

            Section("Runtime") {
                if result.formats.count > 1 {
                    Picker("Backend", selection: Binding(
                        get: { format },
                        set: { newValue in
                            selectedFormat = newValue
                            // Quant choices only exist for GGUF layouts.
                            selectedQuant = isGGUF(newValue) ? result.defaultQuant : nil
                        }
                    )) {
                        ForEach(result.formats) { option in
                            Text(option.displayName).tag(option)
                        }
                    }
                } else {
                    LabeledContent("Backend", value: format.displayName)
                }

                if isGGUF(format), !result.ggufQuants.isEmpty {
                    Picker("Quantization", selection: Binding(
                        get: { selectedQuant ?? result.defaultQuant ?? result.ggufQuants[0] },
                        set: { selectedQuant = $0 }
                    )) {
                        ForEach(result.ggufQuants) { quant in
                            Text("\(quant.label) · \(formatGB(Double(quant.size) / ImportAnalyzer.bytesPerGB))")
                                .tag(quant)
                        }
                    }
                }
            }

            Section("Fit on this \(device.marketingName)") {
                verdictRow(
                    title: "RAM",
                    detail: "\(formatGB(estimate.ramGB)) needed · "
                        + "\(formatGB(max(estimate.usableRAMGB, 0))) usable",
                    color: verdictColor(estimate.ramVerdict),
                    symbol: verdictSymbol(estimate.ramVerdict)
                )
                verdictRow(
                    title: "Disk",
                    detail: "\(formatGB(estimate.diskGB)) download · "
                        + "\(formatGB(estimate.freeDiskGB)) free",
                    color: estimate.diskFits ? .green : .red,
                    symbol: estimate.diskFits ? "checkmark.circle.fill" : "xmark.circle.fill"
                )
                if estimate.ramVerdict == .tight {
                    Text("Close to the limit — expect slow first loads and possible "
                         + "termination if other apps are running.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }

            Section {
                Button {
                    Task { await startDownload(result, format: format, estimate: estimate) }
                } label: {
                    HStack {
                        if isStartingDownload {
                            ProgressView().controlSize(.small)
                        }
                        Label("Download", systemImage: "arrow.down.circle.fill")
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isStartingDownload || estimate.ramVerdict == .wontFit || !estimate.diskFits)

                if estimate.ramVerdict == .wontFit {
                    Text("This model needs more RAM than the device can give it.")
                        .font(.caption)
                        .foregroundStyle(.red)
                } else if !estimate.diskFits {
                    Text("Not enough free storage for the download.")
                        .font(.caption)
                        .foregroundStyle(.red)
                }
            }

            Section {
                Button {
                    webLink = HuggingFaceAPI
                        .onlineDemoURL(repo: result.repo, spaces: spaces)
                        .map(WebLink.init)
                } label: {
                    Label(spaces.isEmpty ? "Open on Hugging Face" : "Try online",
                          systemImage: "safari")
                }
            }

            Section {
                Button("Check another link") {
                    step = .input
                    selectedFormat = nil
                    selectedQuant = nil
                    spaces = []
                }
            }
        }
    }

    private func verdictRow(
        title: String, detail: String, color: Color, symbol: String
    ) -> some View {
        HStack(spacing: 10) {
            Image(systemName: symbol)
                .foregroundStyle(color)
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.subheadline)
                Text(detail).font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Actions

    private func check() async {
        errorMessage = nil
        guard let repo = HuggingFaceAPI.parseRepoID(from: link) else {
            errorMessage = HuggingFaceError.invalidLink(link).localizedDescription
            return
        }

        step = .checking
        do {
            let info = try await HuggingFaceAPI.fetchRepoInfo(repo: repo)
            guard !info.isPrivate else { throw HuggingFaceError.restricted(repo) }
            let analysis = try ImportAnalyzer.analyze(info)
            selectedFormat = analysis.detected
            selectedQuant = isGGUF(analysis.detected) ? analysis.defaultQuant : nil
            step = .analyzed(analysis)
            spaces = await HuggingFaceAPI.fetchSpaces(repo: repo)
        } catch {
            errorMessage = (error as? LocalizedError)?.errorDescription
                ?? error.localizedDescription
            step = .input
        }
    }

    private func startDownload(
        _ result: ImportAnalysis,
        format: ImportFormat,
        estimate: ResourceEstimate
    ) async {
        isStartingDownload = true
        defer { isStartingDownload = false }

        let quant = isGGUF(format) ? (selectedQuant ?? result.defaultQuant) : nil
        let contextLength = await HuggingFaceAPI.fetchContextLength(
            repo: result.repo, revision: result.revision
        )

        let model = ImportedModelBuilder.makeModel(
            analysis: result,
            format: format,
            quant: quant,
            estimate: estimate,
            contextLength: contextLength
        )

        InstalledModelsStore.shared.register(model)
        ModelDownloader.shared.download(model)
        dismiss()
    }

    // MARK: - Helpers

    private func estimate(for result: ImportAnalysis, format: ImportFormat) -> ResourceEstimate {
        let quant = isGGUF(format) ? (selectedQuant ?? result.defaultQuant) : nil
        let sizing = result.sizing(layout: format.layout, quant: quant)

        return ImportAnalyzer.estimate(
            layout: format.layout,
            weightBytes: sizing.weights,
            downloadBytes: sizing.download,
            device: device,
            freeDiskBytes: device.freeDiskSpace
        )
    }

    private func isGGUF(_ format: ImportFormat) -> Bool {
        format.layout == .singleGGUF || format.layout == .ggufWithMmproj
    }

    private func verdictColor(_ verdict: ResourceVerdict) -> Color {
        switch verdict {
        case .fits:    return .green
        case .tight:   return .orange
        case .wontFit: return .red
        }
    }

    private func verdictSymbol(_ verdict: ResourceVerdict) -> String {
        switch verdict {
        case .fits:    return "checkmark.circle.fill"
        case .tight:   return "exclamationmark.triangle.fill"
        case .wontFit: return "xmark.circle.fill"
        }
    }

    private func formatGB(_ value: Double) -> String {
        value < 1 ? String(format: "%.2f GB", value) : String(format: "%.1f GB", value)
    }
}
