//
//  ImportModelSheet.swift
//  local_ai_test
//
//  Two ways in, one sheet. "Paste a Hugging Face link, get a model card":
//  enter a link → inspect what the repo actually contains → download. Or
//  "Import from Files": pick a .gguf the user already has → copy it in.
//

import SwiftUI
import UIKit
import UniformTypeIdentifiers

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
    /// Result of the pre-download architecture check. nil while unchecked or
    /// for repos with no MLX format at all.
    @State private var mlxVerdict: MLXArchitectureVerdict?

    // Local ".gguf from Files" import.
    @State private var showFileImporter = false
    @State private var pendingReplace: PendingLocalImport?
    @State private var showReplacePrompt = false
    /// Descriptor of the file currently being copied in, used to read the
    /// copy's progress back out of the downloader.
    @State private var importingModel: AIModel?

    private enum Step {
        case input
        case checking
        case analyzed(ImportAnalysis)
        /// Copying a picked file into the app container; carries its name.
        case copying(String)
    }

    /// A picked file that would overwrite an already-imported model, held
    /// until the user answers the confirmation.
    private struct PendingLocalImport {
        let candidate: LocalGGUFCandidate
        let url: URL
    }

    /// `.gguf` has no registered UTI, so the extension-derived type is used
    /// when the system can build one and plain data is kept alongside it —
    /// without the fallback the file would not be selectable at all.
    private static let ggufContentTypes: [UTType] = {
        if let gguf = UTType(filenameExtension: "gguf") { return [gguf, .data] }
        return [.data]
    }()

    var body: some View {
        NavigationStack {
            Group {
                switch step {
                case .input:                inputStep
                case .checking:             checkingStep
                case .analyzed(let result): analyzedStep(result)
                case .copying(let filename): copyingStep(filename)
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
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: Self.ggufContentTypes,
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    Task { await prepareLocalImport(url) }
                case .failure(let error):
                    errorMessage = error.localizedDescription
                }
            }
            .alert(
                "Replace imported model?",
                isPresented: $showReplacePrompt,
                presenting: pendingReplace
            ) { pending in
                Button("Cancel", role: .cancel) { pendingReplace = nil }
                Button("Replace", role: .destructive) {
                    pendingReplace = nil
                    Task { await runLocalImport(candidate: pending.candidate, sourceURL: pending.url) }
                }
            } message: { pending in
                Text("\(pending.candidate.filename) is already imported. Replacing it "
                     + "overwrites the copy that is on the device now.")
            }
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

            Section {
                Button {
                    errorMessage = nil
                    showFileImporter = true
                } label: {
                    Label("Import from Files", systemImage: "folder.badge.plus")
                }
            } header: {
                Text("Local file")
            } footer: {
                Text("Pick a .gguf file already on this device or in iCloud Drive. "
                     + "It is copied into the app and runs through llama.cpp. "
                     + "Vision projectors (mmproj) cannot be imported on their own.")
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

    // MARK: - Local file copy

    private func copyingStep(_ filename: String) -> some View {
        VStack(spacing: 14) {
            ProgressView(value: copyProgress)
                .progressViewStyle(.linear)
                .frame(maxWidth: 280)
            Text("Copying \(filename)")
                .font(.subheadline)
                .lineLimit(2)
                .multilineTextAlignment(.center)
            Text("\(Int(copyProgress * 100))% copied")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("Large files take a while. You can close this sheet - the copy "
                 + "keeps going and the model card shows the progress.")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(24)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Read back from the downloader so the sheet and the model card behind
    /// it never disagree about how far the copy has got.
    private var copyProgress: Double {
        guard let model = importingModel,
              case .downloading(let progress, _, _) = ModelDownloader.shared.state(for: model)
        else { return 0 }
        return progress
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

                // Checked before any bytes move: mlx-swift builds a model by
                // looking up config.json's model_type, and an architecture it
                // has no implementation for otherwise only fails after the
                // whole repo is already on disk.
                if format.layout == .huggingFaceMLX, let message = mlxVerdict?.message {
                    Label(message, systemImage: "xmark.octagon.fill")
                        .font(.caption)
                        .foregroundStyle(.red)
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
                .disabled(isStartingDownload || estimate.ramVerdict == .wontFit
                          || !estimate.diskFits || blocksDownload(format))

                if blocksDownload(format) {
                    Text("The MLX engine in this build cannot run this architecture. "
                         + "Nothing is downloaded.")
                        .font(.caption)
                        .foregroundStyle(.red)
                } else if estimate.ramVerdict == .wontFit {
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
                    mlxVerdict = nil
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
        mlxVerdict = nil
        guard let repo = HuggingFaceAPI.parseRepoID(from: link) else {
            errorMessage = HuggingFaceError.invalidLink(link).localizedDescription
            return
        }

        step = .checking
        do {
            let info = try await HuggingFaceAPI.fetchRepoInfo(repo: repo)
            guard !info.isPrivate else { throw HuggingFaceError.restricted(repo) }
            let analysis = try ImportAnalyzer.analyze(info)
            // Resolved while the sheet is still on the "checking" screen, so
            // the verdict is already in hand when the Download button first
            // appears and cannot be beaten to it by a fast tap.
            mlxVerdict = await ImportAnalyzer.mlxArchitectureVerdict(for: analysis)
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

    // MARK: - Local file import

    /// Works out what the picked file is and, unless it would overwrite an
    /// existing import, starts copying it straight away.
    private func prepareLocalImport(_ url: URL) async {
        errorMessage = nil

        // Files handed over by the picker live outside the app sandbox; the
        // scope has to be open for every read of them, inspection included.
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }

        let candidate: LocalGGUFCandidate
        do {
            candidate = try await Task.detached(priority: .userInitiated) {
                try LocalGGUFFile.inspect(fileURL: url)
            }.value
        } catch {
            errorMessage = describe(error)
            return
        }

        if InstalledModelsStore.shared.knownModel(id: candidate.modelID) != nil {
            pendingReplace = PendingLocalImport(candidate: candidate, url: url)
            showReplacePrompt = true
            return
        }

        await runLocalImport(candidate: candidate, sourceURL: url)
    }

    /// Copies the file in and registers it. The security scope is reopened
    /// here because the duplicate confirmation returns from
    /// `prepareLocalImport` before the user answers.
    private func runLocalImport(candidate: LocalGGUFCandidate, sourceURL: URL) async {
        errorMessage = nil
        let accessing = sourceURL.startAccessingSecurityScopedResource()
        defer { if accessing { sourceURL.stopAccessingSecurityScopedResource() } }

        // Judged by the same estimator the Hugging Face path uses, so an
        // imported GGUF and a downloaded one report the same numbers. Free
        // space is read fresh rather than taken from the device scan.
        let estimate = ImportAnalyzer.estimate(
            layout: .singleGGUF,
            weightBytes: candidate.fileSizeBytes,
            downloadBytes: candidate.fileSizeBytes,
            device: device,
            freeDiskBytes: LocalGGUFFile.freeDiskBytes()
        )
        guard estimate.diskFits else {
            errorMessage = LocalImportError
                .notEnoughSpace(neededGB: estimate.diskGB, freeGB: estimate.freeDiskGB)
                .localizedDescription
            return
        }
        // Mirrors the Hugging Face path's Download-button gate: a GGUF that
        // fits on disk but not in RAM would import fine and then fail every
        // load preflight, leaving the user with a model that can never run.
        guard estimate.ramVerdict != .wontFit else {
            errorMessage = "This model needs more RAM than this device can give it."
            return
        }

        let model = LocalGGUFModelBuilder.makeModel(candidate: candidate, estimate: estimate)
        importingModel = model
        step = .copying(candidate.filename)

        do {
            try await ModelDownloader.shared.installLocalFile(
                model,
                from: sourceURL,
                filename: candidate.filename,
                totalBytes: candidate.fileSizeBytes
            )
            dismiss()
        } catch is CancellationError {
            // Cancelled from the model card behind the sheet; nothing to say.
            importingModel = nil
            step = .input
        } catch {
            errorMessage = describe(error)
            importingModel = nil
            step = .input
        }
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

    /// Only MLX layouts are gated on the architecture check — llama.cpp and
    /// Core ML decide what they can run on their own terms.
    private func blocksDownload(_ format: ImportFormat) -> Bool {
        format.layout == .huggingFaceMLX && (mlxVerdict?.blocksDownload ?? false)
    }

    private func describe(_ error: any Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
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
