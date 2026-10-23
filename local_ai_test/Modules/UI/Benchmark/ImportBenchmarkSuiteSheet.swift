//
//  ImportBenchmarkSuiteSheet.swift
//  local_ai_test
//
//  "Bring your own benchmark suite", built to look and fail like
//  `ImportModelSheet`: two ways in (a pasted link, a file from Files), one
//  inline error line, no half-finished state left behind.
//
//  The one thing it adds is a REVIEW step. A model import either copies the
//  file or does not; a suite import can succeed and still have quietly
//  dropped a third of the file's tests, and a score for two thirds of an exam
//  is not the score the file describes. So nothing is saved until the user
//  has seen how many tasks were recognised, how many were skipped, and why
//  each one was.
//

import SwiftUI
import UIKit
import UniformTypeIdentifiers

struct ImportBenchmarkSuiteSheet: View {

    /// Called with the stored suite once it is on disk, so the caller can
    /// select it straight away.
    var onImported: (StoredBenchmarkSuite) -> Void = { _ in }

    @Environment(\.dismiss) private var dismiss

    @State private var link = ""
    @State private var step: Step = .input
    @State private var errorMessage: String?
    @State private var showFileImporter = false

    private enum Step {
        case input
        case loading
        case review(SuiteImportResult)
    }

    /// YAML has no registered system UTI on iOS, so the extension-derived
    /// types are used where the system can build them and plain data is kept
    /// alongside — without the fallback a `.yaml` would not be selectable.
    private static let suiteContentTypes: [UTType] = {
        var types: [UTType] = [.json]
        for suffix in ["yaml", "yml"] {
            if let type = UTType(filenameExtension: suffix) { types.append(type) }
        }
        types.append(contentsOf: [.plainText, .text, .data])
        return types
    }()

    var body: some View {
        NavigationStack {
            Group {
                switch step {
                case .input:               inputStep
                case .loading:             loadingStep
                case .review(let result):  reviewStep(result)
                }
            }
            .navigationTitle("Import suite")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: Self.suiteContentTypes,
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    importFile(url)
                case .failure(let error):
                    errorMessage = error.localizedDescription
                }
            }
        }
    }

    // MARK: - Step 1: pick a source

    private var inputStep: some View {
        Form {
            Section {
                TextField("https://example.com/promptfooconfig.yaml", text: $link, axis: .vertical)
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
                Text("Link")
            } footer: {
                Text("A direct link to the file itself, not to a repo page.")
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
                    Task { await importLink() }
                } label: {
                    Label("Fetch suite", systemImage: "arrow.down.circle")
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
                Text("Two formats are read: this app's own benchmarks.json schema, "
                     + "and a promptfoo config (promptfooconfig.yaml).")
            }

            Section("What can be checked") {
                Text("String assertions map across: equals, contains, contains-all, "
                     + "contains-any, not-contains, regex, starts-with and is-json.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("Assertions that run code (javascript, python) or ask another "
                     + "model to grade (llm-rubric, model-graded, g-eval) are refused. "
                     + "A score here has to mean the same thing on every run and every "
                     + "device, and a judge model's verdict does not.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("At most \(SuiteImportLimits.maxTasks) tasks per suite, "
                     + "\(SuiteImportLimits.maxTokensPerTask) generated tokens per task.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var loadingStep: some View {
        VStack(spacing: 14) {
            ProgressView().scaleEffect(1.3)
            Text("Reading the suite…")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Step 2: review before anything is saved

    @ViewBuilder
    private func reviewStep(_ result: SuiteImportResult) -> some View {
        Form {
            Section("Suite") {
                LabeledContent("Name", value: result.suite.name)
                LabeledContent("Format", value: result.format.displayName)
                LabeledContent("Tasks", value: "\(result.recognizedTaskCount)")
                LabeledContent("Categories", value: result.suite.document.categories
                    .joined(separator: ", "))
            }

            if !result.skipped.isEmpty {
                Section {
                    ForEach(result.skipped) { entry in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(entry.name)
                                .font(.footnote).bold()
                            Text(entry.reason)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .padding(.vertical, 1)
                    }
                } header: {
                    Text("\(result.skipped.count) skipped")
                } footer: {
                    Text("These tests are not part of the suite and are not counted in "
                         + "its score.")
                }
            }

            if !result.notes.isEmpty {
                Section("Worth knowing") {
                    ForEach(result.notes, id: \.self) { note in
                        Text(note)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Section {
                Button {
                    save(result)
                } label: {
                    Label("Add suite", systemImage: "checkmark.circle.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)

                Button("Choose a different file") {
                    step = .input
                }
            } footer: {
                Text("Scores from this suite are never compared against the built-in "
                     + "one - they are different exams.")
            }

            if let errorMessage {
                Section {
                    Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
                        .font(.footnote)
                        .foregroundStyle(.red)
                }
            }
        }
    }

    // MARK: - Actions

    private func importLink() async {
        errorMessage = nil
        guard let url = BenchmarkSuiteImporter.parseURL(from: link) else {
            errorMessage = "That does not look like a link to a file."
            return
        }
        step = .loading
        do {
            let result = try await BenchmarkSuiteImporter.makeSuite(from: url)
            step = .review(result)
        } catch {
            errorMessage = describe(error)
            step = .input
        }
    }

    private func importFile(_ url: URL) {
        errorMessage = nil
        step = .loading

        // Files handed over by the picker live outside the app sandbox, so
        // the scope has to be open for the read.
        let accessing = url.startAccessingSecurityScopedResource()
        defer { if accessing { url.stopAccessingSecurityScopedResource() } }

        do {
            let data = try Data(contentsOf: url)
            let result = try BenchmarkSuiteImporter.makeSuite(
                data: data,
                source: url.lastPathComponent
            )
            step = .review(result)
        } catch {
            errorMessage = describe(error)
            step = .input
        }
    }

    private func save(_ result: SuiteImportResult) {
        do {
            try CustomBenchmarkSuiteStore.shared.add(result.suite)
            onImported(result.suite)
            dismiss()
        } catch {
            errorMessage = describe(error)
        }
    }

    private func describe(_ error: any Error) -> String {
        (error as? LocalizedError)?.errorDescription ?? error.localizedDescription
    }
}
