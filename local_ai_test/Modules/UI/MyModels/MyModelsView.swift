//
//  MyModelsView.swift
//  local_ai_test
//
//  "My Models": every model on the device in one list, catalog or not, with
//  its size and a Remove action. It is reached from a single row on the
//  Models tab so the catalog below keeps its layout.
//

import SwiftUI

// MARK: - Shared state

/// What the My Models row and screen both read: the entries and the measured
/// folder sizes. Measuring walks every model folder, so it runs off the main
/// actor and only when the set of installed models changes.
@Observable
final class MyModelsStore {

    static let shared = MyModelsStore()
    private init() {}

    private(set) var measuredSizes: [String: Int64] = [:]

    private var store: InstalledModelsStore { InstalledModelsStore.shared }

    var entries: [MyModelsEntry] {
        MyModels.entries(
            known: store.knownModels,
            installedIDs: store.installedIDs,
            catalogIDs: Set(ModelCatalog.all.map(\.id)),
            loadedModelID: InferenceManager.shared.currentModelId,
            measuredSizes: measuredSizes
        )
    }

    func refreshSizes() async {
        let downloader = ModelDownloader.shared
        let targets = store.knownModels
            .filter(MyModels.canDelete)
            .map { ($0.id, downloader.modelDirectory(for: $0)) }
        let sizes = await Task.detached(priority: .utility) {
            Dictionary(
                targets.map { ($0.0, ModelDownloader.directorySize(of: $0.1)) },
                uniquingKeysWith: { first, _ in first }
            )
        }.value
        measuredSizes = sizes
    }

    /// Same delete path the catalog cards use. A loaded model is unloaded
    /// first so the engine is not left pointing at files that are gone.
    func remove(_ entry: MyModelsEntry) {
        let model = entry.model
        Task {
            if InferenceManager.shared.currentModelId == model.id {
                await InferenceManager.shared.unloadCurrent()
            }
            ModelDownloader.shared.delete(model)
            if MyModels.unregistersOnRemoval(model) {
                store.unregister(model)
            }
        }
    }
}

// MARK: - Row on the Models tab

/// One row in the style of the Device block. Hidden while there is nothing
/// to manage, so a first launch looks exactly as before.
struct MyModelsSummaryRow: View {

    private var myModels: MyModelsStore { MyModelsStore.shared }
    private var store: InstalledModelsStore { InstalledModelsStore.shared }

    var body: some View {
        let entries = myModels.entries
        if !entries.isEmpty {
            NavigationLink {
                MyModelsView()
            } label: {
                HStack {
                    Label("My Models", systemImage: "square.stack.3d.up")
                        .foregroundStyle(.primary)
                    Spacer()
                    Text(summary(entries))
                        .foregroundStyle(.secondary)
                    Image(systemName: "chevron.right")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.tertiary)
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 11)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
            }
            .buttonStyle(.plain)
            .task(id: store.installedIDs) { await myModels.refreshSizes() }
        }
    }

    private func summary(_ entries: [MyModelsEntry]) -> String {
        let count = entries.count == 1 ? "1 model" : "\(entries.count) models"
        return "\(count) · \(MyModels.formatSize(MyModels.totalBytes(entries)))"
    }
}

// MARK: - Screen

struct MyModelsView: View {

    @State private var pendingRemoval: MyModelsEntry?

    private var myModels: MyModelsStore { MyModelsStore.shared }
    private var store: InstalledModelsStore { InstalledModelsStore.shared }
    private var downloader: ModelDownloader { ModelDownloader.shared }

    var body: some View {
        let entries = myModels.entries
        List {
            Section {
                HStack {
                    Label("Used by models", systemImage: "internaldrive")
                    Spacer()
                    Text(MyModels.formatSize(MyModels.totalBytes(entries)))
                        .foregroundStyle(.secondary)
                }
            } footer: {
                Text("Removing a model deletes its files from this device.")
            }

            if entries.isEmpty {
                ContentUnavailableView(
                    "No models on this device",
                    systemImage: "square.stack.3d.up",
                    description: Text("Models you install or import show up here.")
                )
            } else {
                Section("On this device") {
                    ForEach(entries) { entry in
                        row(entry)
                            .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                Button("Remove") { pendingRemoval = entry }
                                    .tint(.red)
                            }
                    }
                }
            }
        }
        .navigationTitle("My Models")
        .navigationBarTitleDisplayMode(.inline)
        .task(id: store.installedIDs) { await myModels.refreshSizes() }
        .alert(
            "Remove Model?",
            isPresented: Binding(
                get: { pendingRemoval != nil },
                set: { if !$0 { pendingRemoval = nil } }
            ),
            presenting: pendingRemoval
        ) { entry in
            Button("Cancel", role: .cancel) {}
            Button("Remove", role: .destructive) { myModels.remove(entry) }
        } message: { entry in
            Text(entry.removalMessage)
        }
    }

    private func row(_ entry: MyModelsEntry) -> some View {
        let state = downloader.state(for: entry.model)
        return VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(entry.model.name)
                    .font(.headline)
                    .lineLimit(1)
                Spacer(minLength: 8)
                statusBadge(entry, state: state)
            }

            Text(detailLine(entry))
                .font(.caption)
                .foregroundStyle(.secondary)

            if entry.isOrphaned {
                Label("No longer in the catalog. It still works in chat and benchmarks.",
                      systemImage: "info.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            HStack {
                if entry.status == .notDownloaded, !state.isActive,
                   !entry.model.huggingFaceRepo.isEmpty {
                    Button {
                        downloader.download(entry.model)
                    } label: {
                        Label(state == .idle ? "Download" : "Retry",
                              systemImage: "arrow.down.circle")
                    }
                    .font(.caption)
                }
                Spacer()
                Button("Remove") { pendingRemoval = entry }
                    .font(.caption)
                    .foregroundStyle(.red)
            }
            // Without it the whole row becomes one tap target and both
            // buttons fire together.
            .buttonStyle(.borderless)
        }
        .padding(.vertical, 2)
    }

    private func detailLine(_ entry: MyModelsEntry) -> String {
        var parts = [entry.model.backend.displayName]
        if entry.sizeBytes > 0 {
            parts.append(MyModels.formatSize(entry.sizeBytes))
        }
        if entry.model.source == .imported {
            parts.append("Imported")
        }
        return parts.joined(separator: " · ")
    }

    @ViewBuilder
    private func statusBadge(_ entry: MyModelsEntry, state: ModelDownloadState) -> some View {
        switch entry.status {
        case .inUse:
            badge("In use", color: .green)
        case .available:
            badge("Available", color: .secondary)
        case .notDownloaded:
            switch state {
            case .fetchingManifest:
                badge("Preparing", color: .blue)
            case .downloading(let progress, _, _):
                badge("Downloading \(Int(progress * 100))%", color: .blue)
            case .compiling:
                badge("Compiling", color: .blue)
            case .error:
                badge("Download failed", color: .red)
            case .idle, .installed:
                badge("Not downloaded", color: .secondary)
            }
        }
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text)
            .font(.caption2).bold()
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule())
    }
}
