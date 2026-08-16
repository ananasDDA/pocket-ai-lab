//
//  InstalledModelsStore.swift
//  local_ai_test
//
//  Shared store for models the app knows about, backed by a JSON registry in
//  Documents/LocalAIModels/. The registry — not the static catalog — is the
//  source of truth for what survives a relaunch, because user-imported models
//  exist nowhere else.
//

import SwiftUI

@Observable
final class InstalledModelsStore {

    static let shared = InstalledModelsStore()

    /// Every model the app has a record of: curated entries that were
    /// installed at least once, plus every imported entry (even while it is
    /// still downloading — otherwise a relaunch mid-download would lose the
    /// only copy of its descriptor).
    private(set) var knownModels: [AIModel] = []

    /// Ids of `knownModels` whose files are on disk.
    private(set) var installedIDs: Set<String> = []

    var installedModels: [AIModel] {
        knownModels.filter { installedIDs.contains($0.id) }
    }

    var importedModels: [AIModel] {
        knownModels.filter { $0.source == .imported }
    }

    var selectedModel: AIModel? {
        didSet { save() }
    }

    private init() {
        load()
    }

    // MARK: - Mutation

    /// Records a model without claiming its files exist yet. Used by the
    /// import flow so the descriptor is persisted before the download starts.
    func register(_ model: AIModel) {
        upsert(model)
        save()
    }

    /// Drops a model from the registry entirely. Only meaningful for imported
    /// entries — curated ones come back from the catalog anyway.
    func unregister(_ model: AIModel) {
        knownModels.removeAll { $0.id == model.id }
        installedIDs.remove(model.id)
        if selectedModel?.id == model.id {
            selectedModel = installedModels.first
        }
        save()
    }

    func markInstalled(_ model: AIModel) {
        upsert(model)
        installedIDs.insert(model.id)
        if selectedModel == nil { selectedModel = model }
        save()
    }

    func isInstalled(_ model: AIModel) -> Bool {
        installedIDs.contains(model.id)
    }

    func markUninstalled(_ model: AIModel) {
        installedIDs.remove(model.id)
        if selectedModel?.id == model.id {
            selectedModel = installedModels.first
        }
        save()
    }

    /// Resolves a descriptor the app persisted earlier. The downloader needs
    /// this for imported models, which the catalog knows nothing about.
    func knownModel(id: String) -> AIModel? {
        knownModels.first { $0.id == id }
    }

    private func upsert(_ model: AIModel) {
        if let index = knownModels.firstIndex(where: { $0.id == model.id }) {
            knownModels[index] = model
        } else {
            knownModels.append(model)
        }
    }

    // MARK: - Persistence

    private struct Registry: Codable {
        var schemaVersion: Int
        var models: [AIModel]
        var installedIDs: [String]
        var selectedModelID: String?
    }

    private static let registrySchemaVersion = 1

    @ObservationIgnored
    private lazy var registryURL: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("LocalAIModels", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(".installed-registry.json")
    }()

    /// Guards against the registry being rewritten while `load()` is still
    /// populating it through the same setters the UI uses.
    @ObservationIgnored
    private var isLoading = false

    private func load() {
        guard let data = try? Data(contentsOf: registryURL),
              let registry = try? JSONDecoder().decode(Registry.self, from: data),
              registry.schemaVersion == Self.registrySchemaVersion
        else { return }

        isLoading = true
        knownModels = registry.models
        installedIDs = Set(registry.installedIDs)
        if let selectedID = registry.selectedModelID {
            selectedModel = knownModels.first { $0.id == selectedID }
        }
        isLoading = false
    }

    private func save() {
        guard !isLoading else { return }
        let registry = Registry(
            schemaVersion: Self.registrySchemaVersion,
            models: knownModels,
            installedIDs: Array(installedIDs),
            selectedModelID: selectedModel?.id
        )
        guard let data = try? JSONEncoder().encode(registry) else { return }
        // Atomic: a crash mid-write must not leave a truncated registry that
        // would silently drop every imported model on next launch.
        try? data.write(to: registryURL, options: .atomic)
    }
}
