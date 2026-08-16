//
//  RemoteCatalog.swift
//  local_ai_test
//
//  The model catalog ships as JSON so it can be corrected without an App
//  Store release: the app reads the bundled `catalog.json` on launch and
//  then quietly tries to pull a newer copy from the project's GitHub repo.
//  Every failure path falls back to what is already on disk, so the models
//  tab is never empty because the network misbehaved.
//

import Foundation
import Observation

// MARK: - Wire format

/// Top-level shape of `catalog.json`.
nonisolated struct CatalogDocument: Codable, Sendable {
    let schemaVersion: Int
    let models: [AIModel]

    /// Bumping this in the app makes it ignore older remote documents, and
    /// bumping it in the remote file makes older app builds ignore it. Either
    /// way the bundled copy keeps working.
    static let supportedSchemaVersion = 1
}

// MARK: - On-disk locations

enum CatalogFile {

    /// Path of `catalog.json` inside the GitHub repo. The bundled resource and
    /// this path must stay in sync — the remote copy is literally the same
    /// file, served raw.
    static let remoteURL = URL(
        string: "https://raw.githubusercontent.com/ananasDDA/pocket-ai-lab/main/local_ai_test/Resources/catalog.json"
    )

    /// Downloaded copy. Application Support (not Documents) because the user
    /// has no business seeing it in Files.app, and it is reproducible.
    static var cacheURL: URL? {
        guard let dir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first else { return nil }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("catalog-cache.json")
    }

    static func decode(_ data: Data) -> [AIModel]? {
        guard let document = try? JSONDecoder().decode(CatalogDocument.self, from: data),
              document.schemaVersion == CatalogDocument.supportedSchemaVersion,
              !document.models.isEmpty
        else { return nil }
        return document.models
    }

    /// Cached remote copy, else the bundled resource. `nil` means the caller
    /// should fall back to the compiled-in catalog.
    static func loadLocal() -> [AIModel]? {
        if let cacheURL, let data = try? Data(contentsOf: cacheURL), let models = decode(data) {
            return models
        }
        if let url = bundledURL, let data = try? Data(contentsOf: url), let models = decode(data) {
            return models
        }
        return nil
    }

    static var bundledURL: URL? {
        // `Bundle.main` covers the app and app-hosted unit tests; the
        // fallbacks keep the lookup working if the resource ever moves into
        // its own bundle.
        if let url = Bundle.main.url(forResource: "catalog", withExtension: "json") {
            return url
        }
        return Bundle.allBundles.compactMap {
            $0.url(forResource: "catalog", withExtension: "json")
        }.first
    }

    static func writeCache(_ data: Data) {
        guard let cacheURL else { return }
        try? data.write(to: cacheURL, options: .atomic)
    }
}

// MARK: - Snapshot

/// Lock-protected snapshot of the catalog in effect. `ModelCatalog.all` reads
/// through this, and the download delegate resolves model ids from a
/// background queue — so it must not be tied to an actor.
final class CatalogStorage {

    static let shared = CatalogStorage()

    private let lock = NSLock()
    private var storage: [AIModel]?

    private init() {}

    var current: [AIModel] {
        lock.withLock {
            if let storage { return storage }
            let loaded = CatalogFile.loadLocal() ?? ModelCatalog.builtIn
            storage = loaded
            return loaded
        }
    }

    func replace(with models: [AIModel]) {
        guard !models.isEmpty else { return }
        lock.withLock { storage = models }
    }
}

// MARK: - Remote refresh

@MainActor
@Observable
final class RemoteCatalog {

    static let shared = RemoteCatalog()

    /// Mirrors `ModelCatalog.all` so SwiftUI redraws after a refresh lands.
    private(set) var models: [AIModel]
    private(set) var lastRefresh: Date?

    @ObservationIgnored
    private var isRefreshing = false

    private init() {
        models = CatalogStorage.shared.current
    }

    /// Best-effort update. Never throws and never surfaces an error: a stale
    /// catalog is strictly better than an empty one.
    func refresh() async {
        guard !isRefreshing, let url = CatalogFile.remoteURL else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.cachePolicy = .reloadIgnoringLocalCacheData

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let fetched = CatalogFile.decode(data)
        else { return }

        CatalogFile.writeCache(data)
        CatalogStorage.shared.replace(with: fetched)
        models = fetched
        lastRefresh = Date()
    }
}
