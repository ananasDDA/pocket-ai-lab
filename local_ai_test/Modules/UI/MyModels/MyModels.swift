//
//  MyModels.swift
//  local_ai_test
//
//  Pure model behind the "My Models" screen: which models are on the device,
//  how much space they take, and what removing one does. Kept free of
//  SwiftUI and of the shared stores so it can be tested on its own.
//

import Foundation

nonisolated struct MyModelsEntry: Identifiable, Equatable, Sendable {

    nonisolated enum Status: Equatable, Sendable {
        /// Loaded by the inference engine right now.
        case inUse
        /// On disk and ready to load.
        case available
        /// An imported entry whose files are not (fully) on disk: still
        /// downloading, failed, or cancelled. Only imported models are listed
        /// in this state, because the registry is their only home.
        case notDownloaded
    }

    let model: AIModel
    let status: Status
    /// A curated model the catalog in effect no longer lists. It keeps
    /// working, but no catalog card offers its Remove button any more.
    let isOrphaned: Bool
    /// Measured size of the model's folder when known, otherwise the
    /// descriptor's download size (0 for entries with nothing on disk yet).
    let sizeBytes: Int64
    let sizeIsMeasured: Bool

    var id: String { model.id }

    var isInstalled: Bool { status != .notDownloaded }

    /// Whether the app can download it again after it is removed: a catalog
    /// model, or an import that came from a Hugging Face link. A file picked
    /// from the Files app, or a model the catalog dropped, is gone for good.
    var canReinstall: Bool {
        switch model.source {
        case .curated: return !isOrphaned
        case .imported: return !model.huggingFaceRepo.isEmpty
        }
    }

    /// Text of the removal confirmation. Same wording the catalog cards use,
    /// except that it does not promise a reinstall the app cannot offer.
    var removalMessage: String {
        let size = MyModels.formatSize(sizeBytes)
        let unload = status == .inUse ? " It will be unloaded first." : ""
        let reinstall = canReinstall
            ? " You can reinstall it later."
            : " This cannot be undone from the app."
        return "This will delete \(model.name) (\(size)) from your device.\(unload)\(reinstall)"
    }
}

nonisolated enum MyModels {

    static let bytesPerGB: Double = 1_073_741_824

    /// Built-in models have nothing on disk and nothing to delete.
    static func canDelete(_ model: AIModel) -> Bool {
        model.backend != .appleIntelligence
    }

    /// Every model with files on the device, plus imported entries that are
    /// still on their way (their only listing in the app). Catalog membership
    /// does not matter: a model that left the catalog is listed like any
    /// other, flagged `isOrphaned`.
    ///
    /// Order: the loaded model first, then by name, so rows do not jump
    /// around while sizes are being measured.
    static func entries(
        known: [AIModel],
        installedIDs: Set<String>,
        catalogIDs: Set<String>,
        loadedModelID: String?,
        measuredSizes: [String: Int64] = [:]
    ) -> [MyModelsEntry] {
        var seen = Set<String>()
        var result: [MyModelsEntry] = []
        for model in known where canDelete(model) && seen.insert(model.id).inserted {
            let installed = installedIDs.contains(model.id)
            guard installed || model.source == .imported else { continue }

            let status: MyModelsEntry.Status
            if !installed {
                status = .notDownloaded
            } else if model.id == loadedModelID {
                status = .inUse
            } else {
                status = .available
            }

            let measured = measuredSizes[model.id]
            let size: Int64
            if let measured, measured > 0 {
                size = measured
            } else if installed {
                size = Int64(model.diskSizeGB * bytesPerGB)
            } else {
                size = 0
            }

            result.append(MyModelsEntry(
                model: model,
                status: status,
                isOrphaned: model.source == .curated && !catalogIDs.contains(model.id),
                sizeBytes: size,
                sizeIsMeasured: (measured ?? 0) > 0
            ))
        }
        return result.sorted { a, b in
            if (a.status == .inUse) != (b.status == .inUse) { return a.status == .inUse }
            let byName = a.model.name.localizedStandardCompare(b.model.name)
            if byName != .orderedSame { return byName == .orderedAscending }
            return a.id < b.id
        }
    }

    /// Space the listed models take on this device.
    static func totalBytes(_ entries: [MyModelsEntry]) -> Int64 {
        entries.reduce(0) { $0 + $1.sizeBytes }
    }

    /// Whether removing it should also drop the registry record. Imported
    /// models live only in the registry, so a record left behind would come
    /// back as an empty row; curated ones keep theirs (analytics and old
    /// benchmark reports resolve names through it).
    static func unregistersOnRemoval(_ model: AIModel) -> Bool {
        model.source == .imported
    }

    static func formatSize(_ bytes: Int64) -> String {
        let gb = Double(bytes) / bytesPerGB
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        return String(format: "%.0f MB", Double(bytes) / 1_048_576)
    }
}
