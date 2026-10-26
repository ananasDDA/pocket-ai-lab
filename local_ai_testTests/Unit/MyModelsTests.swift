//
//  MyModelsTests.swift
//  local_ai_testTests
//
//  The pure part of the My Models screen: which models it lists, the sizes
//  and total it shows, which ones are flagged as gone from the catalog, and
//  what removing one promises.
//

import Testing
import Foundation
@testable import local_ai_test

struct MyModelsTests {

    private static let gb: Double = 1_073_741_824

    private static func model(
        _ id: String,
        name: String? = nil,
        backend: ModelBackend = .llamaCpp,
        disk: Double = 1.0,
        repo: String? = nil,
        source: ModelSource = .curated
    ) -> AIModel {
        AIModel(
            id: id,
            name: name ?? id,
            family: "Test",
            parameterSize: "1B",
            quantization: "Q4",
            backend: backend,
            ramRequiredGB: 1,
            diskSizeGB: disk,
            huggingFaceRepo: repo ?? id,
            contextLength: 4096,
            quality: .good,
            source: source
        )
    }

    // A device with one of everything.
    private static let apple = model("apple-intelligence", name: "Apple Intelligence",
                                     backend: .appleIntelligence, disk: 0, repo: "")
    private static let current = model("org/current", name: "Current", disk: 1.6)
    private static let notInstalled = model("org/never", name: "Never installed")
    private static let orphan = model("org/dropped", name: "Dropped", disk: 2.4)
    private static let hfImport = model("u/hf", name: "HF import", disk: 0.8, source: .imported)
    private static let hfDownloading = model("u/pending", name: "Pending import", source: .imported)
    private static let filesImport = model("local/file.gguf", name: "Files import", disk: 0.5,
                                           repo: "", source: .imported)

    private static let known = [apple, current, notInstalled, orphan, hfImport, hfDownloading, filesImport]
    private static let installed: Set<String> = [apple.id, current.id, orphan.id, hfImport.id, filesImport.id]
    private static let catalog: Set<String> = [apple.id, current.id, notInstalled.id]

    private func entries(
        loaded: String? = nil,
        measured: [String: Int64] = [:]
    ) -> [MyModelsEntry] {
        MyModels.entries(
            known: Self.known,
            installedIDs: Self.installed,
            catalogIDs: Self.catalog,
            loadedModelID: loaded,
            measuredSizes: measured
        )
    }

    private func entry(_ id: String, in list: [MyModelsEntry]) throws -> MyModelsEntry {
        try #require(list.first { $0.id == id }, "\(id) is not listed")
    }

    // MARK: - What is listed

    @Test func listsEveryDownloadedModelWhateverItsSource() {
        let ids = Set(entries().map(\.id))
        #expect(ids == [Self.current.id, Self.orphan.id, Self.hfImport.id,
                        Self.hfDownloading.id, Self.filesImport.id])
    }

    @Test func builtInAppleIntelligenceIsNeverListed() {
        #expect(!entries().contains { $0.model.backend == .appleIntelligence })
        #expect(!MyModels.canDelete(Self.apple))
    }

    @Test func curatedModelsThatWereNeverInstalledAreNotListed() {
        #expect(!entries().contains { $0.id == Self.notInstalled.id })
    }

    /// Imported entries live only in the registry, so one that is still
    /// downloading (or failed) has to be listed to stay reachable.
    @Test func importedModelsAreListedBeforeTheirFilesArrive() throws {
        let pending = try entry(Self.hfDownloading.id, in: entries())
        #expect(pending.status == .notDownloaded)
        #expect(!pending.isInstalled)
        #expect(pending.sizeBytes == 0)
    }

    @Test func everyListedModelCanBeDeleted() {
        for entry in entries() {
            #expect(MyModels.canDelete(entry.model), "\(entry.id)")
        }
    }

    @Test func duplicateRegistryRecordsAreListedOnce() {
        let list = MyModels.entries(
            known: [Self.current, Self.current],
            installedIDs: [Self.current.id],
            catalogIDs: Self.catalog,
            loadedModelID: nil
        )
        #expect(list.count == 1)
    }

    // MARK: - Orphans

    @Test func onlyCuratedModelsMissingFromTheCatalogAreOrphaned() throws {
        let list = entries()
        #expect(try entry(Self.orphan.id, in: list).isOrphaned)
        #expect(try !entry(Self.current.id, in: list).isOrphaned)
        // Imported models were never in the catalog; that is not "dropped".
        #expect(try !entry(Self.hfImport.id, in: list).isOrphaned)
        #expect(try !entry(Self.filesImport.id, in: list).isOrphaned)
    }

    @Test func aModelDroppedByTheCatalogStaysUsable() throws {
        let orphan = try entry(Self.orphan.id, in: entries())
        #expect(orphan.status == .available)
        #expect(orphan.isInstalled)
    }

    /// The post-1.1 catalog drops these three; a 1.1 user who installed one
    /// must still find it here.
    @Test func modelsRemovedAfter1_1AreListedAsOrphans() {
        let removed = [
            Self.model("bartowski/Phi-3.5-mini-instruct-GGUF", disk: 2.4),
            Self.model("mlx-community/Phi-3.5-mini-instruct-4bit", backend: .mlx, disk: 2.3),
            Self.model("anziank/grio-qwen2.5-1.5b-instruct-coreml-stateful-int8", backend: .coreML, disk: 2.3),
        ]
        let list = MyModels.entries(
            known: removed,
            installedIDs: Set(removed.map(\.id)),
            catalogIDs: Self.catalog,
            loadedModelID: nil
        )
        #expect(list.count == 3)
        #expect(list.allSatisfy { $0.isOrphaned && $0.isInstalled })
    }

    // MARK: - Status

    @Test func theLoadedModelIsInUseAndListedFirst() throws {
        let list = entries(loaded: Self.orphan.id)
        #expect(list.first?.id == Self.orphan.id)
        #expect(try entry(Self.orphan.id, in: list).status == .inUse)
        #expect(list.dropFirst().allSatisfy { $0.status != .inUse })
    }

    @Test func restIsSortedByName() {
        let names = entries().map(\.model.name)
        #expect(names == names.sorted { $0.localizedStandardCompare($1) == .orderedAscending })
    }

    // MARK: - Sizes

    @Test func measuredSizeWinsOverTheCatalogFigure() throws {
        let measured = Int64(2.1 * Self.gb)
        let list = entries(measured: [Self.orphan.id: measured])
        let orphan = try entry(Self.orphan.id, in: list)
        #expect(orphan.sizeBytes == measured)
        #expect(orphan.sizeIsMeasured)

        let current = try entry(Self.current.id, in: list)
        #expect(current.sizeBytes == Int64(1.6 * Self.gb))
        #expect(!current.sizeIsMeasured)
    }

    /// An empty or unreadable folder measures 0; fall back rather than
    /// claiming an installed model takes no space.
    @Test func zeroMeasurementFallsBackToTheCatalogFigure() throws {
        let list = entries(measured: [Self.current.id: 0])
        #expect(try entry(Self.current.id, in: list).sizeBytes == Int64(1.6 * Self.gb))
    }

    @Test func partialDownloadsCountOnlyWhatIsOnDisk() throws {
        let partial = Int64(300 * 1_048_576)
        let list = entries(measured: [Self.hfDownloading.id: partial])
        #expect(try entry(Self.hfDownloading.id, in: list).sizeBytes == partial)
    }

    @Test func totalIsTheSumOfTheRows() {
        let list = entries(measured: [Self.hfImport.id: Int64(0.9 * Self.gb)])
        let expected = Int64(1.6 * Self.gb) + Int64(2.4 * Self.gb)
            + Int64(0.9 * Self.gb) + Int64(0.5 * Self.gb)
        #expect(MyModels.totalBytes(list) == expected)
        #expect(MyModels.totalBytes([]) == 0)
    }

    @Test func sizesFormatLikeTheCards() {
        #expect(MyModels.formatSize(Int64(2.4 * Self.gb)) == "2.4 GB")
        #expect(MyModels.formatSize(300 * 1_048_576) == "300 MB")
    }

    // MARK: - Removal

    @Test func reinstallIsOnlyPromisedWhereTheAppCanDoIt() throws {
        let list = entries()
        #expect(try entry(Self.current.id, in: list).canReinstall)
        #expect(try entry(Self.hfImport.id, in: list).canReinstall)
        #expect(try !entry(Self.orphan.id, in: list).canReinstall)
        #expect(try !entry(Self.filesImport.id, in: list).canReinstall)
    }

    @Test func removalMessageMatchesWhatHappens() throws {
        let list = entries(loaded: Self.current.id)
        let current = try entry(Self.current.id, in: list).removalMessage
        #expect(current.contains("Current (1.6 GB)"))
        #expect(current.contains("unloaded first"))
        #expect(current.contains("You can reinstall it later."))

        let orphan = try entry(Self.orphan.id, in: list).removalMessage
        #expect(!orphan.contains("unloaded"))
        #expect(!orphan.contains("reinstall"))
    }

    @Test func onlyImportedModelsLeaveTheRegistryOnRemoval() {
        #expect(MyModels.unregistersOnRemoval(Self.hfImport))
        #expect(MyModels.unregistersOnRemoval(Self.filesImport))
        #expect(!MyModels.unregistersOnRemoval(Self.current))
        #expect(!MyModels.unregistersOnRemoval(Self.orphan))
    }
}
