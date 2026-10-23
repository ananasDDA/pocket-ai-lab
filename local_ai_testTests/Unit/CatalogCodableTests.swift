//
//  CatalogCodableTests.swift
//  local_ai_testTests
//
//  The catalog now travels as JSON (bundled resource, remote refresh, and the
//  installed-models registry), so the encoding has to survive a round trip and
//  the bundled document has to match what the app compiled in.
//

import Testing
import Foundation
@testable import local_ai_test

struct AIModelCodableTests {

    private func roundTrip(_ model: AIModel) throws -> AIModel {
        let data = try JSONEncoder().encode(model)
        return try JSONDecoder().decode(AIModel.self, from: data)
    }

    @Test func roundTripsEveryCatalogEntry() throws {
        for model in ModelCatalog.builtIn {
            #expect(try roundTrip(model) == model, "round trip changed \(model.id)")
        }
    }

    @Test func roundTripsAnImportedEntry() throws {
        let model = AIModel(
            id: "owner/repo#Q4_K_M",
            name: "Repo (Q4_K_M)",
            family: "Qwen",
            parameterSize: "7B",
            quantization: "Q4_K_M",
            backend: .llamaCppVision,
            capabilities: .vision,
            fileLayout: .ggufWithMmproj,
            ramRequiredGB: 5.4,
            diskSizeGB: 4.2,
            huggingFaceRepo: "owner/repo",
            contextLength: 32768,
            quality: .good,
            preferredGGUFFilename: "repo-Q4_K_M.gguf",
            mmprojFilename: "mmproj-F16.gguf",
            source: .imported,
            revision: "abc123"
        )
        #expect(try roundTrip(model) == model)
    }

    @Test func capabilitiesEncodeAsReadableNames() throws {
        let data = try JSONEncoder().encode(ModelCapabilities.videoVision)
        let names = try JSONDecoder().decode([String].self, from: data)
        #expect(names == ["textIn", "imageIn", "videoIn", "textOut"])
    }

    @Test func capabilitiesStillDecodeFromRawBitmask() throws {
        let raw = ModelCapabilities.vision.rawValue
        let decoded = try JSONDecoder().decode(
            ModelCapabilities.self, from: Data("\(raw)".utf8)
        )
        #expect(decoded == .vision)
    }

    @Test func minimalJSONFillsInDefaults() throws {
        let json = #"{"id":"a/b","name":"B","family":"F","backend":"mlx"}"#
        let model = try JSONDecoder().decode(AIModel.self, from: Data(json.utf8))
        #expect(model.source == .curated)
        #expect(model.revision == nil)
        #expect(model.capabilities == .textOnly)
        #expect(model.fileLayout == .huggingFaceMLX)
        #expect(model.contextLength == 4096)
        #expect(model.quality == .good)
    }
}

struct BundledCatalogTests {

    @Test func bundledCatalogDecodes() throws {
        let url = try #require(CatalogFile.bundledURL, "catalog.json is not in the app bundle")
        let data = try Data(contentsOf: url)
        let document = try JSONDecoder().decode(CatalogDocument.self, from: data)
        #expect(document.schemaVersion == CatalogDocument.supportedSchemaVersion)
        #expect(!document.models.isEmpty)
    }

    /// The bundled JSON and the compiled-in fallback are two copies of the
    /// same list; drift between them would silently change the app's catalog.
    @Test func bundledCatalogMatchesBuiltIn() throws {
        let url = try #require(CatalogFile.bundledURL)
        let document = try JSONDecoder().decode(
            CatalogDocument.self, from: try Data(contentsOf: url)
        )
        #expect(document.models == ModelCatalog.builtIn)
    }

    /// The bundled file is the one copy of the catalog this build is
    /// guaranteed to be able to read, so its version IS the supported version.
    /// A mismatch here means the app would reject its own resource on launch
    /// and fall back to `ModelCatalog.builtIn` — see RELEASE_CHECKLIST step 5.
    @Test func bundledCatalogDeclaresTheSupportedSchema() throws {
        let url = try #require(CatalogFile.bundledURL)
        let document = try JSONDecoder().decode(
            CatalogDocument.self, from: try Data(contentsOf: url)
        )
        #expect(document.schemaVersion == CatalogDocument.supportedSchemaVersion)
        #expect(CatalogDocument.supportedSchemaVersion == 3)
    }

    /// Schema 3 exists so that a v2 catalog published from an older branch
    /// cannot replace the 2026 GGUF entries. The bundled file must carry
    /// them, at the version this build accepts.
    @Test func theBundledV3CatalogCarriesThe2026Entries() throws {
        let url = try #require(CatalogFile.bundledURL)
        let models = try #require(CatalogFile.decode(try Data(contentsOf: url)))
        let ids = Set(models.map(\.id))
        for id in [
            "unsloth/Qwen3.5-0.8B-GGUF", "LiquidAI/LFM2-1.2B-GGUF",
            "unsloth/Qwen3.5-2B-GGUF", "LiquidAI/LFM2-2.6B-GGUF",
            "unsloth/Qwen3.5-4B-GGUF", "unsloth/gemma-4-E2B-it-GGUF"
        ] {
            #expect(ids.contains(id), "\(id) missing from the bundled catalog")
        }
    }

    /// Version skew is rejected in BOTH directions, which is the whole point:
    /// a v1 document is the copy still sitting in the public repo (and in the
    /// cache written from it), and accepting it would overwrite the newer
    /// bundled catalog on every launch.
    @Test func aSchemaVersion1DocumentIsRejected() {
        let json = #"{"schemaVersion":1,"models":[{"id":"a/b","name":"B","family":"F","backend":"mlx"}]}"#
        #expect(CatalogFile.decode(Data(json.utf8)) == nil)
    }

    /// A v2 document is what `main` would publish before this catalog lands;
    /// accepting it would drop the 2026 entries on every launch.
    @Test func aSchemaVersion2DocumentIsRejected() {
        let json = #"{"schemaVersion":2,"models":[{"id":"a/b","name":"B","family":"F","backend":"mlx"}]}"#
        #expect(CatalogFile.decode(Data(json.utf8)) == nil)
    }

    @Test func aSchemaVersion3DocumentIsAccepted() throws {
        let json = #"{"schemaVersion":3,"models":[{"id":"a/b","name":"B","family":"F","backend":"mlx"}]}"#
        let models = try #require(CatalogFile.decode(Data(json.utf8)))
        #expect(models.map(\.id) == ["a/b"])
    }

    @Test func aNewerSchemaVersionIsAlsoRejected() {
        let json = #"{"schemaVersion":99,"models":[{"id":"a/b","name":"B","family":"F","backend":"mlx"}]}"#
        #expect(CatalogFile.decode(Data(json.utf8)) == nil)
    }

    @Test func emptyModelListIsRejected() {
        let json = #"{"schemaVersion":3,"models":[]}"#
        #expect(CatalogFile.decode(Data(json.utf8)) == nil)
    }

    /// A cache written by a build with a different supported version is
    /// deleted rather than re-read and re-rejected forever. The existing cache
    /// is restored afterwards so running the suite does not cost the device
    /// its live catalog.
    @Test func aCacheFromAnotherSchemaIsPurged() throws {
        let cacheURL = try #require(CatalogFile.cacheURL)
        let backup = try? Data(contentsOf: cacheURL)
        defer {
            if let backup {
                try? backup.write(to: cacheURL, options: .atomic)
            } else {
                try? FileManager.default.removeItem(at: cacheURL)
            }
        }

        let stale = #"{"schemaVersion":1,"models":[{"id":"a/b","name":"B","family":"F","backend":"mlx"}]}"#
        try Data(stale.utf8).write(to: cacheURL, options: .atomic)

        #expect(CatalogFile.purgeIncompatibleCache())
        #expect(!FileManager.default.fileExists(atPath: cacheURL.path))
        // Second call has nothing left to do.
        #expect(!CatalogFile.purgeIncompatibleCache())
    }

    @Test func activeCatalogIsNeverEmpty() {
        #expect(!ModelCatalog.all.isEmpty)
    }
}
