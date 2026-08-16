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

    @Test func documentWithWrongSchemaIsRejected() {
        let json = #"{"schemaVersion":99,"models":[]}"#
        #expect(CatalogFile.decode(Data(json.utf8)) == nil)
    }

    @Test func emptyModelListIsRejected() {
        let json = #"{"schemaVersion":1,"models":[]}"#
        #expect(CatalogFile.decode(Data(json.utf8)) == nil)
    }

    @Test func activeCatalogIsNeverEmpty() {
        #expect(!ModelCatalog.all.isEmpty)
    }
}
