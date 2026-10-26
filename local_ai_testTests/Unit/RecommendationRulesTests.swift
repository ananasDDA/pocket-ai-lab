//
//  RecommendationRulesTests.swift
//  local_ai_testTests
//
//  Rules the Setup screen relies on, checked against the catalog in this
//  checkout on every iPhone memory class. The assertions are about the rule
//  (a chat model, never a deprecated one, the recommended card not repeated)
//  rather than a particular model id, so catalog edits that only move
//  quality values or drop entries keep them green.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - Fixtures

@MainActor
private enum Phones {

    static let gib = 1_073_741_824.0

    /// Figures as `DeviceScanner` reports them on real hardware: `totalRAM`
    /// as iOS rounds it, a jetsam limit of about half of that, and a Metal
    /// working set of about 0.7 x RAM.
    static func phone(
        ramGiB: Double,
        jetsamGiB: Double,
        metalGiB: Double,
        diskGiB: Double = 100,
        ai: AppleIntelligenceStatus = .notSupported
    ) -> DeviceInfo {
        DeviceInfo(
            identifier: "iPhone-test",
            marketingName: "iPhone Test",
            chip: "A-test",
            totalRAM: UInt64(ramGiB * gib),
            jetsamLimit: Int64(jetsamGiB * gib),
            metalWorkingSet: Int64(metalGiB * gib),
            iOSVersion: "26.1",
            freeDiskSpace: Int64(diskGiB * gib),
            appleIntelligence: ai
        )
    }

    static func threeGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 2.75, jetsamGiB: 1.4, metalGiB: 1.9, ai: ai)
    }
    static func fourGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 3.66, jetsamGiB: 1.9, metalGiB: 2.56, ai: ai)
    }
    static func sixGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 5.5, jetsamGiB: 2.9, metalGiB: 3.85, ai: ai)
    }
    static func eightGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 7.3, jetsamGiB: 3.7, metalGiB: 5.1, ai: ai)
    }
    static func twelveGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 11.5, jetsamGiB: 5.75, metalGiB: 8.0, ai: ai)
    }

    /// Every class with and without Apple Intelligence, labelled for failures.
    static var all: [(String, DeviceInfo)] {
        var result: [(String, DeviceInfo)] = []
        for ai in [AppleIntelligenceStatus.notSupported, .available] {
            let tag = ai == .available ? "AI on" : "AI off"
            result += [
                ("3 GB, \(tag)", threeGB(ai: ai)),
                ("4 GB, \(tag)", fourGB(ai: ai)),
                ("6 GB, \(tag)", sixGB(ai: ai)),
                ("8 GB, \(tag)", eightGB(ai: ai)),
                ("12 GB, \(tag)", twelveGB(ai: ai)),
            ]
        }
        return result
    }
}

@MainActor
private func bundledCatalog() throws -> [AIModel] {
    let url = try #require(CatalogFile.bundledURL, "catalog.json is not in the app bundle")
    return try #require(CatalogFile.decode(try Data(contentsOf: url)))
}

private func testModel(
    _ id: String,
    backend: ModelBackend = .llamaCpp,
    ram: Double,
    disk: Double = 1,
    quality: ModelQuality = .good,
    deprecated: Bool = false
) -> AIModel {
    AIModel(
        id: id,
        name: id,
        family: "Test",
        parameterSize: "1B",
        quantization: "Q4",
        backend: backend,
        capabilities: backend.canChat ? .textOnly : [.audioIn, .textOut],
        ramRequiredGB: ram,
        diskSizeGB: disk,
        huggingFaceRepo: "test/\(id)",
        contextLength: 4096,
        quality: quality,
        deprecated: deprecated
    )
}

/// Same descriptor with the flag set, as the post-1.1 remote catalog ships it.
private func markedDeprecated(_ m: AIModel) -> AIModel {
    AIModel(
        id: m.id, name: m.name, family: m.family, parameterSize: m.parameterSize,
        quantization: m.quantization, backend: m.backend, capabilities: m.capabilities,
        fileLayout: m.fileLayout, ramRequiredGB: m.ramRequiredGB, diskSizeGB: m.diskSizeGB,
        huggingFaceRepo: m.huggingFaceRepo, requiresAppleIntelligence: m.requiresAppleIntelligence,
        contextLength: m.contextLength, quality: m.quality, requiresCompilation: m.requiresCompilation,
        preferredGGUFFilename: m.preferredGGUFFilename, mmprojFilename: m.mmprojFilename,
        source: m.source, revision: m.revision, licenseName: m.licenseName,
        licenseURL: m.licenseURL, preferredPackage: m.preferredPackage, deprecated: true
    )
}

// MARK: - Recommendation rules on the bundled catalog

@MainActor
struct RecommendationRulesTests {

    private func clearsMargin(_ model: AIModel, on device: DeviceInfo) -> Bool {
        model.ramRequiredGB <= device.budgetGB(for: model.backend) * ModelRecommender.recommendationMargin
    }

    @Test func recommendationIsAlwaysACurrentChatModel() throws {
        let catalog = try bundledCatalog()
        for (label, device) in Phones.all {
            let result = ModelRecommender.shared.recommend(for: device, models: catalog)
            let model = try #require(result.recommended, "\(label): nothing recommended")
            #expect(model.backend.canChat, "\(label): \(model.id)")
            #expect(!model.deprecated, "\(label): \(model.id)")
            #expect(result.allCompatible.contains { $0.id == model.id }, "\(label)")
            if device.appleIntelligence != .available {
                #expect(model.backend != .appleIntelligence, "\(label)")
            }
        }
    }

    /// The original bug: Whisper fits every phone, so on small ones it won.
    @Test func smallPhonesAreNotOfferedASpeechModel() throws {
        let catalog = try bundledCatalog()
        for device in [Phones.threeGB(), Phones.fourGB(), Phones.sixGB(), Phones.sixGB(ai: .available)] {
            let model = try #require(ModelRecommender.shared.recommend(for: device, models: catalog).recommended)
            #expect(model.backend != .whisperCpp && model.backend != .coreMLWhisper
                    && model.backend != .coreMLKokoro, "\(model.id)")
        }
    }

    /// The pick is the best current chat model that clears the margin, or
    /// the best current chat model at all when none does.
    @Test func recommendationIsTheBestCandidateThatLeavesHeadroom() throws {
        let catalog = try bundledCatalog()
        for (label, device) in Phones.all {
            let result = ModelRecommender.shared.recommend(for: device, models: catalog)
            let model = try #require(result.recommended)
            let candidates = result.allCompatible.filter(ModelRecommender.isRecommendable)
            let index = try #require(candidates.firstIndex { $0.id == model.id })
            for better in candidates[..<index] {
                #expect(!clearsMargin(better, on: device), "\(label): \(better.id) was skipped")
            }
            if !clearsMargin(model, on: device) {
                #expect(index == 0, "\(label)")
                #expect(!candidates.contains { clearsMargin($0, on: device) }, "\(label)")
            }
        }
    }

    @Test func speechModelsStayInTheCompatibleList() throws {
        let catalog = try bundledCatalog()
        let speech = catalog.filter { !$0.backend.canChat }
        try #require(!speech.isEmpty, "catalog has no speech model to check")
        for (label, device) in Phones.all {
            let result = ModelRecommender.shared.recommend(for: device, models: catalog)
            for model in speech where model.ramRequiredGB <= device.budgetGB(for: model.backend) {
                #expect(result.allCompatible.contains { $0.id == model.id }, "\(label): \(model.id)")
            }
            // Nothing is lost between the two lists.
            #expect(Set((result.allCompatible + result.incompatible).map(\.id)) == Set(catalog.map(\.id)),
                    "\(label)")
        }
    }

    // MARK: Other compatible models (Setup list under the recommended card)

    @Test func otherCompatibleDropsExactlyTheRecommendedModel() throws {
        let catalog = try bundledCatalog()
        for (label, device) in Phones.all {
            let result = ModelRecommender.shared.recommend(for: device, models: catalog)
            let recommended = try #require(result.recommended)
            #expect(result.otherCompatible.count == result.allCompatible.count - 1, "\(label)")
            #expect(!result.otherCompatible.contains { $0.id == recommended.id }, "\(label)")
            #expect(result.otherCompatible.map(\.id)
                    == result.allCompatible.map(\.id).filter { $0 != recommended.id }, "\(label)")
        }
    }

    /// 8 GB phone: the first model in sort order misses the 0.85 margin, a
    /// smaller one is recommended. The first one used to vanish from the
    /// screen while the recommended one was shown twice.
    @Test func firstSortedModelStaysVisibleWhenItIsNotTheRecommendation() throws {
        let big = testModel("big", ram: 5.0, quality: .excellent)
        let fits = testModel("fits", ram: 2.2, quality: .great)
        let small = testModel("small", ram: 1.4)
        let result = ModelRecommender.shared.recommend(for: Phones.eightGB(), models: [small, fits, big])

        #expect(result.allCompatible.first?.id == "big")
        #expect(result.recommended?.id == "fits")
        #expect(result.otherCompatible.map(\.id) == ["big", "small"])
    }

    @Test func otherCompatibleIsEverythingWhenNothingIsRecommended() {
        let whisper = testModel("whisper", backend: .whisperCpp, ram: 0.8)
        let result = ModelRecommender.shared.recommend(for: Phones.sixGB(), models: [whisper])
        #expect(result.recommended == nil)
        #expect(result.otherCompatible.map(\.id) == ["whisper"])
    }

    // MARK: Nothing to chat with

    @Test func onlySpeechFitsSaysSoPlainly() {
        let whisper = testModel("whisper", backend: .whisperCpp, ram: 0.8, quality: .great)
        let huge = testModel("huge", ram: 40)
        let result = ModelRecommender.shared.recommend(for: Phones.sixGB(), models: [whisper, huge])

        #expect(result.recommended == nil)
        #expect(result.allCompatible.map(\.id) == ["whisper"])
        #expect(result.reason.contains("None of the chat models"))
        #expect(result.reason.contains("hold a conversation"))
        #expect(result.reason.contains("enough memory"))
        #expect(!result.reason.contains("Free up storage"))
    }

    @Test func chatModelBlockedOnlyByDiskPointsAtStorage() {
        let whisper = testModel("whisper", backend: .whisperCpp, ram: 0.8, disk: 0.15)
        let chat = testModel("chat", ram: 1.4, disk: 3.0)
        let device = Phones.phone(ramGiB: 7.3, jetsamGiB: 3.7, metalGiB: 5.1, diskGiB: 1.0)
        let result = ModelRecommender.shared.recommend(for: device, models: [whisper, chat])

        #expect(result.recommended == nil)
        #expect(result.reason.contains("Free up storage"))
    }

    // MARK: Deprecated entries

    @Test func deprecatedModelsAreNeverRecommended() {
        let old = testModel("old", ram: 2.0, quality: .excellent, deprecated: true)
        let current = testModel("current", ram: 1.4)
        let result = ModelRecommender.shared.recommend(for: Phones.eightGB(), models: [old, current])
        #expect(result.recommended?.id == "current")
        #expect(result.allCompatible.map(\.id) == ["current", "old"])
    }

    @Test func onlyDeprecatedChatModelsFitSaysSo() {
        let old = testModel("old", ram: 2.0, deprecated: true)
        let huge = testModel("huge", ram: 40)
        let result = ModelRecommender.shared.recommend(for: Phones.eightGB(), models: [old, huge])
        #expect(result.recommended == nil)
        #expect(result.allCompatible.map(\.id) == ["old"])
        #expect(result.reason.contains("older models"))
    }

    @Test func deprecatedModelsSortAfterEveryCurrentOne() {
        let models = [
            testModel("old-excellent", ram: 2.0, quality: .excellent, deprecated: true),
            testModel("current-good", ram: 1.0),
            testModel("old-good", ram: 3.0, deprecated: true),
            testModel("current-great", ram: 1.5, quality: .great),
            testModel("speech", backend: .whisperCpp, ram: 0.5),
        ]
        let ordered = ModelRecommender.inListOrder(models).map(\.id)
        #expect(ordered == ["current-great", "current-good", "speech", "old-excellent", "old-good"])

        // Incompatible lists keep catalog order apart from the deprecated split.
        let incompatible = ModelRecommender.inListOrder(models, byQuality: false).map(\.id)
        #expect(incompatible == ["current-good", "current-great", "speech", "old-excellent", "old-good"])
    }

    @Test func deprecatedModelsSortLastAmongIncompatibleToo() {
        let oldHuge = testModel("old-huge", ram: 40, deprecated: true)
        let huge = testModel("huge", ram: 50)
        let result = ModelRecommender.shared.recommend(for: Phones.eightGB(), models: [oldHuge, huge])
        #expect(result.incompatible.map(\.id) == ["huge", "old-huge"])
    }

    /// The post-1.1 remote catalog flags the older models instead of
    /// deleting them. Simulate that on the catalog in this checkout.
    @Test func flaggingOlderModelsMovesThemToTheBottomEverywhere() throws {
        let olderIDs: Set<String> = [
            "bartowski/Phi-3.5-mini-instruct-GGUF",
            "mlx-community/Phi-3.5-mini-instruct-4bit",
            "anziank/grio-qwen2.5-1.5b-instruct-coreml-stateful-int8",
        ]
        var catalog = try bundledCatalog()
        let present = catalog.filter { olderIDs.contains($0.id) }
        // The pruned catalog may no longer carry them; flag the lowest-quality
        // chat models instead so the rule is still exercised.
        let flagged: Set<String> = present.isEmpty
            ? Set(catalog.filter { $0.backend.canChat && $0.quality == .good && $0.ramRequiredGB > 0 }
                    .prefix(3).map(\.id))
            : olderIDs
        try #require(!flagged.isEmpty)
        catalog = catalog.map { flagged.contains($0.id) ? markedDeprecated($0) : $0 }

        for (label, device) in Phones.all {
            let result = ModelRecommender.shared.recommend(for: device, models: catalog)
            #expect(!flagged.contains(result.recommended?.id ?? ""), "\(label)")
            let deprecatedFlags = result.allCompatible.map(\.deprecated)
            #expect(deprecatedFlags == deprecatedFlags.sorted { !$0 && $1 }, "\(label): \(result.allCompatible.map(\.id))")
            let otherFlags = result.otherCompatible.map(\.deprecated)
            #expect(otherFlags == otherFlags.sorted { !$0 && $1 }, "\(label)")
        }
    }

    @Test func listOrderWithinCurrentModelsIsStillByQuality() throws {
        let catalog = try bundledCatalog()
        for (label, device) in Phones.all {
            let current = ModelRecommender.shared.recommend(for: device, models: catalog)
                .allCompatible.filter { !$0.deprecated }
            let qualities = current.map(\.quality.rawValue)
            #expect(qualities == qualities.sorted(by: >), "\(label)")
        }
    }
}

// MARK: - `deprecated` in catalog JSON

@MainActor
struct DeprecatedFieldCodingTests {

    private static let entryWithoutKey = """
    {"id":"x/y","name":"Y","family":"F","backend":"llamaCpp","ramRequiredGB":1,"diskSizeGB":1,"huggingFaceRepo":"x/y","quality":1}
    """

    private func entry(deprecated: Bool?) -> Data {
        guard let deprecated else { return Data(Self.entryWithoutKey.utf8) }
        let json = Self.entryWithoutKey.dropLast() + ",\"deprecated\":\(deprecated)}"
        return Data(json.utf8)
    }

    private func encodedKeys(_ model: AIModel) throws -> [String: Any] {
        let data = try JSONEncoder().encode(model)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func absentKeyDecodesAsCurrentAndIsNotWrittenBack() throws {
        let model = try JSONDecoder().decode(AIModel.self, from: entry(deprecated: nil))
        #expect(!model.deprecated)
        #expect(try encodedKeys(model)["deprecated"] == nil)
    }

    @Test func trueIsReadAndWrittenBack() throws {
        let model = try JSONDecoder().decode(AIModel.self, from: entry(deprecated: true))
        #expect(model.deprecated)
        #expect(try encodedKeys(model)["deprecated"] as? Bool == true)
        #expect(try JSONDecoder().decode(AIModel.self, from: JSONEncoder().encode(model)) == model)
    }

    @Test func explicitFalseIsReadAndOmittedOnWrite() throws {
        let model = try JSONDecoder().decode(AIModel.self, from: entry(deprecated: false))
        #expect(!model.deprecated)
        #expect(try encodedKeys(model)["deprecated"] == nil)
    }

    /// No schema bump: a catalog document at the current schema that carries
    /// the new key still passes `CatalogFile.decode`.
    @Test func catalogWithTheKeyDecodesAtTheCurrentSchema() throws {
        let url = try #require(CatalogFile.bundledURL)
        var document = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        var models = try #require(document["models"] as? [[String: Any]])
        // Flag one entry that does not carry the key yet; the catalog may
        // already flag others.
        let index = try #require(models.lastIndex { $0["deprecated"] == nil })
        let flaggedID = try #require(models[index]["id"] as? String)
        let alreadyFlagged = Set(models.filter { $0["deprecated"] as? Bool == true }.compactMap { $0["id"] as? String })
        models[index]["deprecated"] = true
        document["models"] = models
        let data = try JSONSerialization.data(withJSONObject: document)

        let decoded = try #require(CatalogFile.decode(data))
        #expect(decoded.count == models.count)
        #expect(Set(decoded.filter(\.deprecated).map(\.id)) == alreadyFlagged.union([flaggedID]))
    }

    /// Re-encoding the bundled catalog adds no key to current entries.
    @Test func bundledCatalogRoundTripsWithoutTheKey() throws {
        for model in try bundledCatalog() {
            let keys = try encodedKeys(model)
            #expect((keys["deprecated"] != nil) == model.deprecated, "\(model.id)")
        }
    }
}
