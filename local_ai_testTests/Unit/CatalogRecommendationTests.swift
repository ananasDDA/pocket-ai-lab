//
//  CatalogRecommendationTests.swift
//  local_ai_testTests
//
//  The recommender is fixed code in a shipped build; the only lever left
//  between releases is the catalog it reads (`quality`, never a doctored
//  `ramRequiredGB`). These tests pin what the catalog in this checkout makes
//  the recommender pick on each iPhone memory class, so a catalog edit that
//  silently changes the first model a new user is offered fails here.
//
//  Device figures are the ones `DeviceScanner` reports on real hardware:
//  `totalRAM` as iOS rounds it (~5.5 GiB on 6 GB phones, ~7.3 on 8 GB,
//  11.5 on 12 GB), a jetsam limit of about half of that with the
//  increased-memory-limit entitlement, and a Metal working set of about
//  0.7 x RAM (exactly 8 GiB on the iPhone Air). A second set leaves the
//  kernel figures at 0, which is the `totalRAM - 3` / `0.67 x RAM` fallback
//  the older recommender tests use.
//

import Testing
import Foundation
@testable import local_ai_test

@Suite(.serialized)
struct CatalogRecommendationTests {

    private static let gemma4E2B = "unsloth/gemma-4-E2B-it-GGUF"
    private static let lfm2_2_6B = "LiquidAI/LFM2-2.6B-GGUF"
    private static let lfm2_1_2B = "LiquidAI/LFM2-1.2B-GGUF"

    /// Lost the 2026-10 evaluation but stay in the catalog so existing
    /// installs keep their delete button: lowest tier, end of both copies,
    /// never recommended.
    private static let legacyIDs = [
        "mlx-community/Phi-3.5-mini-instruct-4bit",
        "bartowski/Phi-3.5-mini-instruct-GGUF",
        "anziank/grio-qwen2.5-1.5b-instruct-coreml-stateful-int8",
    ]

    // MARK: - Fixtures

    private static let gib = 1_073_741_824.0

    private static func phone(
        ramGiB: Double,
        jetsamGiB: Double,
        metalGiB: Double,
        ai: AppleIntelligenceStatus
    ) -> DeviceInfo {
        DeviceInfo(
            identifier: "iPhone-test",
            marketingName: "iPhone Test",
            chip: "A-test",
            totalRAM: UInt64(ramGiB * gib),
            jetsamLimit: Int64(jetsamGiB * gib),
            metalWorkingSet: Int64(metalGiB * gib),
            iOSVersion: "26.1",
            freeDiskSpace: Int64(100 * gib),
            appleIntelligence: ai
        )
    }

    private static func sixGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 5.5, jetsamGiB: 2.9, metalGiB: 3.85, ai: ai)
    }

    private static func eightGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 7.3, jetsamGiB: 3.7, metalGiB: 5.1, ai: ai)
    }

    private static func twelveGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 11.5, jetsamGiB: 5.75, metalGiB: 8.0, ai: ai)
    }

    private static func fourGB() -> DeviceInfo {
        phone(ramGiB: 3.66, jetsamGiB: 1.9, metalGiB: 2.56, ai: .notSupported)
    }

    private static func threeGB() -> DeviceInfo {
        phone(ramGiB: 2.8, jetsamGiB: 1.4, metalGiB: 1.96, ai: .notSupported)
    }

    private static func sixteenGB(ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        phone(ramGiB: 15.5, jetsamGiB: 7.75, metalGiB: 10.8, ai: ai)
    }

    /// Every class above, with and without Apple Intelligence where the
    /// hardware could have it, plus the fallback-math devices.
    private static func allDeviceClasses() -> [DeviceInfo] {
        var devices = [threeGB(), fourGB()]
        for ai in [AppleIntelligenceStatus.notSupported, .available] {
            devices += [sixGB(ai: ai), eightGB(ai: ai), twelveGB(ai: ai), sixteenGB(ai: ai)]
            devices += [6.0, 8.0, 12.0].map { fallback(ramGiB: $0, ai: ai) }
        }
        return devices
    }

    /// Kernel figures left at 0: the fallback math.
    private static func fallback(ramGiB: Double, ai: AppleIntelligenceStatus) -> DeviceInfo {
        phone(ramGiB: ramGiB, jetsamGiB: 0, metalGiB: 0, ai: ai)
    }

    // MARK: - Catalog under test

    private func bundledModels() throws -> [AIModel] {
        let url = try #require(CatalogFile.bundledURL, "catalog.json is not in the app bundle")
        return try #require(CatalogFile.decode(try Data(contentsOf: url)))
    }

    /// `ModelCatalog.all` on a phone is whatever catalog the device last
    /// fetched from the public repo. Run the recommender against the catalog
    /// in this checkout instead, and put the live one back afterwards.
    private func recommend(_ device: DeviceInfo) throws -> RecommendationResult {
        let bundled = try bundledModels()
        let previous = CatalogStorage.shared.current
        CatalogStorage.shared.replace(with: bundled)
        defer { CatalogStorage.shared.replace(with: previous) }
        return ModelRecommender.shared.recommend(for: device)
    }

    // MARK: - Budgets the pins rely on

    @Test func fixtureBudgetsMatchTheDeviceClasses() {
        #expect(abs(Self.sixGB().mmapCeilingGB - 3.025) < 0.01)
        #expect(abs(Self.eightGB().mmapCeilingGB - 5.1) < 0.01)
        #expect(abs(Self.twelveGB().mmapCeilingGB - 8.0) < 0.01)
        #expect(abs(Self.sixGB().usableRAMGB - 1.9) < 0.01)
        #expect(abs(Self.eightGB().usableRAMGB - 3.7) < 0.01)
        #expect(abs(Self.twelveGB().usableRAMGB - 5.75) < 0.01)
    }

    // MARK: - Recommendation per device class

    @Test func twelveGBPhonesGetGemma4E2B() throws {
        for ai in [AppleIntelligenceStatus.notSupported, .available] {
            let result = try recommend(Self.twelveGB(ai: ai))
            #expect(result.recommended?.id == Self.gemma4E2B, "AI \(ai)")
            // Recommended AND first in the list, so the Setup screen shows it once.
            #expect(result.allCompatible.first?.id == Self.gemma4E2B, "AI \(ai)")
        }
    }

    /// Gemma 4 E2B needs 5.0 GB against a ~5.1 GB GGUF budget: it fits, but
    /// not inside the 85% margin, so LFM2 2.6B is the recommendation. Its RAM
    /// figure is measured and stays honest.
    @Test func eightGBPhonesGetLFM2_2_6B() throws {
        for ai in [AppleIntelligenceStatus.notSupported, .available] {
            let result = try recommend(Self.eightGB(ai: ai))
            #expect(result.recommended?.id == Self.lfm2_2_6B, "AI \(ai)")
            #expect(result.allCompatible.contains { $0.id == Self.gemma4E2B }, "AI \(ai)")
        }
    }

    @Test func sixGBPhonesGetLFM2_2_6B() throws {
        let result = try recommend(Self.sixGB())
        #expect(result.recommended?.id == Self.lfm2_2_6B)
        #expect(result.allCompatible.first?.id == Self.lfm2_2_6B)
        #expect(!result.allCompatible.contains { $0.id == Self.gemma4E2B })
    }

    /// Below 6 GB nothing `.excellent` fits; LFM2 1.2B has to outrank the
    /// Whisper entries, which fit every device.
    @Test func fourGBPhonesGetLFM2_1_2B() throws {
        let result = try recommend(Self.fourGB())
        #expect(result.recommended?.id == Self.lfm2_1_2B)
    }

    @Test(arguments: [6.0, 8.0, 12.0])
    func fallbackBudgetsPickTheSameModels(ramGiB: Double) throws {
        let expected = ramGiB >= 12 ? Self.gemma4E2B : Self.lfm2_2_6B
        for ai in [AppleIntelligenceStatus.notSupported, .available] {
            let result = try recommend(Self.fallback(ramGiB: ramGiB, ai: ai))
            #expect(result.recommended?.id == expected, "\(ramGiB) GB, AI \(ai)")
        }
    }

    /// The recommendation is a chat model: never Apple Intelligence (which
    /// fits everywhere at 0 GB) and never a speech-to-text entry.
    @Test func recommendationIsAlwaysALocalChatModel() throws {
        // The 3 GB class is left out on purpose: nothing chat-capable clears
        // the margin there, so the 1.1 recommender falls back to Whisper Base.
        for device in Self.allDeviceClasses() where device.totalRAM > UInt64(3 * Self.gib) {
            let model = try #require(try recommend(device).recommended)
            #expect(model.backend != .appleIntelligence, "\(model.id)")
            #expect(model.capabilities.contains(.textIn), "\(model.id)")
            #expect(!model.capabilities.contains(.audioIn), "\(model.id)")
        }
    }

    // MARK: - Catalog content

    /// The recommender's choice rests on exactly these two being the top tier.
    @Test func onlyGemma4E2BAndLFM2_2_6BAreExcellent() throws {
        let excellent = Set(try bundledModels().filter { $0.quality == .excellent }.map(\.id))
        #expect(excellent == [Self.gemma4E2B, Self.lfm2_2_6B])
    }

    @Test func legacyModelsAreNeverRecommended() throws {
        for device in Self.allDeviceClasses() {
            let result = try recommend(device)
            let pick = result.recommended?.id ?? "none"
            #expect(!Self.legacyIDs.contains(pick),
                    "\(pick) recommended at \(device.totalRAMFormatted), AI \(device.appleIntelligence)")
        }
    }

    @Test func legacyModelsAreGoodAndLastInBothCopies() throws {
        for (name, models) in [("catalog.json", try bundledModels()), ("builtIn", ModelCatalog.builtIn)] {
            #expect(models.suffix(Self.legacyIDs.count).map(\.id) == Self.legacyIDs, "\(name)")
            for model in models where Self.legacyIDs.contains(model.id) {
                #expect(model.quality == .good, "\(name): \(model.id)")
            }
        }
    }

    // MARK: - The "deprecated" key

    /// Raw `models` array of the bundled file, before any decoding.
    private func rawEntries() throws -> [[String: Any]] {
        let url = try #require(CatalogFile.bundledURL)
        let object = try JSONSerialization.jsonObject(with: try Data(contentsOf: url))
        let document = try #require(object as? [String: Any])
        return try #require(document["models"] as? [[String: Any]])
    }

    /// The legacy entries carry `"deprecated": true` in both copies, and
    /// the decoded flag is what keeps them last and out of the
    /// recommendation in 1.2. (Branch catalog/post-1.1 pins the other half:
    /// the 1.1 decoder skips the key.)
    @Test func legacyEntriesAreFlaggedDeprecatedInBothCopies() throws {
        let raw = try rawEntries()
        let flagged = raw.filter { ($0["deprecated"] as? Bool) == true }.compactMap { $0["id"] as? String }
        #expect(flagged == Self.legacyIDs)
        #expect(ModelCatalog.builtIn.filter(\.deprecated).map(\.id) == Self.legacyIDs)
        #expect(try bundledModels().filter(\.deprecated).map(\.id) == Self.legacyIDs)
    }

    /// Same check as `BundledCatalogTests.bundledCatalogMatchesBuiltIn`,
    /// kept next to the pins: they are only meaningful while the copy the
    /// recommender reads here equals the compiled-in fallback.
    @Test func bothCatalogCopiesMatch() throws {
        #expect(try bundledModels() == ModelCatalog.builtIn)
    }
}
