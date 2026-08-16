//
//  HFMetadataStoreTests.swift
//  local_ai_testTests
//
//  Pure parts of the Hugging Face card metadata pipeline: the compact count
//  formatter, the cache TTL rule, the light `/api/models/{repo}` decoder and
//  the pipeline-tag presentation. No network.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - Compact counts

struct CompactCountTests {

    @Test func leavesSmallNumbersAlone() {
        #expect(HFMetadataStore.compactCount(0) == "0")
        #expect(HFMetadataStore.compactCount(7) == "7")
        #expect(HFMetadataStore.compactCount(999) == "999")
    }

    @Test func abbreviatesThousands() {
        #expect(HFMetadataStore.compactCount(1_000) == "1.0K")
        #expect(HFMetadataStore.compactCount(1_234) == "1.2K")
        #expect(HFMetadataStore.compactCount(999_999) == "1000.0K")
    }

    @Test func abbreviatesMillions() {
        #expect(HFMetadataStore.compactCount(1_000_000) == "1.0M")
        #expect(HFMetadataStore.compactCount(3_400_000) == "3.4M")
        #expect(HFMetadataStore.compactCount(12_345_678) == "12.3M")
    }

    @Test func handlesNegativesWithoutCrashing() {
        #expect(HFMetadataStore.compactCount(-5) == "-5")
    }
}

// MARK: - Cache TTL

struct MetadataCacheTTLTests {

    @Test func freshEntryIsNotExpired() {
        let now = Date()
        #expect(HFMetadataStore.isExpired(fetchedAt: now, now: now) == false)
        #expect(HFMetadataStore.isExpired(fetchedAt: now.addingTimeInterval(-60), now: now) == false)
    }

    @Test func expiresExactlyAtTheTTLBoundary() {
        let now = Date()
        let ttl = HFMetadataStore.ttl
        #expect(HFMetadataStore.isExpired(
            fetchedAt: now.addingTimeInterval(-ttl + 1), now: now) == false)
        #expect(HFMetadataStore.isExpired(fetchedAt: now.addingTimeInterval(-ttl), now: now))
        #expect(HFMetadataStore.isExpired(fetchedAt: now.addingTimeInterval(-ttl - 1), now: now))
    }

    @Test func defaultTTLIsADay() {
        #expect(HFMetadataStore.ttl == 24 * 60 * 60)
    }

    /// A timestamp in the future means the clock moved; such an entry would
    /// otherwise never age out.
    @Test func futureTimestampCountsAsExpired() {
        let now = Date()
        #expect(HFMetadataStore.isExpired(fetchedAt: now.addingTimeInterval(3_600), now: now))
    }

    @Test func honoursACustomTTL() {
        let now = Date()
        #expect(HFMetadataStore.isExpired(
            fetchedAt: now.addingTimeInterval(-30), now: now, ttl: 60) == false)
        #expect(HFMetadataStore.isExpired(
            fetchedAt: now.addingTimeInterval(-90), now: now, ttl: 60))
    }
}

// MARK: - Owner parsing

struct RepoOwnerTests {

    @Test func splitsOwnerFromRepo() {
        #expect(HFMetadataStore.owner(of: "mlx-community/gemma-2-2b-it-4bit") == "mlx-community")
        #expect(HFMetadataStore.owner(of: "apple/mistral-coreml") == "apple")
    }

    @Test func rejectsIncompleteIDs() {
        #expect(HFMetadataStore.owner(of: "") == nil)
        #expect(HFMetadataStore.owner(of: "owner") == nil)
        #expect(HFMetadataStore.owner(of: "owner/") == nil)
    }
}

// MARK: - Light metadata decoder

struct HFRepoStatsDecodingTests {

    private func decode(_ json: String) throws -> HFRepoStats {
        let data = Data(json.utf8)
        return try JSONDecoder().decode(HFRepoStats.self, from: data)
    }

    @Test func decodesAFullResponse() throws {
        let stats = try decode("""
        {
          "id": "mlx-community/Qwen2.5-3B-Instruct-4bit",
          "downloads": 12345,
          "likes": 67,
          "pipeline_tag": "text-generation",
          "lastModified": "2024-09-25T17:12:35.000Z"
        }
        """)
        #expect(stats.downloads == 12_345)
        #expect(stats.likes == 67)
        #expect(stats.pipelineTag == "text-generation")
        #expect(stats.lastModified != nil)
    }

    /// The API omits fields for plenty of repos; a missing counter must not
    /// cost the card its whole metadata line.
    @Test func toleratesMissingFields() throws {
        let stats = try decode("""
        { "id": "someone/repo" }
        """)
        #expect(stats.downloads == 0)
        #expect(stats.likes == 0)
        #expect(stats.pipelineTag == nil)
        #expect(stats.lastModified == nil)
    }

    @Test func ignoresUnknownFields() throws {
        let stats = try decode("""
        {
          "downloads": 5,
          "likes": 1,
          "siblings": [{"rfilename": "model.safetensors"}],
          "cardData": {"license": "apache-2.0"}
        }
        """)
        #expect(stats.downloads == 5)
        #expect(stats.likes == 1)
    }

    @Test func parsesTimestampsWithAndWithoutFractionalSeconds() throws {
        let withFraction = try decode("""
        { "lastModified": "2024-09-25T17:12:35.000Z" }
        """)
        let withoutFraction = try decode("""
        { "lastModified": "2024-09-25T17:12:35Z" }
        """)
        #expect(withFraction.lastModified == withoutFraction.lastModified)
        #expect(withFraction.lastModified?.timeIntervalSince1970 == 1_727_284_355)
    }

    @Test func dropsUnparseableTimestamps() throws {
        let stats = try decode("""
        { "lastModified": "yesterday" }
        """)
        #expect(stats.lastModified == nil)
    }

    /// A repo body large enough to matter must not stall the card: the decoder
    /// simply ignores `siblings`, so the light endpoint stays light.
    @Test func equatableComparesAllFields() {
        let base = HFRepoStats(downloads: 1, likes: 2, pipelineTag: "text-generation")
        #expect(base == HFRepoStats(downloads: 1, likes: 2, pipelineTag: "text-generation"))
        #expect(base != HFRepoStats(downloads: 1, likes: 3, pipelineTag: "text-generation"))
    }
}

// MARK: - Pipeline tag presentation

struct PipelineTagTests {

    @Test func humanisesTheTag() {
        #expect(HFModelHeader.pipelineLabel("text-generation") == "Text Generation")
        #expect(HFModelHeader.pipelineLabel("automatic-speech-recognition")
                == "Automatic Speech Recognition")
        #expect(HFModelHeader.pipelineLabel("text_to_speech") == "Text To Speech")
    }

    @Test func mapsKnownTagsToSymbols() {
        #expect(HFModelHeader.pipelineSymbol("text-generation") == "text.alignleft")
        #expect(HFModelHeader.pipelineSymbol("image-text-to-text") == "photo.on.rectangle")
        #expect(HFModelHeader.pipelineSymbol("automatic-speech-recognition") == "waveform")
        #expect(HFModelHeader.pipelineSymbol("text-to-speech") == "speaker.wave.2")
    }

    @Test func fallsBackForUnknownTags() {
        #expect(HFModelHeader.pipelineSymbol("reinforcement-learning") == "sparkles")
        #expect(HFModelHeader.pipelineSymbol("") == "sparkles")
    }
}
