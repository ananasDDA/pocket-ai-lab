//
//  LlamaCppVisionPromptTests.swift
//
//  Unit tests for the pure-Swift helpers that glue `ChatTurn`/
//  `PromptFormatter` output into libmtmd's `<__media__>` marker protocol.
//  These tests run on every target (even without the `llama` module linked)
//  because both helpers have a fallback stub implementation in the `#else`
//  branch of LlamaCppVisionEngine.swift.
//

import Testing
import Foundation
@testable import local_ai_test

struct LlamaCppVisionPromptTests {

    // MARK: - Marker injection

    @Test func mediaMarkerInjectionInsertsOneMarkerPerImage() {
        let base = PromptFormatter.format(
            turns: [.user("Describe the pictures.")],
            family: .gemma
        )
        let out = LlamaCppVisionEngine.injectMediaMarkers(
            into: base,
            family: .gemma,
            count: 2
        )
        let occurrences = out.components(separatedBy: "<__media__>").count - 1
        #expect(occurrences == 2)
    }

    @Test func mediaMarkerInjectionPlacesMarkersBeforeAssistantOpen() {
        let base = PromptFormatter.format(
            turns: [.user("Describe.")],
            family: .gemma
        )
        let out = LlamaCppVisionEngine.injectMediaMarkers(
            into: base,
            family: .gemma,
            count: 1
        )
        // Marker must sit before the <start_of_turn>model\n sentinel.
        let markerIdx = out.range(of: "<__media__>")?.lowerBound
        let assistantIdx = out.range(of: "<start_of_turn>model\n", options: .backwards)?.lowerBound
        #expect(markerIdx != nil && assistantIdx != nil)
        if let m = markerIdx, let a = assistantIdx {
            #expect(m < a)
        }
    }

    @Test func mediaMarkerInjectionNoopForZeroImages() {
        let base = "irrelevant"
        let out = LlamaCppVisionEngine.injectMediaMarkers(
            into: base,
            family: .plain,
            count: 0
        )
        #expect(out == base)
    }

    // Smoke-test every supported chat family: the helper must (a) inject
    // exactly `count` markers, and (b) not corrupt the assistant-open tag
    // by inserting inside/after it.
    @Test func mediaMarkerInjectionWorksForAllFamilies() {
        let families: [PromptFamily] = [.llama3, .llama2, .gemma, .qwen, .chatml, .mistral, .phi, .plain]
        for family in families {
            let prompt = PromptFormatter.format(
                turns: [.user("Describe."), .system("You are a helpful vision model.")],
                family: family
            )
            let out = LlamaCppVisionEngine.injectMediaMarkers(
                into: prompt,
                family: family,
                count: 3
            )
            let occurrences = out.components(separatedBy: "<__media__>").count - 1
            #expect(occurrences == 3, "family: \(family.rawValue)")
        }
    }

    // MARK: - Attachment extraction

    @Test func extractsImageDataFromLastUserTurnOnly() {
        let oldImage = Data(repeating: 0xAA, count: 16)
        let newImage1 = Data(repeating: 0xBB, count: 32)
        let newImage2 = Data(repeating: 0xCC, count: 64)

        let turns: [ChatTurn] = [
            .user("old", attachments: [.image(oldImage)]),
            .assistant("ok"),
            .user("new", attachments: [.image(newImage1), .image(newImage2)])
        ]
        let extracted = LlamaCppVisionEngine.extractImages(from: turns)
        #expect(extracted == [newImage1, newImage2])
    }

    @Test func extractsEmptyWhenLastTurnHasNoImages() {
        let turns: [ChatTurn] = [
            .user("just text")
        ]
        #expect(LlamaCppVisionEngine.extractImages(from: turns).isEmpty)
    }

    @Test func extractsIgnoresAudioButKeepsVideoFrames() {
        let jpeg = Data([0xFF, 0xD8, 0xFF])
        let frame1 = Data([0xFF, 0xD8, 0x01])
        let frame2 = Data([0xFF, 0xD8, 0x02])
        let turns: [ChatTurn] = [
            .user("mixed", attachments: [
                .audio(Data(count: 10), sampleRate: 16000),
                .videoFrames([frame1, frame2], durationSeconds: 1),
                .image(jpeg)
            ])
        ]
        // Video frames are flattened into the image list (libmtmd has no
        // video chunk type — each frame becomes a separate <__media__>).
        // Audio is still skipped (needs mtmd_bitmap_init_from_audio, TBD).
        let extracted = LlamaCppVisionEngine.extractImages(from: turns)
        #expect(extracted == [frame1, frame2, jpeg])
    }

    @Test func videoAttachmentExpandsIntoMultipleImages() {
        let frames = (0..<8).map { Data([0xFF, 0xD8, UInt8($0)]) }
        let turns: [ChatTurn] = [
            .user("describe", attachments: [.videoFrames(frames, durationSeconds: 10)])
        ]
        let extracted = LlamaCppVisionEngine.extractImages(from: turns)
        #expect(extracted == frames)
    }

    @Test func extractsEmptyWhenNoUserTurns() {
        let turns: [ChatTurn] = [.system("sys"), .assistant("asst")]
        #expect(LlamaCppVisionEngine.extractImages(from: turns).isEmpty)
    }

    // MARK: - Availability

    // Gated on the compile-time flag: on a build without the `llama`
    // module linked, `isLinked` is expected to be false. Once the
    // vendored XCFramework is linked, it should be true and
    // `unavailableReason` should be nil.
    @Test func backendAvailabilityTracksLlamaModule() {
        if BackendAvailability.hasLlama {
            #expect(ModelBackend.llamaCppVision.isLinked)
            #expect(ModelBackend.llamaCppVision.unavailableReason == nil)
        } else {
            #expect(!ModelBackend.llamaCppVision.isLinked)
            #expect(ModelBackend.llamaCppVision.unavailableReason != nil)
        }
    }
}
