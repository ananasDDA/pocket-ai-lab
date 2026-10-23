//
//  LlamaCppVisionPromptTests.swift
//
//  Unit tests for `VisionPromptBuilder`, the pure-Swift step that turns a
//  conversation into the (prompt, images) pair libmtmd expects. It has no
//  dependency on the `llama` module, so these run on every target.
//

import Testing
import Foundation
@testable import local_ai_test

struct LlamaCppVisionPromptTests {

    private let bigBudget = 8192

    // MARK: - Markers

    @Test func oneMarkerPerImage() {
        let prepared = VisionPromptBuilder.build(
            turns: [.user("Describe the pictures.", attachments: [
                .image(Data([0x01])), .image(Data([0x02]))
            ])],
            family: .gemma,
            budgetTokens: bigBudget
        )
        let occurrences = prepared.prompt.components(separatedBy: "<__media__>").count - 1
        #expect(occurrences == 2)
        #expect(prepared.images == [Data([0x01]), Data([0x02])])
    }

    @Test func markersSitInsideTheUserTurnNotTheAssistantOpen() {
        let prepared = VisionPromptBuilder.build(
            turns: [.user("Describe.", attachments: [.image(Data([0x01]))])],
            family: .gemma,
            budgetTokens: bigBudget
        )
        let markerIdx = prepared.prompt.range(of: "<__media__>")?.lowerBound
        let assistantIdx = prepared.prompt.range(
            of: "<start_of_turn>model\n",
            options: .backwards
        )?.lowerBound
        #expect(markerIdx != nil && assistantIdx != nil)
        if let m = markerIdx, let a = assistantIdx {
            #expect(m < a)
        }
    }

    @Test func noMarkersWithoutAttachments() {
        let prepared = VisionPromptBuilder.build(
            turns: [.user("just text")],
            family: .plain,
            budgetTokens: bigBudget
        )
        #expect(!prepared.prompt.contains("<__media__>"))
        #expect(prepared.images.isEmpty)
    }

    @Test func markerCountMatchesImageCountForEveryFamily() {
        let families: [PromptFamily] = [.llama3, .llama2, .gemma, .gemma4, .qwen, .chatml, .mistral, .phi, .plain]
        for family in families {
            let prepared = VisionPromptBuilder.build(
                turns: [
                    .system("You are a helpful vision model."),
                    .user("Describe.", attachments: [
                        .image(Data([0x01])), .image(Data([0x02])), .image(Data([0x03]))
                    ])
                ],
                family: family,
                budgetTokens: bigBudget
            )
            let occurrences = prepared.prompt.components(separatedBy: "<__media__>").count - 1
            #expect(occurrences == 3, "family: \(family.rawValue)")
            #expect(prepared.images.count == 3, "family: \(family.rawValue)")
        }
    }

    // MARK: - Media replay

    // The regression behind the 1.0.1 report: the KV cache is rebuilt from
    // the prompt on every turn, so an image that is only in an earlier turn
    // has to be replayed or the model can no longer see it.
    @Test func imagesFromEarlierTurnsAreReplayed() {
        let first = Data([0xAA])
        let second = Data([0xBB])
        let prepared = VisionPromptBuilder.build(
            turns: [
                .user("What is this?", attachments: [.image(first)]),
                .assistant("A car."),
                .user("And this one?", attachments: [.image(second)])
            ],
            family: .gemma,
            budgetTokens: bigBudget
        )
        #expect(prepared.images == [first, second])
        let occurrences = prepared.prompt.components(separatedBy: "<__media__>").count - 1
        #expect(occurrences == 2)
    }

    @Test func audioIsSkippedAndVideoFramesBecomeImages() {
        let frame1 = Data([0xFF, 0xD8, 0x01])
        let frame2 = Data([0xFF, 0xD8, 0x02])
        let jpeg = Data([0xFF, 0xD8, 0xFF])
        let prepared = VisionPromptBuilder.build(
            turns: [.user("mixed", attachments: [
                .audio(Data(count: 10), sampleRate: 16000),
                .videoFrames([frame1, frame2], durationSeconds: 1),
                .image(jpeg)
            ])],
            family: .gemma,
            budgetTokens: bigBudget
        )
        #expect(prepared.images == [frame1, frame2, jpeg])
        let occurrences = prepared.prompt.components(separatedBy: "<__media__>").count - 1
        #expect(occurrences == 3)
    }

    // MARK: - Window fitting

    @Test func oldestTurnsAreDroppedUntilThePromptFits() {
        var turns: [ChatTurn] = [.system("sys")]
        for i in 0..<20 {
            turns.append(.user("question \(i) " + String(repeating: "x", count: 400)))
            turns.append(.assistant("answer \(i) " + String(repeating: "y", count: 400)))
        }
        turns.append(.user("the actual question"))

        let (kept, dropped) = VisionPromptBuilder.fit(turns: turns, budgetTokens: 1000)
        #expect(dropped > 0)
        let estimate = kept.reduce(0) { $0 + VisionPromptBuilder.estimatedTokens(of: $1) }
        #expect(estimate <= 1000)
        // The instructions and the question being asked always survive.
        #expect(kept.first?.role == .system)
        #expect(kept.last?.content == "the actual question")
    }

    @Test func nothingIsDroppedWhenTheConversationFits() {
        let turns: [ChatTurn] = [.user("hi"), .assistant("hello"), .user("how are you?")]
        let (kept, dropped) = VisionPromptBuilder.fit(turns: turns, budgetTokens: 8192)
        #expect(dropped == 0)
        #expect(kept.count == turns.count)
    }

    @Test func imagesDominateTheTokenEstimate() {
        let withImage = ChatTurn.user("hi", attachments: [.image(Data([0x01]))])
        let textOnly = ChatTurn.user("hi")
        #expect(VisionPromptBuilder.estimatedTokens(of: withImage)
                >= VisionPromptBuilder.estimatedTokens(of: textOnly) + 256)
    }

    // A single turn bigger than the whole budget is handed over untouched:
    // llama.cpp then reports a window it cannot serve, which is the truth,
    // rather than us silently dropping the question itself.
    @Test func theLastTurnIsNeverDropped() {
        let huge = ChatTurn.user(String(repeating: "x", count: 100_000))
        let (kept, _) = VisionPromptBuilder.fit(turns: [huge], budgetTokens: 100)
        #expect(kept.count == 1)
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

    // MARK: - Media cap

    // Every image in the prompt is re-encoded on every turn, so a long
    // photo thread must not grow without bound.
    @Test func onlyTheNewestImagesAreReplayed() {
        var turns: [ChatTurn] = []
        for i in 0..<6 {
            turns.append(.user("page \(i)", attachments: [.image(Data([UInt8(i)]))]))
            turns.append(.assistant("parsed \(i)"))
        }
        let capped = VisionPromptBuilder.capMedia(turns: turns, limit: 4)
        let replayed = capped.flatMap(\.attachments)
        #expect(replayed.count == 4)
        // The newest four, oldest two dropped.
        #expect(capped.compactMap { turn -> UInt8? in
            guard case .image(let d)? = turn.attachments.first else { return nil }
            return d.first
        } == [2, 3, 4, 5])
        // The text of the stripped turns survives.
        #expect(capped.count == turns.count)
        #expect(capped.first?.content == "page 0")
    }

    @Test func aShortThreadKeepsEveryImage() {
        let turns: [ChatTurn] = [
            .user("a", attachments: [.image(Data([0x01]))]),
            .assistant("ok"),
            .user("b", attachments: [.image(Data([0x02]))])
        ]
        let capped = VisionPromptBuilder.capMedia(turns: turns, limit: 4)
        #expect(capped.flatMap(\.attachments).count == 2)
    }

    // A video expands into frames, and the frames count against the cap.
    @Test func videoFramesCountTowardsTheCap() {
        let frames = (0..<8).map { Data([UInt8($0)]) }
        let turns: [ChatTurn] = [
            .user("clip", attachments: [.videoFrames(frames, durationSeconds: 4)]),
            .assistant("ok"),
            .user("photo", attachments: [.image(Data([0xFF]))])
        ]
        let capped = VisionPromptBuilder.capMedia(turns: turns, limit: 4)
        // The clip does not fit whole, so it goes whole; the photo stays.
        #expect(capped.flatMap(\.attachments).count == 1)
        #expect(capped.last?.attachments.count == 1)
    }

    @Test func markerCountMatchesTheCappedImages() {
        var turns: [ChatTurn] = []
        for i in 0..<6 {
            turns.append(.user("page \(i)", attachments: [.image(Data([UInt8(i)]))]))
            turns.append(.assistant("parsed \(i)"))
        }
        let prepared = VisionPromptBuilder.build(turns: turns, family: .ernie, budgetTokens: 8192)
        let markers = prepared.prompt.components(separatedBy: "<__media__>").count - 1
        #expect(markers == prepared.images.count)
        #expect(prepared.images.count == VisionPromptBuilder.maxReplayedImages)
    }
}
