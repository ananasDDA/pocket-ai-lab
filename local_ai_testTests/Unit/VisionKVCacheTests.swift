//
//  VisionKVCacheTests.swift
//
//  The prefix arithmetic behind `LlamaVisionContext`'s KV-cache reuse,
//  exercised on hand-built chunk lists: no model, no `llama` module.
//

import Foundation
import Testing
@testable import local_ai_test

struct VisionKVCacheTests {

    // MARK: Fixtures

    private static let bos: Int32 = 2

    private func image(_ name: String, tokens: Int = 256, positions: Int? = nil) -> KVChunk {
        .media(KVMedia(id: name, tokens: tokens, positions: positions ?? tokens))
    }

    /// BOS + "user: <image>" opening, the image, then the question and the
    /// generation prompt - the shape libmtmd returns for a first turn.
    private var firstTurn: [KVChunk] {
        [.text([Self.bos, 10, 11]), image("cat"), .text([12, 13, 14, 15])]
    }

    // MARK: Matching

    @Test func identicalPromptsReuseAllButTheLastToken() {
        let total = KVPrefix.totalPositions(firstTurn)
        #expect(KVPrefix.commonPositions(firstTurn, firstTurn) == total)
        // The last position is decoded again so sampling has logits.
        #expect(KVPrefix.reusablePositions(cached: firstTurn, new: firstTurn) == total - 1)
    }

    // The usual follow-up: the cache holds the previous prompt plus the
    // reply as one text run; the new prompt re-tokenizes the reply and adds
    // the next question. Everything cached is reused, the image included.
    @Test func aFollowUpTurnReusesThePromptTheImageAndTheReply() {
        let reply: [Int32] = [40, 41, 42]
        let cached = KVPrefix.appending(reply, to: firstTurn)
        let next: [KVChunk] = [
            .text([Self.bos, 10, 11]), image("cat"),
            .text([12, 13, 14, 15] + reply + [99, 50, 51, 52, 15])
        ]
        let reusable = KVPrefix.reusablePositions(cached: cached, new: next)
        #expect(reusable == KVPrefix.totalPositions(cached))
        let cut = KVPrefix.cut(next, atPosition: reusable)
        #expect(cut == KVCut(chunk: 2, offset: 7))
        let reused = KVPrefix.prefix(next, positions: reusable) ?? []
        #expect(KVPrefix.mediaCount(reused) == 1)
    }

    @Test func textRunBoundariesDoNotMatter() {
        let cached: [KVChunk] = [.text([1, 2]), .text([3, 4])]
        let next: [KVChunk] = [.text([1, 2, 3, 4, 5])]
        #expect(KVPrefix.commonPositions(cached, next) == 4)
    }

    @Test func divergenceInTheMiddleStopsThePrefixThere() {
        let cached: [KVChunk] = [.text([Self.bos, 10, 11, 12, 13]), image("cat"), .text([20, 21])]
        let next: [KVChunk] = [.text([Self.bos, 10, 77, 12, 13]), image("cat"), .text([20, 21])]
        let reusable = KVPrefix.reusablePositions(cached: cached, new: next)
        #expect(reusable == 2)
        // Nothing after the divergence counts, the identical image included.
        #expect(KVPrefix.mediaCount(KVPrefix.prefix(next, positions: reusable) ?? []) == 0)
    }

    // A different picture at the same place in the conversation must not be
    // mistaken for the cached one: the cut lands right before it.
    @Test func anImageReplacedAtTheSamePositionIsNotReused() {
        let cached = KVPrefix.appending([40, 41], to: firstTurn)
        let next: [KVChunk] = [.text([Self.bos, 10, 11]), image("dog"), .text([12, 13, 14, 15, 40, 41, 50])]
        let reusable = KVPrefix.reusablePositions(cached: cached, new: next)
        #expect(reusable == 3)
        #expect(KVPrefix.cut(next, atPosition: reusable) == KVCut(chunk: 1, offset: 0))
    }

    @Test func aSameSizedImageWithAnotherIDDoesNotMatch() {
        let a: [KVChunk] = [.text([1]), image("a", tokens: 64)]
        let b: [KVChunk] = [.text([1]), image("b", tokens: 64)]
        #expect(KVPrefix.commonPositions(a, b) == 1)
    }

    // The vision builder drops the oldest turns once the transcript outgrows
    // the window. The new prompt then shares only the BOS with the cache.
    @Test func aTrimmedFrontOfTheConversationKeepsOnlyTheBOS() {
        let cached: [KVChunk] = [
            .text([Self.bos, 10, 11]), image("cat"), .text([12, 13, 40, 41, 60, 61, 62])
        ]
        let next: [KVChunk] = [.text([Self.bos, 60, 61, 62, 70, 71])]
        #expect(KVPrefix.reusablePositions(cached: cached, new: next) == 1)
    }

    @Test func nothingSharedReusesNothing() {
        #expect(KVPrefix.reusablePositions(cached: [.text([5, 6])], new: [.text([7, 8])]) == 0)
        #expect(KVPrefix.reusablePositions(cached: [], new: firstTurn) == 0)
    }

    // M-RoPE: an image spans fewer positions than it has tokens. The cut
    // after it is counted in positions, which is what llama_memory_seq_rm
    // takes.
    @Test func mropeImagesAreMeasuredInPositions() {
        let cached: [KVChunk] = [.text([Self.bos, 10]), image("cat", tokens: 300, positions: 20), .text([11, 12, 13])]
        let next: [KVChunk] = [.text([Self.bos, 10]), image("cat", tokens: 300, positions: 20), .text([11, 12, 99])]
        #expect(KVPrefix.reusablePositions(cached: cached, new: next) == 2 + 20 + 2)
        #expect(KVPrefix.totalTokens(next) == 2 + 300 + 3)
        #expect(KVPrefix.cut(next, atPosition: 24) == KVCut(chunk: 2, offset: 2))
    }

    @Test func aCutInsideAnImageIsRefused() {
        #expect(KVPrefix.cut(firstTurn, atPosition: 3 + 100) == nil)
        #expect(KVPrefix.prefix(firstTurn, positions: 3 + 100) == nil)
        #expect(KVPrefix.cut(firstTurn, atPosition: 3) == KVCut(chunk: 1, offset: 0))
        #expect(KVPrefix.cut(firstTurn, atPosition: 3 + 256) == KVCut(chunk: 2, offset: 0))
        let total = KVPrefix.totalPositions(firstTurn)
        #expect(KVPrefix.cut(firstTurn, atPosition: total) == KVCut(chunk: 3, offset: 0))
        #expect(KVPrefix.cut(firstTurn, atPosition: total + 1) == nil)
    }

    // A prompt that ends in an image (never sent by the app, but legal) is
    // re-evaluated from before that image when nothing else is new.
    @Test func aTrailingImageIsReevaluatedWhole() {
        let prompt: [KVChunk] = [.text([Self.bos, 10]), image("cat", tokens: 8)]
        #expect(KVPrefix.reusablePositions(cached: prompt, new: prompt) == 2)
    }

    @Test func prefixKeepsWholeChunksAndSplitsText() {
        let cut = KVPrefix.prefix(firstTurn, positions: 3 + 256 + 2)
        #expect(cut == [.text([Self.bos, 10, 11]), image("cat"), .text([12, 13])])
        #expect(KVPrefix.prefix(firstTurn, positions: 0) == [])
    }

    @Test func appendingMergesIntoATrailingTextRun() {
        #expect(KVPrefix.appending([7], to: [.text([1])]) == [.text([1, 7])])
        #expect(KVPrefix.appending([7], to: [image("x")]) == [image("x"), .text([7])])
        #expect(KVPrefix.appending([], to: [image("x")]) == [image("x")])
    }

    // MARK: Checkpoints

    @Test func theNewestCheckpointInsideThePrefixIsChosen() {
        #expect(KVPrefix.checkpoint(atOrBefore: 100, among: [40, 96, 120]) == 96)
        #expect(KVPrefix.checkpoint(atOrBefore: 96, among: [40, 96]) == 96)
        #expect(KVPrefix.checkpoint(atOrBefore: 30, among: [40, 96]) == nil)
        #expect(KVPrefix.checkpoint(atOrBefore: 30, among: [0]) == nil)
    }

    // Qwen 3 / 3.5: the prompt ends in "<|im_start|>assistant\n" plus the
    // four-token empty reasoning block; the snapshot goes right before it.
    @Test func theCheckpointGoesFourPositionsBeforeTheEnd() {
        let prompt: [KVChunk] = [.text([Self.bos, 10]), image("cat", tokens: 8), .text([11, 12, 13, 14, 15, 16])]
        let total = KVPrefix.totalPositions(prompt)
        #expect(KVPrefix.checkpointPosition(in: prompt, reused: 0, offsetFromEnd: 4) == total - 4)
        // Already inside the reused prefix: nothing to snapshot this turn.
        #expect(KVPrefix.checkpointPosition(in: prompt, reused: total - 4, offsetFromEnd: 4) == nil)
    }

    @Test func noCheckpointRightAfterOrInsideAnImage() {
        // The last text run is exactly four tokens: the target is the
        // boundary after the image.
        let atBoundary: [KVChunk] = [.text([Self.bos, 10]), image("cat", tokens: 8), .text([11, 12, 13, 14])]
        #expect(KVPrefix.checkpointPosition(in: atBoundary, reused: 0, offsetFromEnd: 4) == nil)
        let inside: [KVChunk] = [.text([Self.bos]), image("cat", tokens: 8), .text([11, 12])]
        #expect(KVPrefix.checkpointPosition(in: inside, reused: 0, offsetFromEnd: 4) == nil)
    }

    // MARK: Tokens and identity

    @Test func aDoubledLeadingBOSIsCollapsed() {
        #expect(KVPrefix.droppingDuplicateLeadingBOS([2, 2, 5], bos: 2) == [2, 5])
        #expect(KVPrefix.droppingDuplicateLeadingBOS([2, 5, 2], bos: 2) == [2, 5, 2])
        #expect(KVPrefix.droppingDuplicateLeadingBOS([2], bos: 2) == [2])
        #expect(KVPrefix.droppingDuplicateLeadingBOS([], bos: 2) == [])
    }

    @Test func mediaIDsFollowContentAndSize() {
        func id(_ bytes: [UInt8], _ w: UInt32, _ h: UInt32) -> String {
            bytes.withUnsafeBytes { KVPrefix.mediaID(width: w, height: h, bytes: $0) }
        }
        let pixels: [UInt8] = [1, 2, 3, 4, 5, 6]
        #expect(id(pixels, 2, 1) == id(pixels, 2, 1))
        #expect(id(pixels, 2, 1) != id(pixels, 1, 2))
        #expect(id(pixels, 2, 1) != id([1, 2, 3, 4, 5, 7], 2, 1))
        #expect(id(pixels, 2, 1).count == 64)
    }
    // MARK: Two conversations on one cache

    // What the first device run of the multi-turn suite measured: its two
    // tests ran in parallel through the one shared engine, so every turn
    // found the OTHER conversation in the cache and reused only the shared
    // opening (BOS, the user header and the first word), re-encoding its
    // photo each time. The same turns run one conversation at a time reuse
    // the whole previous prompt and the photo.
    @Test func alternatingConversationsShareOnlyTheirOpening() {
        let photoA = image("photo-a", tokens: 266)
        let photoB = image("photo-b", tokens: 256)
        func prompt(_ question: [Int32], _ photo: KVChunk, _ history: [Int32]) -> [KVChunk] {
            [.text([Self.bos, 100, 101, 102] + question), photo, .text([103, 104, 105] + history)]
        }
        let a1 = prompt([7, 8], photoA, [])
        let b1 = prompt([7, 9], photoB, [])
        let a2 = prompt([7, 8], photoA, [40, 41, 50, 51])
        let b2 = prompt([7, 9], photoB, [60, 61, 70, 71])

        // Interleaved: A1, B1, A2, B2 - each turn sees the other's cache.
        let cacheAfterB1 = KVPrefix.appending([60, 61], to: b1)
        let reuseA2 = KVPrefix.reusablePositions(cached: cacheAfterB1, new: a2)
        #expect(reuseA2 == 5)
        #expect(KVPrefix.mediaCount(KVPrefix.prefix(a2, positions: reuseA2) ?? []) == 0)
        let cacheAfterA2 = KVPrefix.appending([42], to: a2)
        #expect(KVPrefix.reusablePositions(cached: cacheAfterA2, new: b2) == 5)

        // One at a time: A2 follows A1 and reuses all of it, photo included.
        let cacheAfterA1 = KVPrefix.appending([40, 41], to: a1)
        let serial = KVPrefix.reusablePositions(cached: cacheAfterA1, new: a2)
        #expect(serial >= KVPrefix.totalPositions(a1))
        #expect(KVPrefix.mediaCount(KVPrefix.prefix(a2, positions: serial) ?? []) == 1)
    }
}
