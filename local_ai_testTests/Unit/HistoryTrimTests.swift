//
//  HistoryTrimTests.swift
//
//  How the llama.cpp engines cut an overlong conversation down to the
//  window: whole exchanges, oldest first, in batches; the system turn and
//  the question always stay; a short first exchange is kept; a single
//  oversized turn loses its middle but keeps its role header. Measured with
//  a stand-in tokenizer (one token per whitespace-separated word) over the
//  app's own fallback templates, so no model is needed.
//

import Foundation
import Testing
@testable import local_ai_test

struct HistoryTrimTests {

    // MARK: Fixtures

    private static func words(_ n: Int, _ tag: String) -> String {
        (0..<n).map { "\(tag)\($0)" }.joined(separator: " ")
    }

    private static func count(_ text: String) -> Int {
        text.split(whereSeparator: { $0.isWhitespace }).count
    }

    private static func fit(
        _ turns: [ChatTurn],
        room: Int,
        family: PromptFamily = .chatml
    ) async -> HistoryTrim.Fit {
        await HistoryTrim.fit(
            turns: turns,
            room: room,
            count: { count($0) },
            measure: { count(PromptFormatter.format(turns: $0, family: family)) }
        )
    }

    private static let system = ChatTurn.system("You are Lumi, a concise and friendly assistant.")

    /// System, a short first exchange with facts, then `n` long exchanges
    /// and a final question.
    private static func conversation(exchanges n: Int, wordsPerTurn: Int = 60) -> [ChatTurn] {
        var turns: [ChatTurn] = [
            system,
            .user("My name is Priya and my cat is Ziggy."),
            .assistant("Nice to meet you, Priya and Ziggy!")
        ]
        for i in 0..<n {
            turns.append(.user("q\(i) " + words(wordsPerTurn, "u\(i)-")))
            turns.append(.assistant("a\(i) " + words(wordsPerTurn, "a\(i)-")))
        }
        turns.append(.user("What is my cat called?"))
        return turns
    }

    // MARK: What always stays

    @Test func theSystemTurnAndTheQuestionAlwaysStay() async {
        let turns = Self.conversation(exchanges: 20)
        let fit = await Self.fit(turns, room: 400)
        #expect(fit.droppedTurns > 0)
        #expect(fit.kept.first == Self.system)
        #expect(fit.kept.last == turns.last)
        #expect(fit.promptTokens <= 400)
        #expect(fit.truncatedTokens == 0)
    }

    @Test func nothingChangesWhenTheConversationFits() async {
        let turns = Self.conversation(exchanges: 2)
        let fit = await Self.fit(turns, room: 4000)
        #expect(fit.kept == turns)
        #expect(fit.droppedTurns == 0)
        #expect(fit.firstUserTurnIntact)
    }

    // The facts users come back to: a short first exchange is kept while
    // the exchanges after it go.
    @Test func aShortFirstExchangeIsKept() async {
        let turns = Self.conversation(exchanges: 20)
        let fit = await Self.fit(turns, room: 400)
        #expect(fit.kept[1] == turns[1])
        #expect(fit.kept[2] == turns[2])
        #expect(fit.firstUserTurnIntact)
    }

    @Test func aLongFirstExchangeGoesFirst() async {
        var turns = Self.conversation(exchanges: 20)
        turns[1] = .user("Read this: " + Self.words(150, "doc-"))
        let fit = await Self.fit(turns, room: 400)
        #expect(!fit.kept.contains(turns[1]))
        #expect(!fit.firstUserTurnIntact)
        #expect(fit.kept.first == Self.system)
    }

    // MARK: Batches

    @Test func theScheduleDropsOldestFirstInBatches() {
        // Room 200: batches of 50, pinning up to 50. Overflow 60 rounds up
        // to 100, which the oldest unpinned exchanges cover at the fourth.
        let lists = HistoryTrim.schedule(
            unitCosts: [10, 30, 30, 30, 30, 30, 30],
            firstUnitIsExchange: true,
            room: 200,
            total: 260
        )
        #expect(lists == [[1, 2, 3, 4], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 0]])
        // Overflow 90 rounds to the same 100: the same exchanges go.
        let grown = HistoryTrim.schedule(
            unitCosts: [10, 30, 30, 30, 30, 30, 30],
            firstUnitIsExchange: true,
            room: 200,
            total: 290
        )
        #expect(grown.first == [1, 2, 3, 4])
        // Too big to pin: the first exchange is just the oldest one.
        let unpinned = HistoryTrim.schedule(
            unitCosts: [80, 30, 30], firstUnitIsExchange: true, room: 200, total: 230
        )
        #expect(unpinned == [[0], [0, 1], [0, 1, 2]])
        // Nothing to drop when it fits.
        #expect(HistoryTrim.schedule(unitCosts: [80], firstUnitIsExchange: true, room: 200, total: 150) == [[]])
    }

    @Test func droppedExchangesAreTheOldestAndAtLeastABatch() async {
        let turns = Self.conversation(exchanges: 20)
        let room = 400
        let fit = await Self.fit(turns, room: room)
        // Survivors after the pinned first exchange are one contiguous run
        // of the newest exchanges.
        let survivors = fit.kept.compactMap { turn -> Int? in
            guard turn.role == .user, turn.content.hasPrefix("q") else { return nil }
            return Int(turn.content.dropFirst().prefix { $0.isNumber })
        }
        #expect(!survivors.isEmpty)
        #expect(survivors == Array(survivors.first!...19))
        let dropped = Self.count(PromptFormatter.format(turns: turns, family: .chatml)) - fit.promptTokens
        #expect(dropped >= room / HistoryTrim.batchDivisor)
    }

    // The point of batching: once a batch is gone, the next turns keep the
    // same start, so the KV prefix cache keeps hitting instead of missing
    // on every message as a "just enough" trim would.
    @Test func thePrefixStaysStableAcrossConsecutiveTurns() async {
        // Batches of 400 tokens; every exchange adds about 100.
        let room = 1600
        var turns: [ChatTurn] = [Self.system, .user("My name is Priya."), .assistant("Hi Priya!")]
        var previousKept: [ChatTurn]?
        var previousDropped = 0
        var stableTurns = 0
        var changes = 0
        var dropEvents: [Int] = []
        for i in 0..<60 {
            turns.append(.user("q\(i) " + Self.words(40, "u\(i)-")))
            let fit = await Self.fit(turns, room: room)
            #expect(fit.promptTokens <= room)
            if let previousKept {
                if fit.droppedTurns == previousDropped {
                    // Same start: the previous prompt's turns are a prefix
                    // of this one's (the reply and the question are new).
                    #expect(Array(fit.kept.prefix(previousKept.count)) == previousKept, "turn \(i)")
                    stableTurns += 1
                } else {
                    #expect(fit.droppedTurns > previousDropped, "turn \(i): drops never shrink")
                    changes += 1
                    dropEvents.append(i)
                }
            }
            previousKept = fit.kept + [.assistant("a\(i) " + Self.words(60, "a\(i)-"))]
            previousDropped = fit.droppedTurns
            turns.append(previousKept!.last!)
        }
        #expect(changes > 0)
        #expect(stableTurns >= 2 * changes, "\(changes) drop events, \(stableTurns) stable turns")
        // No two consecutive turns both moved the start.
        for (a, b) in zip(dropEvents, dropEvents.dropFirst()) {
            #expect(b - a >= 2, "drops on turns \(a) and \(b)")
        }
    }

    // MARK: Cutting inside a turn

    @Test func aSingleOversizedTurnLosesItsMiddleAndKeepsItsHeader() async {
        let article = "START " + Self.words(3000, "w") + " Summarise this in five bullets."
        let turns: [ChatTurn] = [Self.system, .user(article)]
        let fit = await Self.fit(turns, room: 500)
        #expect(fit.promptTokens <= 500)
        #expect(fit.truncatedTokens > 2000)
        #expect(fit.droppedTurns == 0)
        #expect(!fit.firstUserTurnIntact)
        let rendered = PromptFormatter.format(turns: fit.kept, family: .chatml)
        #expect(rendered.contains("<|im_start|>system\n" + Self.system.content))
        #expect(rendered.contains("<|im_start|>user\nSTART "))
        #expect(rendered.contains("[...]"))
        #expect(rendered.contains("Summarise this in five bullets.<|im_end|>"))
    }

    @Test func truncatingMiddleKeepsStartAndEnd() {
        let text = "abcdefghijklmnopqrstuvwxyz"
        let cut = HistoryTrim.truncatingMiddle(text, keepingCharacters: 10)
        #expect(cut.hasPrefix("abcdef"))
        #expect(cut.hasSuffix("wxyz"))
        #expect(cut.contains("[...]"))
        #expect(HistoryTrim.truncatingMiddle(text, keepingCharacters: 100) == text)
    }

    // MARK: Gemma folding

    /// The first rendered Gemma turn up to the user's own text: the labelled
    /// instructions (`SystemPrompt.foldedIntoFirstUser`).
    private static var gemmaFoldedOpening: String {
        "<start_of_turn>user\n" + SystemPrompt.foldedInstructionsHeader + "\n" + system.content
            + "\n\n" + SystemPrompt.foldedMessageHeader + "\n"
    }

    // Gemma-family templates have no system role and fold the system text
    // into the first user turn they render. Dropping whole exchanges never
    // loses it: it moves to whichever user turn is first now.
    @Test func gemmaFoldedSystemTextSurvivesTheTrim() async {
        var turns = Self.conversation(exchanges: 20)
        turns[1] = .user("Read this: " + Self.words(150, "doc-"))   // not pinned: it goes
        let fit = await Self.fit(turns, room: 400, family: .gemma)
        #expect(!fit.kept.contains(turns[1]))
        let rendered = PromptFormatter.format(turns: fit.kept, family: .gemma)
        #expect(rendered.hasPrefix(Self.gemmaFoldedOpening))
        #expect(rendered.contains("What is my cat called?"))
    }

    @Test func gemmaFoldedSystemTextSurvivesACut() async {
        let turns: [ChatTurn] = [Self.system, .user("START " + Self.words(2000, "w") + " END")]
        let fit = await Self.fit(turns, room: 300, family: .gemma)
        let rendered = PromptFormatter.format(turns: fit.kept, family: .gemma)
        #expect(fit.truncatedTokens > 0)
        #expect(rendered.hasPrefix(Self.gemmaFoldedOpening + "START "))
    }

    // MARK: Layout

    @Test func layoutProtectsSystemTurnsAndTheTail() {
        let turns: [ChatTurn] = [
            .system("s"), .user("u0"), .assistant("a0"), .assistant("a0b"),
            .system("mid"), .user("u1"), .assistant("a1"), .user("u2")
        ]
        let layout = HistoryTrim.layout(turns)
        #expect(layout.units == [1..<4, 5..<7])
        #expect(layout.firstUserIndex == 1)
        #expect(layout.kept(dropping: [0, 1]) == [.system("s"), .system("mid"), .user("u2")])
        #expect(layout.dropsFirstUserTurn([0]))
        #expect(!layout.dropsFirstUserTurn([1]))
    }

    // MARK: Vision estimate path

    @Test func theEstimatedFitNeverCutsInsideATurn() {
        let huge = ChatTurn.user(String(repeating: "x", count: 100_000))
        let fit = HistoryTrim.fitEstimated(turns: [Self.system, huge], room: 100, cost: { $0.content.count / 4 })
        #expect(fit.kept == [Self.system, huge])
        #expect(fit.truncatedTokens == 0)
    }
}
