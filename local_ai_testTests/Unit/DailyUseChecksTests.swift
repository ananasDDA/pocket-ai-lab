//
//  DailyUseChecksTests.swift
//
//  Pins the automatic checks of the daily-use device suite. They run on
//  transcripts from real models, so each case below is shaped like
//  something a small model actually writes: markdown, reasoning blocks,
//  code fences, a total repeated with a currency sign.
//

import Foundation
import Testing
@testable import local_ai_test

struct DailyUseChecksTests {

    // MARK: - Context

    private static let history: [ChatTurn] = [
        .user("Hi! My name is Priya and I live in Lisbon."),
        .assistant("Nice to meet you, Priya!"),
        .user("What is my name?")
    ]

    private static func report(
        droppedFront: Int = 0,
        droppedTurns: Int = 0,
        firstIntact: Bool = true
    ) -> PromptCacheReport {
        PromptCacheReport(
            promptTokens: 100, reusedTokens: 0, evaluatedTokens: 100,
            droppedFrontTokens: droppedFront, droppedTurns: droppedTurns,
            firstUserTurnIntact: firstIntact, rewind: .empty
        )
    }

    // The bug the grader found: the text engine trims tokens AFTER
    // rendering, so the rendered prompt still holds the first turn.
    @Test func firstTurnIsLostWhenTheEngineTrimmedTokensFromTheFront() {
        let rendered = "<|im_start|>user\nHi! My name is Priya and I live in Lisbon.<|im_end|>..."
        #expect(DailyUseChecks.firstTurnKept(
            turns: Self.history, renderedText: rendered, cache: Self.report(droppedFront: 812)
        ) == false)
    }

    @Test func firstTurnIsLostWhenTheEngineDroppedIt() {
        #expect(DailyUseChecks.firstTurnKept(
            turns: Self.history, renderedText: "What is my name?",
            cache: Self.report(droppedTurns: 2, firstIntact: false)
        ) == false)
    }

    // The engines keep a short first exchange while older middle turns go:
    // dropped turns alone do not mean the first one is gone.
    @Test func firstTurnIsKeptWhenOnlyMiddleTurnsWereDropped() {
        let rendered = "user\nHi! My name is Priya and I live in Lisbon. ... What is my name?"
        #expect(DailyUseChecks.firstTurnKept(
            turns: Self.history, renderedText: rendered, cache: Self.report(droppedTurns: 4)
        ) == true)
    }

    @Test func firstTurnIsNotIntactWhenItWasCutInside() {
        let rendered = "user\nHi! My name is Priya and I live in Lisbon.\n[...]\n..."
        #expect(DailyUseChecks.firstTurnKept(
            turns: Self.history, renderedText: rendered, cache: Self.report(firstIntact: false)
        ) == false)
    }

    @Test func aSystemTurnIsNotHistory() {
        let turns: [ChatTurn] = [.system("Be brief."), .user("Hello")]
        #expect(DailyUseChecks.firstTurnKept(turns: turns, renderedText: "Hello", cache: Self.report()) == nil)
    }

    @Test func firstTurnIsKeptWhenNothingWasCutAndTheTextHasIt() {
        let rendered = "<start_of_turn>user\nHi! My name is Priya and I live in Lisbon.<end_of_turn>..."
        #expect(DailyUseChecks.firstTurnKept(
            turns: Self.history, renderedText: rendered, cache: Self.report()
        ) == true)
        // Engines without a cache report fall back to the rendered text.
        #expect(DailyUseChecks.firstTurnKept(turns: Self.history, renderedText: rendered, cache: nil) == true)
        #expect(DailyUseChecks.firstTurnKept(turns: Self.history, renderedText: "other", cache: nil) == false)
    }

    @Test func firstTurnKeptIsUnknownWithoutHistoryOrEvidence() {
        #expect(DailyUseChecks.firstTurnKept(
            turns: [.user("Hello")], renderedText: "Hello", cache: Self.report()
        ) == nil)
        #expect(DailyUseChecks.firstTurnKept(turns: Self.history, renderedText: nil, cache: nil) == nil)
        #expect(DailyUseChecks.firstTurnKept(turns: Self.history, renderedText: nil, cache: Self.report()) == true)
    }

    // MARK: - Fact recall

    @Test func recallFindsAllThreeFacts() {
        let answer = "Of course! Your name is **Priya**, you live in Lisbon, and your cat is called Ziggy."
        let result = DailyUseChecks.recalledFacts(in: answer, facts: ["Priya", "Lisbon", "Ziggy"])
        #expect(result.score == 3)
    }

    @Test func recallIsCaseInsensitiveAndCountsPartialAnswers() {
        let answer = "you're PRIYA from lisbon! I'm afraid I don't remember your cat's name."
        let result = DailyUseChecks.recalledFacts(in: answer, facts: ["Priya", "Lisbon", "Ziggy"])
        #expect(result.score == 2)
        #expect(result.found["Ziggy"] == false)
    }

    @Test func recallDoesNotMatchInsideOtherWords() {
        let result = DailyUseChecks.recalledFacts(in: "I like bananas.", facts: ["Ana"])
        #expect(result.score == 0)
    }

    @Test func recallIgnoresTheReasoningBlock() {
        let answer = "<think>The user said Priya, Lisbon, Ziggy.</think>\n\nSorry, I don't know."
        let result = DailyUseChecks.recalledFacts(in: answer, facts: ["Priya", "Lisbon", "Ziggy"])
        #expect(result.score == 0)
        #expect(DailyUseChecks.startsWithReasoning(answer))
    }

    // MARK: - Bullets

    @Test func bulletCountReadsCommonListShapes() {
        let answer = """
        Here is a summary:

        - First point
        * Second point
        • Third point
          - Indented fourth point
        5. Numbered fifth point

        **Bold heading, not a bullet**
        Plain prose line.
        """
        #expect(DailyUseChecks.bulletCount(in: answer) == 5)
    }

    @Test func bulletCountReadsNumberedLists() {
        let answer = "1) One\n2) Two\n3. Three\n10. Ten"
        #expect(DailyUseChecks.bulletCount(in: answer) == 4)
    }

    @Test func bulletCountIgnoresAMarkdownRuleAndBoldLines() {
        #expect(DailyUseChecks.bulletCount(in: "---\n**Note:** text\n-no space") == 0)
    }

    // MARK: - Words

    @Test func wordCountSplitsOnWhitespace() {
        #expect(DailyUseChecks.wordCount(in: "Dear Mark,\n\nI hope you are well.") == 7)
        #expect(DailyUseChecks.wordCount(in: "   ") == 0)
    }

    // MARK: - JSON

    @Test func jsonInsideACodeFencePasses() {
        let answer = """
        Sure:
        ```json
        {"date": "2026-03-12", "time": "3:30 PM", "place": "Riverside Health Clinic", "people": ["Dr. Elena Ruiz", "Ana"]}
        ```
        """
        let check = DailyUseChecks.jsonKeys(in: answer, required: ["date", "time", "place", "people"])
        #expect(check.passed)
        #expect(check.values["people"]?.contains("ruiz") == true)
        #expect(check.values["time"] == "3:30 pm")
    }

    @Test func jsonWithAMissingKeyFails() {
        let check = DailyUseChecks.jsonKeys(
            in: #"{"date": "March 12", "time": "15:30", "location": "clinic", "people": "Ana"}"#,
            required: ["date", "time", "place", "people"]
        )
        #expect(check.parsed)
        #expect(check.missing == ["place"])
        #expect(!check.passed)
    }

    @Test func notJSONFails() {
        let check = DailyUseChecks.jsonKeys(in: "date: March 12", required: ["date"])
        #expect(!check.parsed)
        #expect(!check.passed)
    }

    // MARK: - Numbers

    @Test func billSplitFinalNumber() {
        let answer = "Total with tip: $84.00 x 1.18 = $99.12. Split four ways, each person pays $24.78."
        #expect(DailyUseChecks.finalNumber(in: answer) == 24.78)
        #expect(DailyUseChecks.containsNumber(99.12, in: answer))
        #expect(!DailyUseChecks.containsNumber(25, in: answer))
    }

    @Test func numbersReadThousandsSeparators() {
        #expect(DailyUseChecks.numbers(in: "It costs 1,200 dollars, not 12.") == [1200, 12])
    }

    // MARK: - Clock times

    @Test func clockTimesReadTwelveAndTwentyFourHourForms() {
        #expect(DailyUseChecks.clockTimes(in: "4:00 PM in London is 11:00 AM in New York") == [960, 660])
        #expect(DailyUseChecks.clockTimes(in: "at 11 a.m. or 4pm") == [660, 960])
        #expect(DailyUseChecks.clockTimes(in: "16:00 there, 11:00 here") == [960, 660])
        #expect(DailyUseChecks.clockTimes(in: "12 AM and 12 PM") == [0, 720])
        #expect(DailyUseChecks.clockTimes(in: "New York is 5 hours behind") == [])
    }

    @Test func clockAnswerRejectsAWrongTimeEvenNextToTheRightOne() {
        let allowed: Set<Int> = [960]
        #expect(DailyUseChecks.clockAnswerMatches(
            "The call is at 11:00 AM your time (4:00 PM in London).", expected: 660, allowedOthers: allowed
        ))
        #expect(!DailyUseChecks.clockAnswerMatches(
            "The call is at 9:00 PM for you.", expected: 660, allowedOthers: allowed
        ))
        #expect(!DailyUseChecks.clockAnswerMatches(
            "Either 11 AM or 11 PM.", expected: 660, allowedOthers: allowed
        ))
    }

    // MARK: - Language

    // The scenario's own prompts are the non-English samples, so this file
    // stays English-only and the detector is pinned on exactly what the
    // suite sends.
    @Test func everyNonEnglishPromptIsReadAsItsOwnLanguage() throws {
        let chats = try DailyFixtures.nonEnglishChats()
        #expect(Set(chats.map(\.language)).isSuperset(of: ["es", "de", "ja"]))
        for chat in chats {
            for turn in chat.turns {
                let guess = DailyUseChecks.detectLanguage(turn)
                #expect(guess.language == chat.language, "\(chat.name): \(guess.detail)")
            }
        }
    }

    // The translate scenario's failure: the model repeats the paragraph
    // in English, which the old non-empty check passed.
    @Test func anEnglishEchoIsEnglish() {
        let echo = "Here is the translation: The train to the coast leaves every morning at eight. "
            + "I usually buy a coffee at the station before I get on."
        #expect(DailyUseChecks.detectLanguage(echo).language == "en")
    }

    @Test func anEnglishPrefaceDoesNotOutweighTheTranslation() throws {
        let spanish = try #require(try DailyFixtures.nonEnglishChats().first { $0.language == "es" })
        let answer = "Sure! " + spanish.turns.joined(separator: " ")
        #expect(DailyUseChecks.detectLanguage(answer).language == "es")
    }

    @Test func scriptsDecideBeforeWords() {
        func text(_ range: ClosedRange<UInt32>) -> String {
            String(String.UnicodeScalarView(range.compactMap(Unicode.Scalar.init)))
        }
        let cyrillic = text(0x430...0x44F)
        let han = text(0x4E00...0x4E1F)
        let hiragana = text(0x3042...0x3060)
        #expect(DailyUseChecks.detectLanguage(cyrillic + " " + cyrillic).language == "ru")
        #expect(DailyUseChecks.detectLanguage(han).language == "zh")
        #expect(DailyUseChecks.detectLanguage(han + hiragana).language == "ja")
        #expect(DailyUseChecks.detectLanguage("12 + 30 = 42").language == nil)
        #expect(DailyUseChecks.detectLanguage("Okay.").language == nil)
    }

    // MARK: - Formatting

    @Test func markdownTableWithTheAskedColumns() throws {
        let answer = """
        Here is a comparison:

        | **Option** | Cost (per month) | Time | Pros |
        |:-----------|-----------------:|------|------|
        | Bicycle    | ~5 EUR           | 20 min | Exercise |
        | Bus        | 40 EUR           | 25 min | Relaxing |
        | Car        | 120 EUR          | 12 min | Fast |

        Hope this helps!
        """
        let table = try #require(DailyUseChecks.markdownTable(in: answer))
        #expect(table.rows.count == 3)
        #expect(DailyUseChecks.missingColumns(["Option", "Cost", "Time", "Pros"], in: table).isEmpty)
        #expect(DailyUseChecks.missingColumns(["Option", "Cons"], in: table) == ["Cons"])
    }

    @Test func aListIsNotATable() {
        #expect(DailyUseChecks.markdownTable(in: "- Bicycle: cheap\n- Bus: slow\n- Car: fast") == nil)
        // A header without a separator row is not a markdown table either.
        #expect(DailyUseChecks.markdownTable(in: "Option | Cost\nBike | 5") == nil)
    }

    @Test func sentenceCounting() {
        #expect(DailyUseChecks.sentenceCount(
            in: "A password manager stores your passwords in an encrypted vault and fills them in for you."
        ) == 1)
        #expect(DailyUseChecks.sentenceCount(in: "It stores passwords. It also creates strong ones!") == 2)
        #expect(DailyUseChecks.sentenceCount(in: "It costs 3.5 dollars a month") == 1)
        #expect(DailyUseChecks.sentenceCount(in: "**Password manager**\nIt keeps your logins safe.") == 2)
        #expect(DailyUseChecks.sentenceCount(in: "<think>Short.</think>\nOne sentence here.") == 1)
    }

    @Test func numberedStepsInCommonShapes() {
        let plain = "1. Fill a pot with water.\n2. Bring it to a boil.\n3) Lower the egg in.\n4. Wait 9 minutes."
        #expect(DailyUseChecks.numberedSteps(in: plain) == [1, 2, 3, 4])
        #expect(DailyUseChecks.stepsAreSequential(DailyUseChecks.numberedSteps(in: plain)))

        let bold = "**Step 1:** Boil water.\n\n**Step 2:** Add the egg.\n\n### 3. Cool it."
        #expect(DailyUseChecks.numberedSteps(in: bold) == [1, 2, 3])

        #expect(!DailyUseChecks.stepsAreSequential([1, 2, 2, 3]))
        #expect(!DailyUseChecks.stepsAreSequential([1, 2]))
        #expect(!DailyUseChecks.stepsAreSequential(DailyUseChecks.numberedSteps(in: "- Boil\n- Add\n- Cool")))
    }

    // MARK: - Money

    @Test func euroAmountsEitherSideOfTheNumber() {
        let checklist = """
        - Hostel: €25/night x 2 = €50
        - Food: 20-30 euros a day
        - Metro pass: EUR 15
        - Total: about 280 €
        """
        #expect(DailyUseChecks.amounts(in: checklist, currency: .euro) == [25, 50, 20, 30, 15, 280])
        #expect(DailyUseChecks.amounts(in: "Total: €310 (a little over)", currency: .euro).max() == 310)
        #expect(DailyUseChecks.amounts(in: "Lunch 12,50 €", currency: .euro) == [12.5])
        #expect(DailyUseChecks.amounts(in: "It costs $40.", currency: .euro).isEmpty)
    }

    @Test func dollarAmounts() {
        #expect(DailyUseChecks.amounts(in: "Rent is $1,450 and the fee $75.", currency: .dollar) == [1450, 75])
        #expect(DailyUseChecks.amounts(in: "Parking costs 120 dollars a month.", currency: .dollar) == [120])
    }

    // MARK: - Diet

    @Test func meatSuggestionsIgnoreRuledOutMeat() {
        #expect(DailyUseChecks.meatSuggestions(
            in: "Try a vegetarian francesinha, skip the bacalhau, and order the meat-free soup."
        ).isEmpty)
        #expect(DailyUseChecks.meatSuggestions(in: "Lunch: a bifana at a local cafe (3 euros).") == ["bifana"])
        #expect(DailyUseChecks.meatSuggestions(in: "Dinner: grilled sardines by the river.") == ["sardines"])
        #expect(DailyUseChecks.mentioned(DailyUseChecks.vegetarianWords, in: "Plant-based bowl") == ["plant-based"])
    }

    // MARK: - Grounding

    @Test func notStatedWording() {
        #expect(DailyUseChecks.saysNotStated("The lease doesn't mention parking at all."))
        #expect(DailyUseChecks.saysNotStated("Parking is not specified in the lease, so ask the landlord."))
        #expect(DailyUseChecks.saysNotStated("There is no information about parking in this lease."))
        #expect(!DailyUseChecks.saysNotStated("Yes, one parking space is included for $100 per month."))
    }

    @Test func admittingNotKnowing() {
        #expect(DailyUseChecks.admitsNotKnowing(
            "I'm not familiar with a paper by that title. It may not exist, or it may be very obscure."
        ))
        #expect(DailyUseChecks.admitsNotKnowing("I couldn’t find any information about that phone."))
        #expect(DailyUseChecks.admitsNotKnowing("I don't have information about Dr. Okonkwo-Lindqvist."))
        #expect(!DailyUseChecks.admitsNotKnowing(
            "The Nokia Lumina X12 Pro has a 108 MP main camera and a 5,000 mAh battery."
        ))
    }

    // MARK: - Dates and days

    @Test func datesInCommonForms() {
        for text in ["April 1, 2027", "1 April 2027", "Apr 1st", "the 1st of April", "2027-04-01",
                     "01/04/2027", "04/01/2027"] {
            #expect(DailyUseChecks.mentionsDate(text, month: 4, day: 1), "\(text)")
        }
        #expect(!DailyUseChecks.mentionsDate("April 10, 2027", month: 4, day: 1))
        #expect(!DailyUseChecks.mentionsDate("March 1, 2027", month: 4, day: 1))
        #expect(DailyUseChecks.mentionsDate("It ends on March 31, 2028.", month: 3, day: 31))
    }

    // The grader's case: the right clock time on the wrong day.
    @Test func dayShiftIsCaught() {
        #expect(DailyUseChecks.mentionsDayShift("The call is at 11:00 AM (previous day) for you."))
        #expect(DailyUseChecks.mentionsDayShift("11 AM tomorrow"))
        #expect(!DailyUseChecks.mentionsDayShift("The call is at 11:00 AM your time."))
        #expect(!DailyUseChecks.mentionsDayShift("11:00 AM on the same day, not the previous day."))
    }

    // MARK: - Code

    @Test func dedupeOrderStaticCheck() {
        let fromKeys = "```python\ndef dedupe(items):\n    \"\"\"Keep order.\"\"\"\n    return list(dict.fromkeys(items))\n```"
        #expect(DailyUseChecks.dedupeKeepsOrder(fromKeys).passed)

        let loop = """
        ```python
        def remove_duplicates(items):
            seen = set()
            result = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        ```
        """
        #expect(DailyUseChecks.dedupeKeepsOrder(loop).passed)

        let comprehension = "def dedupe(xs):\n    seen = set()\n    return [x for x in xs if not (x in seen or seen.add(x))]"
        #expect(DailyUseChecks.dedupeKeepsOrder(comprehension).passed)

        let setOnly = "def dedupe(items):\n    return list(set(items))"
        #expect(!DailyUseChecks.dedupeKeepsOrder(setOnly).passed)

        let stringKeys = """
        def dedupe(items):
            seen = set()
            out = []
            for item in items:
                if str(item) not in seen:
                    seen.add(str(item))
                    out.append(item)
            return out
        """
        let check = DailyUseChecks.dedupeKeepsOrder(stringKeys)
        #expect(!check.passed)
        #expect(check.reason.contains("str(item)"))

        #expect(!DailyUseChecks.dedupeKeepsOrder("Use dict.fromkeys on the list.").passed)
    }

    // MARK: - Text and errors

    @Test func mentionsIgnoresCaseAndPunctuation() {
        #expect(DailyUseChecks.mentions("Maple Street Grocery", in: "The shop is MAPLE STREET GROCERY."))
        #expect(!DailyUseChecks.mentions("Maple Street", in: "Maple St."))
    }

    @Test func errorKindsFromEngineMessages() {
        #expect(DailyUseChecks.errorKind(
            "llama.cpp: the conversation no longer fits the 8192-token context window"
        ) == "context_full")
        #expect(DailyUseChecks.errorKind("Not enough free memory for vision inference") == "out_of_memory")
        #expect(DailyUseChecks.errorKind("llama.cpp: decode(token) failed") == "engine_error")
        #expect(DailyUseChecks.errorKind("Generation failed: guardrailViolation") == "guardrail")
        #expect(DailyUseChecks.errorKind("unsupportedLanguageOrLocale") == "unsupported_language")
    }
}
