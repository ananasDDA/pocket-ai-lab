//
//  CoreMLContextBudgetTests.swift
//  local_ai_testTests
//
//  `CoreMLTextEngine` has to fit a prompt into a fixed context window, and the
//  shipped Core ML entry's window is 512 tokens - small enough that a couple
//  of chat turns reach it. Trimming the prompt to "just barely fits" leaves
//  the reply a single token, which reads to a user as a model that answers
//  with one word and then stops. These lock the reservation in.
//
//  The helper takes only integers, so the rule is testable without a model,
//  a tokenizer or a device.
//

import Testing
@testable import local_ai_test

struct CoreMLContextBudgetTests {

    /// The regression: a 600-token prompt on the 512-token entry used to be
    /// trimmed to 511, leaving room for exactly one generated token.
    @Test func trimLeavesRoomForAnActualReply() {
        let prompt = Array(1...600)
        let trimmed = CoreMLTextEngine.trimToContext(
            prompt, contextLimit: 512, maxNewTokens: 128, bosTokenId: nil
        )
        #expect(trimmed.count == 512 - 128)
        #expect(512 - trimmed.count >= 128, "reply budget must survive the trim")
    }

    /// The reservation is capped at half the window, so the 1024-token default
    /// reply budget cannot swallow a 512-token context whole.
    @Test func reservationNeverExceedsHalfTheWindow() {
        let trimmed = CoreMLTextEngine.trimToContext(
            Array(1...5_000), contextLimit: 512, maxNewTokens: 1_024, bosTokenId: nil
        )
        #expect(trimmed.count == 256)
    }

    /// A modest reply budget is honoured as-is rather than rounded up to half.
    @Test func smallReplyBudgetKeepsMoreOfThePrompt() {
        let trimmed = CoreMLTextEngine.trimToContext(
            Array(1...5_000), contextLimit: 4_096, maxNewTokens: 256, bosTokenId: nil
        )
        #expect(trimmed.count == 4_096 - 256)
    }

    @Test func aPromptThatAlreadyFitsIsUntouched() {
        let prompt = Array(1...100)
        let trimmed = CoreMLTextEngine.trimToContext(
            prompt, contextLimit: 512, maxNewTokens: 128, bosTokenId: nil
        )
        #expect(trimmed == prompt)
    }

    /// The oldest history is what goes; the newest turn is what the user just
    /// typed and must survive.
    @Test func trimDropsTheOldestTokens() {
        let trimmed = CoreMLTextEngine.trimToContext(
            Array(1...600), contextLimit: 512, maxNewTokens: 128, bosTokenId: nil
        )
        #expect(trimmed.last == 600)
        #expect(trimmed.first == 600 - (512 - 128) + 1)
    }

    /// Chat templates put BOS first and models trained with it degrade without
    /// it, so the trimmed sequence is re-rooted on it instead of starting mid
    /// history.
    @Test func leadingBOSSurvivesTheTrim() {
        let prompt = [1] + Array(2...600)
        let trimmed = CoreMLTextEngine.trimToContext(
            prompt, contextLimit: 512, maxNewTokens: 128, bosTokenId: 1
        )
        #expect(trimmed.count == 512 - 128)
        #expect(trimmed.first == 1)
        #expect(trimmed.last == 600)
    }

    /// No BOS to preserve when the prompt does not start with one.
    @Test func noBOSIsInventedWhenThePromptHasNone() {
        let trimmed = CoreMLTextEngine.trimToContext(
            Array(2...600), contextLimit: 512, maxNewTokens: 128, bosTokenId: 1
        )
        #expect(trimmed.first != 1)
    }

    /// Degenerate windows must still produce a non-empty prompt rather than
    /// trapping on a negative `suffix`.
    @Test func degenerateContextLimitsStayInBounds() {
        #expect(CoreMLTextEngine.trimToContext(
            Array(1...10), contextLimit: 1, maxNewTokens: 128, bosTokenId: nil
        ).count == 1)
        #expect(CoreMLTextEngine.trimToContext(
            Array(1...10), contextLimit: 0, maxNewTokens: 128, bosTokenId: nil
        ).count == 1)
        #expect(CoreMLTextEngine.trimToContext(
            Array(1...10), contextLimit: 512, maxNewTokens: 0, bosTokenId: nil
        ).count == 10)
    }
}
