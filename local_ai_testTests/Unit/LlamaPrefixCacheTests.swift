//
//  LlamaPrefixCacheTests.swift
//
//  The prompt-prefix arithmetic behind `LlamaContext`'s KV-cache reuse.
//  Guarded on the module: the vendored XCFramework has a device slice only,
//  so a simulator build has no `llama` and no `llama_token`.
//

#if canImport(llama)

import Testing
import llama
@testable import local_ai_test

struct LlamaPrefixCacheTests {

    @Test func identicalSequencesShareEverything() {
        let tokens: [llama_token] = [1, 2, 3, 4]
        #expect(LlamaContext.commonPrefixLength(tokens, tokens) == 4)
    }

    // The usual chat shape: the next turn is the previous transcript plus
    // the reply and the new question, so the whole cache is reusable.
    @Test func anExtendedSequenceReusesTheWholePrefix() {
        let cached: [llama_token] = [1, 2, 3]
        let next: [llama_token] = [1, 2, 3, 4, 5]
        #expect(LlamaContext.commonPrefixLength(cached, next) == 3)
    }

    @Test func divergenceStopsTheSharedPrefix() {
        let cached: [llama_token] = [1, 2, 3, 4]
        let next: [llama_token] = [1, 2, 9, 4]
        #expect(LlamaContext.commonPrefixLength(cached, next) == 2)
    }

    // What a trimmed conversation looks like: same BOS, different history.
    @Test func onlyTheBosSurvivesATrim() {
        let cached: [llama_token] = [1, 20, 21, 22]
        let next: [llama_token] = [1, 40, 41]
        #expect(LlamaContext.commonPrefixLength(cached, next) == 1)
    }

    @Test func anEmptyCacheSharesNothing() {
        #expect(LlamaContext.commonPrefixLength([], [1, 2, 3]) == 0)
        #expect(LlamaContext.commonPrefixLength([1, 2, 3], []) == 0)
    }
}

#endif
