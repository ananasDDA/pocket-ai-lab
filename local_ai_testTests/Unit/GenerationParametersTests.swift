//
//  GenerationParametersTests.swift
//

import Testing
@testable import local_ai_test

struct GenerationParametersTests {

    @Test func defaultIsSane() {
        let p = GenerationParameters.default
        #expect(p.temperature >= 0 && p.temperature <= 2)
        #expect(p.topP > 0 && p.topP <= 1)
        #expect(p.topK >= 1)
        #expect(p.maxTokens > 0)
    }

    @Test func deterministicHasZeroTemp() {
        #expect(GenerationParameters.deterministic.temperature == 0)
        #expect(GenerationParameters.deterministic.topK == 1)
        #expect(GenerationParameters.deterministic.seed != nil)
    }
}
