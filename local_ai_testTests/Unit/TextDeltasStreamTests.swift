//
//  TextDeltasStreamTests.swift
//
//  Verifies `AsyncThrowingStream<ChatOutput>.textDeltas()` drops non-text
//  events and propagates errors.
//

import Testing
import Foundation
@testable import local_ai_test

struct TextDeltasStreamTests {

    @Test func filtersOnlyTextDeltas() async throws {
        let source = AsyncThrowingStream<ChatOutput, Error> { cont in
            cont.yield(.diagnostic(.firstTokenLatency(ms: 50)))
            cont.yield(.textDelta("Hello"))
            cont.yield(.audioChunk(Data(), sampleRate: 24_000, channels: 1))
            cont.yield(.textDelta(" world"))
            cont.yield(.diagnostic(.finished(tokensGenerated: 2, tokensPerSecond: 10)))
            cont.finish()
        }
        var collected = ""
        for try await chunk in source.textDeltas() {
            collected += chunk
        }
        #expect(collected == "Hello world")
    }

    @Test func propagatesErrors() async {
        let source = AsyncThrowingStream<ChatOutput, Error> { cont in
            cont.yield(.textDelta("partial"))
            cont.finish(throwing: InferenceError.generationFailed("boom"))
        }
        var collected = ""
        var caught: Error?
        do {
            for try await chunk in source.textDeltas() { collected += chunk }
        } catch {
            caught = error
        }
        #expect(collected == "partial")
        #expect(caught != nil)
    }
}
