//
//  InferenceManagerLifecycleTests.swift
//
//  Does not touch real models — focuses on state transitions and the
//  availableMemoryMB utility.
//

import Testing
import Foundation
@testable import local_ai_test

struct InferenceManagerLifecycleTests {

    @Test @MainActor func initialStateIsIdle() {
        #expect(InferenceManager.shared.state == .idle)
    }

    @Test @MainActor func availableMemoryReportsSomething() {
        // `os_proc_available_memory()` returns 0 or -1 in the iOS Simulator and
        // for macCatalyst-style hosts, so we only verify the function doesn't
        // crash and returns a defined sentinel/positive value.
        let mem = InferenceManager.availableMemoryMB()
        #expect(mem == -1 || mem > 0)
    }

    @Test @MainActor func unloadingFromIdleIsSafe() async {
        await InferenceManager.shared.unloadCurrent()
        #expect(InferenceManager.shared.state == .idle)
    }

    @Test @MainActor func resetConversationFromIdleIsSafe() async {
        await InferenceManager.shared.resetConversation()
    }

    @Test @MainActor func generateBeforeLoadThrows() async {
        await InferenceManager.shared.unloadCurrent()
        let stream = InferenceManager.shared.generate(turns: [.user("hi")])
        do {
            for try await _ in stream {}
            Issue.record("Expected modelNotLoaded error")
        } catch let error as InferenceError {
            switch error {
            case .modelNotLoaded: break
            default: Issue.record("Wrong error: \(error)")
            }
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
    }
}
