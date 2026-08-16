//
//  BackendSwitchStressTests.swift
//
//  Rapidly switches between installed models. Verifies that RAM returns to
//  a baseline after each unload (i.e. the engine chain cleans up after
//  itself and MLX.GPU.clearCache() is effective).
//

import Testing
@testable import local_ai_test

@Suite(.tags(.device, .stress))
struct BackendSwitchStressTests {

    @Test @MainActor func rapidSwitchDoesNotLeak() async throws {
        let installed = InstalledModelsStore.shared.installedModels
        let pool = installed.prefix(3).map { $0 }
        guard pool.count >= 2 else {
            Issue.record("Need at least 2 installed models to run switch stress")
            return
        }

        let baseline = InferenceManager.availableMemoryMB()
        #expect(baseline > 0)

        for round in 0..<5 {
            for model in pool {
                await InferenceManager.shared.prepare(model: model)
                guard case .ready = InferenceManager.shared.state else {
                    Issue.record("Prepare failed for \(model.id) in round \(round)")
                    return
                }
                await InferenceManager.shared.unloadCurrent()
                try await Task.sleep(nanoseconds: 500_000_000)
            }
        }

        let final = InferenceManager.availableMemoryMB()
        let delta = baseline - final
        // Allow some drift (20 MB) for OS caches, but not more than 150 MB.
        #expect(delta < 150, "Memory leaked by \(delta) MB after switch stress")
    }
}
