//
//  CoreMLTextDeviceTests.swift
//

import Testing
@testable import local_ai_test

@Suite(.tags(.device, .coreML))
struct CoreMLTextDeviceTests {

    private let modelId = "apple/mistral-coreml"

    @Test @MainActor func generatesResponseFromCoreML() async throws {
        guard let model = ModelCatalog.model(id: modelId) else { return }
        guard InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("Core ML Mistral not installed; skipping")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("Core ML load failed: \(InferenceManager.shared.state)")
            return
        }

        var out = ""
        for try await output in InferenceManager.shared.generate(
            turns: [.user("Reply with the single word 'OK'.")],
            parameters: .deterministic
        ) {
            if case .textDelta(let s) = output { out += s }
            if out.count > 40 { break }
        }
        #expect(!out.isEmpty)
    }
}
