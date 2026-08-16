//
//  LlamaCppDeviceTests.swift
//

import Testing
@testable import local_ai_test

@Suite(.tags(.device, .llamaCpp))
struct LlamaCppDeviceTests {

    private let modelId = "bartowski/Llama-3.2-3B-Instruct-GGUF"

    @Test @MainActor func generatesResponseFromGGUF() async throws {
        guard let model = ModelCatalog.model(id: modelId) else { return }
        guard InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("llama.cpp model not installed; skipping")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("llama.cpp load failed: \(InferenceManager.shared.state)")
            return
        }

        var out = ""
        for try await output in InferenceManager.shared.generate(
            turns: [.user("What is 2+2?")], parameters: .deterministic
        ) {
            if case .textDelta(let s) = output { out += s }
            if out.count > 40 { break }
        }
        #expect(out.contains("4"))
    }
}
