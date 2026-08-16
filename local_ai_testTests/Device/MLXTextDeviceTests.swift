//
//  MLXTextDeviceTests.swift
//
//  Smoke tests for the text-only MLX backend. Require Llama 3.2 3B installed.
//

import Testing
@testable import local_ai_test

@Suite(.tags(.device, .mlx))
struct MLXTextDeviceTests {

    private let modelId = "mlx-community/Llama-3.2-3B-Instruct-4bit"

    @Test @MainActor func loadsAndGeneratesNonEmpty() async throws {
        guard let model = ModelCatalog.model(id: modelId) else { Issue.record("Model not in catalog"); return }
        guard InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("MLX Llama 3.2 3B not installed on this device; skipping")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("Prepare failed: \(InferenceManager.shared.state)")
            return
        }

        var collected = ""
        let stream = InferenceManager.shared.generate(
            turns: [.user("Say hi in one word.")],
            parameters: .deterministic
        )
        for try await output in stream {
            if case .textDelta(let s) = output { collected += s }
            if collected.count > 80 { break }
        }
        #expect(!collected.isEmpty)
    }

    @Test @MainActor func multiTurnContextWorks() async throws {
        guard let model = ModelCatalog.model(id: modelId) else { return }
        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else { return }

        let turns: [ChatTurn] = [
            .system("You are terse."),
            .user("The magic word is 'pineapple'. Remember it."),
            .assistant("Got it."),
            .user("What was the magic word?")
        ]

        var out = ""
        for try await output in InferenceManager.shared.generate(turns: turns, parameters: .deterministic) {
            if case .textDelta(let s) = output { out += s }
            if out.count > 200 { break }
        }
        #expect(out.lowercased().contains("pineapple"))
    }
}
