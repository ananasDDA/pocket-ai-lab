//
//  LoadTimeTests.swift
//
//  Measures cold vs warm load time. Core ML compile time is included in
//  the first cold load and not in subsequent warms.
//

import Testing
import Foundation
@testable import local_ai_test

@Suite(.tags(.device, .performance))
struct LoadTimeTests {

    @Test @MainActor func measureMLXColdWarm() async throws {
        try await measureLoad(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit", tag: "MLX/Llama-3.2-3B")
    }

    @Test @MainActor func measureLlamaCppColdWarm() async throws {
        try await measureLoad(modelId: "bartowski/Llama-3.2-3B-Instruct-GGUF", tag: "llama.cpp/Llama-3.2-3B")
    }

    @Test @MainActor func measureCoreMLWithCompile() async throws {
        try await measureLoad(modelId: "apple/mistral-coreml", tag: "CoreML/Mistral")
    }

    @MainActor
    private func measureLoad(modelId: String, tag: String) async throws {
        guard let model = ModelCatalog.model(id: modelId),
              InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("\(tag): not installed, skipping")
            return
        }

        await InferenceManager.shared.unloadCurrent()
        let coldStart = Date()
        await InferenceManager.shared.prepare(model: model)
        let coldElapsed = Date().timeIntervalSince(coldStart)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("\(tag): cold load failed: \(InferenceManager.shared.state)")
            return
        }

        await InferenceManager.shared.unloadCurrent()
        let warmStart = Date()
        await InferenceManager.shared.prepare(model: model)
        let warmElapsed = Date().timeIntervalSince(warmStart)

        print("[PERF] \(tag): cold=\(String(format: "%.2f", coldElapsed))s warm=\(String(format: "%.2f", warmElapsed))s")
        #expect(warmElapsed <= coldElapsed * 1.2)
    }
}
