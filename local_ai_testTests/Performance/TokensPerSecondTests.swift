//
//  TokensPerSecondTests.swift
//
//  Records baseline tok/s for each (model, backend) pair. Uses
//  XCTestMetric-style measurement via a hand-rolled timer since
//  swift-testing doesn't yet have an official perf-metric API.
//

import Testing
import Foundation
@testable import local_ai_test

@Suite(.tags(.device, .performance))
struct TokensPerSecondTests {

    @Test @MainActor func measureLlama32_3B_MLX() async throws {
        try await measure(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit",
                          tag: "MLX/Llama-3.2-3B")
    }

    @Test @MainActor func measureLlama32_3B_GGUF() async throws {
        try await measure(modelId: "bartowski/Llama-3.2-3B-Instruct-GGUF",
                          tag: "llama.cpp/Llama-3.2-3B")
    }

    @Test @MainActor func measureGemma3_4B_Vision() async throws {
        try await measure(modelId: "mlx-community/gemma-3-4b-it-4bit",
                          tag: "MLX/Gemma-3-4B")
    }

    // MARK: - Helper

    @MainActor
    private func measure(modelId: String, tag: String) async throws {
        guard let model = ModelCatalog.model(id: modelId),
              InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("\(tag): model not installed, skipping")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else { return }

        // Warm up
        _ = try await drainFirst(n: 8, prompt: "Write 1 word.")

        let start = Date()
        let tokens = try await drainFirst(n: 128, prompt: "Write 30 words about iOS.")
        let elapsed = Date().timeIntervalSince(start)
        let tps = Double(tokens) / elapsed

        print("[PERF] \(tag): \(String(format: "%.2f", tps)) tok/s over \(tokens) tokens")
        #expect(tps > 0.1, "Unreasonable tok/s \(tps) for \(tag)")
    }

    @MainActor
    private func drainFirst(n: Int, prompt: String) async throws -> Int {
        var tokens = 0
        for try await output in InferenceManager.shared.generate(
            turns: [.user(prompt)],
            parameters: GenerationParameters(
                temperature: 0.7, topP: 0.9, topK: 40, repetitionPenalty: 1.1,
                maxTokens: n, seed: nil
            )
        ) {
            if case .textDelta = output { tokens += 1 }
            if tokens >= n { break }
        }
        return tokens
    }
}
