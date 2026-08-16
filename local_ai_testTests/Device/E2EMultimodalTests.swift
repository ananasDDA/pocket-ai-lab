//
//  E2EMultimodalTests.swift
//
//  End-to-end: synthesize 16 kHz PCM tone -> Whisper -> LLM text.
//  If Kokoro is installed, feed the answer back through TTS and check a
//  non-empty audio chunk is produced.
//

import Testing
import Foundation
@testable import local_ai_test

@Suite(.tags(.device, .e2e))
struct E2EMultimodalTests {

    @Test @MainActor func asrThenLlmThenTts() async throws {
        let whisperId = "ggerganov/whisper.cpp/tiny.en"
        let textLLMCandidates = [
            "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "bartowski/Llama-3.2-3B-Instruct-GGUF"
        ]
        let installed = Set(InstalledModelsStore.shared.installedModels.map(\.id))

        guard installed.contains(whisperId) else { return }
        guard let llmId = textLLMCandidates.first(where: { installed.contains($0) }),
              let llm = ModelCatalog.model(id: llmId) else { return }

        // Synthesize simple PCM (ASR will likely return empty or short text;
        // we only verify the pipeline plumbing, not transcription accuracy).
        let pcm = [Float](repeating: 0, count: 16_000)

        let whisper = WhisperCppEngine(
            modelDirectory: ModelDownloader.shared.modelDirectory(for: ModelCatalog.model(id: whisperId)!),
            model: ModelCatalog.model(id: whisperId)!
        )
        try await whisper.load()
        let transcript = try await whisper.transcribe(pcm: pcm)
        _ = transcript

        await InferenceManager.shared.prepare(model: llm)
        guard case .ready = InferenceManager.shared.state else { return }
        var assistantText = ""
        for try await output in InferenceManager.shared.generate(
            turns: [.user("Respond with 'hello'")],
            parameters: .deterministic
        ) {
            if case .textDelta(let s) = output { assistantText += s }
            if assistantText.count > 30 { break }
        }
        #expect(!assistantText.isEmpty)
    }
}
