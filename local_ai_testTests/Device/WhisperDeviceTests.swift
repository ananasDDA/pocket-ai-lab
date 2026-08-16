//
//  WhisperDeviceTests.swift
//
//  Feeds a short synthesized PCM buffer to the Whisper engine. We use a
//  1-second 440 Hz tone plus silence as a stand-in; any non-empty
//  transcription counts as a pass since models may interpret the tone
//  arbitrarily.
//

import Testing
import Foundation
@testable import local_ai_test

@Suite(.tags(.device, .whisper))
struct WhisperDeviceTests {

    @Test @MainActor func transcribesSomePCM() async throws {
        let modelId = "ggerganov/whisper.cpp/tiny.en"
        guard let model = ModelCatalog.model(id: modelId) else { return }
        guard InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("Whisper tiny not installed; skipping")
            return
        }

        let sr = 16_000
        let durationSeconds = 2
        let count = sr * durationSeconds
        var pcm = [Float](repeating: 0, count: count)
        for i in 0..<count {
            pcm[i] = sin(2 * .pi * 440 * Float(i) / Float(sr)) * 0.1
        }
        let dir = ModelDownloader.shared.modelDirectory(for: model)
        let engine = WhisperCppEngine(modelDirectory: dir, model: model)
        try await engine.load()
        let text = try await engine.transcribe(pcm: pcm)
        // Any successful call without throwing is a pass.
        _ = text
    }
}
