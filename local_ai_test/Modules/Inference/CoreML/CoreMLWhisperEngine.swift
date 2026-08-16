//
//  CoreMLWhisperEngine.swift
//  local_ai_test
//
//  Thin front-end that routes to `WhisperCppEngine` (which itself can use a
//  Core ML encoder when a `*-encoder.mlmodelc` sits next to the ggml weights).
//  The split exists purely for catalog clarity - some Apple repos publish
//  "Whisper Core ML" bundles separately from ggml weights.
//

import Foundation

final class CoreMLWhisperEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .speechToText

    private let modelDirectory: URL
    private let inner: WhisperCppEngine

    var isLoaded: Bool { inner.isLoaded }

    init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
        let synthetic = AIModel(
            id: "coreml-whisper",
            name: "Whisper",
            family: "Whisper",
            parameterSize: "varies",
            quantization: "ggml",
            backend: .whisperCpp,
            capabilities: .speechToText,
            fileLayout: .whisperGGML,
            ramRequiredGB: 0.5,
            diskSizeGB: 0.2,
            huggingFaceRepo: "",
            contextLength: 0,
            quality: .good
        )
        self.inner = WhisperCppEngine(modelDirectory: modelDirectory, model: synthetic)
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        try await inner.load(progress: progress)
    }

    func unload() async { await inner.unload() }
    func resetConversation() async { await inner.resetConversation() }

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        inner.generate(turns: turns, parameters: parameters)
    }

    func transcribe(pcm: [Float]) async throws -> String {
        try await inner.transcribe(pcm: pcm)
    }
}
