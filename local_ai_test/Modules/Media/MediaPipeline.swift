//
//  MediaPipeline.swift
//  local_ai_test
//
//  Glue that connects microphone -> Whisper -> LLM -> Kokoro TTS so the UI
//  layer can call a single high-level async function to do voice chat.
//
//  The pipeline owns two companion engines (ASR and TTS) independently from
//  the main chat engine in InferenceManager. It lazily downloads + loads
//  the defaults when first used.
//

import Foundation

@MainActor
@Observable
final class MediaPipeline {

    static let shared = MediaPipeline()

    private(set) var transcript: String = ""
    private(set) var isTranscribing = false
    private(set) var isSpeaking = false

    @ObservationIgnored
    private var whisperEngine: WhisperCppEngine?
    @ObservationIgnored
    private var ttsEngine: CoreMLKokoroEngine?
    @ObservationIgnored
    private var speakAssistantAloud = false

    var speakAssistantEnabled: Bool {
        get { speakAssistantAloud }
        set { speakAssistantAloud = newValue }
    }

    private init() {}

    // MARK: - ASR

    /// Transcribes a blob of 16 kHz mono f32 PCM captured by AudioRecorder.
    func transcribe(pcm: [Float], using model: AIModel) async throws -> String {
        isTranscribing = true
        defer { isTranscribing = false }

        if whisperEngine == nil {
            let dir = ModelDownloader.shared.modelDirectory(for: model)
            whisperEngine = WhisperCppEngine(modelDirectory: dir, model: model)
            try await whisperEngine?.load()
        }
        guard let engine = whisperEngine else { throw InferenceError.modelNotLoaded }
        let text = try await engine.transcribe(pcm: pcm)
        self.transcript = text
        return text
    }

    // MARK: - TTS

    /// Speaks a single sentence through the Kokoro engine. Caller is expected
    /// to chunk text at sentence boundaries before calling.
    func speak(_ text: String, using model: AIModel) async throws {
        isSpeaking = true
        defer { isSpeaking = false }

        if ttsEngine == nil {
            let dir = ModelDownloader.shared.modelDirectory(for: model)
            ttsEngine = CoreMLKokoroEngine(modelDirectory: dir)
            try await ttsEngine?.load()
        }
        guard let tts = ttsEngine else { throw InferenceError.modelNotLoaded }

        let pcmData = try await tts.synthesize(text: text)
        AudioPlayer.shared.enqueue(data: pcmData, sampleRate: 24_000, channels: 1)
    }

    /// Pipes an assistant stream into the TTS engine. Splits on sentence
    /// boundaries; each completed sentence is synthesized + enqueued to the
    /// player. The returned stream is a passthrough of `textDelta` events so
    /// the UI can still update the bubble in real time.
    func streamToSpeech(
        source: AsyncThrowingStream<ChatOutput, Error>,
        using model: AIModel
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                do {
                    var buffer = ""
                    for try await output in source {
                        continuation.yield(output)
                        if case .textDelta(let delta) = output {
                            buffer += delta
                            while let range = buffer.rangeOfCharacter(from: ["!", ".", "?", "\n"]) {
                                let sentence = String(buffer[..<range.upperBound])
                                    .trimmingCharacters(in: .whitespacesAndNewlines)
                                buffer = String(buffer[range.upperBound...])
                                if !sentence.isEmpty, self?.speakAssistantAloud == true {
                                    try? await self?.speak(sentence, using: model)
                                }
                            }
                        }
                    }
                    if !buffer.isEmpty, self?.speakAssistantAloud == true {
                        try? await self?.speak(buffer, using: model)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }
}
