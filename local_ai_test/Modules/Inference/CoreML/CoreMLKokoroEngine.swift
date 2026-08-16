//
//  CoreMLKokoroEngine.swift
//  local_ai_test
//
//  Text-to-speech via a Kokoro-82M Core ML bundle
//  (https://huggingface.co/hexgrad/Kokoro-82M + CoreML community ports).
//  Kokoro natively outputs 24 kHz mono f32 PCM.
//
//  Integration shape: the engine streams `ChatOutput.audioChunk` for each
//  synthesized sentence so the UI can start playback while the remainder is
//  still synthesizing.
//

import Foundation
import CoreML

final class CoreMLKokoroEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .textToSpeech

    private let modelDirectory: URL
    private var model: MLModel?

    var isLoaded: Bool { model != nil }

    init(modelDirectory: URL) {
        self.modelDirectory = modelDirectory
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        let mlpackage = try locateMLPackage()
        progress(.compiling(0.0))
        let compiled = try await CoreMLCompiler.compileIfNeeded(
            mlpackageURL: mlpackage,
            cacheDirectory: modelDirectory,
            estimatedDurationSeconds: 30
        ) { p in progress(.compiling(p)) }

        progress(.loadingWeights(0.5))
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: compiled, configuration: config)
        progress(.ready)
    }

    func unload() async {
        model = nil
    }

    func resetConversation() async {}

    /// Splits the most-recent user text into sentences, synthesizes each one
    /// and streams the resulting PCM. The exact Core ML I/O feature names
    /// depend on the specific Kokoro port; those are resolved at run time
    /// from `model.modelDescription.inputDescriptionsByName` to remain
    /// version-tolerant.
    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated; MLModel
            // prediction is synchronous but Kokoro-82M is small enough that
            // per-sentence latency stays acceptable.
            let task = Task { @MainActor [weak self] in
                guard let self, let model = self.model else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                let text = turns.last(where: { $0.role == .user })?.content ?? ""
                let sentences = Self.splitSentences(text)

                do {
                    for sentence in sentences where !sentence.isEmpty {
                        if Task.isCancelled { throw InferenceError.cancelled }
                        let pcm = try Self.synthesize(text: sentence, model: model)
                        let data = pcm.withUnsafeBufferPointer {
                            Data(buffer: $0)
                        }
                        continuation.yield(.audioChunk(data, sampleRate: 24_000, channels: 1))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func locateMLPackage() throws -> URL {
        let items = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        guard let pkg = items.first(where: { $0.pathExtension == "mlpackage" }) else {
            throw InferenceError.generationFailed("No Kokoro .mlpackage found in \(modelDirectory.path)")
        }
        return pkg
    }

    private static func splitSentences(_ text: String) -> [String] {
        text.split(whereSeparator: { ".!?\n".contains($0) })
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
    }

    /// Runs the Core ML model for one sentence. The actual feature plumbing
    /// (text preprocessing, voice embedding lookup, phonemization) depends
    /// on the specific Kokoro port. This placeholder looks up the first
    /// string input and the first multi-array output and returns its
    /// contents as Float PCM. Real Kokoro ports typically require a
    /// companion phonemizer step, which lives in a higher layer.
    private static func synthesize(text: String, model: MLModel) throws -> [Float] {
        let description = model.modelDescription
        guard let textInputName = description.inputDescriptionsByName.first(where: { $0.value.type == .string })?.key else {
            throw InferenceError.generationFailed("Kokoro: no string input found in model")
        }
        let features: [String: Any] = [textInputName: text]
        let provider = try MLDictionaryFeatureProvider(dictionary: features)
        let output = try model.prediction(from: provider)

        guard let outName = description.outputDescriptionsByName.first(where: { $0.value.type == .multiArray })?.key,
              let array = output.featureValue(for: outName)?.multiArrayValue else {
            throw InferenceError.generationFailed("Kokoro: no audio output found")
        }
        let count = array.count
        var result = [Float](repeating: 0, count: count)
        for i in 0..<count { result[i] = array[i].floatValue }
        return result
    }

    /// Direct synthesis API used by the media pipeline to play assistant
    /// responses aloud without going through the chat interface.
    func synthesize(text: String) async throws -> Data {
        guard let model else { throw InferenceError.modelNotLoaded }
        let pcm = try Self.synthesize(text: text, model: model)
        return pcm.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
