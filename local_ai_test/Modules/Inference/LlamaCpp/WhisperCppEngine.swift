//
//  WhisperCppEngine.swift
//  local_ai_test
//
//  Speech-to-text via whisper.cpp (https://github.com/ggerganov/whisper.cpp).
//  Supports Core ML encoder acceleration if a matching `*-encoder.mlmodelc`
//  is present next to the ggml weights.
//
//  Capabilities: audioIn -> textOut. Use as a pre-processor in the media
//  pipeline: record PCM -> transcribe -> push the transcript as a user
//  ChatTurn into the active text LLM.
//

import Foundation

#if canImport(whisper)

import whisper

final class WhisperCppEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .speechToText

    private let modelDirectory: URL
    private let model: AIModel
    private var context: WhisperContext?

    var isLoaded: Bool { context != nil }

    init(modelDirectory: URL, model: AIModel) {
        self.modelDirectory = modelDirectory
        self.model = model
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        progress(.loadingWeights(0.0))
        let ggmlURL = try locateGGML()
        context = try await WhisperContext.create(modelPath: ggmlURL.path)
        progress(.ready)
    }

    func unload() async {
        let ctx = context
        context = nil
        await ctx?.free()
    }

    func resetConversation() async {}

    /// Standard chat surface: accept an `audio` attachment, produce transcribed
    /// text via `textDelta` output. Audio must be 16 kHz mono f32.
    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated; the blocking
            // whisper_full call runs inside the WhisperContext actor.
            let task = Task { @MainActor [weak self] in
                guard let self, let ctx = self.context else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                guard let audioData = turns.last(where: { $0.role == .user })?.attachments
                    .compactMap({ att -> Data? in
                        if case let .audio(data, _, _) = att { return data }
                        return nil
                    }).first
                else {
                    continuation.finish(throwing: InferenceError.unsupportedAttachment("audio attachment required"))
                    return
                }

                do {
                    // Data is not guaranteed to be Float-aligned, and baseAddress
                    // is nil for empty buffers — copy via a properly-aligned array.
                    let pcm = [Float](unsafeUninitializedCapacity: audioData.count / MemoryLayout<Float>.size) { dest, initializedCount in
                        let byteCount = dest.count * MemoryLayout<Float>.size
                        if byteCount > 0 {
                            audioData.copyBytes(to: UnsafeMutableRawBufferPointer(dest), count: byteCount)
                        }
                        initializedCount = dest.count
                    }
                    guard !pcm.isEmpty else {
                        throw InferenceError.unsupportedAttachment("audio attachment is empty")
                    }

                    let segments = try await ctx.transcribe(pcm: pcm)
                    for segment in segments {
                        continuation.yield(.textDelta(segment))
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Convenience synchronous entry point used by the media pipeline.
    func transcribe(pcm: [Float]) async throws -> String {
        let stream = generate(
            turns: [.user("", attachments: [.audio(
                Data(bytes: pcm, count: pcm.count * MemoryLayout<Float>.size),
                sampleRate: 16_000
            )])],
            parameters: .default
        )
        var result = ""
        for try await output in stream {
            if case .textDelta(let s) = output { result += s }
        }
        return result
    }

    private func locateGGML() throws -> URL {
        let items = (try? FileManager.default.contentsOfDirectory(at: modelDirectory, includingPropertiesForKeys: nil)) ?? []
        guard let bin = items.first(where: { $0.lastPathComponent.hasPrefix("ggml-") && $0.pathExtension == "bin" }) else {
            throw InferenceError.generationFailed("No ggml-*.bin file found in \(modelDirectory.path)")
        }
        return bin
    }
}

/// Actor isolating all direct calls into the whisper.cpp C API, mirroring
/// `LlamaContext`. Keeps the blocking `whisper_full` call off the main actor
/// and serializes it against `free()` so unload-during-transcribe cannot
/// use-after-free.
actor WhisperContext {
    private var ctx: OpaquePointer?

    static func create(modelPath: String) async throws -> WhisperContext {
        var params = whisper_context_default_params()
        params.use_gpu = true
        guard let ctx = whisper_init_from_file_with_params(modelPath, params) else {
            throw InferenceError.generationFailed(
                "whisper.cpp: failed to load \((modelPath as NSString).lastPathComponent)"
            )
        }
        return WhisperContext(ctx: ctx)
    }

    private init(ctx: OpaquePointer) {
        self.ctx = ctx
    }

    func free() {
        if let ctx { whisper_free(ctx) }
        ctx = nil
    }

    func transcribe(pcm: [Float]) throws -> [String] {
        guard let ctx else { throw InferenceError.modelNotLoaded }

        var wParams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        wParams.n_threads = Int32(max(1, ProcessInfo.processInfo.processorCount - 1))
        wParams.print_realtime = false
        wParams.print_progress = false
        wParams.print_special = false
        wParams.print_timestamps = false
        wParams.translate = false
        wParams.single_segment = false

        let rc = pcm.withUnsafeBufferPointer { ptr in
            whisper_full(ctx, wParams, ptr.baseAddress, Int32(pcm.count))
        }
        guard rc == 0 else {
            throw InferenceError.generationFailed("whisper.cpp: whisper_full rc=\(rc)")
        }

        var segments: [String] = []
        let nSegments = whisper_full_n_segments(ctx)
        for i in 0..<nSegments {
            if let cstr = whisper_full_get_segment_text(ctx, i) {
                segments.append(String(cString: cstr))
            }
        }
        return segments
    }
}

#else

final class WhisperCppEngine: InferenceEngine, @unchecked Sendable {
    let capabilities: ModelCapabilities = .speechToText
    var isLoaded: Bool { false }
    init(modelDirectory: URL, model: AIModel) {}
    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        throw InferenceError.backendUnavailable(
            "WhisperCppEngine requires the 'whisper' module. See Scripts/build_whisper_xcframework.sh."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("whisper.cpp not linked")) }
    }
    func transcribe(pcm: [Float]) async throws -> String {
        throw InferenceError.backendUnavailable("whisper.cpp not linked")
    }
}

#endif
