//
//  LlamaCppVisionEngine.swift
//  local_ai_test
//
//  llama.cpp multimodal vision via the mmproj projector (LLaVA, MiniCPM-V,
//  Gemma 3 Vision, Qwen2-VL, …). Requires both the main model .gguf and a
//  companion `mmproj-*.gguf` in the same directory.
//
//  The heavy lifting happens inside `LlamaVisionContext`, which wraps
//  `libmtmd` (bundled in our custom `llama.xcframework` built via
//  `Scripts/build_llama_mtmd_xcframework.sh`). This engine is the glue:
//    • finds the two GGUFs on disk,
//    • formats the conversation into a text prompt,
//    • injects `<__media__>` markers for each image attachment,
//    • extracts raw image `Data` and forwards everything to the context,
//    • translates the piece callback into the streaming `ChatOutput` API.
//
//  Reference implementation: tools/mtmd/mtmd-cli.cpp in llama.cpp.
//

import Foundation

/// The literal token libmtmd looks for in the prompt to splice in an
/// image/audio chunk. Hard-coded on the C side (see `mtmd_default_marker`
/// in `mtmd.cpp`); we mirror it here to avoid a round-trip just to read it.
private let kMediaMarker = "<__media__>"

#if canImport(llama)

import llama

final class LlamaCppVisionEngine: InferenceEngine, @unchecked Sendable {

    let capabilities: ModelCapabilities = .videoVision

    private let modelDirectory: URL
    private let model: AIModel
    private var context: LlamaVisionContext?

    var isLoaded: Bool { context != nil }

    init(modelDirectory: URL, model: AIModel) {
        self.modelDirectory = modelDirectory
        self.model = model
    }

    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        progress(.loadingWeights(0.0))
        let ggufURL = try locateMainGGUF()
        let mmprojURL = try locateMmproj()
        context = try await LlamaVisionContext.create(
            modelPath: ggufURL.path,
            mmprojPath: mmprojURL.path
        )
        progress(.ready)
    }

    func unload() async {
        let ctx = context
        context = nil
        // Await actual release of the C-side weights + Metal buffers so the
        // caller can safely load the next model without doubling peak RAM.
        await ctx?.cleanup()
    }

    func resetConversation() async {
        await context?.reset()
    }

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            // @MainActor: engine state is main-actor isolated; the heavy
            // decode work runs inside the LlamaVisionContext actor.
            let task = Task { @MainActor [weak self] in
                guard let self, let ctx = self.context else {
                    continuation.finish(throwing: InferenceError.modelNotLoaded)
                    return
                }
                do {
                    // Guard: vision inference needs ~500 MB+ for Metal compute
                    // buffers. If the device is severely memory-constrained the
                    // Metal allocator returns NULL and llama.cpp crashes
                    // (EXC_BAD_ACCESS in ggml_metal_buffer_is_shared). Bail
                    // early with a user-readable error rather than crashing.
                    let freeMB = InferenceManager.availableMemoryMB()
                    if freeMB > 0, freeMB < 350 {
                        throw InferenceError.generationFailed(
                            "Not enough free memory for vision inference " +
                            "(\(freeMB) MB free, need ≥ 350 MB). " +
                            "Close other apps and try again."
                        )
                    }

                    // 1. Assemble the text prompt via the model family's
                    //    template (same path as the text engine so chat
                    //    structure stays consistent between backends).
                    let family = PromptFormatter.detectFamily(modelId: self.model.id)
                    var prompt = PromptFormatter.format(turns: turns, family: family)

                    // 2. Pull image bytes from the last user turn. Video
                    //    attachments are flattened into their sampled
                    //    frames (VideoFrameExtractor produces ~8 JPEGs by
                    //    default); libmtmd has no native "video" chunk
                    //    type — it sees each frame as a separate image.
                    //    Audio via `mtmd_bitmap_init_from_audio` is
                    //    deferred to a follow-up.
                    let images = Self.extractImages(from: turns)

                    // 3. Splice one `<__media__>` marker per image into the
                    //    tail of the last user message — right before the
                    //    assistant-open tag that `PromptFormatter` appends.
                    //    Placing markers *after* the text means the model
                    //    sees the question first, which mirrors the
                    //    `mtmd-cli` behaviour and works for every arch
                    //    (Gemma 3 / LLaVA / Qwen2-VL).
                    if !images.isEmpty {
                        prompt = Self.injectMediaMarkers(
                            into: prompt,
                            family: family,
                            count: images.count
                        )
                    }

                    let start = Date()

                    // Token index is passed from the actor, so no shared
                    // mutable state is captured across concurrency domains.
                    let tokensGenerated = try await ctx.generate(
                        prompt: prompt,
                        images: images,
                        parameters: parameters
                    ) { piece, index in
                        if index == 0 {
                            let ms = Date().timeIntervalSince(start) * 1000
                            continuation.yield(.diagnostic(.firstTokenLatency(ms: ms)))
                        }
                        continuation.yield(.textDelta(piece))
                    }

                    let elapsed = Date().timeIntervalSince(start)
                    continuation.yield(.diagnostic(.finished(
                        tokensGenerated: tokensGenerated,
                        tokensPerSecond: elapsed > 0 ? Double(tokensGenerated) / elapsed : 0
                    )))
                    continuation.finish()
                } catch {
                    continuation.finish(
                        throwing: Task.isCancelled
                            ? InferenceError.cancelled
                            : InferenceError.generationFailed(error.localizedDescription)
                    )
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: - File discovery

    private func locateMainGGUF() throws -> URL {
        if let preferred = model.preferredGGUFFilename {
            let candidate = modelDirectory.appendingPathComponent(preferred)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        let items = (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )) ?? []
        guard let gguf = items.first(where: {
            $0.pathExtension.lowercased() == "gguf" && !$0.lastPathComponent.hasPrefix("mmproj")
        }) else {
            throw InferenceError.generationFailed("No main .gguf file found in \(modelDirectory.path)")
        }
        return gguf
    }

    private func locateMmproj() throws -> URL {
        if let preferred = model.mmprojFilename {
            let candidate = modelDirectory.appendingPathComponent(preferred)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        let items = (try? FileManager.default.contentsOfDirectory(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        )) ?? []
        guard let mmproj = items.first(where: { $0.lastPathComponent.hasPrefix("mmproj") }) else {
            throw InferenceError.generationFailed("No mmproj-*.gguf file found in \(modelDirectory.path)")
        }
        return mmproj
    }
}

// MARK: - Attachment / marker helpers (shared; also used by unit tests)

extension LlamaCppVisionEngine {

    /// Returns image bytes from the **last user turn** only. Earlier turns'
    /// attachments are ignored because the KV-cache replays prior messages
    /// as text chunks; re-tokenising their images would misalign `nPast`.
    ///
    /// `.videoFrames` attachments are flattened into their individual JPEG
    /// samples and surface as multiple images — one `<__media__>` marker
    /// per frame, which is exactly what libmtmd expects. Gemma 3 / LLaVA
    /// / Qwen2-VL interpret a sequence of images as a temporal sequence
    /// without any special "video" chunk type.
    ///
    /// `.audio` is skipped for now (needs `mtmd_bitmap_init_from_audio` +
    /// PCM decoding; deferred to a follow-up).
    static func extractImages(from turns: [ChatTurn]) -> [Data] {
        guard let lastUser = turns.last(where: { $0.role == .user }) else { return [] }
        var out: [Data] = []
        for attachment in lastUser.attachments {
            switch attachment {
            case .image(let data):
                out.append(data)
            case .videoFrames(let frames, _):
                out.append(contentsOf: frames)
            case .audio:
                continue
            }
        }
        return out
    }

    /// Inserts `<__media__>` markers into the formatted prompt, one per
    /// image. The exact insertion point depends on the chat template — we
    /// want the markers to live inside the *last user turn*, not in the
    /// assistant-open tag the formatter appends.
    ///
    /// Strategy: find the template-specific "assistant opens its turn"
    /// substring, and inject markers immediately before it. This keeps the
    /// markers inside the user's message body on every supported family.
    static func injectMediaMarkers(
        into prompt: String,
        family: PromptFamily,
        count: Int
    ) -> String {
        guard count > 0 else { return prompt }
        let markers = String(repeating: kMediaMarker + "\n", count: count)

        // These needles match `PromptFormatter`'s trailing "assistant
        // opens" snippets. When the formatter changes, these need to move
        // in lockstep (covered by `LlamaCppVisionPromptTests`).
        let needle: String
        switch family {
        case .llama3:  needle = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        case .gemma:   needle = "<start_of_turn>model\n"
        case .qwen, .chatml: needle = "<|im_start|>assistant\n"
        case .phi:     needle = "<|assistant|>\n"
        case .plain:   needle = "\n\nassistant:"
        case .llama2, .mistral:
            // These families don't have a separate assistant-open tag;
            // the formatter ends with `[/INST]` right after the user's
            // content, so markers go immediately before that.
            needle = "[/INST]"
        }

        if let range = prompt.range(of: needle, options: .backwards) {
            var copy = prompt
            copy.insert(contentsOf: markers, at: range.lowerBound)
            return copy
        }
        // Fallback: just append. mtmd only cares that the marker count
        // matches the bitmap count, not where they are.
        return prompt + "\n" + markers
    }
}

#else

final class LlamaCppVisionEngine: InferenceEngine, @unchecked Sendable {
    let capabilities: ModelCapabilities = .videoVision
    var isLoaded: Bool { false }
    init(modelDirectory: URL, model: AIModel) {}
    func load(progress: @escaping @Sendable (LoadProgress) -> Void) async throws {
        throw InferenceError.backendUnavailable(
            "llama.cpp vision requires the 'llama' module with libmtmd. " +
            "Build a custom XCFramework via Scripts/build_llama_mtmd_xcframework.sh."
        )
    }
    func unload() async {}
    func resetConversation() async {}
    func generate(turns: [ChatTurn], parameters: GenerationParameters) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { $0.finish(throwing: InferenceError.backendUnavailable("llama.cpp not linked")) }
    }
}

// Stubs so unit tests (which test the helpers, not the C API) still
// compile when `llama` isn't linked (e.g. on CI without the XCFramework).
extension LlamaCppVisionEngine {
    static func extractImages(from turns: [ChatTurn]) -> [Data] {
        guard let lastUser = turns.last(where: { $0.role == .user }) else { return [] }
        var out: [Data] = []
        for attachment in lastUser.attachments {
            switch attachment {
            case .image(let data):
                out.append(data)
            case .videoFrames(let frames, _):
                out.append(contentsOf: frames)
            case .audio:
                continue
            }
        }
        return out
    }

    static func injectMediaMarkers(
        into prompt: String,
        family: PromptFamily,
        count: Int
    ) -> String {
        guard count > 0 else { return prompt }
        let markers = String(repeating: kMediaMarker + "\n", count: count)
        return prompt + "\n" + markers
    }
}

#endif
