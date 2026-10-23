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

    func embeddedChatTemplate() async -> String? {
        await context?.embeddedChatTemplate()
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

                    // Assemble the transcript, its media markers and the
                    // bitmaps in one pass, trimmed to what the context
                    // window can hold. The engine is stateless — the KV
                    // cache is rebuilt from this prompt on every turn — so
                    // the media of EVERY turn is replayed, not just the
                    // last one's.
                    let family = PromptFormatter.detectFamily(modelId: self.model.id)
                    let prepared = VisionPromptBuilder.build(
                        turns: turns,
                        family: family,
                        budgetTokens: Int(LlamaVisionContext.contextTokens)
                            - parameters.maxTokens - 64
                    )
                    // The markers are part of the user turn's text by now,
                    // so the model's own template renders them in place.
                    let rendered = ChatTemplateResolver.render(
                        turns: prepared.turns,
                        modelID: self.model.id,
                        embedded: await ctx.embeddedChatTemplate()
                    )
                    continuation.yield(.diagnostic(.promptRendered(rendered)))
                    let prompt = rendered.text
                    let images = prepared.images

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
            GGUFNaming.isWeights($0.lastPathComponent)
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
        guard let mmproj = items.first(where: { GGUFNaming.isProjector($0.lastPathComponent) }) else {
            throw InferenceError.generationFailed("No mmproj-*.gguf file found in \(modelDirectory.path)")
        }
        return mmproj
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

#endif
