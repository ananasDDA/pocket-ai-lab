//
//  InferenceManager.swift
//  local_ai_test
//
//  Orchestrates loading, unloading and switching between inference engines.
//  Routes multimodal requests to the appropriate backend based on the
//  selected model's capabilities. Handles memory pressure by unloading
//  the current engine.
//

import Foundation
import UIKit
import Metal

enum InferenceState: Equatable {
    case idle
    case loading
    case compiling(progress: Double)
    case loadingWeights(progress: Double)
    case ready
    case generating
    case error(String)
}

/// Detailed, user-copyable description of the last engine failure.
struct EngineFailure: Sendable, Identifiable {
    enum Phase: String, Sendable { case load, generate }

    let id = UUID()
    let phase: Phase
    let modelId: String
    let modelName: String
    let backend: String
    /// One-line headline ("Backend not available in this build").
    let summary: String
    /// Multi-line explanation, safe to copy to issue trackers.
    let detail: String
    let underlyingType: String
    let timestamp: Date = .init()

    /// Full text that gets copied to the clipboard / share sheet.
    var exportText: String {
        """
        local_ai_test — engine failure
        time:     \(ISO8601DateFormatter().string(from: timestamp))
        phase:    \(phase.rawValue)
        model:    \(modelName) (\(modelId))
        backend:  \(backend)
        type:     \(underlyingType)
        summary:  \(summary)

        detail:
        \(detail)
        """
    }
}

@MainActor
@Observable
final class InferenceManager {

    static let shared = InferenceManager()

    private init() {
        NotificationCenter.default.addObserver(
            forName: UIApplication.didReceiveMemoryWarningNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                self.handleMemoryWarning()
            }
        }
    }

    private func handleMemoryWarning() {
        guard let engine = currentEngine else { return }
        // iOS fires this for SYSTEM-wide pressure too — routine while a
        // multi-GB model loads (the kernel is evicting background apps, not
        // threatening us). Dropping the model on every warning made loads
        // self-destruct. Only unload when OUR OWN allowance is nearly gone —
        // that is when the next KV-cache growth would be a jetsam kill.
        let remainingMB = Self.availableMemoryMB()
        guard remainingMB >= 0, remainingMB < 512 else { return }

        currentEngine = nil
        currentModelId = nil
        state = .error("Model unloaded due to memory pressure. Try a smaller model.")
        Task { await engine.unload() }
    }

    private(set) var state: InferenceState = .idle
    private var currentEngine: InferenceEngine?
    private(set) var currentModelId: String?

    /// Last load/generation failure. Survives across state transitions so
    /// the UI can show a detailed, copyable error after the label rolls
    /// back to `.idle` or `.ready`.
    private(set) var lastError: EngineFailure?

    /// The capabilities of the currently loaded model (or empty set).
    var currentCapabilities: ModelCapabilities {
        currentEngine?.capabilities ?? []
    }

    /// Clear the last-error banner. Called from the UI after the user
    /// acknowledges the error or picks a different model.
    func clearLastError() { lastError = nil }

    // MARK: - Public

    func prepare(model: AIModel) async {
        if currentModelId == model.id, currentEngine?.isLoaded == true {
            state = .ready
            return
        }

        // Fully unload the previous engine and *wait* for its resources to
        // actually be released — loading the next model while the old one is
        // still resident doubles peak RAM and gets the app jetsam-killed.
        // A short pause afterwards lets iOS reclaim the freed pages.
        if let engine = currentEngine {
            currentEngine = nil
            currentModelId = nil
            await engine.unload()
            try? await Task.sleep(nanoseconds: 200_000_000)
        }
        currentEngine = nil
        currentModelId = nil

        state = .loading
        lastError = nil

        // Fail fast when the user picked a model whose backend isn't linked
        // into this build. Surfaces a detailed message instead of the
        // generic "Model is not loaded into memory."
        if !model.backend.isLinked {
            let reason = model.backend.unavailableReason
                ?? "Backend \(model.backend.displayName) is not linked into this build."
            let failure = EngineFailure(
                phase: .load,
                modelId: model.id,
                modelName: model.name,
                backend: model.backend.displayName,
                summary: "Backend not available in this build",
                detail: reason,
                underlyingType: "BackendUnavailable"
            )
            lastError = failure
            state = .error(failure.summary)
            return
        }

        // Preflight: refuse to start a load that cannot fit in the memory
        // budget. Without this, loading an oversized model is a silent
        // jetsam kill — the app just vanishes with no crash report.
        //
        // The budget depends on how the backend holds its weights:
        //   • MLX / Core ML copy weights into anonymous Metal buffers —
        //     dirty memory, charged 1:1 against the jetsam allowance
        //     (os_proc_available_memory).
        //   • llama.cpp / whisper.cpp mmap their GGUF — clean file-backed
        //     pages that jetsam does NOT charge; the binding ceiling is the
        //     Metal working set (~0.67 × RAM).
        let budgetMB = Self.memoryBudgetMB(for: model.backend)
        let neededMB = Int64(model.ramRequiredGB * 1024)
        if budgetMB > 0, neededMB > budgetMB {
            let failure = EngineFailure(
                phase: .load,
                modelId: model.id,
                modelName: model.name,
                backend: model.backend.displayName,
                summary: "Not enough memory to load this model",
                detail: "\(model.name) needs ~\(neededMB) MB but this device allows "
                    + "~\(budgetMB) MB for the \(model.backend.displayName) backend. "
                    + "Close other apps and try again, or pick a smaller model / "
                    + "lower quantization.",
                underlyingType: "MemoryPreflight"
            )
            lastError = failure
            state = .error(failure.summary)
            return
        }

        do {
            let engine = try makeEngine(for: model)
            try await engine.load { [weak self] progress in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    switch progress {
                    case .downloading:           self.state = .loading
                    case .compiling(let p):      self.state = .compiling(progress: p)
                    case .loadingWeights(let p): self.state = .loadingWeights(progress: p)
                    case .ready:                 self.state = .ready
                    }
                }
                _ = self
            }
            currentEngine = engine
            currentModelId = model.id
            state = .ready
        } catch {
            let failure = EngineFailure(
                phase: .load,
                modelId: model.id,
                modelName: model.name,
                backend: model.backend.displayName,
                summary: "Failed to load model",
                detail: error.localizedDescription,
                underlyingType: String(describing: type(of: error))
            )
            lastError = failure
            state = .error(failure.summary + " — " + error.localizedDescription)
        }
    }

    // MARK: - Multimodal entry point (preferred)

    func generate(
        turns: [ChatTurn],
        parameters: GenerationParameters = GenerationParameters.default
    ) -> AsyncThrowingStream<ChatOutput, Error> {
        guard let engine = currentEngine, engine.isLoaded else {
            // Prefer replaying the original load-time failure (if any) so
            // the user sees *why* the engine is missing, not a generic
            // "Model is not loaded into memory."
            let underlying: Error = lastError.map { InferenceError.backendUnavailable($0.detail) }
                ?? InferenceError.modelNotLoaded
            return AsyncThrowingStream { $0.finish(throwing: underlying) }
        }

        state = .generating
        let stream = engine.generate(turns: turns, parameters: parameters)

        return AsyncThrowingStream { continuation in
            let task = Task { @MainActor [weak self] in
                do {
                    for try await output in stream {
                        continuation.yield(output)
                    }
                    continuation.finish()
                    self?.state = .ready
                } catch {
                    continuation.finish(throwing: error)
                    guard let self else { return }
                    let userCancelled = error is CancellationError
                        || (error as? InferenceError).map {
                            if case .cancelled = $0 { return true }
                            return false
                        } ?? false
                    if userCancelled {
                        self.state = .ready
                        return
                    }
                    let failure = EngineFailure(
                        phase: .generate,
                        modelId: self.currentModelId ?? "(unknown)",
                        modelName: self.currentModelId ?? "(unknown)",
                        backend: self.currentEngine.map { String(describing: type(of: $0)) } ?? "(none)",
                        summary: "Generation failed",
                        detail: error.localizedDescription,
                        underlyingType: String(describing: type(of: error))
                    )
                    self.lastError = failure
                    self.state = .error(failure.summary + " — " + error.localizedDescription)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    // MARK: - Legacy text-only adapter (kept for binary-compat with ChatViewModel)

    func generate(
        messages: [(role: String, content: String)],
        temperature: Float = 0.7,
        maxTokens: Int = 2048
    ) -> AsyncThrowingStream<String, Error> {
        let turns: [ChatTurn] = messages.map { msg in
            let role = ChatRole(rawValue: msg.role) ?? .user
            return ChatTurn(role: role, content: msg.content)
        }
        let params = GenerationParameters(
            temperature: temperature,
            topP: 0.9,
            topK: 40,
            repetitionPenalty: 1.1,
            maxTokens: maxTokens,
            seed: nil
        )
        return generate(turns: turns, parameters: params).textDeltas()
    }

    func unloadCurrent() async {
        let engine = currentEngine
        currentEngine = nil
        currentModelId = nil
        state = .idle
        await engine?.unload()
    }

    func resetConversation() async {
        await currentEngine?.resetConversation()
    }

    /// Best-effort free memory check. Returns -1 if unavailable.
    static func availableMemoryMB() -> Int64 {
        let mib = Int64(os_proc_available_memory()) / (1024 * 1024)
        return mib > 0 ? mib : -1
    }

    /// Loadable-model budget in MB for a backend — see the preflight comment
    /// in `prepare(model:)` for the reasoning. 0 when nothing is measurable.
    static func memoryBudgetMB(for backend: ModelBackend) -> Int64 {
        switch backend {
        case .llamaCpp, .llamaCppVision, .whisperCpp:
            let ramMB = Int64(ProcessInfo.processInfo.physicalMemory) / (1024 * 1024)
            let workingSetMB = Int64(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0) / (1024 * 1024)
            if workingSetMB > 0 { return min(workingSetMB, ramMB * 7 / 10) }
            return ramMB * 2 / 3
        default:
            return max(availableMemoryMB(), 0)
        }
    }

    // MARK: - Factory

    private func makeEngine(for model: AIModel) throws -> InferenceEngine {
        let directory = ModelDownloader.shared.modelDirectory(for: model)

        switch model.backend {
        case .appleIntelligence:
            if #available(iOS 26, *) { return AppleIntelligenceEngine() }
            throw InferenceError.backendUnavailable("Apple Intelligence requires iOS 26+")

        case .mlx:
            return MLXTextEngine(modelDirectory: directory)

        case .mlxVision:
            #if canImport(MLXVLM)
            return MLXVisionEngine(modelDirectory: directory)
            #else
            throw InferenceError.backendUnavailable(
                "MLX Vision backend requires the MLXVLM Swift package. See MULTI_BACKEND_INTEGRATION.md."
            )
            #endif

        case .mlxAudio:
            #if canImport(MLXVLM)
            return MLXAudioEngine(modelDirectory: directory)
            #else
            throw InferenceError.backendUnavailable(
                "MLX Audio backend requires the MLXVLM Swift package (Qwen2-Audio)."
            )
            #endif

        case .coreML:
            return CoreMLTextEngine(modelDirectory: directory, modelId: model.id)

        case .coreMLWhisper:
            return CoreMLWhisperEngine(modelDirectory: directory)

        case .coreMLKokoro:
            return CoreMLKokoroEngine(modelDirectory: directory)

        case .llamaCpp:
            return LlamaCppTextEngine(modelDirectory: directory, model: model)

        case .llamaCppVision:
            return LlamaCppVisionEngine(modelDirectory: directory, model: model)

        case .whisperCpp:
            return WhisperCppEngine(modelDirectory: directory, model: model)
        }
    }
}
