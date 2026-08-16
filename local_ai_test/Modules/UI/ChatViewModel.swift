//
//  ChatViewModel.swift
//  local_ai_test
//
//  Drives the main chat tab. Now multimodal-aware: the send() call bundles
//  staged attachments into the user turn, wires the engine's ChatOutput
//  stream into the bubble, and optionally pipes the text through Kokoro TTS.
//

import SwiftUI

/// Shown as a placeholder chip while photo / video / file data is prepared off the main thread.
enum PendingStagedAttachmentKind: Equatable, Sendable {
    case none
    case loadingPhoto
    case preparingVideo
    case importingFile
}

@MainActor
@Observable
final class ChatViewModel {

    private(set) var messages: [ChatMessage] = []
    private(set) var isGenerating = false
    var inputText: String = ""

    /// Attachments staged for the next send.
    var stagedAttachments: [ChatAttachment] = []

    /// Non-`.none` while the UI waits on `PhotosPicker` / `VideoFrameExtractor` / file read.
    var pendingStagedAttachment: PendingStagedAttachmentKind = .none

    /// "Speak response" toggle. When true, assistant text is piped through
    /// Kokoro on the fly. Requires a Kokoro model to be installed —
    /// enforced by `trySetSpeakResponses(_:)`.
    private(set) var speakResponses: Bool = false

    /// Hugging Face repo id of the bundled Kokoro catalog entry. Kept in
    /// sync with `ModelCatalog`'s Kokoro model; surfaced so the UI can deep
    /// link into the Models tab and focus this exact card.
    static let kokoroCatalogID = "coreml-community/Kokoro-82M-CoreML"

    private let store = InstalledModelsStore.shared
    private let inference = InferenceManager.shared
    private let media = MediaPipeline.shared

    /// Owns the in-flight `send()` work so the stop button can cancel generation.
    @ObservationIgnored
    private var generationTask: Task<Void, Never>?

    /// Identity of the in-flight generation. A cancelled task's cleanup must
    /// not clobber state that already belongs to a newer generation.
    @ObservationIgnored
    private var generationID: UUID?

    var selectedModel: AIModel? { store.selectedModel }
    var installedModels: [AIModel] { store.installedModels }
    var inferenceState: InferenceState { inference.state }

    var currentCapabilities: ModelCapabilities {
        inference.currentCapabilities
    }

    /// Whether a Kokoro TTS model is installed and ready to be used as the
    /// sink of the Speak-response pipeline.
    var isKokoroInstalled: Bool { findInstalledKokoro() != nil }

    /// Enables `speakResponses` only when Kokoro is actually installed.
    /// Returns `false` when the caller asked to enable speech but the
    /// required model is missing — letting the UI present a "please
    /// install Kokoro" prompt instead.
    @discardableResult
    func trySetSpeakResponses(_ newValue: Bool) -> Bool {
        if newValue {
            guard isKokoroInstalled else {
                speakResponses = false
                media.speakAssistantEnabled = false
                return false
            }
        }
        speakResponses = newValue
        // Keep the pipeline's own gate in sync — streamToSpeech() checks it
        // per sentence, so toggling mid-generation takes effect immediately.
        media.speakAssistantEnabled = newValue
        return true
    }

    // MARK: - Model switching

    func selectModel(_ model: AIModel) {
        store.selectedModel = model
        messages = []
        stagedAttachments = []
        pendingStagedAttachment = .none
        Task {
            await inference.resetConversation()
            await inference.prepare(model: model)
        }
    }

    func prepareCurrentModel() async {
        guard let model = selectedModel else { return }
        await inference.prepare(model: model)
    }

    // MARK: - Attachments

    func attachImage(_ data: Data) {
        if let normalized = ImagePreprocessor.normalize(data) {
            stagedAttachments.append(.image(normalized))
        } else {
            stagedAttachments.append(.image(data))
        }
    }

    func attachAudio(pcm: [Float], sampleRate: Int = 16_000) {
        let data = pcm.withUnsafeBufferPointer { Data(buffer: $0) }
        stagedAttachments.append(.audio(data, sampleRate: sampleRate, channels: 1))
    }

    func attachVideo(_ result: VideoFrameExtractor.ExtractResult) {
        stagedAttachments.append(.videoFrames(result.frames, durationSeconds: result.durationSeconds))
    }

    func removeAttachment(at index: Int) {
        guard stagedAttachments.indices.contains(index) else { return }
        stagedAttachments.remove(at: index)
    }

    // MARK: - Send message

    /// Starts generation in a detached task so the stop control can cancel it.
    func send() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachments = stagedAttachments
        guard !(text.isEmpty && attachments.isEmpty), !isGenerating else { return }
        inputText = ""
        stagedAttachments = []

        messages.append(ChatMessage(role: .user, text: text, attachments: attachments))

        generationTask?.cancel()
        let generationID = UUID()
        self.generationID = generationID
        generationTask = Task { @MainActor [weak self] in
            guard let self else { return }
            defer {
                // Only clean up if a newer send() hasn't taken over —
                // a cancelled task must not clobber its successor's state.
                if self.generationID == generationID {
                    self.isGenerating = false
                    self.generationTask = nil
                    self.generationID = nil
                }
            }
            self.isGenerating = true

            if self.inference.state != .ready, let model = self.selectedModel {
                await self.inference.prepare(model: model)
            }
            if Task.isCancelled { return }

            self.messages.append(ChatMessage(role: .assistant, text: ""))
            let assistantIndex = self.messages.count - 1

            do {
                let turns: [ChatTurn] = self.messages.dropLast().map { msg in
                    ChatTurn(
                        role: msg.chatRole,
                        content: msg.text,
                        attachments: msg.attachments
                    )
                }

                var stream = self.inference.generate(turns: turns, parameters: .default)
                if self.speakResponses, let kokoro = self.findInstalledKokoro() {
                    stream = self.media.streamToSpeech(source: stream, using: kokoro)
                }

                for try await output in stream {
                    // clearChat() may have emptied `messages` while a chunk
                    // was in flight — never index into a mutated array.
                    guard self.messages.indices.contains(assistantIndex) else { break }
                    switch output {
                    case .textDelta(let chunk):
                        self.messages[assistantIndex].text += chunk
                    case .audioChunk(let data, let sr, let ch):
                        AudioPlayer.shared.enqueue(data: data, sampleRate: sr, channels: ch)
                    case .diagnostic:
                        break
                    }
                }

                if self.messages.indices.contains(assistantIndex) {
                    self.messages[assistantIndex].text = self.messages[assistantIndex].text
                        .trimmingCharacters(in: .whitespacesAndNewlines)

                    if self.messages[assistantIndex].text.isEmpty {
                        self.messages[assistantIndex].text = "(Empty response)"
                    }
                }
            } catch is CancellationError {
                AudioPlayer.shared.stop()
                if self.messages.indices.contains(assistantIndex),
                   self.messages[assistantIndex].text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    self.messages.remove(at: assistantIndex)
                }
            } catch {
                let cancelled: Bool = {
                    if case InferenceError.cancelled = error { return true }
                    return false
                }()
                if cancelled {
                    AudioPlayer.shared.stop()
                    if self.messages.indices.contains(assistantIndex),
                       self.messages[assistantIndex].text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        self.messages.remove(at: assistantIndex)
                    }
                } else if self.messages.indices.contains(assistantIndex),
                          self.messages[assistantIndex].text.isEmpty {
                    self.messages[assistantIndex].text = "Error: \(error.localizedDescription)"
                }
            }
        }
    }

    /// Stops streaming generation (and Kokoro playback) when the user taps the stop button.
    func cancelGeneration() {
        generationTask?.cancel()
    }

    func clearChat() {
        generationTask?.cancel()
        generationTask = nil
        generationID = nil
        isGenerating = false
        messages = []
        stagedAttachments = []
        pendingStagedAttachment = .none
        Task { await inference.resetConversation() }
    }

    // MARK: - Helpers

    private func findInstalledKokoro() -> AIModel? {
        installedModels.first(where: { $0.backend == .coreMLKokoro })
    }
}
