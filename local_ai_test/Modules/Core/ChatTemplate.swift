//
//  ChatTemplate.swift
//  local_ai_test
//
//  Which recipe turns a conversation into the single string the model sees.
//
//  Every instruct model was trained with its own markers - Qwen writes
//  `<|im_start|>user`, Gemma `<start_of_turn>user`, PaddleOCR-VL `User: `.
//  Feed a model someone else's markers and it reads noise at the start of
//  every message: the answers get worse for a reason nothing in the UI
//  explains. Until now this app guessed the format from the repo id, which
//  works for the catalog and fails for anything imported.
//
//  A GGUF carries its own template in the metadata, so the guess is only
//  ever a last resort now. The order is: a template the user set for this
//  model, then the model's own, then our guess by family.
//

import Foundation

/// Where the transcript format came from, for the UI to show and for the
/// engines to record on a message.
enum ChatTemplateSource: Equatable, Sendable {
    /// The template stored inside the model file.
    case model
    /// A template the user pasted for this model.
    case custom
    /// This app's built-in format for the family the model id looks like.
    case fallback(PromptFamily)

    var label: String {
        switch self {
        case .model:              return "Model's own"
        case .custom:             return "Custom"
        case .fallback(let f):    return "App fallback (\(f.rawValue))"
        }
    }

    /// Whether the format is the model's real one rather than a guess.
    var isAuthoritative: Bool {
        switch self {
        case .model, .custom: return true
        case .fallback:       return false
        }
    }
}

/// The exact string handed to the model, and where its format came from.
struct RenderedPrompt: Equatable, Sendable, Identifiable {
    /// Identity for `sheet(item:)`: the transcript itself. Two identical
    /// transcripts are the same thing to look at.
    var id: String { text }

    let text: String
    let source: ChatTemplateSource
    /// The template itself, when one was applied - what the UI shows when
    /// the user wants to read it.
    let template: String?
}

/// Per-model template overrides the user typed in. Small enough for
/// `UserDefaults`; a template is a few kilobytes of text at most.
@MainActor
final class ChatTemplateStore {

    static let shared = ChatTemplateStore()

    private let defaults: UserDefaults
    private let key = "customChatTemplates"

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    private var all: [String: String] {
        get { defaults.dictionary(forKey: key) as? [String: String] ?? [:] }
        set { defaults.set(newValue, forKey: key) }
    }

    func custom(for modelID: String) -> String? {
        let stored = all[modelID]?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (stored?.isEmpty == false) ? stored : nil
    }

    func setCustom(_ template: String?, for modelID: String) {
        var dict = all
        let trimmed = template?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let trimmed, !trimmed.isEmpty {
            dict[modelID] = trimmed
        } else {
            dict.removeValue(forKey: modelID)
        }
        all = dict
    }

    func hasCustom(for modelID: String) -> Bool { custom(for: modelID) != nil }
}

/// Picks the transcript format for a model and renders it.
///
/// Order, most authoritative first:
///   1. a template the user set for this model,
///   2. the template stored in the model file,
///   3. this app's format for the family the id looks like.
///
/// A template llama.cpp cannot render falls through to the next source
/// rather than failing the message: an unusable custom template should cost
/// the user a worse transcript, not a dead chat.
enum ChatTemplateResolver {

    @MainActor
    static func render(
        turns: [ChatTurn],
        modelID: String,
        embedded: String?,
        store: ChatTemplateStore = .shared
    ) -> RenderedPrompt {
        if let custom = store.custom(for: modelID),
           let text = LlamaChatTemplate.render(turns: turns, template: custom) {
            return RenderedPrompt(text: text, source: .custom, template: custom)
        }
        if let embedded,
           let text = LlamaChatTemplate.render(turns: turns, template: embedded) {
            return RenderedPrompt(
                text: closingReasoning(text, modelID: modelID),
                source: .model,
                template: embedded
            )
        }
        let family = PromptFormatter.detectFamily(modelId: modelID)
        return RenderedPrompt(
            text: closingReasoning(PromptFormatter.format(turns: turns, family: family), modelID: modelID),
            source: .fallback(family),
            template: nil
        )
    }

    /// What Qwen 3 / 3.5's own template writes after the assistant header
    /// when thinking is off (`enable_thinking=false`, the default for every
    /// request this app makes).
    static let qwen3EmptyReasoning = "<think>\n\n</think>\n\n"

    /// Qwen 3 and 3.5 ship a Jinja template that llama.cpp b8882 does not
    /// interpret: it sniffs `<|im_start|>` and writes plain ChatML, which
    /// drops the empty reasoning block the real template ends with. Without
    /// it the model opens its reply with `<think>` and spends the whole
    /// token budget reasoning. The id test is "qwen3", which covers Qwen 3
    /// and 3.5 and leaves Qwen 2.5 alone. A user's custom template is never
    /// touched: that one says exactly what it wants.
    static func closingReasoning(_ text: String, modelID: String) -> String {
        guard modelID.lowercased().contains("qwen3"),
              text.hasSuffix("<|im_start|>assistant\n")
        else { return text }
        return text + qwen3EmptyReasoning
    }

    /// What the UI shows before a message is ever sent: which source would
    /// be used, without rendering a transcript.
    @MainActor
    static func source(
        modelID: String,
        embedded: String?,
        store: ChatTemplateStore = .shared
    ) -> ChatTemplateSource {
        if let custom = store.custom(for: modelID), LlamaChatTemplate.isSupported(custom) {
            return .custom
        }
        if let embedded, LlamaChatTemplate.isSupported(embedded) {
            return .model
        }
        return .fallback(PromptFormatter.detectFamily(modelId: modelID))
    }
}
