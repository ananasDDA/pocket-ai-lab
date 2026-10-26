//
//  SystemPrompt.swift
//  local_ai_test
//
//  The instructions every chat starts with.
//
//  Small models left to themselves answer "briefly" with 900 tokens, give
//  two translations and a grammar lesson when asked for one translation, and
//  write a plain sum as LaTeX that the chat cannot render. One short system
//  turn fixes most of it. It is added by the chat screen only: benchmarks
//  build their own turns (`BenchmarkEngine.qualityTurns`) so their scores
//  stay comparable with every report made before this existed.
//
//  The text is a single, global setting rather than one per model: what a
//  person wants from a reply (shorter, in another language, more detail) is
//  about them, not about the model file, and re-entering it for each of six
//  installed models would be the price of a per-model store. The chat
//  template, by contrast, is a property of the model file and stays per
//  model (`ChatTemplateStore`).
//

import Foundation
import Observation

enum SystemPrompt {

    /// The built-in instructions. English, about 65 tokens (Llama 3's
    /// tokenizer; Gemma's larger vocabulary needs fewer). Each sentence
    /// answers something the graded daily-use run of 2026-10-24 (with and
    /// without a system prompt) showed:
    ///
    ///   * "Be concise." Two to three times fewer tokens, far fewer replies
    ///     cut at the 512-token cap, constraints (budget, vegetarian) kept.
    ///   * "For any calculation, show brief steps" - with "Be concise" alone
    ///     Gemma 4 and Qwen 3.5 answered the bill split with a bare, wrong
    ///     number; without a prompt both worked through it and got it right.
    ///   * "Reply in the language of the user's latest message." Gemma 3
    ///     answered German and Japanese in English, and Qwen 3.5 without a
    ///     prompt mixed Chinese characters into other languages. "Latest" so
    ///     a chat that switches language follows the switch; it comes before
    ///     the translation rule, which overrides it.
    ///   * "When asked to translate, rewrite or extract, reply with only the
    ///     result." One translation, no commentary, no preamble.
    ///   * "If you can't verify something, say so" - eight of nine runs
    ///     invented a paper, a phone or a person, and "Be concise" alone cut
    ///     the hedge Gemma 4 gave without a prompt. It also covers a question
    ///     about something never said in the chat, which the previous
    ///     wording named explicitly - and which Gemma 3, reading it folded
    ///     into the user's first message, answered unasked ("I have no
    ///     information about earlier conversations.").
    ///   * "Write math in plain text" because the chat's Markdown renderer
    ///     (`AttributedString(markdown:)`) has no LaTeX support.
    nonisolated static let defaultText =
        "Be concise. For any calculation, show brief steps, then the answer. "
        + "Reply in the language of the user's latest message. "
        + "When asked to translate, rewrite or extract, reply with only the result. "
        + "If you can't verify something, say so instead of inventing details. "
        + "Write math in plain text, not LaTeX."

    /// The conversation the chat screen sends: the history, with the system
    /// prompt as its first turn. A history that already opens with a system
    /// turn is left alone, and an empty prompt adds nothing.
    static func turns(history: [ChatTurn], systemPrompt: String?) -> [ChatTurn] {
        let text = systemPrompt?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !text.isEmpty, history.first?.role != .system else { return history }
        return [.system(text)] + history
    }

    /// The conversation exactly as the chat screen hands it to the engine:
    /// the system prompt (for text models, unless the user turned it off),
    /// then the history. Benchmarks never come through here.
    static func requestTurns(
        history: [ChatTurn],
        systemPrompt: String?,
        capabilities: ModelCapabilities?
    ) -> [ChatTurn] {
        guard let capabilities, applies(to: capabilities) else { return history }
        return turns(history: history, systemPrompt: systemPrompt)
    }

    /// Whether the system prompt applies to a model at all: only models that
    /// read text and write text. Speech-to-text and text-to-speech engines
    /// would ignore it at best.
    nonisolated static func applies(to capabilities: ModelCapabilities) -> Bool {
        capabilities.contains(.textIn) && capabilities.contains(.textOut)
    }

    /// Opens a system text folded into a user turn. Without it Gemma 3 read
    /// the instructions as the start of the user's message and answered
    /// them ("I have no information about earlier conversations." under a
    /// table), and took their English for the language of the message.
    nonisolated static let foldedInstructionsHeader = "Instructions for all your replies:"
    /// Separates the folded instructions from what the user actually wrote,
    /// so "the user's latest message" has one meaning.
    nonisolated static let foldedMessageHeader = "User's message:"

    /// The system turn merged into the first user turn, for models without
    /// a system role: Gemma 2 and 3 on llama.cpp (`ChatTemplateResolver`
    /// folds before llama.cpp's own Gemma format would, with a bare blank
    /// line) and in the app's fallback formats (Gemma, Mistral), and on MLX
    /// and Core ML when a model's Jinja template rejects a system role
    /// (Gemma 2's raises "System role not supported"). The instructions and
    /// the user's text are each labelled, so the model reads the first as
    /// instructions rather than as part of the conversation:
    ///
    ///     Instructions for all your replies:
    ///     Be concise. ...
    ///
    ///     User's message:
    ///     Hallo! ...
    ///
    /// A turn with no text of its own (a photo alone) gets the instructions
    /// only.
    static func foldedIntoFirstUser(_ turns: [ChatTurn]) -> [ChatTurn] {
        let systemText = turns
            .filter { $0.role == .system }
            .map { $0.content.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        var rest = turns.filter { $0.role != .system }
        guard !systemText.isEmpty else { return rest }
        let instructions = foldedInstructionsHeader + "\n" + systemText
        if let first = rest.firstIndex(where: { $0.role == .user }) {
            let turn = rest[first]
            let content = turn.content.isEmpty
                ? instructions
                : instructions + "\n\n" + foldedMessageHeader + "\n" + turn.content
            rest[first] = ChatTurn(role: .user, content: content, attachments: turn.attachments)
        } else {
            rest.insert(.user(instructions), at: 0)
        }
        return rest
    }

    /// The instructions Apple's Foundation Models session is built with:
    /// the last non-empty system turn, or nil.
    static func instructions(in turns: [ChatTurn]) -> String? {
        let text = turns
            .last(where: { $0.role == .system })?
            .content
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return (text?.isEmpty == false) ? text : nil
    }
}

/// The user's version of the system prompt, kept in `UserDefaults`.
///
/// Three states, deliberately distinct:
///   * nothing stored - the built-in default, which follows app updates;
///   * a non-empty text - the user's own instructions;
///   * an empty text - the user turned the system prompt off.
@MainActor
@Observable
final class SystemPromptStore {

    static let shared = SystemPromptStore()

    @ObservationIgnored private let defaults: UserDefaults
    @ObservationIgnored private let key = "chatSystemPrompt"

    /// Bumped on every write so SwiftUI views reading `effective` refresh;
    /// `UserDefaults` itself is not observable.
    private var revision = 0

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    private var stored: String? {
        _ = revision
        return defaults.string(forKey: key)
    }

    /// What the chat sends: the user's text, the default, or nil when the
    /// user turned it off.
    var effective: String? {
        let text = (stored ?? SystemPrompt.defaultText)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text.isEmpty ? nil : text
    }

    /// The text to show in the editor: what is in effect, or empty when off.
    var editableText: String { effective ?? "" }

    var isDefault: Bool { stored == nil }

    var isOff: Bool { effective == nil }

    /// Stores the user's text. Saving the default text verbatim goes back to
    /// "default" so a later app update's wording reaches this user too.
    func set(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed == SystemPrompt.defaultText {
            defaults.removeObject(forKey: key)
        } else {
            defaults.set(trimmed, forKey: key)
        }
        revision += 1
    }

    func resetToDefault() {
        defaults.removeObject(forKey: key)
        revision += 1
    }
}
