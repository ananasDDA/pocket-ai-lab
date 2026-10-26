//
//  ChatMessage.swift
//  local_ai_test
//

import Foundation

struct ChatMessage: Identifiable {
    let id = UUID()
    let role: Role
    var text: String
    var attachments: [ChatAttachment]
    let timestamp: Date = .now
    /// For a user message: the exact transcript this message produced,
    /// history and media markers included, and the template that wrote it.
    /// Filled in when generation starts; nil for messages sent before the
    /// engine could report one.
    var sentPrompt: RenderedPrompt?
    /// For an assistant message: generation ended because it reached the
    /// reply's token cap, not because the model finished. The chat offers
    /// "Continue" under such a reply instead of leaving it looking broken.
    var stoppedAtLimit = false
    /// For an assistant message: the text is the app's explanation of an
    /// empty reply, not something the model wrote. Sent back to the model as
    /// an empty turn so the explanation never becomes part of the transcript.
    var isNotice = false

    init(role: Role, text: String, attachments: [ChatAttachment] = []) {
        self.role = role
        self.text = text
        self.attachments = attachments
    }

    enum Role: String {
        case user = "user"
        case assistant = "assistant"
    }

    var chatRole: ChatRole {
        switch role {
        case .user: return .user
        case .assistant: return .assistant
        }
    }
}

/// How chat messages are rendered in `ChatView`. Persisted in `@AppStorage`.
enum ChatDisplayStyle: String, CaseIterable, Identifiable {
    /// Classic iMessage-like bubbles with alternating alignment.
    case bubble
    /// Document-style chat layout: assistant replies rendered as
    /// full-width formatted Markdown, no bubble chrome.
    case document
    /// Exactly what crosses the engine boundary: the rendered prompt for
    /// each user turn and the model's raw reply, monospaced, unformatted.
    case raw

    var id: String { rawValue }

    var label: String {
        switch self {
        case .bubble:   return "Bubbles"
        case .document: return "Document (Markdown)"
        case .raw:      return "Raw (as sent)"
        }
    }

    var systemImage: String {
        switch self {
        case .bubble:   return "text.bubble"
        case .document: return "doc.text"
        case .raw:      return "curlybraces"
        }
    }
}
