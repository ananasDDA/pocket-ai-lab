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

    var id: String { rawValue }

    var label: String {
        switch self {
        case .bubble:   return "Bubbles"
        case .document: return "Document (Markdown)"
        }
    }

    var systemImage: String {
        switch self {
        case .bubble:   return "text.bubble"
        case .document: return "doc.text"
        }
    }
}
