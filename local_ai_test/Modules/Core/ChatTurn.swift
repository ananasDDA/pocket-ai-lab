//
//  ChatTurn.swift
//  local_ai_test
//
//  A single exchange in a conversation. Role + text + optional attachments.
//  The inference pipeline consumes `[ChatTurn]` and returns a stream of
//  `ChatOutput`s.
//

import Foundation

enum ChatRole: String, Sendable, Codable {
    case system
    case user
    case assistant
}

struct ChatTurn: Sendable, Hashable {
    let role: ChatRole
    let content: String
    let attachments: [ChatAttachment]

    init(role: ChatRole, content: String, attachments: [ChatAttachment] = []) {
        self.role = role
        self.content = content
        self.attachments = attachments
    }

    static func system(_ text: String) -> ChatTurn {
        ChatTurn(role: .system, content: text)
    }

    static func user(_ text: String, attachments: [ChatAttachment] = []) -> ChatTurn {
        ChatTurn(role: .user, content: text, attachments: attachments)
    }

    static func assistant(_ text: String) -> ChatTurn {
        ChatTurn(role: .assistant, content: text)
    }

    var hasAttachments: Bool { !attachments.isEmpty }
}
