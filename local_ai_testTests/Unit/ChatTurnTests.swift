//
//  ChatTurnTests.swift
//

import Testing
import Foundation
@testable import local_ai_test

struct ChatTurnTests {

    @Test func textOnlyTurn() {
        let t = ChatTurn.user("hello")
        #expect(t.role == .user)
        #expect(t.content == "hello")
        #expect(!t.hasAttachments)
    }

    @Test func imageTurn() {
        let data = Data(repeating: 0xFF, count: 100)
        let t = ChatTurn.user("look", attachments: [.image(data)])
        #expect(t.hasAttachments)
        #expect(t.attachments.first?.kind == .image)
        #expect(t.attachments.first?.approximateBytes == 100)
    }

    @Test func audioAttachmentApproxBytes() {
        let d = Data(repeating: 0, count: 16_000 * 4)
        let att: ChatAttachment = .audio(d, sampleRate: 16_000)
        #expect(att.approximateBytes == 64_000)
        #expect(att.kind == .audio)
    }

    @Test func videoFramesKind() {
        let frames = [Data(count: 10), Data(count: 20), Data(count: 5)]
        let att: ChatAttachment = .videoFrames(frames, durationSeconds: 3)
        #expect(att.kind == .video)
        #expect(att.approximateBytes == 35)
    }

    @Test func systemUserAssistantConstructors() {
        #expect(ChatTurn.system("sys").role == .system)
        #expect(ChatTurn.user("u").role == .user)
        #expect(ChatTurn.assistant("a").role == .assistant)
    }
}
