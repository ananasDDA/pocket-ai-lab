//
//  ChatTemplateTests.swift
//
//  Verifies PromptFormatter produces the expected template shape for each
//  model family. Templates are not pixel-perfect; we check structural
//  anchors (role tags, start/end markers) rather than exact whitespace.
//

import Testing
@testable import local_ai_test

struct ChatTemplateTests {

    private let turns: [ChatTurn] = [
        .system("You are a helpful assistant."),
        .user("Hello"),
        .assistant("Hi there!"),
        .user("What's the weather?")
    ]

    @Test func llama3HasEotTokens() {
        let out = PromptFormatter.format(turns: turns, family: .llama3)
        #expect(out.contains("<|begin_of_text|>"))
        #expect(out.contains("<|start_header_id|>system<|end_header_id|>"))
        #expect(out.contains("<|start_header_id|>user<|end_header_id|>"))
        #expect(out.contains("<|start_header_id|>assistant<|end_header_id|>"))
        #expect(out.contains("<|eot_id|>"))
    }

    @Test func gemmaHasTurnTokens() {
        let out = PromptFormatter.format(turns: turns, family: .gemma)
        #expect(out.contains("<start_of_turn>user"))
        #expect(out.contains("<start_of_turn>model"))
        #expect(out.contains("<end_of_turn>"))
    }

    @Test func chatmlHasImStartEnd() {
        let out = PromptFormatter.format(turns: turns, family: .chatml)
        #expect(out.contains("<|im_start|>system"))
        #expect(out.contains("<|im_end|>"))
        #expect(out.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test func qwenUsesChatML() {
        let chatml = PromptFormatter.format(turns: turns, family: .chatml)
        let qwen = PromptFormatter.format(turns: turns, family: .qwen)
        #expect(chatml == qwen)
    }

    @Test func mistralHasInst() {
        let out = PromptFormatter.format(turns: turns, family: .mistral)
        #expect(out.contains("[INST]"))
        #expect(out.contains("[/INST]"))
    }

    @Test func phiHasEndTokens() {
        let out = PromptFormatter.format(turns: turns, family: .phi)
        #expect(out.contains("<|user|>"))
        #expect(out.contains("<|end|>"))
        #expect(out.hasSuffix("<|assistant|>\n"))
    }

    @Test func detectFamilyReturnsExpected() {
        #expect(PromptFormatter.detectFamily(modelId: "mlx-community/Llama-3.2-3B-Instruct-4bit") == .llama3)
        #expect(PromptFormatter.detectFamily(modelId: "mlx-community/gemma-3-4b-it-4bit") == .gemma)
        #expect(PromptFormatter.detectFamily(modelId: "mlx-community/Qwen2.5-3B-Instruct-4bit") == .qwen)
        #expect(PromptFormatter.detectFamily(modelId: "mlx-community/Phi-3.5-mini-instruct-4bit") == .phi)
        #expect(PromptFormatter.detectFamily(modelId: "mlx-community/Mistral-7B-Instruct-v0.3-4bit") == .mistral)
        #expect(PromptFormatter.detectFamily(modelId: "unknown/model") == .plain)
    }
}
