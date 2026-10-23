//
//  ChatTemplateTests2.swift
//
//  Choosing the transcript format: the user's template, then the model's
//  own, then the app's guess by family.
//
//  Named to sit beside the existing ChatTemplateTests, which covers the
//  built-in `PromptFormatter` families.
//

import Testing
import Foundation
@testable import local_ai_test

@MainActor
struct ChatTemplateResolverTests {

    /// A ChatML template llama.cpp implements natively - the same string
    /// Qwen ships in its GGUF metadata.
    private let chatML = """
    {% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' \
    + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}\
    {% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
    """

    private func store(_ name: String) -> ChatTemplateStore {
        let defaults = UserDefaults(suiteName: "template-tests-\(name)-\(UUID().uuidString)")!
        return ChatTemplateStore(defaults: defaults)
    }

    private let turns: [ChatTurn] = [.user("Hello.")]

    // MARK: - Store

    @Test func aCustomTemplateIsKeptPerModel() {
        let s = store("per-model")
        s.setCustom("A", for: "repo/one")
        #expect(s.custom(for: "repo/one") == "A")
        #expect(s.custom(for: "repo/two") == nil)
        #expect(s.hasCustom(for: "repo/one"))
    }

    @Test func clearingRemovesTheOverride() {
        let s = store("clear")
        s.setCustom("A", for: "repo/one")
        s.setCustom(nil, for: "repo/one")
        #expect(s.custom(for: "repo/one") == nil)
        #expect(!s.hasCustom(for: "repo/one"))
    }

    @Test func whitespaceOnlyIsNoTemplateAtAll() {
        let s = store("blank")
        s.setCustom("   \n ", for: "repo/one")
        #expect(s.custom(for: "repo/one") == nil)
    }

    // MARK: - Resolution order

    @Test func withoutAnyTemplateTheAppFormatsByFamily() {
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "unsloth/gemma-3-4b-it-GGUF",
            embedded: nil,
            store: store("fallback")
        )
        #expect(rendered.source == .fallback(.gemma))
        #expect(rendered.template == nil)
        #expect(rendered.text.contains("<start_of_turn>"))
    }

    // Only meaningful where llama.cpp is linked: the renderer is what turns
    // a template string into a transcript.
    @Test func theModelsOwnTemplateWinsOverTheGuess() throws {
        try #require(BackendAvailability.hasLlama)
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            // An id that would otherwise be formatted as Gemma.
            modelID: "unsloth/gemma-3-4b-it-GGUF",
            embedded: chatML,
            store: store("embedded")
        )
        #expect(rendered.source == .model)
        #expect(rendered.template == chatML)
        #expect(rendered.text.contains("<|im_start|>"))
        #expect(!rendered.text.contains("<start_of_turn>"))
    }

    /// Gemma 4's GGUF template matches none of the formats llama.cpp b8882
    /// sniffs for, so rendering it fails and the Gemma 4 fallback - not
    /// Gemma 3's - is what the model gets. If a llama.cpp bump ever learns
    /// the format, this test starts failing and the fallback can go.
    @Test func gemma4sOwnTemplateFallsBackToTheGemma4Format() throws {
        try #require(BackendAvailability.hasLlama)
        let gemma4Excerpt = """
        {{- bos_token -}}{%- for message in messages -%}        {%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}        {{- '<|turn>' + role + '\n' }}{{- message['content'] | trim -}}{{- '<turn|>\n' -}}        {%- endfor -%}{%- if add_generation_prompt -%}{{- '<|turn>model\n' -}}{%- endif -%}
        """
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "unsloth/gemma-4-E2B-it-GGUF",
            embedded: gemma4Excerpt,
            store: store("gemma4")
        )
        #expect(rendered.source == .fallback(.gemma4))
        #expect(rendered.text == "<|turn>user\nHello.<turn|>\n<|turn>model\n")
    }

    // MARK: - Qwen 3 / 3.5 thinking off

    /// Qwen 3.5's GGUF template is rendered by llama.cpp as plain ChatML, so
    /// the empty reasoning block its own template adds has to be appended
    /// here, or the model spends the reply budget inside <think>.
    @Test func qwen35sOwnTemplateEndsWithAnEmptyReasoningBlock() throws {
        try #require(BackendAvailability.hasLlama)
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "unsloth/Qwen3.5-0.8B-GGUF",
            embedded: chatML,
            store: store("qwen35-model")
        )
        #expect(rendered.source == .model)
        #expect(rendered.text.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    @Test func qwen3sFallbackEndsWithAnEmptyReasoningBlock() {
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "Qwen/Qwen3-4B-GGUF",
            embedded: nil,
            store: store("qwen3-fallback")
        )
        #expect(rendered.source == .fallback(.qwen))
        #expect(rendered.text == "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test func qwen25IsLeftWithoutAReasoningBlock() {
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "mlx-community/Qwen2.5-3B-Instruct-4bit",
            embedded: nil,
            store: store("qwen25")
        )
        #expect(rendered.text.hasSuffix("<|im_start|>assistant\n"))
        #expect(!rendered.text.contains("<think>"))
    }

    @Test func aCustomTemplateIsNeverGivenAReasoningBlock() throws {
        try #require(BackendAvailability.hasLlama)
        let s = store("qwen35-custom")
        s.setCustom(chatML, for: "unsloth/Qwen3.5-2B-GGUF")
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "unsloth/Qwen3.5-2B-GGUF",
            embedded: chatML,
            store: s
        )
        #expect(rendered.source == .custom)
        #expect(rendered.text.hasSuffix("<|im_start|>assistant\n"))
    }

    @Test func aCustomTemplateWinsOverTheModelsOwn() throws {
        try #require(BackendAvailability.hasLlama)
        let s = store("custom-wins")
        s.setCustom(chatML, for: "repo/model")
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "repo/model",
            embedded: "{% for m in messages %}{{ 'nonsense' }}{% endfor %}",
            store: s
        )
        #expect(rendered.source == .custom)
        #expect(rendered.text.contains("<|im_start|>"))
    }

    // A template llama.cpp cannot render must cost a worse transcript, not
    // a dead chat.
    @Test func anUnrenderableTemplateFallsThrough() throws {
        try #require(BackendAvailability.hasLlama)
        let s = store("unrenderable")
        s.setCustom("not a template at all", for: "repo/model")
        let rendered = ChatTemplateResolver.render(
            turns: turns,
            modelID: "repo/model",
            embedded: nil,
            store: s
        )
        #expect(rendered.source == .fallback(.plain))
        #expect(!rendered.text.isEmpty)
    }

    @Test func theSourceIsReportableBeforeAnythingIsSent() throws {
        try #require(BackendAvailability.hasLlama)
        #expect(ChatTemplateResolver.source(
            modelID: "repo/model", embedded: chatML, store: store("source")
        ) == .model)
        #expect(ChatTemplateResolver.source(
            modelID: "mlx-community/Qwen2.5-3B-Instruct-4bit", embedded: nil, store: store("source2")
        ) == .fallback(.qwen))
    }

    // MARK: - Labels

    @Test func everySourceSaysWhatItIs() {
        #expect(ChatTemplateSource.model.label == "Model's own")
        #expect(ChatTemplateSource.custom.label == "Custom")
        #expect(ChatTemplateSource.fallback(.qwen).label == "App fallback (qwen)")
        #expect(ChatTemplateSource.model.isAuthoritative)
        #expect(!ChatTemplateSource.fallback(.plain).isAuthoritative)
    }
}
