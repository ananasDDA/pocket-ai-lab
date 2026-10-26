//
//  PromptRenderingTests.swift
//
//  What the model actually reads, for the three fixes from the 2026-10-24
//  daily-use run: the default system prompt's wording, Gemma 3's folded
//  instructions, and Qwen 3 / 3.5's history written back the way the model
//  generated it so the KV prefix keeps hitting.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - Default system prompt

@MainActor
struct DefaultSystemPromptTests {

    private func store(_ name: String) -> (SystemPromptStore, UserDefaults) {
        let defaults = UserDefaults(suiteName: "sysprompt-v2-\(name)-\(UUID().uuidString)")!
        return (SystemPromptStore(defaults: defaults), defaults)
    }

    /// One sentence per finding of the graded run; see `SystemPrompt.defaultText`.
    @Test func everyRuleIsThere() {
        let text = SystemPrompt.defaultText
        #expect(text.hasPrefix("Be concise."))
        #expect(text.contains("show brief steps, then the answer"))
        #expect(text.contains("language of the user's latest message"))
        #expect(text.contains("reply with only the result"))
        #expect(text.contains("can't verify something, say so"))
        #expect(text.contains("not LaTeX"))
    }

    /// The sentence Gemma 3 answered unasked when it was folded into the
    /// user's first message is gone; the verification rule covers it.
    @Test func noSentenceThatReadsLikeAQuestionAboutTheChat() {
        #expect(!SystemPrompt.defaultText.contains("earlier in this chat"))
    }

    /// The language rule comes before the translation rule, which is the
    /// more specific one and has to win for "translate this into Spanish".
    @Test func translationRuleComesAfterTheLanguageRule() throws {
        let text = SystemPrompt.defaultText
        let language = try #require(text.range(of: "Reply in the language"))
        let translate = try #require(text.range(of: "When asked to translate"))
        #expect(language.lowerBound < translate.lowerBound)
    }

    /// Someone who never edited the prompt stores nothing, so the new wording
    /// reaches them with the update.
    @Test func aUserWhoNeverEditedItGetsTheNewText() {
        let (s, defaults) = store("untouched")
        #expect(defaults.object(forKey: "chatSystemPrompt") == nil)
        #expect(s.isDefault)
        #expect(s.effective == SystemPrompt.defaultText)
    }

    /// A prompt the user wrote is theirs: an app update never rewrites it,
    /// even when it is close to an older default.
    @Test func aCustomPromptIsNotReplacedByTheNewDefault() {
        let (s, defaults) = store("custom")
        let own = "Be concise. Answer in Spanish. Write math in plain text, not LaTeX."
        defaults.set(own, forKey: "chatSystemPrompt")
        #expect(!s.isDefault)
        #expect(s.effective == own)
        #expect(SystemPrompt.requestTurns(
            history: [.user("Hola")], systemPrompt: s.effective, capabilities: .textOnly
        ).first == .system(own))
    }

    @Test func turnedOffStaysOff() {
        let (s, defaults) = store("off")
        defaults.set("", forKey: "chatSystemPrompt")
        #expect(s.isOff)
        #expect(s.effective == nil)
    }
}

// MARK: - Gemma 3: folded instructions

@MainActor
struct GemmaFoldedPromptTests {

    private let german = "Hallo! Ich möchte morgens mehr Energie haben. Hast du drei einfache Tipps für mich?"

    private func templateStore(_ name: String) -> ChatTemplateStore {
        ChatTemplateStore(defaults: UserDefaults(suiteName: "gemma-fold-\(name)-\(UUID().uuidString)")!)
    }

    /// An excerpt of Gemma 3's `chat_template` from its GGUF. llama.cpp
    /// b8882 recognises it by `<start_of_turn>` and renders its own Gemma
    /// format.
    private let gemma3Template = """
    {{ bos_token }}{%- if messages[0]['role'] == 'system' -%}{%- set first_user_prefix = \
    messages[0]['content'] + '\\n\\n' -%}{%- set loop_messages = messages[1:] -%}{%- else -%}\
    {%- set first_user_prefix = "" -%}{%- set loop_messages = messages -%}{%- endif -%}\
    {%- for message in loop_messages -%}{{ '<start_of_turn>' + role + '\\n' + \
    (first_user_prefix if loop.first else "") }}{{ message['content'] | trim }}\
    {{ '<end_of_turn>\\n' }}{%- endfor -%}{%- if add_generation_prompt -%}\
    {{'<start_of_turn>model\\n'}}{%- endif -%}
    """

    private var expected: String {
        "<start_of_turn>user\n"
            + "Instructions for all your replies:\n"
            + SystemPrompt.defaultText + "\n"
            + "\n"
            + "User's message:\n"
            + german + "<end_of_turn>\n"
            + "<start_of_turn>model\n"
    }

    private var request: [ChatTurn] { [.system(SystemPrompt.defaultText), .user(german)] }

    /// The app's own Gemma format (no template in the file, or one llama.cpp
    /// cannot render). Before: the instructions, a blank line and the German
    /// question ran together as one user message.
    @Test func fallbackLabelsInstructionsAndMessage() {
        let rendered = ChatTemplateResolver.render(
            turns: request, modelID: "unsloth/gemma-3-4b-it-GGUF", embedded: nil,
            store: templateStore("fallback")
        )
        #expect(rendered.source == .fallback(.gemma))
        #expect(rendered.text == expected)
        #expect(!rendered.text.contains(SystemPrompt.defaultText + "\n\n" + german))
    }

    /// The model's own template: the resolver folds before llama.cpp would,
    /// so llama.cpp sees no system turn and the transcript is the same as
    /// the fallback's.
    @Test func ownTemplateGetsTheSameLabelledFold() throws {
        try #require(BackendAvailability.hasLlama)
        let rendered = ChatTemplateResolver.render(
            turns: request, modelID: "unsloth/gemma-3-4b-it-GGUF", embedded: gemma3Template,
            store: templateStore("own")
        )
        #expect(rendered.source == .model)
        #expect(rendered.text == expected)
    }

    @Test func onlyGemmaStyleTemplatesAreFoldedFirst() {
        #expect(ChatTemplateResolver.foldsSystemTurn(gemma3Template))
        #expect(!ChatTemplateResolver.foldsSystemTurn("{{'<|im_start|>' + message['role'] }}"))
        #expect(!ChatTemplateResolver.foldsSystemTurn("{{- '<|turn>' + role + '\\n' }}"))
    }

    /// Only the first user turn carries the labels; later turns are the
    /// user's text alone, and the alternation stays user / model.
    @Test func laterTurnsAreLeftAlone() {
        let turns: [ChatTurn] = request + [.assistant("1. Wasser trinken."), .user("Welcher ist am wichtigsten?")]
        let text = PromptFormatter.format(turns: turns, family: .gemma)
        #expect(text.components(separatedBy: SystemPrompt.foldedInstructionsHeader).count == 2)
        #expect(text.components(separatedBy: SystemPrompt.foldedMessageHeader).count == 2)
        #expect(text.hasSuffix(
            "<start_of_turn>model\n1. Wasser trinken.<end_of_turn>\n"
                + "<start_of_turn>user\nWelcher ist am wichtigsten?<end_of_turn>\n<start_of_turn>model\n"
        ))
    }

    /// A photo sent without text: the instructions alone, no empty label.
    @Test func aTurnWithoutTextGetsNoMessageLabel() {
        let image = ChatAttachment.image(Data([1]))
        let folded = SystemPrompt.foldedIntoFirstUser([.system("Be brief."), .user("", attachments: [image])])
        #expect(folded == [.user("Instructions for all your replies:\nBe brief.", attachments: [image])])
    }

    /// Families with a system role keep it: the labels never appear there.
    @Test(arguments: [PromptFamily.gemma4, .qwen, .llama3, .phi, .chatml])
    func familiesWithASystemRoleAreNotFolded(family: PromptFamily) {
        let text = PromptFormatter.format(turns: request, family: family)
        #expect(!text.contains(SystemPrompt.foldedInstructionsHeader))
        #expect(!text.contains(SystemPrompt.foldedMessageHeader))
    }
}

// MARK: - History as generated (KV prefix reuse)

@MainActor
struct HistoryAsGeneratedTests {

    /// The ChatML template Qwen ships; llama.cpp b8882 renders it as plain
    /// ChatML.
    private let chatML = """
    {% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' \
    + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}\
    {% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
    """

    private func templateStore(_ name: String) -> ChatTemplateStore {
        ChatTemplateStore(defaults: UserDefaults(suiteName: "history-\(name)-\(UUID().uuidString)")!)
    }

    private let reply = "Day 1: Ribeira and the river.\nDay 2: Serralves."

    private var turn1: [ChatTurn] {
        [.system(SystemPrompt.defaultText), .user("Plan 3 days in Porto on 300 euros.")]
    }

    private var turn2: [ChatTurn] {
        turn1 + [.assistant(reply), .user("Make day 2 cheaper.")]
    }

    /// The KV cache after turn 1 holds turn 1's prompt plus the reply's
    /// tokens. Turn 2's prompt has to start with exactly that, or the engine
    /// re-reads the reply (the run's "evaluated = previous reply + new
    /// message" on every Qwen turn).
    private func continues(_ next: String, _ previous: String) -> Bool {
        next.hasPrefix(previous + reply)
    }

    @Test(arguments: ["unsloth/Qwen3.5-2B-GGUF", "unsloth/Qwen3.5-0.8B-GGUF", "Qwen/Qwen3-4B-GGUF"])
    func qwen3FallbackHistoryExtendsTheCachedPrompt(modelID: String) {
        let s = templateStore("fallback")
        let first = ChatTemplateResolver.render(turns: turn1, modelID: modelID, embedded: nil, store: s)
        let second = ChatTemplateResolver.render(turns: turn2, modelID: modelID, embedded: nil, store: s)
        #expect(continues(second.text, first.text))
        #expect(second.text.contains(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" + reply + "<|im_end|>\n<|im_start|>user\nMake day 2 cheaper."
        ))
        #expect(second.text.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    @Test func qwen35OwnTemplateHistoryExtendsTheCachedPrompt() throws {
        try #require(BackendAvailability.hasLlama)
        let s = templateStore("own")
        let id = "unsloth/Qwen3.5-2B-GGUF"
        let first = ChatTemplateResolver.render(turns: turn1, modelID: id, embedded: chatML, store: s)
        let second = ChatTemplateResolver.render(turns: turn2, modelID: id, embedded: chatML, store: s)
        #expect(first.source == .model)
        #expect(continues(second.text, first.text))
    }

    /// The first turn - what the model sees before it has said anything - is
    /// exactly what it was.
    @Test func theFirstTurnIsUnchanged() {
        let rendered = ChatTemplateResolver.render(
            turns: turn1, modelID: "unsloth/Qwen3.5-2B-GGUF", embedded: nil, store: templateStore("first")
        )
        #expect(rendered.text == "<|im_start|>system\n" + SystemPrompt.defaultText + "<|im_end|>\n"
            + "<|im_start|>user\nPlan 3 days in Porto on 300 euros.<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    @Test func everyEarlierReplyGetsTheBlockOnce() {
        let turns: [ChatTurn] = turn2 + [.assistant("Cheaper day 2."), .user("Thanks."),
                                         .assistant("<think>\n\n</think>\n\nAlready marked."), .user("Bye.")]
        let sent = ChatTemplateResolver.historyAsGenerated(turns, modelID: "unsloth/Qwen3.5-4B-GGUF")
        let replies = sent.filter { $0.role == .assistant }.map(\.content)
        #expect(replies == [
            "<think>\n\n</think>\n\n" + reply,
            "<think>\n\n</think>\n\nCheaper day 2.",
            "<think>\n\n</think>\n\nAlready marked.",
        ])
        #expect(sent.filter { $0.role != .assistant } == turns.filter { $0.role != .assistant })
    }

    /// Qwen 2.5 never got the block at generation time, so its history is
    /// left as it is.
    @Test func qwen25HistoryIsUntouched() {
        let s = templateStore("qwen25")
        let id = "mlx-community/Qwen2.5-3B-Instruct-4bit"
        let first = ChatTemplateResolver.render(turns: turn1, modelID: id, embedded: nil, store: s)
        let second = ChatTemplateResolver.render(turns: turn2, modelID: id, embedded: nil, store: s)
        #expect(!second.text.contains("<think>"))
        #expect(continues(second.text, first.text))
    }

    /// A user's template is rendered exactly as written: no reasoning block
    /// at the end, none in the history.
    @Test func aCustomTemplateGetsNoBlockInTheHistory() throws {
        try #require(BackendAvailability.hasLlama)
        let s = templateStore("custom")
        s.setCustom(chatML, for: "unsloth/Qwen3.5-2B-GGUF")
        let rendered = ChatTemplateResolver.render(
            turns: turn2, modelID: "unsloth/Qwen3.5-2B-GGUF", embedded: chatML, store: s
        )
        #expect(rendered.source == .custom)
        #expect(!rendered.text.contains("<think>"))
    }

    /// The same property for every family in the catalog: the next turn's
    /// prompt starts with the previous prompt plus the reply, so the KV
    /// prefix cache reuses all of it.
    @Test(arguments: [
        PromptFamily.llama3, .gemma, .gemma4, .qwen, .chatml, .phi, .mistral, .ernie,
    ])
    func everyCatalogFamilyRendersHistoryAsGenerated(family: PromptFamily) {
        let first = PromptFormatter.format(turns: turn1, family: family)
        let second = PromptFormatter.format(turns: turn2, family: family)
        #expect(continues(second, first))
    }
}
