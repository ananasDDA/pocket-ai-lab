//
//  ChatDefaultsTests.swift
//
//  The 1.2 chat defaults: the system prompt every chat starts with, the
//  512-token reply cap and what it frees for history, and the thermal
//  indicator's decision. All pure logic - no model, no device state.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - System prompt: text and insertion

@MainActor
struct SystemPromptInsertionTests {

    private let history: [ChatTurn] = [
        .user("What is 2 + 2?"),
        .assistant("4"),
        .user("And times 3?"),
    ]

    /// About 65 tokens on Llama 3's tokenizer; 300 characters is the cap
    /// that keeps it under 70 on every tokenizer in the catalog.
    @Test func defaultTextIsShortEnglishAndMentionsTheRules() {
        let text = SystemPrompt.defaultText
        #expect(text.count <= 300)
        #expect(BenchmarkEngine.estimatedTokenCount(text) <= 75)
        #expect(text.unicodeScalars.allSatisfy { $0.isASCII })
        #expect(text.contains("concise"))
        #expect(text.contains("LaTeX"))
    }

    @Test func chatRequestStartsWithTheSystemTurn() {
        let turns = SystemPrompt.requestTurns(
            history: history,
            systemPrompt: SystemPrompt.defaultText,
            capabilities: .textOnly
        )
        #expect(turns.count == history.count + 1)
        #expect(turns.first == .system(SystemPrompt.defaultText))
        #expect(Array(turns.dropFirst()) == history)
    }

    @Test func visionModelsGetItToo() {
        let turns = SystemPrompt.requestTurns(
            history: [.user("What is this?", attachments: [.image(Data([1, 2, 3]))])],
            systemPrompt: SystemPrompt.defaultText,
            capabilities: .vision
        )
        #expect(turns.first?.role == .system)
        #expect(turns.last?.attachments.count == 1)
    }

    @Test func speechAndTTSModelsDoNotGetIt() {
        for caps in [ModelCapabilities.speechToText, .textToSpeech] {
            let turns = SystemPrompt.requestTurns(
                history: history, systemPrompt: SystemPrompt.defaultText, capabilities: caps
            )
            #expect(turns == history)
        }
        #expect(SystemPrompt.requestTurns(
            history: history, systemPrompt: SystemPrompt.defaultText, capabilities: nil
        ) == history)
    }

    @Test func anEmptyOrMissingPromptAddsNothing() {
        for prompt in [nil, "", "   \n "] as [String?] {
            #expect(SystemPrompt.turns(history: history, systemPrompt: prompt) == history)
        }
    }

    @Test func anExistingSystemTurnIsNotDoubled() {
        let own: [ChatTurn] = [.system("Answer in French.")] + history
        #expect(SystemPrompt.turns(history: own, systemPrompt: SystemPrompt.defaultText) == own)
    }

    // MARK: Per prompt family (the app's fallback formats)

    private var request: [ChatTurn] {
        SystemPrompt.turns(history: history, systemPrompt: SystemPrompt.defaultText)
    }

    private func occurrences(of needle: String, in text: String) -> Int {
        text.components(separatedBy: needle).count - 1
    }

    @Test(arguments: [
        PromptFamily.llama3, .llama2, .gemma, .gemma4, .qwen, .mistral, .phi, .chatml, .ernie, .plain,
    ])
    func everyFamilySendsTheSystemPromptExactlyOnce(family: PromptFamily) {
        let text = PromptFormatter.format(turns: request, family: family)
        #expect(occurrences(of: SystemPrompt.defaultText, in: text) == 1)
        // The instructions come before the first question in every format.
        let sys = text.range(of: SystemPrompt.defaultText)!.lowerBound
        let question = text.range(of: "What is 2 + 2?")!.lowerBound
        #expect(sys < question)
    }

    @Test func familiesWithASystemRoleUseIt() {
        #expect(PromptFormatter.format(turns: request, family: .qwen)
            .hasPrefix("<|im_start|>system\n" + SystemPrompt.defaultText + "<|im_end|>"))
        #expect(PromptFormatter.format(turns: request, family: .gemma4)
            .hasPrefix("<|turn>system\n" + SystemPrompt.defaultText + "<turn|>"))
        #expect(PromptFormatter.format(turns: request, family: .llama3)
            .contains("<|start_header_id|>system<|end_header_id|>\n\n" + SystemPrompt.defaultText))
        #expect(PromptFormatter.format(turns: request, family: .phi)
            .hasPrefix("<|system|>\n" + SystemPrompt.defaultText))
        #expect(PromptFormatter.format(turns: request, family: .llama2)
            .contains("<<SYS>>\n" + SystemPrompt.defaultText + "\n<</SYS>>"))
    }

    /// Gemma 3 has no system role: the instructions open the first user
    /// turn and the user/model alternation stays intact.
    @Test func gemmaFoldsItIntoTheFirstUserTurn() {
        let text = PromptFormatter.format(turns: request, family: .gemma)
        #expect(!text.contains("<start_of_turn>system"))
        #expect(text.hasPrefix(
            "<start_of_turn>user\nInstructions for all your replies:\n" + SystemPrompt.defaultText
                + "\n\nUser's message:\nWhat is 2 + 2?<end_of_turn>\n"
        ))
        #expect(occurrences(of: "<start_of_turn>user", in: text) == 2)
    }

    @Test func mistralFoldsItIntoTheFirstInstruction() {
        let text = PromptFormatter.format(turns: request, family: .mistral)
        #expect(text.hasPrefix(
            "<s>[INST] Instructions for all your replies:\n" + SystemPrompt.defaultText
                + "\n\nUser's message:\nWhat is 2 + 2? [/INST]"
        ))
        #expect(occurrences(of: "[INST]", in: text) == 2)
    }

    // MARK: Through the template resolver (llama.cpp text and vision)

    private func templateStore(_ name: String) -> ChatTemplateStore {
        ChatTemplateStore(defaults: UserDefaults(suiteName: "sysprompt-\(name)-\(UUID().uuidString)")!)
    }

    @Test func resolverFallbackCarriesTheSystemPrompt() {
        let rendered = ChatTemplateResolver.render(
            turns: request, modelID: "unsloth/gemma-3-4b-it-GGUF", embedded: nil,
            store: templateStore("fallback")
        )
        #expect(rendered.source == .fallback(.gemma))
        #expect(rendered.text.contains(SystemPrompt.defaultText))
    }

    /// The system prompt is a turn, never an edit of the user's template:
    /// the stored custom template is byte-for-byte what the user saved.
    @Test func aCustomTemplateIsNeverRewritten() {
        let store = templateStore("custom")
        let custom = """
        {% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' \
        + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}\
        {% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}
        """
        store.setCustom(custom, for: "someone/model-GGUF")
        let saved = store.custom(for: "someone/model-GGUF")
        let rendered = ChatTemplateResolver.render(
            turns: request, modelID: "someone/model-GGUF", embedded: nil, store: store
        )
        #expect(store.custom(for: "someone/model-GGUF") == saved)
        #expect(rendered.text.contains(SystemPrompt.defaultText))
        if rendered.source == .custom {
            // llama.cpp is linked (device): the template it used is the saved one.
            #expect(rendered.template == saved)
        }
    }

    /// llama.cpp vision trims history to fit; the system turn is never what
    /// goes, and it stays first.
    @Test func visionTrimKeepsTheSystemTurn() {
        let long = String(repeating: "word ", count: 400)
        var turns: [ChatTurn] = [.system(SystemPrompt.defaultText)]
        for i in 0..<10 {
            turns.append(.user("\(i) " + long))
            turns.append(.assistant(long))
        }
        turns.append(.user("Last question?", attachments: [.image(Data([0]))]))
        let prepared = VisionPromptBuilder.build(turns: turns, family: .gemma, budgetTokens: 900)
        #expect(prepared.droppedTurns > 0)
        #expect(prepared.turns.first == .system(SystemPrompt.defaultText))
        #expect(prepared.prompt.contains(SystemPrompt.defaultText))
    }
}

// MARK: - System prompt: engine paths without a transcript string

@MainActor
struct SystemPromptEnginePathTests {

    /// MLX and Core ML retry with this when a model's own Jinja template
    /// refuses a system role (Gemma 2).
    @Test func foldingMergesTheSystemTurnIntoTheFirstUserTurn() {
        let image = ChatAttachment.image(Data([9]))
        let turns: [ChatTurn] = [
            .system("Be brief."),
            .user("Hi", attachments: [image]),
            .assistant("Hello"),
            .user("Bye"),
        ]
        let folded = SystemPrompt.foldedIntoFirstUser(turns)
        #expect(folded == [
            .user("Instructions for all your replies:\nBe brief.\n\nUser's message:\nHi", attachments: [image]),
            .assistant("Hello"),
            .user("Bye"),
        ])
    }

    @Test func foldingWithoutASystemTurnChangesNothing() {
        let turns: [ChatTurn] = [.user("Hi"), .assistant("Hello")]
        #expect(SystemPrompt.foldedIntoFirstUser(turns) == turns)
    }

    @Test func foldingWithoutAUserTurnStillKeepsTheInstructions() {
        #expect(SystemPrompt.foldedIntoFirstUser([.system("Be brief.")])
            == [.user("Instructions for all your replies:\nBe brief.")])
    }

    /// Apple Foundation Models: the system turn becomes the session's
    /// instructions.
    @Test func appleSessionInstructionsComeFromTheSystemTurn() {
        let turns = SystemPrompt.turns(history: [.user("Hi")], systemPrompt: SystemPrompt.defaultText)
        #expect(SystemPrompt.instructions(in: turns) == SystemPrompt.defaultText)
        #expect(SystemPrompt.instructions(in: [.user("Hi")]) == nil)
        #expect(SystemPrompt.instructions(in: [.system("  "), .user("Hi")]) == nil)
    }

    @Test func appleSessionInspectorShowsTheInstructions() {
        guard #available(iOS 26, *) else { return }
        let rendered = AppleIntelligenceEngine.renderedPrompt(
            instructions: SystemPrompt.defaultText, prompt: "Hi"
        )
        #expect(rendered.source == .session)
        #expect(rendered.text.contains(SystemPrompt.defaultText))
        #expect(rendered.text.hasSuffix("Hi"))
    }
}

// MARK: - Benchmarks never get it

@MainActor
struct BenchmarkSystemPromptIsolationTests {

    /// Pins exactly what a quality task sends, so the chat's system prompt
    /// can never leak into scores.
    @Test func aTaskWithoutASystemPromptSendsOnlyItsPrompt() {
        let task = BenchmarkTask(id: "t1", category: "math", prompt: "2+2?", check: .exactMatch("4"))
        #expect(BenchmarkEngine.qualityTurns(for: task) == [.user("2+2?")])
    }

    @Test func aTaskKeepsItsOwnSystemPromptAndNothingElse() {
        let task = BenchmarkTask(
            id: "t2", category: "retrieval", prompt: "Who?",
            systemPrompt: "Answer from the passage only.", check: .exactMatch("x")
        )
        #expect(BenchmarkEngine.qualityTurns(for: task) == [
            .system("Answer from the passage only."),
            .user("Who?"),
        ])
    }

    @Test func noBundledTaskCarriesTheChatPrompt() throws {
        let url = try #require(BenchmarkSuiteFile.bundledURL)
        let suite = try #require(BenchmarkSuiteFile.decode(try Data(contentsOf: url)))
        #expect(!suite.tasks.isEmpty)
        for task in suite.tasks {
            let turns = BenchmarkEngine.qualityTurns(for: task)
            #expect(!turns.contains { $0.content.contains(SystemPrompt.defaultText) })
            #expect(turns.last == .user(task.prompt))
            #expect(turns.count == ((task.systemPrompt?.isEmpty == false) ? 2 : 1))
        }
    }
}

// MARK: - System prompt: user override and reset

@MainActor
struct SystemPromptStoreTests {

    private func store(_ name: String) -> SystemPromptStore {
        SystemPromptStore(defaults: UserDefaults(suiteName: "sysprompt-store-\(name)-\(UUID().uuidString)")!)
    }

    @Test func startsOnTheDefault() {
        let s = store("fresh")
        #expect(s.isDefault)
        #expect(!s.isOff)
        #expect(s.effective == SystemPrompt.defaultText)
    }

    @Test func aCustomPromptIsUsedAndTrimmed() {
        let s = store("custom")
        s.set("  Answer in Spanish.\n")
        #expect(!s.isDefault)
        #expect(s.effective == "Answer in Spanish.")
        #expect(SystemPrompt.requestTurns(
            history: [.user("Hi")], systemPrompt: s.effective, capabilities: .textOnly
        ).first == .system("Answer in Spanish."))
    }

    @Test func savingEmptyTurnsItOff() {
        let s = store("off")
        s.set("   ")
        #expect(s.isOff)
        #expect(!s.isDefault)
        #expect(s.effective == nil)
        #expect(s.editableText.isEmpty)
        #expect(SystemPrompt.requestTurns(
            history: [.user("Hi")], systemPrompt: s.effective, capabilities: .textOnly
        ) == [.user("Hi")])
    }

    @Test func resetGoesBackToTheDefault() {
        let s = store("reset")
        s.set("Something else.")
        s.resetToDefault()
        #expect(s.isDefault)
        #expect(s.effective == SystemPrompt.defaultText)
        s.set("")
        s.resetToDefault()
        #expect(s.effective == SystemPrompt.defaultText)
    }

    /// Saving the default verbatim stores nothing, so a later app update's
    /// wording still reaches this user.
    @Test func savingTheDefaultTextKeepsFollowingTheDefault() {
        let s = store("same")
        s.set(SystemPrompt.defaultText)
        #expect(s.isDefault)
    }

    @Test func itIsOneSettingForEveryModel() {
        let defaults = UserDefaults(suiteName: "sysprompt-shared-\(UUID().uuidString)")!
        SystemPromptStore(defaults: defaults).set("Shared.")
        #expect(SystemPromptStore(defaults: defaults).effective == "Shared.")
    }
}

// MARK: - Reply cap and context budget

@MainActor
struct ReplyCapTests {

    @Test func chatCapIs512AndOnlyTheCapDiffers() {
        let chat = GenerationParameters.chat
        let base = GenerationParameters.default
        #expect(chat.maxTokens == 512)
        #expect(ChatViewModel.parameters == chat)
        var same = chat
        same.maxTokens = base.maxTokens
        #expect(same == base)
    }

    /// Benchmark and the other callers keep their caps.
    @Test func defaultAndDeterministicAreUnchanged() {
        #expect(GenerationParameters.default.maxTokens == 1024)
        #expect(GenerationParameters.deterministic.maxTokens == 1024)
        #expect(GenerationParameters.deterministic.temperature == 0)
    }

    /// llama.cpp text window: 4096 tokens. The reply reservation follows the
    /// cap, so the 512 tokens the lower cap frees go to history.
    @Test func lowerCapGivesTheRoomToHistoryIn4096() {
        #expect(ContextBudget.reservedForCompletion(maxTokens: 512, contextSize: 4096) == 512)
        #expect(ContextBudget.promptRoom(maxTokens: 512, contextSize: 4096) == 3584)
        #expect(ContextBudget.promptRoom(maxTokens: 1024, contextSize: 4096) == 3072)
    }

    @Test func reservationIsCappedAtHalfTheWindow() {
        #expect(ContextBudget.reservedForCompletion(maxTokens: 5000, contextSize: 4096) == 2048)
        #expect(ContextBudget.promptRoom(maxTokens: 5000, contextSize: 4096) == 2048)
        #expect(ContextBudget.reservedForCompletion(maxTokens: 0, contextSize: 4096) == 1)
        #expect(ContextBudget.promptRoom(maxTokens: 512, contextSize: 0) == 1)
    }

    /// Core ML applies the same budget to real token arrays.
    @Test func coreMLTrimUsesTheChatCap() {
        let trimmed = CoreMLTextEngine.trimToContext(
            Array(1...5000), contextLimit: 4096,
            maxNewTokens: GenerationParameters.chat.maxTokens, bosTokenId: nil
        )
        #expect(trimmed.count == 3584)
    }

    @Test func aReplyThatReachedTheCapWasCut() {
        #expect(ReplyLimit.wasCut(tokensGenerated: 512, maxTokens: 512))
        #expect(ReplyLimit.wasCut(tokensGenerated: 513, maxTokens: 512))
        #expect(!ReplyLimit.wasCut(tokensGenerated: 511, maxTokens: 512))
        #expect(!ReplyLimit.wasCut(tokensGenerated: nil, maxTokens: 512))
        #expect(!ReplyLimit.wasCut(tokensGenerated: 0, maxTokens: 0))
    }
}

// MARK: - Thermal indicator

struct ThermalAdvisorTests {

    private func model(
        _ id: String, backend: ModelBackend = .llamaCpp, gb: Double,
        caps: ModelCapabilities = .textOnly
    ) -> AIModel {
        AIModel(
            id: id, name: id, family: "f", parameterSize: "1B", quantization: "Q4",
            backend: backend, capabilities: caps, ramRequiredGB: gb, diskSizeGB: gb,
            huggingFaceRepo: id, contextLength: 4096, quality: .good
        )
    }

    @Test func nothingWhileThePhoneIsCool() {
        let heavy = model("heavy", gb: 3.4)
        for state in [ProcessInfo.ThermalState.nominal, .fair] {
            #expect(ThermalAdvisor.advice(thermal: state, loaded: heavy, installed: [heavy]) == nil)
        }
    }

    @Test func seriousAndCriticalShowTheIndicator() {
        for state in [ProcessInfo.ThermalState.serious, .critical] {
            let advice = ThermalAdvisor.advice(thermal: state, loaded: nil, installed: [])
            #expect(advice?.state == state)
            #expect(advice?.lighterModel == nil)
        }
    }

    @Test func aHeavyModelGetsTheBestLighterInstalledOne() {
        let heavy = model("gemma-4-e2b", backend: .llamaCppVision, gb: 4.1, caps: .vision)
        let tiny = model("qwen-0.8b", backend: .llamaCppVision, gb: 0.8, caps: .vision)
        let small = model("lfm2-2.6b", gb: 1.6)
        let advice = ThermalAdvisor.advice(
            thermal: .serious, loaded: heavy, installed: [heavy, tiny, small]
        )
        #expect(advice?.lighterModel?.id == "lfm2-2.6b")
    }

    @Test func mlxCountsAsHeavyToo() {
        let heavy = model("phi-3.5", backend: .mlx, gb: 2.3)
        let small = model("lfm2-1.2b", gb: 0.8)
        #expect(ThermalAdvisor.advice(thermal: .critical, loaded: heavy, installed: [small])?
            .lighterModel?.id == "lfm2-1.2b")
    }

    @Test func aLightModelGetsNoSuggestion() {
        let light = model("lfm2-1.2b", gb: 0.8)
        let lighter = model("tiny", gb: 0.3)
        #expect(ThermalAdvisor.advice(thermal: .serious, loaded: light, installed: [light, lighter])?
            .lighterModel == nil)
    }

    @Test func noLighterModelInstalledMeansNoSuggestion() {
        let heavy = model("heavy", gb: 3.4)
        let otherHeavy = model("other", gb: 2.5)
        #expect(ThermalAdvisor.advice(thermal: .serious, loaded: heavy, installed: [heavy, otherHeavy])?
            .lighterModel == nil)
    }

    /// Never offered: speech models, other backends, Apple Intelligence.
    @Test func onlyLocalChatModelsAreOffered() {
        let heavy = model("heavy", gb: 3.4)
        let whisper = model("whisper", backend: .whisperCpp, gb: 0.1, caps: .speechToText)
        let apple = model("apple", backend: .appleIntelligence, gb: 0, caps: .textOnly)
        #expect(ThermalAdvisor.advice(thermal: .serious, loaded: heavy, installed: [heavy, whisper, apple])?
            .lighterModel == nil)
        #expect(ThermalAdvisor.advice(thermal: .serious, loaded: apple, installed: [apple, heavy])?
            .lighterModel == nil)
    }

    @Test func explanationIsTheAgreedLine() {
        #expect(ThermalAdvice.explanation
            == "The phone is hot, so replies are slower. A lighter model or a short break helps.")
    }
}
