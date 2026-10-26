//
//  ModelSamplingTests.swift
//
//  Per-model catalog sampling: the optional `sampling` object on catalog
//  entries, how it merges over the chat's global parameters, what each
//  engine receives, and the guarantee that without it nothing changes (the
//  llama.cpp chain, the MLX arguments, the benchmark's greedy parameters).
//

import Foundation
import Testing
@testable import local_ai_test

private let lfm2 = ModelSampling(temperature: 0.3, minP: 0.15, repetitionPenalty: 1.05)
private let qwen35 = ModelSampling(
    temperature: 0.7, topP: 0.8, topK: 20, repetitionPenalty: 1.0, presencePenalty: 1.5
)

private func entry(
    id: String = "owner/model-GGUF",
    source: ModelSource = .curated,
    sampling: ModelSampling? = nil
) -> AIModel {
    AIModel(
        id: id, name: "Model", family: "Family", parameterSize: "1B", quantization: "Q4_K_M",
        backend: .llamaCpp, capabilities: .textOnly, fileLayout: .singleGGUF,
        ramRequiredGB: 1, diskSizeGB: 1, huggingFaceRepo: id, contextLength: 4096,
        quality: .good, source: source, sampling: sampling
    )
}

// MARK: - Catalog JSON

struct ModelSamplingCodableTests {

    private let base = """
    {"id": "LiquidAI/LFM2-1.2B-GGUF", "name": "LFM2 1.2B (GGUF)", "family": "LFM",
     "backend": "llamaCpp", "fileLayout": "singleGGUF", "contextLength": 4096
    """

    private func decode(_ json: String) throws -> AIModel {
        try JSONDecoder().decode(AIModel.self, from: Data(json.utf8))
    }

    private func object(_ model: AIModel) throws -> [String: Any] {
        let data = try JSONEncoder().encode(model)
        return try #require(try JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func anEntryWithoutSamplingDecodesAndEncodesAsBefore() throws {
        let model = try decode(base + "}")
        #expect(model.sampling == nil)
        #expect(try object(model)["sampling"] == nil)
        #expect(try JSONDecoder().decode(AIModel.self, from: JSONEncoder().encode(model)) == model)
    }

    @Test func aFullSamplingObjectDecodes() throws {
        let model = try decode(base + """
        , "sampling": {"temperature": 0.7, "topP": 0.8, "topK": 20, "minP": 0.05,
                       "repetitionPenalty": 1.0, "presencePenalty": 1.5}}
        """)
        #expect(model.sampling == ModelSampling(
            temperature: 0.7, topP: 0.8, topK: 20, minP: 0.05,
            repetitionPenalty: 1.0, presencePenalty: 1.5
        ))
    }

    @Test func membersLeftOutStayNilAndAreNotWritten() throws {
        let model = try decode(base + """
        , "sampling": {"temperature": 0.3, "minP": 0.15, "repetitionPenalty": 1.05}}
        """)
        #expect(model.sampling == lfm2)
        let sampling = try #require(try object(model)["sampling"] as? [String: Any])
        #expect(Set(sampling.keys) == ["temperature", "minP", "repetitionPenalty"])
    }

    @Test func samplingRoundTrips() throws {
        for sampling in [lfm2, qwen35, ModelSampling()] {
            let model = entry(sampling: sampling)
            #expect(try JSONDecoder().decode(AIModel.self, from: JSONEncoder().encode(model)) == model)
        }
    }

    @Test func authorValuesSurviveAsWritten() throws {
        let model = try decode(base + """
        , "sampling": {"temperature": 0.3, "minP": 0.15, "repetitionPenalty": 1.05, "topP": 0.95}}
        """)
        let s = try #require(model.sampling)
        #expect(s.temperature == Float(0.3))
        #expect(s.minP == Float(0.15))
        #expect(s.repetitionPenalty == Float(1.05))
        #expect(s.topP == Float(0.95))
    }

    /// A malformed object must cost the sampling, not the entry: one bad
    /// entry would otherwise reject the whole remote catalog.
    @Test func aMalformedSamplingObjectIsDropped() throws {
        let model = try decode(base + """
        , "sampling": {"temperature": "low"}}
        """)
        #expect(model.id == "LiquidAI/LFM2-1.2B-GGUF")
        #expect(model.sampling == nil)
        #expect(try decode(base + #", "sampling": null}"#).sampling == nil)
        #expect(try decode(base + #", "sampling": [1, 2]}"#).sampling == nil)
    }

    @Test func aCatalogDocumentWithSamplingDecodesAtTheCurrentSchema() throws {
        let json = """
        {"schemaVersion": \(CatalogDocument.supportedSchemaVersion), "models": [
          \(base), "sampling": {"temperature": 0.3}},
          {"id": "b", "name": "B", "family": "F", "backend": "mlx"}
        ]}
        """
        let document = try JSONDecoder().decode(CatalogDocument.self, from: Data(json.utf8))
        #expect(document.models.count == 2)
        #expect(document.models[0].sampling == ModelSampling(temperature: 0.3))
        #expect(document.models[1].sampling == nil)
    }
}

// MARK: - Merge rules

struct SamplingMergeTests {

    @Test func noSamplingChangesNothing() {
        #expect(GenerationParameters.chat.applying(nil) == .chat)
        #expect(GenerationParameters.chat.applying(ModelSampling()) == .chat)
        #expect(ModelSampling().isEmpty)
        #expect(!lfm2.isEmpty)
    }

    /// The new members default to off, so the global sets are what they were.
    @Test func globalParametersAreUnchanged() {
        let chat = GenerationParameters.chat
        #expect(chat.temperature == 0.7 && chat.topP == 0.9 && chat.topK == 40)
        #expect(chat.repetitionPenalty == 1.1 && chat.maxTokens == 512 && chat.seed == nil)
        #expect(chat.minP == 0 && chat.presencePenalty == 0)
        #expect(GenerationParameters.default.minP == 0 && GenerationParameters.default.presencePenalty == 0)
        let greedy = GenerationParameters.deterministic
        #expect(greedy.minP == 0 && greedy.presencePenalty == 0)
    }

    @Test func onlyTheMembersThatAreSetReplaceTheGlobalOnes() {
        var expected = GenerationParameters.chat
        expected.temperature = 0.3
        expected.minP = 0.15
        expected.repetitionPenalty = 1.05
        #expect(GenerationParameters.chat.applying(lfm2) == expected)

        let qwen = GenerationParameters.chat.applying(qwen35)
        #expect(qwen.temperature == 0.7 && qwen.topP == 0.8 && qwen.topK == 20)
        #expect(qwen.repetitionPenalty == 1.0 && qwen.presencePenalty == 1.5 && qwen.minP == 0)
    }

    @Test func capAndSeedAreNeverTouched() {
        var p = GenerationParameters.chat
        p.seed = 42
        p.maxTokens = 300
        let merged = p.applying(qwen35)
        #expect(merged.seed == 42 && merged.maxTokens == 300)
    }

    @Test func implausibleValuesAreIgnored() {
        let bad = ModelSampling(
            temperature: -1, topP: 0, topK: 0, minP: 1, repetitionPenalty: 0, presencePenalty: -0.5
        )
        #expect(GenerationParameters.chat.applying(bad) == .chat)
        let notFinite = ModelSampling(
            temperature: .nan, topP: .infinity, minP: .nan, repetitionPenalty: .infinity,
            presencePenalty: .nan
        )
        #expect(GenerationParameters.chat.applying(notFinite) == .chat)
        #expect(GenerationParameters.chat.applying(ModelSampling(topP: 1.5)) == .chat)
        // Edges that are valid.
        let edges = GenerationParameters.chat.applying(
            ModelSampling(temperature: 0, topP: 1, topK: 1, minP: 0, presencePenalty: 0)
        )
        #expect(edges.temperature == 0 && edges.topP == 1 && edges.topK == 1)
    }

    @Test func aCuratedModelFollowsTheCatalogInEffect() {
        let snapshot = entry(sampling: nil)
        let published = entry(sampling: lfm2)
        #expect(ModelSampling.effective(for: snapshot, catalogEntry: published) == lfm2)
        // Values removed from the catalog are gone for installed models too.
        #expect(ModelSampling.effective(for: published, catalogEntry: snapshot) == nil)
    }

    @Test func otherModelsUseTheirOwnDescriptor() {
        let imported = entry(source: .imported, sampling: qwen35)
        #expect(ModelSampling.effective(for: imported, catalogEntry: entry(sampling: lfm2)) == qwen35)
        let dropped = entry(sampling: lfm2)
        #expect(ModelSampling.effective(for: dropped, catalogEntry: nil) == lfm2)
        #expect(ModelSampling.effective(for: dropped, catalogEntry: entry(id: "other", sampling: qwen35)) == lfm2)
        #expect(ModelSampling.effective(for: nil, catalogEntry: dropped) == nil)
    }
}

// MARK: - llama.cpp chain

struct LlamaSamplerPlanTests {

    private func chain(_ p: GenerationParameters) -> String {
        LlamaSamplerPlan.describe(LlamaSamplerPlan.steps(for: p))
    }

    /// The chain both llama.cpp engines built before per-model sampling,
    /// call for call: penalties(64, repeat, 0.0, 0.0), top_k, top_p(p, 1),
    /// temp, dist. Without catalog sampling it must stay exactly this.
    @Test func theDefaultChainIsTodaysChain() {
        let today = "penalties(64, 1.1, 0.0, 0.0) -> top_k(40) -> top_p(0.9, 1) -> temp(0.7) -> dist"
        #expect(chain(.chat) == today)
        #expect(chain(.default) == today)
        #expect(LlamaSamplerPlan.steps(for: .chat) == [
            .penalties(lastN: 64, repeat: 1.1, frequency: 0, presence: 0),
            .topK(40),
            .topP(0.9, minKeep: 1),
            .temperature(0.7),
            .dist,
        ])
    }

    @Test func theBenchmarkChainIsTodaysGreedyChain() {
        #expect(chain(.deterministic)
                == "penalties(64, 1.0, 0.0, 0.0) -> top_k(1) -> top_p(1.0, 1) -> temp(0.0) -> dist")
    }

    @Test func minPGoesBetweenTopPAndTemperature() {
        #expect(chain(GenerationParameters.chat.applying(lfm2))
                == "penalties(64, 1.05, 0.0, 0.0) -> top_k(40) -> top_p(0.9, 1) -> min_p(0.15, 1) -> temp(0.3) -> dist")
    }

    /// llama_sampler_init_penalties(last_n, repeat, freq, present): the
    /// presence penalty is the fourth argument.
    @Test func presenceRidesInThePenaltiesSampler() {
        #expect(chain(GenerationParameters.chat.applying(qwen35))
                == "penalties(64, 1.0, 0.0, 1.5) -> top_k(20) -> top_p(0.8, 1) -> temp(0.7) -> dist")
    }

    @Test func aZeroMinPAddsNoSampler() {
        var p = GenerationParameters.chat
        p.minP = 0
        #expect(!LlamaSamplerPlan.steps(for: p).contains { if case .minP = $0 { return true }; return false })
    }
}

// MARK: - MLX

struct MLXSamplingTests {

    /// Today's arguments: the text engine passes temperature, topP and the
    /// repetition penalty; vision and audio pass temperature and topP only.
    @Test func defaultsMapAsBefore() {
        #expect(MLXSampling.map(.chat, passRepetitionPenalty: true)
                == MLXSampling(temperature: 0.7, topP: 0.9, repetitionPenalty: 1.1))
        #expect(MLXSampling.map(.chat, passRepetitionPenalty: false)
                == MLXSampling(temperature: 0.7, topP: 0.9, repetitionPenalty: nil))
        #expect(MLXSampling.map(.deterministic, passRepetitionPenalty: true)
                == MLXSampling(temperature: 0, topP: 1, repetitionPenalty: 1))
    }

    @Test func whatMLXCannotTakeIsDropped() {
        let p = GenerationParameters.chat.applying(qwen35)
        #expect(MLXSampling.map(p, passRepetitionPenalty: true)
                == MLXSampling(temperature: 0.7, topP: 0.8, repetitionPenalty: 1.0))
        let l = GenerationParameters.chat.applying(lfm2)
        #expect(MLXSampling.map(l, passRepetitionPenalty: true)
                == MLXSampling(temperature: 0.3, topP: 0.9, repetitionPenalty: 1.05))
    }
}

// MARK: - Daily-use switch

struct DailySamplingSwitchTests {

    @Test func parsesTheModes() {
        #expect(DailySamplingMode.parse(nil) == .catalog)
        #expect(DailySamplingMode.parse("") == .catalog)
        #expect(DailySamplingMode.parse("  ") == .catalog)
        #expect(DailySamplingMode.parse("catalog") == .catalog)
        #expect(DailySamplingMode.parse("GLOBAL") == .global)
        #expect(DailySamplingMode.parse(" authors\n") == .authors)
        #expect(DailySamplingMode.parse("author") == nil)
        #expect(DailySamplingMode.parse("1") == nil)
    }

    @Test func eachModePicksItsSampling() {
        let global = GenerationParameters.chat
        func run(_ mode: DailySamplingMode, catalog: ModelSampling?, authors: ModelSampling?) -> GenerationParameters {
            DailySampling.parameters(mode: mode, global: global, catalogSampling: catalog, authorsSampling: authors)
        }
        #expect(run(.catalog, catalog: nil, authors: lfm2) == global)
        #expect(run(.catalog, catalog: qwen35, authors: lfm2) == global.applying(qwen35))
        #expect(run(.global, catalog: qwen35, authors: lfm2) == global)
        #expect(run(.authors, catalog: qwen35, authors: lfm2) == global.applying(lfm2))
        #expect(run(.authors, catalog: qwen35, authors: nil) == global)
    }

    private func authorsData() throws -> Data {
        #if MAC_STANDIN
        let url = try #require(Bundle.module.url(forResource: "DailyUseAuthorsSampling", withExtension: "json"))
        return try Data(contentsOf: url)
        #else
        return try #require(DailyFixtures.bundledResource(DailyAuthorsSampling.resourceName))
        #endif
    }

    /// Every value in the resource is one the merge accepts, so an
    /// `authors` run really sends what the file says.
    @Test func theAuthorsFileDecodesAndEveryValueIsApplied() throws {
        let table = try DailyAuthorsSampling.decode(try authorsData()).models
        #expect(table.count == 10)
        #expect(table["LiquidAI/LFM2-2.6B-GGUF"] == lfm2)
        #expect(table["unsloth/Qwen3.5-2B-GGUF"] == qwen35)
        #expect(table["unsloth/gemma-4-E2B-it-GGUF"]
                == ModelSampling(temperature: 0.7, topP: 0.95, topK: 64, repetitionPenalty: 1.0))
        for (id, s) in table {
            let p = GenerationParameters.chat.applying(s)
            #expect(s.temperature.map { $0 == p.temperature } ?? true, "\(id) temperature")
            #expect(s.topP.map { $0 == p.topP } ?? true, "\(id) topP")
            #expect(s.topK.map { $0 == p.topK } ?? true, "\(id) topK")
            #expect(s.minP.map { $0 == p.minP } ?? true, "\(id) minP")
            #expect(s.repetitionPenalty.map { $0 == p.repetitionPenalty } ?? true, "\(id) repetitionPenalty")
            #expect(s.presencePenalty.map { $0 == p.presencePenalty } ?? true, "\(id) presencePenalty")
        }
    }

    @Test func theAuthorsFileNamesCatalogModels() throws {
        let table = try DailyAuthorsSampling.decode(try authorsData()).models
        let ids = Set(ModelCatalog.builtIn.map(\.id))
        for id in table.keys {
            #expect(ids.contains(id), "\(id) is not in the catalog")
        }
    }
}

// MARK: - App wiring (app bundle only)

#if !MAC_STANDIN
@MainActor
struct SamplingWiringTests {

    /// 1.2 ships the capability without values: no bundled entry carries
    /// `sampling`, so every chat message is sent exactly as before.
    @Test func theBundledCatalogCarriesNoSampling() throws {
        #expect(ModelCatalog.builtIn.allSatisfy { $0.sampling == nil })
        let url = try #require(CatalogFile.bundledURL)
        let document = try JSONDecoder().decode(CatalogDocument.self, from: Data(contentsOf: url))
        #expect(document.models.allSatisfy { $0.sampling == nil })
    }

    @Test func theChatSendsTheGlobalParametersWithoutCatalogSampling() {
        #expect(ChatViewModel.parameters(for: nil) == .chat)
        #expect(ChatViewModel.parameters(for: entry(source: .imported)) == .chat)
        #expect(ChatViewModel.parameters(for: entry(source: .imported, sampling: lfm2))
                == GenerationParameters.chat.applying(lfm2))
    }

    /// The daily suite's default mode is what the chat screen sends.
    @Test func dailyCatalogModeMirrorsTheChat() {
        for model in ModelCatalog.all + [entry(source: .imported, sampling: qwen35)] {
            let daily = DailySampling.parameters(
                mode: .catalog,
                global: ChatViewModel.parameters,
                catalogSampling: ModelSampling.effective(
                    for: model, catalogEntry: ModelCatalog.model(id: model.id)
                ),
                authorsSampling: lfm2
            )
            #expect(daily == ChatViewModel.parameters(for: model), "\(model.id)")
        }
    }

    /// The benchmark sends greedy parameters whatever the model's sampling.
    @Test func theBenchmarkIsUntouched() {
        let p = BenchmarkEngine.generationParameters(maxTokens: 96)
        var expected = GenerationParameters.deterministic
        expected.maxTokens = 96
        #expect(p == expected)
        #expect(p.temperature == 0 && p.topK == 1 && p.topP == 1 && p.repetitionPenalty == 1)
        #expect(p.minP == 0 && p.presencePenalty == 0 && p.seed == 42)
    }
}
#endif
