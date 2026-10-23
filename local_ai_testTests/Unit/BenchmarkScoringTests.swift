//
//  BenchmarkScoringTests.swift
//  local_ai_testTests
//
//  The rules that decide what a benchmark number means:
//
//    * the chance-aware score, and the promise that it changes nothing for
//      the free-answer suite every existing report was graded on;
//    * the deterministic, category-interleaved run order;
//    * the quality-stage time budget and the partial reports it produces;
//    * stripping a reasoning model's <think> block before grading.
//
//  No model, no device: everything here is a pure function or a value type.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - Fixtures

/// A report around the given quality block, with dull fixed performance
/// numbers that no test here looks at.
private func makeReport(_ quality: QualityResults?) -> BenchmarkReport {
    BenchmarkReport(
        appVersion: "1.0",
        appBuild: "1",
        modelId: "test/model",
        modelName: "Test",
        backend: "llama.cpp",
        quantization: "Q4_K_M",
        parameterSize: "1B",
        device: DeviceSnapshot(
            identifier: "iPhone17,1", marketingName: "iPhone",
            chip: "A18", ramGB: 8, iOSVersion: "26.1"
        ),
        performance: PerformanceMetrics(
            loadSeconds: 1,
            prefillTokensPerSecond: 100,
            generationTokensPerSecond: 20,
            peakMemoryMB: 1000,
            memoryBudgetMB: 4000,
            thermalStateBefore: "nominal",
            thermalStateAfter: "nominal",
            tokensGenerated: 200,
            wallClockSeconds: 30
        ),
        quality: quality
    )
}

/// A category of `count` tasks, the first `passed` of them passing.
private func category(
    _ name: String, count: Int, passed: Int, chance: Double = 0
) -> CategoryScore {
    CategoryScore(
        category: name,
        tasks: (0..<count).map { index in
            TaskResult(
                taskId: "\(name)_\(index)",
                passed: index < passed,
                latencySeconds: 1,
                answer: "x",
                chance: chance
            )
        }
    )
}

/// The shape of the bundled v3 suite: 118 tasks in five categories.
private let v3Shape: [(String, Int)] = [
    ("knowledge", 57), ("instructions", 45), ("math", 8),
    ("multilingual", 4), ("retrieval", 4)
]

private func task(_ id: String, _ category: String, chance: Double = 0) -> BenchmarkTask {
    BenchmarkTask(
        id: id, category: category, prompt: "Q \(id)",
        check: .exactMatch("a"), chance: chance
    )
}

private func v3ShapedTasks() -> [BenchmarkTask] {
    v3Shape.flatMap { name, count in
        (0..<count).map { task("\(name)_\(String(format: "%02d", $0))", name) }
    }
}

// MARK: - Score normalisation

struct ChanceNormalizedScoreTests {

    /// The promise that lets the new formula ship without a suite bump:
    /// with every chance at 0 the score is EXACTLY the old passed/total,
    /// bit for bit, for every pass count the v3 suite can produce.
    @Test func withNoGuessingFloorTheScoreIsTheOldPassRate() {
        let total = v3Shape.reduce(0) { $0 + $1.1 }
        for passedKnowledge in stride(from: 0, through: 57, by: 7) {
            for passedInstructions in stride(from: 0, through: 45, by: 9) {
                let quality = QualityResults(
                    suiteVersion: 3,
                    categories: [
                        category("knowledge", count: 57, passed: passedKnowledge),
                        category("instructions", count: 45, passed: passedInstructions),
                        category("math", count: 8, passed: 3),
                        category("multilingual", count: 4, passed: 4),
                        category("retrieval", count: 4, passed: 1)
                    ]
                )
                let passed = passedKnowledge + passedInstructions + 3 + 4 + 1
                let legacy = Double(passed) / Double(total) * 100
                #expect(makeReport(quality).overallScore == legacy)
                #expect(quality.total == total)
                #expect(quality.completed == total)
                #expect(!quality.truncated)
            }
        }
    }

    /// A report written by an older build (no `chance`, `total`,
    /// `completed` or `truncated` keys anywhere) keeps its number.
    @Test func anOldReportDecodesToTheSameScore() throws {
        let json = """
        {
          "suiteVersion": 3,
          "suiteID": "built-in",
          "categories": [
            {"category": "math", "passed": 3, "total": 4, "tasks": [
              {"taskId": "m1", "passed": true,  "latencySeconds": 1, "answer": "4"},
              {"taskId": "m2", "passed": true,  "latencySeconds": 1, "answer": "5"},
              {"taskId": "m3", "passed": true,  "latencySeconds": 1, "answer": "6"},
              {"taskId": "m4", "passed": false, "latencySeconds": 1, "answer": "0"}
            ]},
            {"category": "retrieval", "passed": 1, "total": 3, "tasks": []}
          ]
        }
        """
        let quality = try JSONDecoder().decode(QualityResults.self, from: Data(json.utf8))
        #expect(quality.completed == 7)
        #expect(quality.total == 7)
        #expect(!quality.truncated)
        #expect(makeReport(quality).overallScore == Double(4) / Double(7) * 100)
        #expect(!makeReport(quality).isPartial)
        #expect(quality.allTasks.allSatisfy { $0.chance == nil })
    }

    /// Open LLM Leaderboard: (accuracy - chance) / (1 - chance).
    @Test func aFourChoiceCategoryIsNormalisedAgainstGuessing() {
        // 2 of 4 right at chance 1/4: (2 - 1) / (4 - 1) = 1/3.
        let half = category("mmlu", count: 4, passed: 2, chance: 0.25)
        #expect(abs(half.normalizedScore - 1.0 / 3.0) < 1e-12)
        #expect(half.passRate == 0.5)

        // Exactly the guessing rate scores zero, and so does worse.
        #expect(category("mmlu", count: 4, passed: 1, chance: 0.25).normalizedScore == 0)
        #expect(category("mmlu", count: 4, passed: 0, chance: 0.25).normalizedScore == 0)
        #expect(category("mmlu", count: 4, passed: 4, chance: 0.25).normalizedScore == 1)
    }

    /// Free-answer and multiple-choice tasks pool into one flat formula:
    /// (Σpass − Σchance) / Σ(1 − chance), not an average of categories.
    @Test func mixedSuitesPoolTheFloorOverAllTasks() throws {
        let quality = QualityResults(
            suiteVersion: 1,
            categories: [
                category("free", count: 6, passed: 3),
                category("choice", count: 4, passed: 3, chance: 0.5)
            ]
        )
        // (6 - 2) / (10 - 2) = 0.5
        let expected = (6.0 - 2.0) / (10.0 - 2.0) * 100
        let score = try #require(makeReport(quality).overallScore)
        #expect(abs(score - expected) < 1e-9)

        // A category average would have said (0.5 + 0.5) / 2 = 50 too here,
        // so check a case where they differ: the flat rule weighs tasks.
        let skewed = QualityResults(
            suiteVersion: 1,
            categories: [
                category("big", count: 57, passed: 57),
                category("small", count: 4, passed: 0)
            ]
        )
        let flat = try #require(makeReport(skewed).overallScore)
        #expect(abs(flat - 57.0 / 61.0 * 100) < 1e-9)
    }

    @Test func theRawOutcomeHelperMatchesTheFormula() {
        #expect(BenchmarkScoring.normalized([]) == 0)
        #expect(BenchmarkScoring.normalized([(true, 0), (false, 0)]) == 0.5)
        #expect(abs(BenchmarkScoring.normalized([(true, 0.25), (false, 0.25)]) - 1.0 / 3.0) < 1e-12)
        // Every task a certain guess: nothing measured, scored 0, no NaN.
        #expect(BenchmarkScoring.normalized([(true, 1), (true, 1)]) == 0)
    }

    @Test func aZeroChanceStaysOutOfTheJSON() throws {
        let free = TaskResult(taskId: "t", passed: true, latencySeconds: 1, answer: "a")
        let freeJSON = String(decoding: try JSONEncoder().encode(free), as: UTF8.self)
        #expect(!freeJSON.contains("chance"))

        let choice = TaskResult(
            taskId: "t", passed: true, latencySeconds: 1, answer: "a", chance: 0.25
        )
        let decoded = try JSONDecoder().decode(
            TaskResult.self, from: try JSONEncoder().encode(choice)
        )
        #expect(decoded.chance == 0.25)
        #expect(decoded.guessingFloor == 0.25)
    }
}

// MARK: - Suite schema

struct BenchmarkTaskChanceSchemaTests {

    private func document(chanceField: String) -> String {
        """
        {
          "schemaVersion": 3,
          "suiteVersion": 1,
          "samplingSeed": 12345,
          "tasks": [
            {"id": "q1", "category": "mmlu", "prompt": "Pick one.",
             "check": {"kind": "exactMatch", "value": "b"}\(chanceField)}
          ]
        }
        """
    }

    @Test func chanceIsOptionalAndDefaultsToZero() throws {
        let suite = try #require(BenchmarkSuiteFile.decode(Data(document(chanceField: "").utf8)))
        #expect(suite.tasks[0].chance == 0)
        #expect(suite.samplingSeed == 12345)
    }

    @Test func aDeclaredChanceIsRead() throws {
        let data = Data(document(chanceField: #", "chance": 0.25"#).utf8)
        let suite = try #require(BenchmarkSuiteFile.decode(data))
        #expect(suite.tasks[0].chance == 0.25)
    }

    /// `25` meant as a percentage must not quietly become "always a guess".
    @Test func anOutOfRangeChanceRejectsTheDocument() {
        #expect(BenchmarkSuiteFile.decode(Data(document(chanceField: #", "chance": 25"#).utf8)) == nil)
        #expect(BenchmarkSuiteFile.decode(Data(document(chanceField: #", "chance": -0.1"#).utf8)) == nil)
    }

    @Test func aZeroChanceIsNotWrittenBack() throws {
        let free = String(decoding: try JSONEncoder().encode(task("t", "x")), as: UTF8.self)
        #expect(!free.contains("chance"))
        let choice = task("t", "x", chance: 0.5)
        let roundTrip = try JSONDecoder().decode(
            BenchmarkTask.self, from: try JSONEncoder().encode(choice)
        )
        #expect(roundTrip.chance == 0.5)
    }

    /// The bundled suite is all free-answer, so no number moves.
    @Test func theBundledSuiteHasNoGuessingFloor() throws {
        let url = try #require(BenchmarkSuiteFile.bundledURL)
        let suite = try #require(BenchmarkSuiteFile.decode(try Data(contentsOf: url)))
        #expect(suite.tasks.allSatisfy { $0.chance == 0 })
        #expect(suite.samplingSeed != nil)
    }

    @Test func theImporterKeepsChanceThroughTheTokenClamp() throws {
        let json = """
        {"schemaVersion": 3, "suiteVersion": 1, "tasks": [
          {"id": "q1", "category": "mmlu", "prompt": "Pick.", "maxTokens": 4096,
           "chance": 0.25, "check": {"kind": "exactMatch", "value": "b"}}
        ]}
        """
        let result = try BenchmarkSuiteImporter.makeSuite(
            text: json, source: "suite.json", suiteID: "custom-test"
        )
        let imported = try #require(result.suite.document.tasks.first)
        #expect(imported.maxTokens == SuiteImportLimits.maxTokensPerTask)
        #expect(imported.chance == 0.25)
    }

    @Test func promptfooMetadataChoicesSetTheFloor() throws {
        let yaml = """
        prompts:
          - "{{question}}"
        tests:
          - description: listed
            vars:
              question: Which?
            metadata:
              choices: [A, B, C, D]
            assert:
              - type: equals
                value: B
          - description: counted
            vars:
              question: Which?
            metadata:
              choices: 2
            assert:
              - type: equals
                value: A
          - description: explicit
            vars:
              question: Which?
            metadata:
              chance: 0.2
            assert:
              - type: equals
                value: C
          - description: free
            vars:
              question: Which?
            assert:
              - type: equals
                value: C
          - description: broken
            vars:
              question: Which?
            metadata:
              choices: 1
            assert:
              - type: equals
                value: C
        """
        let result = try BenchmarkSuiteImporter.makeSuite(
            text: yaml, source: "promptfooconfig.yaml", suiteID: "custom-test"
        )
        let chances = Dictionary(
            uniqueKeysWithValues: result.suite.document.tasks.map { ($0.id, $0.chance) }
        )
        #expect(chances["listed"] == 0.25)
        #expect(chances["counted"] == 0.5)
        #expect(chances["explicit"] == 0.2)
        #expect(chances["free"] == 0)
        #expect(chances["broken"] == nil)
        #expect(result.skipped.contains { $0.name == "broken" })
    }
}

// MARK: - Run order

struct QualityRunOrderTests {

    /// Pins the exact algorithm. If this changes, every device's order
    /// changes with it, so it may only change on purpose. The expected list
    /// was computed by an independent reimplementation of the same
    /// SplitMix64 + FNV-1a + exact-fraction merge.
    @Test func theOrderIsPinned() {
        let tasks = (0..<5).map { task("a\($0)", "a") }
            + [task("b0", "b"), task("b1", "b"), task("c0", "c")]
        #expect(BenchmarkTaskOrder.interleaved(tasks, seed: 42).map(\.id)
                == ["a3", "b0", "a0", "a2", "c0", "a1", "b1", "a4"])
        #expect(BenchmarkTaskOrder.interleaved(tasks, seed: 7).map(\.id)
                == ["a3", "b1", "a1", "a0", "c0", "a4", "b0", "a2"])
    }

    /// Golden order for the bundled suite: the first 20 task ids of the
    /// v3 suite (seed 20260609) as the quality stage runs them. Computed by an
    /// independent reimplementation of the same algorithm. A change here
    /// changes what a time-budgeted run covers on every device, so it may
    /// only happen together with a suite version bump.
    @Test func theBundledV3SuiteRunsInThePinnedOrder() throws {
        let url = try #require(BenchmarkSuiteFile.bundledURL)
        let suite = try #require(BenchmarkSuiteFile.decode(try Data(contentsOf: url)))
        #expect(suite.suiteVersion == 3)
        #expect(suite.tasks.count == 118)
        #expect(suite.samplingSeed == 20260609)
        let firstTwenty = suite.qualityRunOrder(categories: nil, limit: nil).prefix(20).map(\.id)
        #expect(Array(firstTwenty) == [
            "know_mmlu_039", "instr_ifeval_2265", "know_mmlu_047", "instr_letter_z_five",
            "know_mmlu_024", "instr_no_commas", "know_mmlu_020", "math_arith_03",
            "instr_ifeval_2482", "know_mmlu_095", "know_photosynthesis_gas", "instr_uppercase",
            "know_author_solitude", "instr_word_max_summary", "retrieval_tunnel_length",
            "multilingual_es_answer", "know_mmlu_004", "instr_ifeval_3749",
            "know_symbol_potassium", "instr_json_person"
        ])
    }

    /// Published reference values for both primitives.
    @Test func thePrimitivesMatchTheirReferenceValues() {
        var generator = BenchmarkTaskOrder.SplitMix64(seed: 0)
        #expect(generator.next() == 0xe220_a839_7b1d_cdaf)
        #expect(generator.next() == 0x6e78_9e6a_a1b9_65f4)
        #expect(generator.next() == 0x06c4_5d18_8009_454f)
        #expect(BenchmarkTaskOrder.fnv1a("") == 0xcbf2_9ce4_8422_2325)
        #expect(BenchmarkTaskOrder.fnv1a("a") == 0xaf63_dc4c_8601_ec8c)
    }

    @Test func theOrderIsAStablePermutation() {
        let tasks = v3ShapedTasks()
        let first = BenchmarkTaskOrder.interleaved(tasks, seed: 20260609)
        let second = BenchmarkTaskOrder.interleaved(tasks, seed: 20260609)
        #expect(first.map(\.id) == second.map(\.id))
        #expect(Set(first.map(\.id)) == Set(tasks.map(\.id)))
        #expect(first.count == tasks.count)
        #expect(BenchmarkTaskOrder.interleaved(tasks, seed: 1).map(\.id) != first.map(\.id))
    }

    /// The point of the order: a run cut short at ANY task has covered
    /// every category in proportion to its size, within one task.
    @Test func everyPrefixCoversTheCategoriesProportionally() {
        let tasks = v3ShapedTasks()
        let ordered = BenchmarkTaskOrder.interleaved(tasks, seed: 20260609)
        let total = Double(tasks.count)
        var seen: [String: Int] = [:]
        for (index, task) in ordered.enumerated() {
            seen[task.category, default: 0] += 1
            let prefix = Double(index + 1)
            for (name, count) in v3Shape {
                let expected = prefix * Double(count) / total
                #expect(abs(Double(seen[name] ?? 0) - expected) < 1,
                        "category \(name) after \(index + 1) tasks")
            }
        }
        // Even the 4-task categories show up early.
        let firstTwenty = Set(ordered.prefix(20).map(\.category))
        #expect(firstTwenty == Set(v3Shape.map(\.0)))
    }

    @Test func theSuiteUsesItsSamplingSeedAndAppliesTheLimitLast() {
        let suite = BenchmarkSuiteDocument(
            suiteVersion: 3, tasks: v3ShapedTasks(), samplingSeed: 20260609
        )
        let full = suite.qualityRunOrder(categories: nil, limit: nil)
        #expect(full.map(\.id)
                == BenchmarkTaskOrder.interleaved(suite.tasks, seed: 20260609).map(\.id))
        #expect(suite.qualityRunOrder(categories: nil, limit: 5).map(\.id)
                == Array(full.prefix(5)).map(\.id))

        let filtered = suite.qualityRunOrder(categories: ["math", "retrieval"], limit: nil)
        #expect(Set(filtered.map(\.category)) == ["math", "retrieval"])
        #expect(filtered.count == 12)
    }

    /// An imported suite has no seed; its order still never varies.
    @Test func aSeedlessSuiteFallsBackToAStableSeed() {
        let suite = BenchmarkSuiteDocument(
            suiteVersion: 1, tasks: v3ShapedTasks(), id: "custom-abcd"
        )
        #expect(suite.runOrderSeed == BenchmarkTaskOrder.fnv1a("custom-abcd#1"))
        #expect(suite.qualityRunOrder(categories: nil, limit: nil).map(\.id)
                == suite.qualityRunOrder(categories: nil, limit: nil).map(\.id))
    }
}

// MARK: - Time budget and partial reports

struct QualityBudgetTests {

    @Test func theDefaultsAreStandardTwentyMinutesAndFullUnlimited() {
        #expect(BenchmarkDefaults.standardQualityBudget == 20 * 60)
        #expect(BenchmarkOptions.standard.qualityBudget == TimeInterval(20 * 60))
        #expect(BenchmarkOptions.standard.stages == .full)
        #expect(BenchmarkOptions.full.qualityBudget == nil)
        #expect(BenchmarkOptions().qualityBudget == nil)

        #expect(BenchmarkRunMode.standard.options.qualityBudget == TimeInterval(20 * 60))
        #expect(BenchmarkRunMode.full.options.qualityBudget == nil)
        #expect(BenchmarkRunMode.full.options.stages == .full)
        #expect(BenchmarkRunMode.performanceOnly.options.stages == .performanceOnly)
    }

    @Test func nonsenseBudgetsMeanUnlimited() {
        #expect(BenchmarkOptions(qualityBudget: 0).qualityBudget == nil)
        #expect(BenchmarkOptions(qualityBudget: -5).qualityBudget == nil)
        #expect(BenchmarkOptions(qualityBudget: .infinity).qualityBudget == nil)
        #expect(BenchmarkOptions(qualityBudget: 90).qualityBudget == 90)
    }

    /// Stop at a task boundary once the budget is spent, but never before
    /// the first task.
    @Test func theStageStopsAtTheFirstBoundaryPastTheBudget() {
        let options = BenchmarkOptions(qualityBudget: 60)
        #expect(!options.qualityBudgetExhausted(beforeTaskAt: 0, elapsed: 500))
        #expect(!options.qualityBudgetExhausted(beforeTaskAt: 5, elapsed: 59.9))
        #expect(options.qualityBudgetExhausted(beforeTaskAt: 5, elapsed: 60))
        #expect(!BenchmarkOptions.full.qualityBudgetExhausted(beforeTaskAt: 117, elapsed: 86_400))
    }

    /// The build test and the release harness publish full scores.
    @Test func theBuildTestIsUnlimitedUnlessTheEnvironmentSaysOtherwise() {
        #expect(BuildTestOptions.default.benchmarkOptions.qualityBudget == nil)
        #expect(BuildTestOptions(qualityBudget: 600).benchmarkOptions.qualityBudget == 600)

        #expect(BenchmarkHarnessConfig.fromEnvironment([:]).options.qualityBudget == nil)
        #expect(BenchmarkHarnessConfig.fromEnvironment(
            ["BENCH_QUALITY_BUDGET_SEC": "600"]).options.qualityBudget == 600)
        #expect(BenchmarkHarnessConfig.fromEnvironment(
            ["BENCH_QUALITY_BUDGET_SEC": "soon"]).options.qualityBudget == nil)
        #expect(BenchmarkHarnessConfig.fromEnvironment(
            ["BENCH_QUALITY_BUDGET_SEC": "0"]).options.qualityBudget == nil)
    }

    @Test func aBuildTestOptionsFileWithoutABudgetStillDecodes() throws {
        let json = """
        {"stages": "full", "modelIDs": [], "runStandardProtocol": true,
         "thermalGating": true, "cooldownBetweenModels": true}
        """
        let options = try JSONDecoder().decode(BuildTestOptions.self, from: Data(json.utf8))
        #expect(options.qualityBudget == nil)
    }

    private func partialQuality() -> QualityResults {
        QualityResults(
            suiteVersion: 3,
            categories: [
                category("knowledge", count: 35, passed: 25),
                category("instructions", count: 28, passed: 18),
                category("math", count: 5, passed: 2),
                category("multilingual", count: 3, passed: 0),
                category("retrieval", count: 2, passed: 0)
            ],
            total: 118,
            truncated: true
        )
    }

    @Test func aTruncatedRunIsScoredOverTheCompletedTasks() throws {
        let quality = partialQuality()
        #expect(quality.completed == 73)
        #expect(quality.total == 118)

        let report = makeReport(quality)
        #expect(report.isPartial)
        #expect(report.overallScore == Double(45) / Double(73) * 100)
        #expect(report.overallScoreRounded == 62)
        #expect(report.partialTasksLabel == "73 of 118 tasks")
        #expect(report.scoreLabel == "62/100, 73 of 118 tasks")
        #expect(report.summaryLine.contains("partial"))
        #expect(BuildTestFormat.score(report) == "62/100, 73 of 118 tasks")

        let full = makeReport(QualityResults(
            suiteVersion: 3, categories: [category("math", count: 8, passed: 6)]
        ))
        #expect(full.scoreLabel == "75/100")
        #expect(!full.summaryLine.contains("partial"))
    }

    @Test func aTruncatedReportRoundTripsThroughJSON() throws {
        let original = makeReport(partialQuality())
        let decoded = try JSONDecoder().decode(
            BenchmarkReport.self, from: try JSONEncoder().encode(original)
        )
        #expect(decoded.quality?.truncated == true)
        #expect(decoded.quality?.total == 118)
        #expect(decoded.quality?.completed == 73)
        #expect(decoded.overallScore == original.overallScore)
    }

    /// Never comparable: not with a full run, and not even with another
    /// partial run of the same suite.
    @Test func aPartialScoreIsNeverCompared() {
        let partial = makeReport(partialQuality())
        let otherPartial = makeReport(partialQuality())
        let full = makeReport(QualityResults(
            suiteVersion: 3, categories: [category("math", count: 8, passed: 6)]
        ))
        #expect(!partial.isScoreComparable(with: full))
        #expect(!full.isScoreComparable(with: partial))
        #expect(!partial.isScoreComparable(with: otherPartial))
        #expect(full.isScoreComparable(with: full))

        let comparison = BenchmarkRunComparison(current: partial, previous: full)
        #expect(comparison.scoreDelta == nil)
        #expect(comparison.scoreSuppressedByPartialRun)
        #expect(!comparison.scoreSuppressedBySuiteVersion)
        #expect(comparison.suppressionNote?.contains("partial run") == true)
    }
}

// MARK: - Build test aggregates

/// A partial run is scored over fewer tasks, so it must stay out of the
/// build test's average and rankings.
struct BuildTestAggregateTests {

    private let device = DeviceSnapshot(
        identifier: "iPhone18,4", marketingName: "iPhone Air",
        chip: "A19", ramGB: 11.5, iOSVersion: "26.6.1"
    )

    private func entry(_ id: String, passed: Int, of total: Int, partial: Bool = false) -> BuildTestEntry {
        let quality = QualityResults(
            suiteVersion: 3,
            categories: [category("knowledge", count: total, passed: passed)],
            total: partial ? 118 : nil,
            truncated: partial
        )
        return BuildTestEntry(
            modelId: id, modelName: id, backend: "llama.cpp",
            status: .completed, report: makeReport(quality)
        )
    }

    private func report(_ entries: [BuildTestEntry]) -> BuildTestReport {
        BuildTestReport(
            appVersion: "1.0.1", appBuild: "2", stages: .full, device: device,
            options: .default, entries: entries, durationSeconds: 1
        )
    }

    @Test func partialRunsAreLeftOutOfTheAverageAndTheRanking() {
        let sweep = report([
            entry("full-50", passed: 59, of: 118),
            entry("full-70", passed: 83, of: 118),
            // 90/100 over 20 tasks: would top the ranking and lift the mean.
            entry("partial-90", passed: 18, of: 20, partial: true)
        ])
        #expect(sweep.partialEntryCount == 1)
        #expect(sweep.aggregatableEntries.map(\.modelId) == ["full-50", "full-70"])
        #expect(sweep.averageScore == 60)
        #expect(sweep.rankedEntries.map(\.modelId) == ["full-70", "full-50"])
        #expect(sweep.fastestEntry?.modelId != "partial-90")
        #expect(sweep.partialExclusionNote == "1 partial run excluded")
    }

    @Test func aSweepOfOnlyPartialRunsHasNoAggregates() {
        let sweep = report([
            entry("p1", passed: 10, of: 20, partial: true),
            entry("p2", passed: 12, of: 20, partial: true)
        ])
        #expect(sweep.averageScore == nil)
        #expect(sweep.rankedEntries.isEmpty)
        #expect(sweep.fastestEntry == nil)
        #expect(sweep.partialExclusionNote == "2 partial runs excluded")
    }

    @Test func aSweepWithoutPartialRunsShowsNoNote() {
        let sweep = report([entry("full", passed: 59, of: 118)])
        #expect(sweep.partialExclusionNote == nil)
        #expect(sweep.averageScore == 50)
    }
}

// MARK: - Reasoning blocks

struct ReasoningBlockStripTests {

    @Test func aClosedThinkBlockIsRemovedBeforeGrading() {
        let raw = "<think>\nThe capital of France is Paris.\n</think>\n\nParis"
        let stripped = BenchmarkChecker.stripLeadingReasoning(raw)
        #expect(stripped.answer == "Paris")
        #expect(stripped.stripped)
        #expect(!stripped.unterminated)
        #expect(BenchmarkCheck.exactMatch("paris").evaluate(answer: stripped.answer))
        // The reason the strip exists: graded raw, a correct answer fails.
        #expect(!BenchmarkCheck.exactMatch("paris").evaluate(answer: raw))
    }

    @Test func anEmptyThinkBlockIsRemovedToo() {
        #expect(BenchmarkChecker.stripLeadingReasoning("<think>\n\n</think>\n\n42").answer == "42")
        #expect(BenchmarkChecker.stripLeadingReasoning("  <THINK>x</Think> 42").answer == "42")
    }

    /// The token cap landed inside the block: no answer, the task fails.
    @Test func anUnterminatedBlockLeavesNothingToGrade() {
        let stripped = BenchmarkChecker.stripLeadingReasoning("<think>\nLet me count the letters")
        #expect(stripped.unterminated)
        #expect(stripped.answer.isEmpty)
    }

    /// A template that opens the block in the prompt leaves only its close.
    @Test func aBlockOpenedByTheTemplateIsRemoved() {
        let stripped = BenchmarkChecker.stripLeadingReasoning("Counting... done.\n</think>\n\n3")
        #expect(stripped.answer == "3")
        #expect(!stripped.unterminated)
    }

    @Test func ordinaryAnswersAreLeftAlone() {
        for answer in ["Paris", "", "The tag <think> means reasoning.", "{\"a\": 1}"] {
            let stripped = BenchmarkChecker.stripLeadingReasoning(answer)
            #expect(stripped.answer == answer)
            #expect(!stripped.stripped)
            #expect(!stripped.unterminated)
        }
    }
}
