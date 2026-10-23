//
//  BenchmarkSuiteImportTests.swift
//  local_ai_testTests
//
//  The importer that lets people bring their own benchmark suite: the small
//  YAML reader, the promptfoo assertion mapping, and the guardrails.
//
//  What is being pinned here is mostly REFUSALS. A suite decides what a
//  model's score means, so the interesting behaviour is not "a good file
//  imports" but "a file this app only half understands does not import
//  quietly". No network, no model, no device.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - Helpers

/// Runs an import and hands back the refusal, or `nil` when it succeeded.
private func refusal(_ text: String) -> SuiteImportError? {
    do {
        _ = try BenchmarkSuiteImporter.makeSuite(
            text: text, source: "suite.yaml", suiteID: "custom-test"
        )
        return nil
    } catch let error as SuiteImportError {
        return error
    } catch {
        return nil
    }
}

private func importSuite(_ text: String) throws -> SuiteImportResult {
    try BenchmarkSuiteImporter.makeSuite(
        text: text, source: "suite.yaml", suiteID: "custom-test"
    )
}

/// One `assert:` entry, built the way the YAML parser would.
private func assertion(_ type: String, _ value: YAMLValue? = nil) -> YAMLValue {
    var pairs = [YAMLPair(key: "type", value: .scalar(type))]
    if let value { pairs.append(YAMLPair(key: "value", value: value)) }
    return .mapping(pairs)
}

private func mapped(_ type: String, _ value: YAMLValue? = nil) throws -> BenchmarkCheck? {
    guard case .check(let check) = try PromptfooAssertion.map(assertion(type, value)) else {
        return nil
    }
    return check
}

private func unsupportedReason(_ type: String, _ value: YAMLValue? = nil) throws -> String? {
    guard case .unsupported(let reason) = try PromptfooAssertion.map(assertion(type, value)) else {
        return nil
    }
    return reason
}

// MARK: - MiniYAML

struct MiniYAMLTests {

    @Test func readsNestedMappingsAndSequences() throws {
        let yaml = """
        description: Demo suite
        tests:
          - description: capital
            vars:
              question: What is the capital of France?
            assert:
              - type: equals
                value: Paris
        """
        let root = try MiniYAML.parse(yaml)
        #expect(root["description"]?.stringValue == "Demo suite")

        let tests = try #require(root["tests"]?.sequenceValue)
        #expect(tests.count == 1)
        #expect(tests[0]["vars"]?["question"]?.stringValue == "What is the capital of France?")
        let asserts = try #require(tests[0]["assert"]?.sequenceValue)
        #expect(asserts[0]["type"]?.stringValue == "equals")
        #expect(asserts[0]["value"]?.stringValue == "Paris")
    }

    @Test func readsSequenceAtTheKeyIndentation() throws {
        let yaml = """
        prompts:
        - Answer briefly.
        tests:
        - assert:
          - type: contains
            value: yes
        """
        let root = try MiniYAML.parse(yaml)
        #expect(root["prompts"]?.sequenceValue?.count == 1)
        #expect(root["tests"]?.sequenceValue?.count == 1)
    }

    @Test func readsLiteralBlockScalar() throws {
        let yaml = """
        prompt: |
          line one
          line two
        after: x
        """
        let root = try MiniYAML.parse(yaml)
        #expect(root["prompt"]?.stringValue == "line one\nline two\n")
        #expect(root["after"]?.stringValue == "x")
    }

    @Test func readsStrippedAndFoldedBlockScalars() throws {
        let yaml = """
        stripped: |-
          only line
        folded: >-
          one
          two
        """
        let root = try MiniYAML.parse(yaml)
        #expect(root["stripped"]?.stringValue == "only line")
        #expect(root["folded"]?.stringValue == "one two")
    }

    @Test func readsFlowSequences() throws {
        let root = try MiniYAML.parse("required: [x, y, \"z z\"]")
        #expect(root["required"]?.sequenceValue?.compactMap(\.stringValue) == ["x", "y", "z z"])
        #expect(try MiniYAML.parse("empty: []")["empty"]?.sequenceValue?.isEmpty == true)
    }

    @Test func stripsCommentsButNotInsideQuotes() throws {
        let yaml = """
        # a whole-line comment
        a: 1  # trailing comment
        b: "has # inside"
        c: not#a#comment
        """
        let root = try MiniYAML.parse(yaml)
        #expect(root["a"]?.stringValue == "1")
        #expect(root["b"]?.stringValue == "has # inside")
        #expect(root["c"]?.stringValue == "not#a#comment")
    }

    @Test func keepsColonsInsidePlainScalars() throws {
        let root = try MiniYAML.parse("provider: openai:gpt-4o")
        #expect(root["provider"]?.stringValue == "openai:gpt-4o")
    }

    @Test func readsQuotedEscapes() throws {
        let root = try MiniYAML.parse(#"a: "line\nnext \"quoted\"""#)
        #expect(root["a"]?.stringValue == "line\nnext \"quoted\"")
    }

    @Test func scalarsAreNotTypeInferred() throws {
        let root = try MiniYAML.parse("flag: true\nnumber: 42\nnothing: null")
        #expect(root["flag"]?.stringValue == "true")
        #expect(root["number"]?.stringValue == "42")
        #expect(root["nothing"]?.isNull == true)
    }

    // MARK: Refusals

    @Test func refusesTabIndentation() {
        #expect(throws: YAMLError.self) {
            try MiniYAML.parse("a:\n\tb: 1")
        }
    }

    @Test func refusesAnchorsAndAliases() {
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: &anchor value") }
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: *anchor") }
    }

    @Test func refusesFlowMappings() {
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: {b: 1}") }
    }

    @Test func refusesDuplicateKeys() {
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: 1\na: 2") }
    }

    @Test func refusesUnterminatedQuotes() {
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: \"unclosed") }
    }

    @Test func refusesTags() {
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: !!str 5") }
    }

    @Test func refusesMultipleDocuments() {
        #expect(throws: YAMLError.self) { try MiniYAML.parse("a: 1\n---\nb: 2") }
    }

    @Test func refusalCarriesTheLineNumber() {
        do {
            _ = try MiniYAML.parse("a: 1\nb: 2\nc: {d: 3}")
            Issue.record("expected a refusal")
        } catch let error as YAMLError {
            #expect(error.line == 3)
        } catch {
            Issue.record("expected a YAMLError")
        }
    }
}

// MARK: - Assertion mapping

struct PromptfooAssertionMappingTests {

    @Test func equalsBecomesExactMatch() throws {
        #expect(try mapped("equals", .scalar("Paris")) == .exactMatch("Paris"))
    }

    @Test func containsBecomesContainsAny() throws {
        #expect(try mapped("contains", .scalar("Paris")) == .containsAny(["Paris"]))
        #expect(try mapped("icontains", .scalar("Paris")) == .containsAny(["Paris"]))
    }

    @Test func containsAllBecomesContainsAll() throws {
        let value = YAMLValue.sequence([.scalar("red"), .scalar("blue")])
        #expect(try mapped("contains-all", value) == .containsAll(["red", "blue"]))
        #expect(try mapped("icontains-all", value) == .containsAll(["red", "blue"]))
    }

    @Test func containsAnyBecomesContainsAny() throws {
        let value = YAMLValue.sequence([.scalar("yes"), .scalar("correct")])
        #expect(try mapped("contains-any", value) == .containsAny(["yes", "correct"]))
    }

    @Test func notContainsBecomesContainsNone() throws {
        #expect(try mapped("not-contains", .scalar("sorry")) == .containsNone(["sorry"]))
        #expect(try mapped("not-icontains", .scalar("sorry")) == .containsNone(["sorry"]))
    }

    @Test func regexBecomesRegex() throws {
        #expect(try mapped("regex", .scalar("^[A-Z]+$")) == .regex("^[A-Z]+$"))
    }

    @Test func startsWithBecomesStartsWith() throws {
        #expect(try mapped("starts-with", .scalar("Dear")) == .startsWith("Dear"))
    }

    @Test func bareIsJSONBecomesValidJSON() throws {
        #expect(try mapped("is-json") == .validJSON(requiredKeys: []))
    }

    @Test func isJSONWithRequiredKeysBecomesValidJSON() throws {
        let schema = YAMLValue.mapping([
            YAMLPair(key: "type", value: .scalar("object")),
            YAMLPair(key: "required", value: .sequence([.scalar("x"), .scalar("y")]))
        ])
        #expect(try mapped("is-json", schema) == .validJSON(requiredKeys: ["x", "y"]))
    }

    /// A richer schema constrains things this app does not check, so honouring
    /// only the key names would grade more leniently than the file asks.
    @Test func isJSONWithARicherSchemaIsNotSupported() throws {
        let schema = YAMLValue.mapping([
            YAMLPair(key: "type", value: .scalar("object")),
            YAMLPair(key: "properties", value: .mapping([]))
        ])
        let reason = try #require(try unsupportedReason("is-json", schema))
        #expect(reason.contains("properties"))
    }

    @Test func aBrokenRegexIsNotSupported() throws {
        let reason = try #require(try unsupportedReason("regex", .scalar("[unclosed")))
        #expect(reason.contains("[unclosed"))
    }

    @Test func anUnknownAssertionIsNotSupportedAndIsNamed() throws {
        let reason = try #require(try unsupportedReason("latency", .scalar("100")))
        #expect(reason.contains("latency"))
    }

    @Test func aFileReferenceIsNotSupported() throws {
        let reason = try #require(try unsupportedReason("equals", .scalar("file://answer.txt")))
        #expect(reason.contains("file://answer.txt"))
    }

    // MARK: Hard refusals

    @Test func executableAssertionsAreRefusedByName() {
        for type in ["javascript", "python", "exec"] {
            do {
                _ = try PromptfooAssertion.map(assertion(type, .scalar("output.length > 0")))
                Issue.record("expected \(type) to be refused")
            } catch let error as SuiteImportError {
                #expect(error == .executableAssertion(type))
            } catch {
                Issue.record("expected a SuiteImportError for \(type)")
            }
        }
    }

    @Test func judgeAssertionsAreRefusedByName() {
        for type in ["llm-rubric", "model-graded-closedqa", "factuality", "g-eval", "similar"] {
            do {
                _ = try PromptfooAssertion.map(assertion(type, .scalar("is it polite?")))
                Issue.record("expected \(type) to be refused")
            } catch let error as SuiteImportError {
                #expect(error == .judgeAssertion(type))
            } catch {
                Issue.record("expected a SuiteImportError for \(type)")
            }
        }
    }
}

// MARK: - promptfoo documents

struct PromptfooImportTests {

    private let validConfig = """
    description: Capitals
    prompts:
      - "Answer with one word. {{question}}"
    tests:
      - description: France
        vars:
          question: What is the capital of France?
        assert:
          - type: equals
            value: Paris
            metric: geography
      - description: Japan
        vars:
          question: What is the capital of Japan?
        assert:
          - type: contains
            value: Tokyo
    """

    @Test func aValidConfigRoundTripsIntoRunnableTasks() throws {
        let result = try importSuite(validConfig)

        #expect(result.format == .promptfooYAML)
        #expect(result.recognizedTaskCount == 2)
        #expect(result.skipped.isEmpty)
        #expect(result.suite.name == "Capitals")
        #expect(result.suite.document.identity == "custom-test")

        let tasks = result.suite.document.tasks
        #expect(tasks[0].id == "france")
        #expect(tasks[0].category == "geography")
        #expect(tasks[0].prompt == "Answer with one word. What is the capital of France?")
        #expect(tasks[0].check.evaluate(answer: "Paris."))
        #expect(!tasks[0].check.evaluate(answer: "Lyon"))
        #expect(tasks[1].category == PromptfooImport.defaultCategory)
        #expect(tasks[1].check.evaluate(answer: "It is Tokyo."))
    }

    @Test func importedTasksRespectThePerTaskTokenCap() throws {
        let result = try importSuite(validConfig)
        #expect(result.suite.document.tasks.allSatisfy {
            $0.maxTokens <= SuiteImportLimits.maxTokensPerTask
        })
    }

    @Test func severalAssertionsBecomeOneAllOfTask() throws {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - description: both
            vars:
              q: name two primary colours
            assert:
              - type: contains-all
                value: [red, blue]
              - type: not-contains
                value: sorry
        """
        let result = try importSuite(yaml)
        #expect(result.recognizedTaskCount == 1)
        guard case .allOf(let checks) = result.suite.document.tasks[0].check else {
            Issue.record("expected an allOf check")
            return
        }
        #expect(checks.count == 2)
        // Passes only when every assertion holds - the way promptfoo scores it.
        #expect(result.suite.document.tasks[0].check.evaluate(answer: "red and blue"))
        #expect(!result.suite.document.tasks[0].check.evaluate(answer: "sorry, red and blue"))
        #expect(!result.suite.document.tasks[0].check.evaluate(answer: "red only"))
    }

    @Test func anEmptyAllOfNeverPasses() {
        #expect(!BenchmarkCheck.allOf([]).evaluate(answer: "anything"))
    }

    @Test func defaultTestAssertionsApplyToEveryTest() throws {
        let yaml = """
        prompts:
          - "{{q}}"
        defaultTest:
          assert:
            - type: not-contains
              value: as an ai
        tests:
          - description: one
            vars:
              q: hello
            assert:
              - type: contains
                value: hi
        """
        let result = try importSuite(yaml)
        guard case .allOf(let checks) = result.suite.document.tasks[0].check else {
            Issue.record("expected an allOf check")
            return
        }
        #expect(checks.contains(.containsNone(["as an ai"])))
    }

    // MARK: Skips - reported, never silent

    @Test func aTestWithAnUnsupportedAssertionIsSkippedWithItsReason() throws {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - description: fast enough
            vars:
              q: hello
            assert:
              - type: latency
                value: 1000
          - description: usable
            vars:
              q: hi
            assert:
              - type: contains
                value: hello
        """
        let result = try importSuite(yaml)
        #expect(result.recognizedTaskCount == 1)
        #expect(result.skipped.count == 1)
        #expect(result.skipped[0].name == "fast enough")
        #expect(result.skipped[0].reason.contains("latency"))
    }

    @Test func aTestWithAnUnresolvedVariableIsSkipped() throws {
        let yaml = """
        prompts:
          - "{{question}}"
        tests:
          - description: missing var
            vars:
              other: hello
            assert:
              - type: contains
                value: hi
          - description: fine
            vars:
              question: hi there
            assert:
              - type: contains
                value: hi
        """
        let result = try importSuite(yaml)
        #expect(result.recognizedTaskCount == 1)
        #expect(result.skipped.count == 1)
        #expect(result.skipped[0].reason.contains("question"))
    }

    @Test func aTemplateExpressionIsSkippedRatherThanPassedThrough() throws {
        let yaml = """
        prompts:
          - "{{ question | upper }}"
        tests:
          - description: filtered
            vars:
              question: hello
            assert:
              - type: contains
                value: hi
        """
        // Every test skipped means nothing runnable is left, which is a
        // refusal rather than an empty suite.
        #expect(refusal(yaml) == .noRunnableTasks)
    }

    @Test func aTestWithNoAssertionsIsSkipped() throws {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - description: ungraded
            vars:
              q: hello
          - description: graded
            vars:
              q: hi
            assert:
              - type: contains
                value: hi
        """
        let result = try importSuite(yaml)
        #expect(result.recognizedTaskCount == 1)
        #expect(result.skipped.count == 1)
        #expect(result.skipped[0].reason.contains("no assertions"))
    }

    // MARK: Refusals

    @Test func aJudgeAssertionRefusesTheWholeImportAndNamesIt() {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - vars:
              q: hello
            assert:
              - type: llm-rubric
                value: is the answer polite?
        """
        #expect(refusal(yaml) == .judgeAssertion("llm-rubric"))
    }

    @Test func anExecutableAssertionRefusesTheWholeImportAndNamesIt() {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - vars:
              q: hello
            assert:
              - type: python
                value: return len(output) > 0
        """
        #expect(refusal(yaml) == .executableAssertion("python"))
    }

    @Test func malformedYAMLIsRefusedWithItsLine() {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - vars: {q: hello}
        """
        guard case .malformedYAML(let line, _)? = refusal(yaml) else {
            Issue.record("expected a malformedYAML refusal")
            return
        }
        #expect(line == 4)
    }

    @Test func aConfigWithoutTestsIsRefused() {
        #expect(refusal("description: nothing here\nprompts:\n  - hello") == .missingTests)
    }

    @Test func aConfigWithoutAPromptIsRefused() {
        let yaml = """
        tests:
          - vars:
              q: hello
            assert:
              - type: contains
                value: hi
        """
        #expect(refusal(yaml) == .missingPrompt)
    }

    @Test func severalPromptsAreRefused() {
        let yaml = """
        prompts:
          - "first {{q}}"
          - "second {{q}}"
        tests:
          - vars:
              q: hello
            assert:
              - type: contains
                value: hi
        """
        #expect(refusal(yaml) == .multiplePrompts(2))
    }

    @Test func externalReferencesAreRefused() {
        let yaml = """
        prompts:
          - file://prompt.txt
        tests:
          - vars:
              q: hello
            assert:
              - type: contains
                value: hi
        """
        #expect(refusal(yaml) == .externalReference("file://prompt.txt"))
    }

    @Test func duplicateTaskIDsAreRefused() {
        let yaml = """
        prompts:
          - "{{q}}"
        tests:
          - description: same name
            vars:
              q: one
            assert:
              - type: contains
                value: a
          - description: same name
            vars:
              q: two
            assert:
              - type: contains
                value: b
        """
        #expect(refusal(yaml) == .duplicateTaskID("same_name"))
    }
}

// MARK: - The app's own JSON

struct AppJSONSuiteImportTests {

    private func json(tasks: String, extra: String = "") -> String {
        "{\"schemaVersion\":2,\"suiteVersion\":3\(extra),\"tasks\":[\(tasks)]}"
    }

    private func task(_ id: String, maxTokens: Int? = nil) -> String {
        let cap = maxTokens.map { ",\"maxTokens\":\($0)" } ?? ""
        return "{\"id\":\"\(id)\",\"category\":\"math\",\"prompt\":\"2+2?\"\(cap),"
            + "\"check\":{\"kind\":\"exactMatch\",\"value\":\"4\"}}"
    }

    @Test func aValidDocumentRoundTripsIntoRunnableTasks() throws {
        let result = try importSuite(json(tasks: task("a") + "," + task("b"),
                                          extra: ",\"name\":\"My suite\""))
        #expect(result.format == .appJSON)
        #expect(result.recognizedTaskCount == 2)
        #expect(result.suite.name == "My suite")
        #expect(result.suite.document.suiteVersion == 3)
        #expect(result.suite.document.identity == "custom-test")
        #expect(result.suite.document.tasks[0].check.evaluate(answer: "4"))
    }

    @Test func theSuiteIsNamedAfterTheFileWhenItNamesItself() throws {
        let result = try BenchmarkSuiteImporter.makeSuite(
            text: json(tasks: task("a")), source: "my-eval.json", suiteID: "custom-test"
        )
        #expect(result.suite.name == "my-eval.json")
    }

    @Test func malformedJSONIsRefused() {
        guard case .malformedJSON? = refusal("{\"schemaVersion\": 2, \"tasks\": [") else {
            Issue.record("expected a malformedJSON refusal")
            return
        }
    }

    /// An unknown check `kind` fails the WHOLE document, the same rule the
    /// remote suite follows: a partially understood suite is never run.
    @Test func anUnknownCheckKindIsRefused() {
        let bad = "{\"id\":\"x\",\"category\":\"math\",\"prompt\":\"p\","
            + "\"check\":{\"kind\":\"vibeCheck\",\"value\":\"4\"}}"
        guard case .malformedJSON? = refusal(json(tasks: bad)) else {
            Issue.record("expected a malformedJSON refusal")
            return
        }
    }

    @Test func aNewerSchemaVersionIsRefused() {
        let future = BenchmarkSuiteDocument.supportedSchemaVersion + 1
        let text = "{\"schemaVersion\":\(future),\"suiteVersion\":1,\"tasks\":[\(task("a"))]}"
        #expect(refusal(text) == .unsupportedSchemaVersion(future))
    }

    @Test func duplicateTaskIDsAreRefused() {
        #expect(refusal(json(tasks: task("a") + "," + task("a"))) == .duplicateTaskID("a"))
    }

    @Test func anEmptyTaskListIsRefused() {
        #expect(refusal(json(tasks: "")) == .noRunnableTasks)
    }

    @Test func theSizeCapTrips() {
        let overflow = SuiteImportLimits.maxTasks + 1
        let tasks = (1...overflow).map { task("t\($0)") }.joined(separator: ",")
        #expect(refusal(json(tasks: tasks)) == .tooManyTasks(count: overflow))
    }

    @Test func theSizeCapAllowsExactlyTheLimit() throws {
        let tasks = (1...SuiteImportLimits.maxTasks)
            .map { task("t\($0)") }.joined(separator: ",")
        let result = try importSuite(json(tasks: tasks))
        #expect(result.recognizedTaskCount == SuiteImportLimits.maxTasks)
        // A suite this long has to warn about how long it will run.
        #expect(result.notes.contains { $0.contains("minutes") })
    }

    @Test func anOversizedTokenCapIsClampedAndReported() throws {
        let result = try importSuite(json(tasks: task("a", maxTokens: 4096)))
        #expect(result.suite.document.tasks[0].maxTokens == SuiteImportLimits.maxTokensPerTask)
        #expect(result.notes.contains { $0.contains("capped") })
    }

    @Test func anEmptyFileIsRefused() {
        #expect(refusal("   \n  ") == .empty)
    }

    @Test func aJSONArrayIsRefusedAsAnUnknownFormat() {
        #expect(refusal("[1, 2, 3]") == .unrecognizedFormat)
    }

    @Test func anOversizedFileIsRefusedBeforeParsing() {
        let data = Data(repeating: UInt8(ascii: "{"), count: SuiteImportLimits.maxBytes + 1)
        do {
            _ = try BenchmarkSuiteImporter.makeSuite(data: data, source: "big.json")
            Issue.record("expected a refusal")
        } catch let error as SuiteImportError {
            #expect(error == .tooLarge(bytes: SuiteImportLimits.maxBytes + 1))
        } catch {
            Issue.record("expected a SuiteImportError")
        }
    }
}

// MARK: - Suite identity in reports

struct SuiteIdentityComparisonTests {

    private func report(suiteID: String?, suiteVersion: Int, passed: Bool) -> BenchmarkReport {
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
            quality: QualityResults(
                suiteVersion: suiteVersion,
                categories: [CategoryScore(
                    category: "math",
                    tasks: [TaskResult(
                        taskId: "t1", passed: passed, latencySeconds: 1, answer: "4"
                    )]
                )],
                suiteID: suiteID,
                suiteName: suiteID == nil ? nil : "Imported"
            )
        )
    }

    @Test func aMissingSuiteIDReadsAsTheBuiltInSuite() {
        #expect(report(suiteID: nil, suiteVersion: 2, passed: true).suiteIdentity
                == BenchmarkSuiteDocument.builtInID)
    }

    @Test func twoBuiltInRunsAreComparable() {
        let current = report(suiteID: nil, suiteVersion: 2, passed: true)
        let previous = report(suiteID: BenchmarkSuiteDocument.builtInID,
                              suiteVersion: 2, passed: false)
        #expect(current.isScoreComparable(with: previous))
    }

    /// The point of the whole identity field: a custom suite that happens to
    /// number itself the same as ours is still a different exam.
    @Test func aCustomSuiteIsNeverComparedAgainstTheBuiltInOne() {
        let custom = report(suiteID: "custom-abcd1234", suiteVersion: 2, passed: true)
        let builtIn = report(suiteID: nil, suiteVersion: 2, passed: false)
        #expect(!custom.isScoreComparable(with: builtIn))

        let comparison = BenchmarkRunComparison(current: custom, previous: builtIn)
        #expect(comparison.scoreDelta == nil)
        #expect(comparison.scoreSuppressedByDifferentSuites)
        #expect(comparison.suppressionNote?.contains("different task suites") == true)
    }

    @Test func twoDifferentCustomSuitesAreNotComparable() {
        let one = report(suiteID: "custom-aaaa1111", suiteVersion: 1, passed: true)
        let two = report(suiteID: "custom-bbbb2222", suiteVersion: 1, passed: false)
        #expect(!one.isScoreComparable(with: two))
    }

    @Test func theSameCustomSuiteIsComparableWithItself() {
        let one = report(suiteID: "custom-aaaa1111", suiteVersion: 1, passed: true)
        let two = report(suiteID: "custom-aaaa1111", suiteVersion: 1, passed: false)
        #expect(one.isScoreComparable(with: two))
        #expect(BenchmarkRunComparison(current: one, previous: two).scoreDelta == 100)
    }
}

// MARK: - Remote suite floor

/// A published suite may fix and extend the one a build ships, never roll it
/// back. The model catalog shipped this bug once: the remote copy replaced
/// the bundled one unconditionally and kept serving a removed entry.
struct RemoteSuiteFloorTests {

    private func suite(version: Int) -> BenchmarkSuiteDocument {
        BenchmarkSuiteDocument(
            schemaVersion: 2,
            suiteVersion: version,
            tasks: [
                BenchmarkTask(
                    id: "t1",
                    category: "math",
                    prompt: "2+2, digits only.",
                    check: .numeric(expected: 4, tolerance: 0)
                )
            ]
        )
    }

    @Test func aNewerSuiteIsAccepted() {
        #expect(BenchmarkSuiteFile.isAtLeastAsNew(suite(version: 4), as: suite(version: 3)))
    }

    @Test func theSameVersionIsAccepted() {
        // Same version, corrected tasks: that is the point of shipping the
        // suite as a resource that can be republished.
        #expect(BenchmarkSuiteFile.isAtLeastAsNew(suite(version: 3), as: suite(version: 3)))
    }

    @Test func anOlderSuiteIsRejected() {
        #expect(!BenchmarkSuiteFile.isAtLeastAsNew(suite(version: 2), as: suite(version: 3)))
    }

    @Test func anythingBeatsNoBundledSuite() {
        #expect(BenchmarkSuiteFile.isAtLeastAsNew(suite(version: 1), as: nil))
    }
}
