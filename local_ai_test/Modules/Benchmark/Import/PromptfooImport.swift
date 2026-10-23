//
//  PromptfooImport.swift
//  local_ai_test
//
//  Maps a promptfoo `promptfooconfig.yaml` onto this app's tasks.
//
//  promptfoo is the eval format app developers actually write, and most of
//  what a promptfoo test asserts is a plain string check — the same family of
//  checks `BenchmarkCheck` already runs. Those map across exactly:
//
//      equals          -> exactMatch        contains-all -> containsAll
//      contains        -> containsAny       contains-any -> containsAny
//      not-contains    -> containsNone      regex        -> regex
//      starts-with     -> startsWith        is-json      -> validJSON
//
//  The `i`-prefixed spellings (icontains, icontains-all, …) map to the same
//  checks. That is not a shortcut: `BenchmarkChecker.normalize` lower-cases
//  and strips punctuation before comparing, so EVERY string check here is
//  already case-insensitive. The flip side is worth stating plainly — a
//  case-SENSITIVE promptfoo `contains` becomes case-insensitive on import.
//  A suite that grades capitalisation should use a `regex` assertion, which
//  is matched against the raw answer.
//
//  What does NOT come across, and what happens instead:
//
//    • `javascript`, `python` — code. REFUSES THE WHOLE IMPORT.
//    • `llm-rubric`, `model-graded-*`, `g-eval`, `similar`, `factuality`,
//      the context-* metrics — a judge model. REFUSES THE WHOLE IMPORT.
//      A judged score depends on the judge's quantization and the phone it
//      ran on; two runs of the same answers would disagree.
//    • anything else unrecognised (`latency`, `cost`, `webhook`, a JSON
//      Schema richer than `required`) — the TEST IS SKIPPED and listed with
//      the reason. Dropping just the assertion would make the test easier
//      than its author wrote it, which is the silent partial import this
//      importer exists to avoid.
//
//  A test with several assertions becomes ONE task whose check is
//  `.allOf`, so it passes only when all of them hold — exactly as promptfoo
//  scores it. Splitting it into one task per assertion would hand out
//  partial credit for a test its author considers failed.
//
//  Multiple choice. promptfoo has no built-in notion of answer options, so a
//  test declares them in its free-form `metadata`, and the task gets the
//  matching guessing floor (`BenchmarkTask.chance`) for the score:
//
//      metadata:
//        choices: [A, B, C, D]    # a list of n options -> chance 1/n
//        choices: 4               # or just the count   -> chance 1/4
//        chance: 0.25             # or the floor itself, 0...1
//
//  An explicit `chance` wins over `choices`. A value that cannot be read
//  (fewer than two choices, a chance outside 0...1) skips the test with the
//  reason, rather than scoring it as a free answer.
//

import Foundation

nonisolated enum PromptfooImport {

    /// Everything one config produced.
    struct Mapping: Sendable {
        let tasks: [BenchmarkTask]
        let skipped: [SkippedSuiteEntry]
        let notes: [String]
        /// The config's `description`, used as the suite name.
        let name: String?
    }

    /// Category given to a task whose assertions carry no promptfoo `metric`.
    static let defaultCategory = "imported"

    // MARK: Entry

    static func map(_ root: YAMLValue) throws -> Mapping {
        let name = root["description"]?.stringValue

        let topTemplate = try promptTemplate(root)
        let defaultTest = root["defaultTest"]
        let defaultVars = try variables(defaultTest?["vars"]) ?? [:]
        let defaultAsserts = defaultTest?["assert"]?.asSequence ?? []

        guard let testsNode = root["tests"] else { throw SuiteImportError.missingTests }
        if let external = testsNode.stringValue {
            throw SuiteImportError.externalReference(external)
        }
        guard let tests = testsNode.sequenceValue, !tests.isEmpty else {
            throw SuiteImportError.missingTests
        }

        // A config with no prompt anywhere is a config we cannot run at all,
        // which is a refusal rather than "every test was skipped".
        let anyTestPrompt = tests.contains { $0["prompt"] != nil }
        guard topTemplate != nil || anyTestPrompt else {
            throw SuiteImportError.missingPrompt
        }

        var tasks: [BenchmarkTask] = []
        var skipped: [SkippedSuiteEntry] = []
        var multiAssertionTests = 0

        for (offset, test) in tests.enumerated() {
            let label = testLabel(test, index: offset)

            guard test.mappingValue != nil else {
                skipped.append(SkippedSuiteEntry(
                    name: label, reason: "this test is not a key/value block"))
                continue
            }

            // Assertions first: a judge or an executable assertion anywhere
            // in the file stops the import, so it must be seen even if the
            // test would be skipped for some other reason.
            let assertions = test["assert"]?.asSequence ?? []
            let combined = assertions + defaultAsserts
            var checks: [BenchmarkCheck] = []
            var unsupported: String?

            for assertion in combined {
                switch try PromptfooAssertion.map(assertion) {
                case .check(let check):
                    checks.append(check)
                case .unsupported(let reason):
                    // Keep looping: a later assertion may still be a judge
                    // one, and that refuses the whole import.
                    if unsupported == nil { unsupported = reason }
                }
            }

            if let unsupported {
                skipped.append(SkippedSuiteEntry(name: label, reason: unsupported))
                continue
            }
            guard !checks.isEmpty else {
                skipped.append(SkippedSuiteEntry(
                    name: label, reason: "no assertions to grade the answer against"))
                continue
            }

            let vars: [String: String]
            do {
                vars = defaultVars.merging(try variables(test["vars"]) ?? [:]) { _, new in new }
            } catch let error as VariableError {
                skipped.append(SkippedSuiteEntry(name: label, reason: error.reason))
                continue
            }

            guard let template = test["prompt"]?.stringValue ?? topTemplate else {
                skipped.append(SkippedSuiteEntry(
                    name: label, reason: "no prompt for this test"))
                continue
            }

            let prompt: String
            switch Template.render(template, vars: vars) {
            case .text(let rendered):
                prompt = rendered
            case .failure(let reason):
                skipped.append(SkippedSuiteEntry(name: label, reason: reason))
                continue
            }
            guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                skipped.append(SkippedSuiteEntry(
                    name: label, reason: "the prompt is empty once its variables are filled in"))
                continue
            }

            let chance: Double
            switch guessingFloor(test["metadata"]) {
            case .success(let value):
                chance = value
            case .failure(let error):
                skipped.append(SkippedSuiteEntry(name: label, reason: error.reason))
                continue
            }

            if checks.count > 1 { multiAssertionTests += 1 }
            tasks.append(BenchmarkTask(
                id: taskID(test, index: offset),
                category: category(of: combined),
                prompt: prompt,
                maxTokens: SuiteImportLimits.defaultTokensPerTask,
                check: checks.count == 1 ? checks[0] : .allOf(checks),
                chance: chance
            ))
        }

        var notes: [String] = []
        if multiAssertionTests > 0 {
            notes.append("\(multiAssertionTests) test\(multiAssertionTests == 1 ? "" : "s") "
                + "carry more than one assertion; each of those passes only when every "
                + "one of its assertions holds.")
        }
        notes.append("Answers are compared after lower-casing and punctuation stripping, "
            + "so string assertions are case-insensitive here. Use a regex assertion "
            + "where letter case matters.")

        return Mapping(tasks: tasks, skipped: skipped, notes: notes, name: name)
    }

    // MARK: Prompt

    /// The single prompt template a config may define.
    private static func promptTemplate(_ root: YAMLValue) throws -> String? {
        let node = root["prompts"] ?? root["prompt"]
        guard let node, !node.isNull else { return nil }

        let entries = node.asSequence
        guard !entries.isEmpty else { return nil }
        guard entries.count == 1 else {
            throw SuiteImportError.multiplePrompts(entries.count)
        }
        guard let text = entries[0].stringValue else {
            throw SuiteImportError.unsupportedPrompt(
                "only a plain text prompt can be imported")
        }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("file://") {
            throw SuiteImportError.externalReference(trimmed)
        }
        // A chat-format prompt is a JSON array of role/content messages. The
        // engine here takes one user prompt plus an optional system prompt,
        // so honouring half of a multi-turn conversation would score
        // something the file did not describe.
        if trimmed.hasPrefix("[") {
            throw SuiteImportError.unsupportedPrompt(
                "chat-format prompts (a JSON array of messages) are not supported")
        }
        return text
    }

    // MARK: Multiple choice

    /// The test's guessing floor from its `metadata` (see the file header).
    /// 0 when the test declares no choices.
    static func guessingFloor(_ metadata: YAMLValue?) -> Result<Double, VariableError> {
        guard let metadata, !metadata.isNull, metadata.mappingValue != nil else {
            return .success(0)
        }
        if let node = metadata["chance"] {
            guard let text = node.stringValue,
                  let value = Double(text.trimmingCharacters(in: .whitespaces)),
                  value.isFinite, (0...1).contains(value) else {
                return .failure(VariableError(
                    reason: "metadata 'chance' must be a number between 0 and 1"))
            }
            return .success(value)
        }
        if let node = metadata["choices"] {
            let count: Int?
            if let items = node.sequenceValue {
                count = items.count
            } else if let text = node.stringValue {
                count = Int(text.trimmingCharacters(in: .whitespaces))
            } else {
                count = nil
            }
            guard let count, count >= 2 else {
                return .failure(VariableError(
                    reason: "metadata 'choices' must list at least two options, "
                        + "or give their count"))
            }
            return .success(1 / Double(count))
        }
        return .success(0)
    }

    // MARK: Variables

    struct VariableError: Error, Equatable {
        let reason: String
    }

    /// promptfoo vars are usually plain scalars. A list or a nested mapping
    /// would have to be rendered by nunjucks, which is not on this device.
    private static func variables(_ node: YAMLValue?) throws -> [String: String]? {
        guard let node, !node.isNull else { return nil }
        guard let pairs = node.mappingValue else {
            throw VariableError(reason: "'vars' is not a key/value block")
        }
        var result: [String: String] = [:]
        for pair in pairs {
            guard let text = pair.value.stringValue else {
                throw VariableError(
                    reason: "the variable '\(pair.key)' is a list or a nested block, "
                        + "which cannot be substituted into a prompt")
            }
            if text.hasPrefix("file://") {
                throw VariableError(
                    reason: "the variable '\(pair.key)' reads an external file (\(text))")
            }
            result[pair.key] = text
        }
        return result
    }

    // MARK: Naming

    private static func testLabel(_ test: YAMLValue, index: Int) -> String {
        if let described = (test["description"] ?? test["name"])?.stringValue,
           !described.isEmpty {
            return described
        }
        return "test \(index + 1)"
    }

    private static func taskID(_ test: YAMLValue, index: Int) -> String {
        if let described = (test["description"] ?? test["name"])?.stringValue {
            let slugged = slug(described)
            if !slugged.isEmpty { return slugged }
        }
        return "test_\(index + 1)"
    }

    /// promptfoo's `metric` is its own grouping concept, which is what a
    /// category is here — so the first one wins the task's category.
    private static func category(of assertions: [YAMLValue]) -> String {
        for assertion in assertions {
            if let metric = assertion["metric"]?.stringValue {
                let slugged = slug(metric)
                if !slugged.isEmpty { return slugged }
            }
        }
        return defaultCategory
    }

    static func slug(_ text: String) -> String {
        let mapped = text.lowercased().map { character -> Character in
            character.isLetter || character.isNumber ? character : "_"
        }
        return String(String(mapped)
            .split(separator: "_", omittingEmptySubsequences: true)
            .joined(separator: "_")
            .prefix(60))
    }

    // MARK: Templates

    /// The `{{ var }}` substitution, and nothing else.
    ///
    /// promptfoo renders prompts with nunjucks: filters, conditionals, loops,
    /// the lot. None of that runs here, and a template engine that silently
    /// left `{{ name | title }}` in the prompt would send the model a
    /// question containing its own markup. So anything that is not a bare
    /// variable name refuses, and the test is skipped with the reason.
    enum Template {

        enum Outcome {
            case text(String)
            case failure(String)
        }

        static func render(_ template: String, vars: [String: String]) -> Outcome {
            if template.contains("{%") {
                return .failure("the prompt uses a template statement ({% ... %}), "
                    + "which this app cannot render")
            }

            var result = ""
            var rest = Substring(template)

            while let open = rest.range(of: "{{") {
                result += rest[rest.startIndex..<open.lowerBound]
                let afterOpen = rest[open.upperBound...]
                guard let close = afterOpen.range(of: "}}") else {
                    return .failure("the prompt has an unclosed {{ }} placeholder")
                }
                let inner = afterOpen[afterOpen.startIndex..<close.lowerBound]
                    .trimmingCharacters(in: .whitespaces)
                guard !inner.isEmpty else {
                    return .failure("the prompt has an empty {{ }} placeholder")
                }
                guard inner.allSatisfy({ $0.isLetter || $0.isNumber || $0 == "_" }) else {
                    return .failure("the placeholder '{{ \(inner) }}' is a template "
                        + "expression, not a plain variable")
                }
                guard let value = vars[inner] else {
                    return .failure("no value for the variable '\(inner)'")
                }
                result += value
                rest = afterOpen[close.upperBound...]
            }
            result += rest
            return .text(result)
        }
    }
}

// MARK: - Assertions

nonisolated enum PromptfooAssertion {

    enum Outcome {
        case check(BenchmarkCheck)
        /// Recognised as an assertion, but not one that can run here. The
        /// test carrying it is skipped and the reason is reported.
        case unsupported(String)
    }

    /// Assertion types graded by another model.
    static let judgeTypes: Set<String> = [
        "llm-rubric", "model-graded-closedqa", "model-graded-factuality",
        "factuality", "answer-relevance", "context-relevance", "context-recall",
        "context-faithfulness", "select-best", "similar", "classifier",
        "moderation", "g-eval", "pi"
    ]

    /// Assertion types that run code.
    static let executableTypes: Set<String> = ["javascript", "python", "exec"]

    static func isJudge(_ type: String) -> Bool {
        judgeTypes.contains(type)
            || type.hasPrefix("model-graded")
            || type.hasPrefix("llm-")
            || type.contains("g-eval")
    }

    static func isExecutable(_ type: String) -> Bool {
        executableTypes.contains(type)
            || type.hasPrefix("javascript")
            || type.hasPrefix("python")
            || type.hasSuffix(".js")
            || type.hasSuffix(".py")
    }

    /// Maps one `assert` entry.
    ///
    /// - Throws: `SuiteImportError.judgeAssertion` / `.executableAssertion`
    ///   for the two kinds that refuse the whole import.
    static func map(_ node: YAMLValue) throws -> Outcome {
        guard node.mappingValue != nil else {
            return .unsupported("an assertion is not a key/value block")
        }
        guard let rawType = node["type"]?.stringValue else {
            return .unsupported("an assertion has no 'type'")
        }
        let type = rawType.trimmingCharacters(in: .whitespaces).lowercased()

        if isJudge(type) { throw SuiteImportError.judgeAssertion(rawType) }
        if isExecutable(type) { throw SuiteImportError.executableAssertion(rawType) }

        let value = node["value"]

        // A `file://` value is content this app cannot fetch, whatever the
        // assertion type says.
        if let text = value?.stringValue, text.hasPrefix("file://") {
            return .unsupported("the assertion '\(rawType)' reads its value from an "
                + "external file (\(text))")
        }

        switch type {
        case "equals", "equality":
            guard let text = value?.stringValue else { return missingValue(rawType) }
            return .check(.exactMatch(text))

        case "contains", "icontains":
            guard let text = value?.stringValue else { return missingValue(rawType) }
            return .check(.containsAny([text]))

        case "contains-all", "icontains-all":
            guard let list = stringList(value), !list.isEmpty else {
                return missingValue(rawType)
            }
            return .check(.containsAll(list))

        case "contains-any", "icontains-any":
            guard let list = stringList(value), !list.isEmpty else {
                return missingValue(rawType)
            }
            return .check(.containsAny(list))

        case "not-contains", "not-icontains":
            guard let list = stringList(value), !list.isEmpty else {
                return missingValue(rawType)
            }
            return .check(.containsNone(list))

        case "regex":
            guard let pattern = value?.stringValue else { return missingValue(rawType) }
            // Compiled now rather than at run time: an unrunnable pattern
            // fails every answer, and a whole category scoring zero for a
            // typo would look like a model problem.
            guard (try? NSRegularExpression(pattern: pattern)) != nil else {
                return .unsupported("the regex '\(pattern)' is not a valid pattern")
            }
            return .check(.regex(pattern))

        case "starts-with", "istarts-with":
            guard let text = value?.stringValue else { return missingValue(rawType) }
            return .check(.startsWith(text))

        case "is-json":
            return jsonCheck(value, rawType: rawType)

        default:
            return .unsupported("the assertion '\(rawType)' is not one this app can "
                + "check without running code or asking another model")
        }
    }

    /// `is-json` with no value is "the answer contains a JSON object". With a
    /// schema, only the narrow `type: object` + `required: [...]` form maps
    /// exactly onto `validJSON`; a richer schema constrains types, lengths
    /// and nesting that this checker does not look at, and honouring only
    /// the key names would grade the answer more leniently than the file
    /// asks.
    private static func jsonCheck(_ value: YAMLValue?, rawType: String) -> Outcome {
        guard let value, !value.isNull else {
            return .check(.validJSON(requiredKeys: []))
        }
        guard let pairs = value.mappingValue else {
            return .unsupported("the assertion '\(rawType)' has a value this app "
                + "cannot read as a JSON Schema")
        }
        var required: [String] = []
        for pair in pairs {
            switch pair.key {
            case "type":
                guard pair.value.stringValue == "object" else {
                    return .unsupported("the assertion '\(rawType)' validates a schema "
                        + "of type '\(pair.value.stringValue ?? "?")'; only an object "
                        + "schema's required keys can be checked here")
                }
            case "required":
                guard let list = stringList(pair.value) else {
                    return .unsupported("the assertion '\(rawType)' has a 'required' "
                        + "list this app cannot read")
                }
                required = list
            default:
                return .unsupported("the assertion '\(rawType)' uses the JSON Schema "
                    + "keyword '\(pair.key)', which this app does not validate")
            }
        }
        return .check(.validJSON(requiredKeys: required))
    }

    private static func missingValue(_ rawType: String) -> Outcome {
        .unsupported("the assertion '\(rawType)' has no usable 'value'")
    }

    /// A scalar counts as a one-element list, matching promptfoo's own
    /// "one thing or a list of things" fields.
    private static func stringList(_ node: YAMLValue?) -> [String]? {
        guard let node, !node.isNull else { return nil }
        if let text = node.stringValue { return [text] }
        guard let items = node.sequenceValue else { return nil }
        var result: [String] = []
        for item in items {
            guard let text = item.stringValue else { return nil }
            result.append(text)
        }
        return result
    }
}
