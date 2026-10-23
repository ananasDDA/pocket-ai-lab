//
//  BenchmarkSuite.swift
//  local_ai_test
//
//  The task suite the quality stage runs, plus the checkers that grade it.
//
//  Design rule: NO LLM JUDGE. Every task is machine-checkable — an exact
//  string, a substring set, a regex, a number with a tolerance, or a JSON
//  object with required keys. A judge model would make the score depend on
//  the judge's mood, its own quantization and the phone it ran on, and would
//  double the runtime of every benchmark. Deterministic checks mean the same
//  answers always produce the same score, on every device, forever.
//
//  The suite ships as JSON so tasks can be fixed without an App Store
//  release: the bundled `benchmarks.json` is read synchronously, and a
//  background refresh pulls a newer copy from the project's GitHub repo —
//  exactly the pattern `RemoteCatalog` uses for `catalog.json`. A run never
//  waits on the network; it takes whatever is cached or bundled right now.
//

import Foundation
import Observation

// MARK: - Task

/// One verifiable prompt/answer pair.
nonisolated struct BenchmarkTask: Codable, Sendable, Hashable, Identifiable {

    /// Stable id, e.g. "math_word_02". Used as the key in `TaskResult`, so
    /// renaming one breaks comparability with older reports — add instead.
    let id: String
    /// "math", "instructions", "knowledge", "retrieval", "multilingual".
    let category: String
    let prompt: String
    /// Optional system prompt. Kept per-task because the instruction-following
    /// tasks need a bare model with no coaching, while the retrieval ones
    /// benefit from being told to answer from the passage only.
    let systemPrompt: String?
    /// Generation cap for this task. Small on purpose: an unbounded cap turns
    /// one chatty model into a ten-minute benchmark.
    let maxTokens: Int
    let check: BenchmarkCheck

    /// Probability of passing this task by guessing, 0...1. 0 (the default,
    /// and the value of every bundled task) for a free-answer check: nobody
    /// guesses "1081" or a valid JSON object by accident. A multiple-choice
    /// task with n options sets 1/n, so 0.25 for a four-option MMLU item.
    ///
    /// It feeds the chance-normalised score (see the header of
    /// `BenchmarkTypes.swift`): a task's pass counts for (1 - chance) above
    /// the guessing floor, the way the Open LLM Leaderboard normalises
    /// multiple-choice benchmarks. Omitted from the JSON when it is 0.
    let chance: Double

    init(
        id: String,
        category: String,
        prompt: String,
        systemPrompt: String? = nil,
        maxTokens: Int = 96,
        check: BenchmarkCheck,
        chance: Double = 0
    ) {
        self.id = id
        self.category = category
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.maxTokens = maxTokens
        self.check = check
        // Clamped for values built in code; the decoder rejects an
        // out-of-range value outright instead (see below).
        self.chance = chance.isFinite ? min(max(chance, 0), 1) : 0
    }

    private enum CodingKeys: String, CodingKey {
        case id, category, prompt, systemPrompt, maxTokens, check, chance
    }

    /// Hand-written so a hand-edited `benchmarks.json` may omit
    /// `systemPrompt`, `maxTokens` and `chance`.
    ///
    /// A `chance` outside 0...1 fails the whole document, like an unknown
    /// check kind does: clamping a typo such as `25` (meant as 25%) to 1
    /// would silently remove the task from the score.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let chance = try c.decodeIfPresent(Double.self, forKey: .chance) ?? 0
        guard chance.isFinite, (0...1).contains(chance) else {
            throw DecodingError.dataCorruptedError(
                forKey: .chance, in: c,
                debugDescription: "'chance' must be between 0 and 1 (1/n for n choices)"
            )
        }
        self.init(
            id: try c.decode(String.self, forKey: .id),
            category: try c.decode(String.self, forKey: .category),
            prompt: try c.decode(String.self, forKey: .prompt),
            systemPrompt: try c.decodeIfPresent(String.self, forKey: .systemPrompt),
            maxTokens: try c.decodeIfPresent(Int.self, forKey: .maxTokens) ?? 96,
            check: try c.decode(BenchmarkCheck.self, forKey: .check),
            chance: chance
        )
    }

    /// Written out by hand only so a zero `chance` stays out of the file:
    /// an exported free-answer suite reads exactly as it did before the
    /// field existed.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(category, forKey: .category)
        try c.encode(prompt, forKey: .prompt)
        try c.encodeIfPresent(systemPrompt, forKey: .systemPrompt)
        try c.encode(maxTokens, forKey: .maxTokens)
        try c.encode(check, forKey: .check)
        if chance > 0 { try c.encode(chance, forKey: .chance) }
    }

    /// The same task with a different token cap. Used by the importer's
    /// clamp, so a new field can never be dropped by a hand-copied init.
    func withMaxTokens(_ maxTokens: Int) -> BenchmarkTask {
        BenchmarkTask(
            id: id,
            category: category,
            prompt: prompt,
            systemPrompt: systemPrompt,
            maxTokens: maxTokens,
            check: check,
            chance: chance
        )
    }
}

// MARK: - Check

/// How an answer is graded. Encoded with a `kind` discriminator so the JSON
/// stays hand-editable. Every spelling, with a worked example:
///
///     { "kind": "exactMatch",      "value": "paris" }
///     { "kind": "containsAll",     "values": ["red", "blue"] }
///     { "kind": "containsAny",     "values": ["yes", "correct"] }
///     { "kind": "containsNone",    "values": ["sorry", "as an ai"] }
///     { "kind": "regex",           "pattern": "^[A-Z ]+$" }
///     { "kind": "numeric",         "expected": 42, "tolerance": 0.01 }
///     { "kind": "validJSON",       "requiredKeys": ["x", "y"] }
///     { "kind": "letterFrequency", "letter": "e", "min": 3, "max": 8 }
///     { "kind": "wordCount",       "min": 30, "max": 60 }
///     { "kind": "lineCount",       "min": 3,  "max": 3 }
///     { "kind": "startsWith",      "value": "dear reader" }
///     { "kind": "endsWith",        "value": "kind regards" }
///     { "kind": "allOf",           "checks": [ {...}, {...} ] }
///
/// `min` defaults to 0 and `max` may be omitted (or `null`) for "no upper
/// bound", so `{"kind":"wordCount","max":50}` is a legal "at most 50 words".
///
/// FORWARD COMPATIBILITY: an unrecognised `kind` makes the WHOLE document
/// fail to decode, which is deliberate. `BenchmarkSuiteFile.decode` then
/// returns nil and the bundled suite stays in force, so an old app build
/// never runs a partially-understood remote suite and never silently scores
/// a model on a subset of its tasks.
nonisolated enum BenchmarkCheck: Codable, Sendable, Hashable {

    /// Answer must equal the expected string after normalization (trim,
    /// lowercase, punctuation stripped, whitespace collapsed).
    case exactMatch(String)
    /// Every needle must appear in the normalized answer.
    case containsAll([String])
    /// At least one needle must appear in the normalized answer.
    case containsAny([String])
    /// NONE of the needles may appear in the normalized answer — the
    /// forbidden-words check IFEval uses ("write about X without using the
    /// word Y"). An empty needle list passes: nothing was forbidden.
    case containsNone([String])
    /// Regex over the RAW answer, case-SENSITIVE. Used where formatting
    /// matters — uppercase replies, word counts, line shapes. Add `(?i)` to
    /// the pattern when case should be ignored.
    case regex(String)
    /// The answer's extracted number must be within `tolerance` of
    /// `expected`.
    ///
    /// v2: extraction follows the lm-evaluation-harness fallback chain (see
    /// `BenchmarkChecker.extractAnswerNumber`) rather than blindly taking the
    /// last number. The last-number rule is still the FINAL fallback, so
    /// every answer that passed under v1 still passes.
    case numeric(expected: Double, tolerance: Double)
    /// The answer must contain a JSON object carrying every required key.
    case validJSON(requiredKeys: [String])
    /// Occurrences of one letter in the answer, counted case-insensitively
    /// over the RAW text, must fall in `min...max`. IFEval's letter-frequency
    /// constraint ("use the letter e at least three times"). `max == nil`
    /// means no upper bound. A `letter` longer than one character counts
    /// occurrences of that whole substring; an empty `letter` always fails.
    case letterFrequency(letter: String, min: Int, max: Int?)
    /// Whitespace-split word count of the RAW answer, bounded.
    case wordCount(min: Int, max: Int?)
    /// Count of NON-EMPTY lines in the RAW answer, bounded. Blank separator
    /// lines are ignored so "three bullet points" does not fail because the
    /// model double-spaced them.
    case lineCount(min: Int, max: Int?)
    /// Normalized answer must start with the normalized prefix.
    case startsWith(String)
    /// Normalized answer must end with the normalized suffix.
    case endsWith(String)
    /// Every nested check must pass.
    ///
    /// v3, added for imported suites: a promptfoo test carries a LIST of
    /// assertions and passes only when all of them hold. Splitting such a
    /// test into one task per assertion would hand out partial credit for a
    /// test its own author considers failed, so the list is kept together as
    /// one task instead. An EMPTY list fails: a task that verifies nothing
    /// has not verified anything.
    case allOf([BenchmarkCheck])

    private enum CodingKeys: String, CodingKey {
        case kind, value, values, pattern, expected, tolerance, requiredKeys
        case letter, min, max, checks
    }

    private enum Kind: String, Codable {
        case exactMatch, containsAll, containsAny, regex, numeric, validJSON
        // v2 (schemaVersion 2)
        case containsNone, letterFrequency, wordCount, lineCount
        case startsWith, endsWith
        // v3 (schemaVersion 3)
        case allOf
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Shared by every bounded-count kind: `min` defaults to 0 so a
        // ceiling-only task needs no filler, `max` stays optional.
        func bounds() throws -> (Int, Int?) {
            (
                try c.decodeIfPresent(Int.self, forKey: .min) ?? 0,
                try c.decodeIfPresent(Int.self, forKey: .max)
            )
        }

        switch try c.decode(Kind.self, forKey: .kind) {
        case .exactMatch:
            self = .exactMatch(try c.decode(String.self, forKey: .value))
        case .containsAll:
            self = .containsAll(try c.decode([String].self, forKey: .values))
        case .containsAny:
            self = .containsAny(try c.decode([String].self, forKey: .values))
        case .containsNone:
            self = .containsNone(try c.decode([String].self, forKey: .values))
        case .regex:
            self = .regex(try c.decode(String.self, forKey: .pattern))
        case .numeric:
            self = .numeric(
                expected: try c.decode(Double.self, forKey: .expected),
                // A missing tolerance means "exact", which for floating point
                // still needs a hair of slack.
                tolerance: try c.decodeIfPresent(Double.self, forKey: .tolerance) ?? 1e-6
            )
        case .validJSON:
            self = .validJSON(requiredKeys: try c.decode([String].self, forKey: .requiredKeys))
        case .letterFrequency:
            let (lower, upper) = try bounds()
            self = .letterFrequency(
                letter: try c.decode(String.self, forKey: .letter),
                min: lower,
                max: upper
            )
        case .wordCount:
            let (lower, upper) = try bounds()
            self = .wordCount(min: lower, max: upper)
        case .lineCount:
            let (lower, upper) = try bounds()
            self = .lineCount(min: lower, max: upper)
        case .startsWith:
            self = .startsWith(try c.decode(String.self, forKey: .value))
        case .endsWith:
            self = .endsWith(try c.decode(String.self, forKey: .value))
        case .allOf:
            self = .allOf(try c.decode([BenchmarkCheck].self, forKey: .checks))
        }
    }

    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .exactMatch(let value):
            try c.encode(Kind.exactMatch, forKey: .kind)
            try c.encode(value, forKey: .value)
        case .containsAll(let values):
            try c.encode(Kind.containsAll, forKey: .kind)
            try c.encode(values, forKey: .values)
        case .containsAny(let values):
            try c.encode(Kind.containsAny, forKey: .kind)
            try c.encode(values, forKey: .values)
        case .containsNone(let values):
            try c.encode(Kind.containsNone, forKey: .kind)
            try c.encode(values, forKey: .values)
        case .regex(let pattern):
            try c.encode(Kind.regex, forKey: .kind)
            try c.encode(pattern, forKey: .pattern)
        case .numeric(let expected, let tolerance):
            try c.encode(Kind.numeric, forKey: .kind)
            try c.encode(expected, forKey: .expected)
            try c.encode(tolerance, forKey: .tolerance)
        case .validJSON(let keys):
            try c.encode(Kind.validJSON, forKey: .kind)
            try c.encode(keys, forKey: .requiredKeys)
        case .letterFrequency(let letter, let lower, let upper):
            try c.encode(Kind.letterFrequency, forKey: .kind)
            try c.encode(letter, forKey: .letter)
            try c.encode(lower, forKey: .min)
            try c.encodeIfPresent(upper, forKey: .max)
        case .wordCount(let lower, let upper):
            try c.encode(Kind.wordCount, forKey: .kind)
            try c.encode(lower, forKey: .min)
            try c.encodeIfPresent(upper, forKey: .max)
        case .lineCount(let lower, let upper):
            try c.encode(Kind.lineCount, forKey: .kind)
            try c.encode(lower, forKey: .min)
            try c.encodeIfPresent(upper, forKey: .max)
        case .startsWith(let value):
            try c.encode(Kind.startsWith, forKey: .kind)
            try c.encode(value, forKey: .value)
        case .endsWith(let value):
            try c.encode(Kind.endsWith, forKey: .kind)
            try c.encode(value, forKey: .value)
        case .allOf(let checks):
            try c.encode(Kind.allOf, forKey: .kind)
            try c.encode(checks, forKey: .checks)
        }
    }

    /// Grades one answer. Pure and total — never throws, never asks a model.
    func evaluate(answer: String) -> Bool {
        switch self {
        case .exactMatch(let expected):
            return BenchmarkChecker.normalize(answer) == BenchmarkChecker.normalize(expected)

        case .containsAll(let needles):
            let hay = BenchmarkChecker.normalize(answer)
            return !needles.isEmpty && needles.allSatisfy {
                hay.contains(BenchmarkChecker.normalize($0))
            }

        case .containsAny(let needles):
            let hay = BenchmarkChecker.normalize(answer)
            return needles.contains { hay.contains(BenchmarkChecker.normalize($0)) }

        case .containsNone(let needles):
            let hay = BenchmarkChecker.normalize(answer)
            // A needle that normalizes to nothing (pure punctuation) is
            // skipped: `contains("")` is always true and would fail every
            // answer for a typo in the suite.
            return !needles.contains { needle in
                let normalized = BenchmarkChecker.normalize(needle)
                return !normalized.isEmpty && hay.contains(normalized)
            }

        case .regex(let pattern):
            return BenchmarkChecker.matches(pattern: pattern, in: answer)

        case .numeric(let expected, let tolerance):
            guard let found = BenchmarkChecker.extractAnswerNumber(in: answer) else { return false }
            return abs(found - expected) <= max(tolerance, 0)

        case .validJSON(let requiredKeys):
            return BenchmarkChecker.jsonObject(in: answer).map { object in
                requiredKeys.allSatisfy { object.keys.contains($0) }
            } ?? false

        case .letterFrequency(let letter, let lower, let upper):
            guard !letter.isEmpty else { return false }
            return Self.within(
                BenchmarkChecker.occurrences(of: letter, in: answer),
                lower, upper
            )

        case .wordCount(let lower, let upper):
            return Self.within(BenchmarkChecker.wordCount(in: answer), lower, upper)

        case .lineCount(let lower, let upper):
            return Self.within(BenchmarkChecker.nonEmptyLineCount(in: answer), lower, upper)

        case .startsWith(let prefix):
            let normalized = BenchmarkChecker.normalize(prefix)
            return !normalized.isEmpty
                && BenchmarkChecker.normalize(answer).hasPrefix(normalized)

        case .endsWith(let suffix):
            let normalized = BenchmarkChecker.normalize(suffix)
            return !normalized.isEmpty
                && BenchmarkChecker.normalize(answer).hasSuffix(normalized)

        case .allOf(let checks):
            return !checks.isEmpty && checks.allSatisfy { $0.evaluate(answer: answer) }
        }
    }

    /// Inclusive bounds check. `nil` upper bound means unbounded above.
    private static func within(_ count: Int, _ lower: Int, _ upper: Int?) -> Bool {
        guard count >= lower else { return false }
        if let upper { return count <= upper }
        return true
    }
}

// MARK: - Checker primitives

/// The normalization and extraction helpers behind `BenchmarkCheck`. Exposed
/// (rather than hidden inside the enum) so unit tests can pin their behaviour
/// directly — the checkers are the part of the benchmark that must never
/// silently change meaning.
nonisolated enum BenchmarkChecker {

    /// Trim, lowercase, strip punctuation, collapse whitespace. Everything a
    /// small model does to a one-word answer — "Paris.", " paris ", "**Paris**"
    /// — normalizes to the same string.
    static func normalize(_ text: String) -> String {
        let lowered = text.lowercased()
        let stripped = lowered.unicodeScalars.map { scalar -> Character in
            if CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar) {
                return " "
            }
            return Character(scalar)
        }
        return String(stripped)
            .split(whereSeparator: { $0.isWhitespace })
            .joined(separator: " ")
    }

    // MARK: Numbers

    /// Every number in the text, in order, each with the range it occupied.
    ///
    /// The scanner is deliberately explicit rather than a regex, because the
    /// edge cases are the whole point:
    ///   • "1,200"   → 1200      (comma between digits is a thousands mark)
    ///   • "3.14"    → 3.14      (one decimal point, digits on both sides)
    ///   • "42."     → 42        (a sentence-ending period is not a decimal)
    ///   • "-5"      → -5        (minus only where a number can start, so
    ///                            "10-3" is two numbers, not 10 and -3)
    ///   • "24.0"    → 24.0      (compared with a tolerance, so it equals 24)
    ///   • "50%"     → 50        (the sign is just the next character)
    ///   • "3.14.15" → 3.14, 15  (only ONE decimal point per number; the
    ///                            second dot ends the first number and the
    ///                            digits after it scan as a second one)
    /// The Unicode minus sign U+2212 is accepted alongside ASCII "-": models
    /// emit it often enough to matter and it is never anything else.
    static func numberMatches(in text: String) -> [(value: Double, range: Range<String.Index>)] {
        var results: [(Double, Range<String.Index>)] = []
        var index = text.startIndex
        var previous: Character?

        while index < text.endIndex {
            let character = text[index]
            let isMinus = character == "-" || character == "\u{2212}"

            // A number starts at a digit, or at a minus that is followed by a
            // digit and not glued to the right of an alphanumeric.
            var starts = character.isNumber
            if isMinus, !starts {
                let next = text.index(after: index)
                let previousAllows = previous.map { !($0.isNumber || $0.isLetter) } ?? true
                starts = next < text.endIndex && text[next].isNumber && previousAllows
            }
            guard starts else {
                previous = character
                index = text.index(after: index)
                continue
            }

            let start = index
            var cursor = index
            if isMinus { cursor = text.index(after: cursor) }

            var digits = ""
            var seenDot = false
            var lastDigitIndex = cursor

            while cursor < text.endIndex {
                let scalar = text[cursor]
                if scalar.isNumber {
                    digits.append(scalar)
                    lastDigitIndex = cursor
                    cursor = text.index(after: cursor)
                    continue
                }
                // A separator only continues the number when a digit follows.
                let next = text.index(after: cursor)
                guard next < text.endIndex, text[next].isNumber else { break }
                if scalar == "," , !seenDot {
                    // Thousands separator: dropped, not recorded.
                    cursor = next
                    continue
                }
                if scalar == "." , !seenDot {
                    seenDot = true
                    digits.append(".")
                    digits.append(text[next])
                    lastDigitIndex = next
                    cursor = text.index(after: next)
                    continue
                }
                break
            }

            let end = text.index(after: lastDigitIndex)
            if let value = Double(digits), !digits.isEmpty {
                results.append((isMinus ? -value : value, start..<end))
            }
            previous = text[lastDigitIndex]
            index = end
        }
        return results.map { (value: $0.0, range: $0.1) }
    }

    /// Last number appearing anywhere in the answer. The v1 rule, kept intact
    /// and still reachable as the final fallback of
    /// `extractAnswerNumber(in:)`: models put the result at the end, after the
    /// reasoning, so the last number is the answer far more often than the
    /// first one is.
    static func lastNumber(in text: String) -> Double? {
        numberMatches(in: text).last?.value
    }

    /// First number appearing in a slice of the answer.
    static func firstNumber<S: StringProtocol>(in text: S) -> Double? {
        numberMatches(in: String(text)).first?.value
    }

    /// The number a model meant as its answer, using the ordered fallback
    /// chain lm-evaluation-harness settled on. Each step is tried in turn and
    /// the first one that yields a number wins:
    ///
    ///   (a) `####`     - the GSM8K convention. The number that FOLLOWS the
    ///                    LAST "####" that is a delimiter rather than a
    ///                    markdown heading.
    ///   (b) `\boxed{}` - the MATH convention. First number inside the LAST
    ///                    balanced `\boxed{…}` (`\fbox{…}` too).
    ///   (c) "answer is" / "answer:" / "answer =" - the number that FOLLOWS
    ///       the LAST such marker, case-insensitive.
    ///   (d) last number anywhere - the v1 rule.
    ///
    /// The order fixes the two failure modes that cost small models real
    /// marks under the plain last-number rule:
    ///   • ANSWER ECHO: "The answer is 18. Check: 3 * 6 = 18, and 18 / 6 = 3"
    ///     ends on 3, but (c) reads 18.
    ///   • STEP COUNTERS: "…so the total is 27. (Step 4 of 4)" ends on 4,
    ///     but (c) reads 27 when the model said "answer", and a suite that
    ///     asks for "#### <number>" gets (a) unconditionally.
    ///
    /// Steps (a) and (c) both require the number to sit IMMEDIATELY after the
    /// marker, because a marker followed by prose is not an answer marker at
    /// all: "#### Step 4: Final" is a heading and "the answer is the product
    /// of 47 and 23, which is 1081" is a sentence. Both fall through to (d),
    /// which reads them correctly.
    static func extractAnswerNumber(in text: String) -> Double? {
        // (a) GSM8K delimiter.
        if let value = gsm8kDelimiterNumber(in: text) { return value }
        // (b) LaTeX box.
        if let value = boxedNumber(in: text) { return value }
        // (c) Explicit answer marker.
        if let value = answerMarkerNumber(in: text) { return value }
        // (d) v1 behaviour.
        return lastNumber(in: text)
    }

    /// Number introduced by the last genuine `####` delimiter.
    ///
    /// A model writing markdown emits "#### Step 4: Final", and taking the
    /// first number after that marker returns the heading's step number
    /// instead of the answer. Only a `####` followed by optional whitespace
    /// and then immediately a number counts; occurrences are checked from the
    /// end backwards, so a heading after the real delimiter does not hide it.
    private static func gsm8kDelimiterNumber(in text: String) -> Double? {
        var search = text.startIndex..<text.endIndex
        var ends: [String.Index] = []
        while let found = text.range(of: "####", range: search) {
            ends.append(found.upperBound)
            guard found.upperBound < text.endIndex else { break }
            search = found.upperBound..<text.endIndex
        }
        for end in ends.reversed() {
            var cursor = end
            while cursor < text.endIndex, text[cursor].isWhitespace {
                cursor = text.index(after: cursor)
            }
            guard numberStarts(in: text, at: cursor) else { continue }
            if let value = firstNumber(in: text[cursor...]) { return value }
        }
        return nil
    }

    /// How far past an answer marker a number may sit and still count as the
    /// answer. Wide enough for "answer is: **$", far too narrow for a
    /// sentence.
    private static let answerMarkerWindow = 24

    /// Characters allowed to sit between an answer marker and its number.
    /// Whitespace is handled separately; these are the decorations models put
    /// around a result — markdown emphasis, a colon, an equals sign, a
    /// currency symbol.
    private static let answerMarkerGap: Set<Character> = ["$", "*", ":", "=", "`", "\""]

    /// Number introduced by the last answer marker, when one immediately
    /// follows it.
    ///
    /// "The answer is 18." reads 18. "The answer is the product of 47 and 23,
    /// which is 1081." reads NOTHING — the marker is followed by prose, and
    /// taking the first number after it would return 47. Returning nil lets
    /// the caller fall through to the last-number rule, which gets 1081.
    private static func answerMarkerNumber(in text: String) -> Double? {
        guard let after = lastAnswerMarkerEnd(in: text) else { return nil }
        var cursor = after
        var skipped = 0
        while cursor < text.endIndex, skipped < answerMarkerWindow {
            let character = text[cursor]
            guard character.isWhitespace || answerMarkerGap.contains(character) else { break }
            cursor = text.index(after: cursor)
            skipped += 1
        }
        guard skipped < answerMarkerWindow, numberStarts(in: text, at: cursor) else { return nil }
        return firstNumber(in: text[cursor...])
    }

    /// True when a number begins exactly at `index` — a digit, or a minus
    /// sign (ASCII or U+2212) directly followed by one, matching what
    /// `numberMatches(in:)` will accept there.
    private static func numberStarts(in text: String, at index: String.Index) -> Bool {
        guard index < text.endIndex else { return false }
        let character = text[index]
        if character.isNumber { return true }
        guard character == "-" || character == "\u{2212}" else { return false }
        let next = text.index(after: index)
        return next < text.endIndex && text[next].isNumber
    }

    /// Markers a model uses to announce its result. Matched
    /// case-insensitively; the LAST one in the answer wins, because a model
    /// that revises itself puts the final claim last.
    private static let answerMarkers = [
        "answer is", "answer:", "answer =", "answer of", "result is", "result:"
    ]

    /// Index just past the last answer marker, or `nil` when there is none.
    private static func lastAnswerMarkerEnd(in text: String) -> String.Index? {
        var best: String.Index?
        for marker in answerMarkers {
            guard let range = text.range(
                of: marker, options: [.caseInsensitive, .backwards]
            ) else { continue }
            if best == nil || range.upperBound > best! { best = range.upperBound }
        }
        return best
    }

    /// First number inside the LAST `\boxed{…}` / `\fbox{…}` group, with
    /// nested braces honoured so `\boxed{\frac{1}{2}}` does not stop at the
    /// first "}".
    private static func boxedNumber(in text: String) -> Double? {
        var openings: [String.Index] = []
        for command in ["\\boxed{", "\\fbox{"] {
            var search = text.startIndex..<text.endIndex
            while let found = text.range(of: command, range: search) {
                openings.append(found.upperBound)
                guard found.upperBound < text.endIndex else { break }
                search = found.upperBound..<text.endIndex
            }
        }
        guard let start = openings.max() else { return nil }

        var depth = 1
        var cursor = start
        while cursor < text.endIndex {
            let character = text[cursor]
            if character == "{" { depth += 1 }
            if character == "}" {
                depth -= 1
                if depth == 0 { return firstNumber(in: text[start..<cursor]) }
            }
            cursor = text.index(after: cursor)
        }
        // Unbalanced: take whatever followed the opening brace.
        return firstNumber(in: text[start...])
    }

    // MARK: Reasoning blocks

    /// An answer with a leading reasoning block taken off.
    struct ReasoningStrip: Equatable, Sendable {
        /// What the check grades.
        let answer: String
        /// A block was found and removed.
        let stripped: Bool
        /// The block was opened but never closed: generation stopped (token
        /// cap or timeout) while the model was still reasoning, so there is
        /// no answer to grade and the task fails.
        let unterminated: Bool
    }

    /// Removes a LEADING `<think>...</think>` block before grading.
    ///
    /// Qwen 3 and Qwen 3.5 reason inside such a block before they answer.
    /// The pinned llama.cpp renders their template as plain ChatML, without
    /// the empty think block that switches reasoning off, so a correct
    /// "Paris" arrives as "<think>...</think>\n\nParis" and an exact-match
    /// check would fail it. Quality grading only; chat shows the raw text.
    ///
    /// Three shapes are handled:
    ///   * `<think>...</think> answer` - the block is dropped;
    ///   * `... </think> answer` with no opening tag - a template that opens
    ///     the block in the prompt itself; everything up to the closing tag
    ///     is reasoning and is dropped;
    ///   * `<think>...` never closed - `unterminated`, empty answer.
    /// Tags are matched case-insensitively. A `<think>` that is not at the
    /// start (after whitespace) is part of the answer and left alone.
    static func stripLeadingReasoning(_ text: String) -> ReasoningStrip {
        let open = "<think>"
        let close = "</think>"
        let body = text.drop(while: { $0.isWhitespace })

        if body.lowercased().hasPrefix(open) {
            let afterOpen = body.dropFirst(open.count)
            guard let end = afterOpen.range(of: close, options: .caseInsensitive) else {
                return ReasoningStrip(answer: "", stripped: true, unterminated: true)
            }
            let answer = afterOpen[end.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ReasoningStrip(answer: answer, stripped: true, unterminated: false)
        }

        if let end = text.range(of: close, options: .caseInsensitive),
           text[..<end.lowerBound].range(of: open, options: .caseInsensitive) == nil {
            let answer = text[end.upperBound...]
                .trimmingCharacters(in: .whitespacesAndNewlines)
            return ReasoningStrip(answer: answer, stripped: true, unterminated: false)
        }

        return ReasoningStrip(answer: text, stripped: false, unterminated: false)
    }

    // MARK: Counting (IFEval-style checks)

    /// Case-insensitive occurrences of a needle in the RAW answer, counted
    /// with overlap allowed so "aaa" contains two "aa". Raw rather than
    /// normalized because a letter-frequency constraint is about the text the
    /// model actually wrote.
    static func occurrences(of needle: String, in text: String) -> Int {
        guard !needle.isEmpty else { return 0 }
        let hay = Array(text.lowercased())
        let pin = Array(needle.lowercased())
        guard hay.count >= pin.count else { return 0 }

        var count = 0
        for start in 0...(hay.count - pin.count) {
            var matched = true
            for offset in 0..<pin.count where hay[start + offset] != pin[offset] {
                matched = false
                break
            }
            if matched { count += 1 }
        }
        return count
    }

    /// Whitespace-split word count of the RAW answer. "Word" means "run of
    /// non-whitespace", the same definition `wc -w` uses — so "state-of-the-art"
    /// is one word and "3 + 4" is three.
    static func wordCount(in text: String) -> Int {
        text.split(whereSeparator: { $0.isWhitespace }).count
    }

    /// Lines with at least one non-whitespace character. Blank separators do
    /// not count, so a three-item list written with blank lines between the
    /// items still counts as three lines.
    static func nonEmptyLineCount(in text: String) -> Int {
        text.split(whereSeparator: { $0.isNewline })
            .filter { $0.contains { !$0.isWhitespace } }
            .count
    }

    /// Regex match anywhere in the RAW, untrimmed answer.
    ///
    /// Deliberately CASE-SENSITIVE: the instruction-following tasks include
    /// "reply in capitals" and "reply in lower case", which a case-insensitive
    /// matcher would pass for free. Patterns that do not care about case say
    /// so themselves with an inline `(?i)` or `(?i:…)` flag.
    ///
    /// An invalid pattern fails the task rather than crashing the run — a
    /// broken remote suite must not take the app down.
    static func matches(pattern: String, in text: String) -> Bool {
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: []
        ) else { return false }
        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.firstMatch(in: text, options: [], range: range) != nil
    }

    /// First balanced `{...}` block in the answer, decoded as a dictionary.
    /// Models wrap JSON in prose and code fences constantly, so the block is
    /// carved out rather than requiring the whole answer to parse.
    static func jsonObject(in text: String) -> [String: Any]? {
        guard let start = text.firstIndex(of: "{") else { return nil }

        var depth = 0
        var inString = false
        var escaped = false
        var index = start

        while index < text.endIndex {
            let character = text[index]
            if escaped {
                escaped = false
            } else if character == "\\" {
                escaped = true
            } else if character == "\"" {
                inString.toggle()
            } else if !inString {
                if character == "{" { depth += 1 }
                if character == "}" {
                    depth -= 1
                    if depth == 0 {
                        let slice = String(text[start...index])
                        guard let data = slice.data(using: .utf8) else { return nil }
                        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
                    }
                }
            }
            index = text.index(after: index)
        }
        return nil
    }
}

// MARK: - Suite document

/// Top-level shape of `benchmarks.json`, and of an imported suite.
nonisolated struct BenchmarkSuiteDocument: Codable, Sendable, Hashable {

    /// Bumped whenever the task set changes in a way that makes old scores
    /// incomparable. Stamped into `QualityResults.suiteVersion`.
    let suiteVersion: Int
    let tasks: [BenchmarkTask]

    /// Identity of the suite. `nil` in the bundled and remote
    /// `benchmarks.json` — the file predates imports, and a missing id means
    /// `builtInID`. An imported suite always carries one.
    ///
    /// A score is a mark on ONE exam. `suiteVersion` alone stopped being
    /// enough the moment users could bring their own tasks: two unrelated
    /// suites both numbered 1 would otherwise look comparable. Reports carry
    /// this id next to the version and comparisons check both.
    let id: String?

    /// Human name, shown in the picker and in reports. `nil` for the bundled
    /// suite, which is named in the UI instead.
    let name: String?

    /// Seed `Scripts/build_benchmark_suite.py` drew the task sample with.
    /// The quality stage reuses it to fix the order tasks run in (see
    /// `qualityRunOrder(categories:limit:)`), so a suite runs in the same
    /// order on every device and every run. `nil` in an imported suite,
    /// which falls back to a seed derived from its identity.
    let samplingSeed: Int?

    /// Highest `schemaVersion` this build understands.
    ///
    /// v1: exactMatch / containsAll / containsAny / regex / numeric /
    ///     validJSON.
    /// v2: adds containsNone, letterFrequency, wordCount, lineCount,
    ///     startsWith, endsWith — the IFEval-style constraint checks.
    /// v3: adds allOf, plus the optional `id`/`name` fields an imported
    ///     suite needs. Purely additive: a v1 or v2 document still decodes
    ///     unchanged, which is why `minimumSchemaVersion` stays at 1.
    ///
    /// The optional per-task `chance` and the top-level `samplingSeed` are
    /// read at every schema version and default to 0 / nil, so they did not
    /// need a bump. One caveat for whoever publishes the first remote suite
    /// that sets `chance`: a build older than the field ignores it and would
    /// grade those tasks without the guessing floor, so that suite should
    /// raise its schemaVersion (and this constant with it) to keep older
    /// builds on their bundled copy.
    ///
    /// A document is accepted when its schemaVersion falls in
    /// `minimumSchemaVersion...supportedSchemaVersion`. A HIGHER version is
    /// REJECTED outright: it was written for a newer build and may contain
    /// check kinds this one would misread or drop, and silently scoring a
    /// model on the subset we happen to understand is worse than running the
    /// bundled suite. See `BenchmarkSuiteFile.decode`.
    static let supportedSchemaVersion = 3

    /// Oldest schema still accepted. v1 suites keep working, so a build that
    /// ships a v2 `benchmarks.json` can still read a v1 file left in the
    /// cache by a previous install.
    static let minimumSchemaVersion = 1

    /// True when this build can run the document in full.
    static func canRun(schemaVersion: Int) -> Bool {
        (minimumSchemaVersion...supportedSchemaVersion).contains(schemaVersion)
    }

    let schemaVersion: Int

    /// Id a suite carries when its document does not name one — the suite
    /// that ships with the app.
    static let builtInID = "built-in"

    init(
        schemaVersion: Int = BenchmarkSuiteDocument.supportedSchemaVersion,
        suiteVersion: Int,
        tasks: [BenchmarkTask],
        id: String? = nil,
        name: String? = nil,
        samplingSeed: Int? = nil
    ) {
        self.schemaVersion = schemaVersion
        self.suiteVersion = suiteVersion
        self.tasks = tasks
        self.id = id
        self.name = name
        self.samplingSeed = samplingSeed
    }

    static let empty = BenchmarkSuiteDocument(suiteVersion: 0, tasks: [])

    /// Id used when comparing reports. Never `nil`, so a bundled-suite report
    /// written before imports existed lines up with a bundled-suite report
    /// written after.
    var identity: String { id ?? Self.builtInID }

    var isBuiltIn: Bool { identity == Self.builtInID }

    var isEmpty: Bool { tasks.isEmpty }

    /// Categories in the order they first appear in the file — the suite is
    /// written grouped, and the UI shows the groups in that order.
    var categories: [String] {
        var seen: Set<String> = []
        return tasks.compactMap { seen.insert($0.category).inserted ? $0.category : nil }
    }

    func tasks(in category: String) -> [BenchmarkTask] {
        tasks.filter { $0.category == category }
    }

    /// Tasks matching an options filter: a `nil` category set means "all".
    /// Suite order; the quality stage runs `qualityRunOrder` instead.
    func tasks(categories: Set<String>?, limit: Int?) -> [BenchmarkTask] {
        var selected = categories.map { wanted in
            tasks.filter { wanted.contains($0.category) }
        } ?? tasks
        if let limit, selected.count > limit {
            selected = Array(selected.prefix(limit))
        }
        return selected
    }

    /// Seed behind the run order: the file's `samplingSeed`, else a stable
    /// hash of the suite's identity and version (never Swift's `Hasher`,
    /// which is randomised per process).
    var runOrderSeed: UInt64 {
        if let samplingSeed { return UInt64(bitPattern: Int64(samplingSeed)) }
        return BenchmarkTaskOrder.fnv1a("\(identity)#\(suiteVersion)")
    }

    /// The tasks the quality stage runs, in the order it runs them: the
    /// category filter, then `BenchmarkTaskOrder.interleaved`, then the
    /// `limit`. Ordering BEFORE the limit means a smoke run's handful of
    /// tasks samples the categories too, instead of being five knowledge
    /// questions.
    func qualityRunOrder(categories: Set<String>?, limit: Int?) -> [BenchmarkTask] {
        let selected = tasks(categories: categories, limit: nil)
        var ordered = BenchmarkTaskOrder.interleaved(selected, seed: runOrderSeed)
        if let limit, ordered.count > limit {
            ordered = Array(ordered.prefix(limit))
        }
        return ordered
    }
}

// MARK: - Run order

/// The order the quality stage runs a suite in.
///
/// A run with a time budget can stop part-way (see
/// `BenchmarkOptions.qualityBudget`), and a suite is written grouped by
/// category. Run in file order, a cut-short run would be all knowledge and
/// no math. So the order is a deterministic, PROPORTIONAL round-robin: every
/// category's tasks are spread evenly over the whole run, and after any
/// prefix of k tasks each category has had about k * n_c / N of them (n_c
/// tasks in the category, N in total). The truncated score is then still a
/// pass rate over a fair cross-section of the suite.
///
/// Proportional rather than one-per-category-per-round on purpose: with
/// categories of 57 and 4 tasks, a plain rotation would finish the 4-task
/// categories in the first 20 tasks and make a short run's flat score
/// mostly retrieval and multilingual.
///
/// Fully deterministic. Same suite, same seed, same order on every device
/// and every run: the within-category shuffle uses SplitMix64 seeded from
/// the suite's seed and an FNV-1a hash of the category name, and the merge
/// compares exact integer fractions, never floating point.
nonisolated enum BenchmarkTaskOrder {

    /// See the type comment. Categories keep the order they first appear in
    /// `tasks`; that order also breaks ties between equal positions.
    static func interleaved(_ tasks: [BenchmarkTask], seed: UInt64) -> [BenchmarkTask] {
        var categoryOrder: [String] = []
        var byCategory: [String: [BenchmarkTask]] = [:]
        for task in tasks {
            if byCategory[task.category] == nil {
                categoryOrder.append(task.category)
                byCategory[task.category] = []
            }
            byCategory[task.category]?.append(task)
        }

        // Shuffled inside each category: a file is usually sorted by source
        // and subject, and a cut-short run should not see only the first
        // subjects of a category.
        let streams: [[BenchmarkTask]] = categoryOrder.map { category in
            var generator = SplitMix64(seed: seed ^ fnv1a(category))
            return shuffled(byCategory[category] ?? [], using: &generator)
        }

        // Task i of a category with n tasks sits at position (2i + 1) / 2n
        // of the run. Merging by that position spreads each category evenly.
        var cursors = Array(repeating: 0, count: streams.count)
        var result: [BenchmarkTask] = []
        result.reserveCapacity(tasks.count)
        while result.count < tasks.count {
            var best: Int?
            for index in streams.indices where cursors[index] < streams[index].count {
                guard let current = best else { best = index; continue }
                // a/b < c/d  <=>  a*d < c*b (all positive, no overflow at
                // suite sizes).
                let lhs = (2 * cursors[index] + 1) * streams[current].count
                let rhs = (2 * cursors[current] + 1) * streams[index].count
                if lhs < rhs { best = index }
            }
            guard let pick = best else { break }
            result.append(streams[pick][cursors[pick]])
            cursors[pick] += 1
        }
        return result
    }

    /// Fisher-Yates with an explicit generator. Not `Array.shuffled(using:)`:
    /// the standard library does not promise that its shuffle consumes the
    /// generator the same way in every release, and the order must never
    /// change under an OS update.
    static func shuffled<T>(_ items: [T], using generator: inout SplitMix64) -> [T] {
        var result = items
        guard result.count > 1 else { return result }
        for index in stride(from: result.count - 1, to: 0, by: -1) {
            let other = Int(generator.next() % UInt64(index + 1))
            result.swapAt(index, other)
        }
        return result
    }

    /// 64-bit FNV-1a over the UTF-8 bytes. Stable across processes and
    /// platforms, unlike `Hasher`.
    static func fnv1a(_ text: String) -> UInt64 {
        var hash: UInt64 = 0xcbf2_9ce4_8422_2325
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 0x0000_0100_0000_01b3
        }
        return hash
    }

    /// SplitMix64 (Steele, Lea, Flood 2014): tiny, fast and fully specified,
    /// which is all a reproducible shuffle needs.
    struct SplitMix64 {
        private var state: UInt64

        init(seed: UInt64) { state = seed }

        mutating func next() -> UInt64 {
            state = state &+ 0x9e37_79b9_7f4a_7c15
            var z = state
            z = (z ^ (z >> 30)) &* 0xbf58_476d_1ce4_e5b9
            z = (z ^ (z >> 27)) &* 0x94d0_49bb_1331_11eb
            return z ^ (z >> 31)
        }
    }
}

// MARK: - On-disk locations

/// Where the suite comes from. Mirrors `CatalogFile` — same three-tier story
/// (cached remote → bundled → nothing), same silent failure paths.
nonisolated enum BenchmarkSuiteFile {

    /// Path of `benchmarks.json` inside the GitHub repo. The bundled resource
    /// and this path must stay in sync — the remote copy is literally the
    /// same file, served raw.
    static let remoteURL = URL(
        string: "https://raw.githubusercontent.com/ananasDDA/pocket-ai-lab/main/local_ai_test/Resources/benchmarks.json"
    )

    /// Downloaded copy. Application Support, not Documents: reproducible and
    /// none of the user's business in Files.app.
    static var cacheURL: URL? {
        guard let dir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first else { return nil }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("benchmarks-cache.json")
    }

    /// Decodes a suite, or returns nil for anything this build must not run.
    ///
    /// Three ways a document is rejected, all of them ending in "use the
    /// bundled copy instead":
    ///   1. it does not parse — including a task carrying a check `kind` this
    ///      build has never heard of, which fails the whole document on
    ///      purpose (see `BenchmarkCheck`);
    ///   2. its `schemaVersion` is outside
    ///      `minimumSchemaVersion...supportedSchemaVersion` — in particular a
    ///      HIGHER version from the remote file, which is the forward-compat
    ///      guard: a newer suite meant for a newer app never runs here;
    ///   3. it has no tasks.
    static func decode(_ data: Data) -> BenchmarkSuiteDocument? {
        guard let document = try? JSONDecoder().decode(BenchmarkSuiteDocument.self, from: data),
              BenchmarkSuiteDocument.canRun(schemaVersion: document.schemaVersion),
              !document.tasks.isEmpty
        else { return nil }
        return document
    }

    /// Cached remote copy, else the bundled resource. `nil` means the quality
    /// stage cannot run at all (`BenchmarkError.suiteUnavailable`).
    ///
    /// The cache only wins while it is at least as new as the suite compiled
    /// into this build. An app update that ships more tasks must not be
    /// undone by a cache written by the previous version, and a stale file
    /// left in the repo must not roll every device back to a smaller exam.
    /// The model catalog taught us this one: a remote copy that replaced the
    /// bundled one unconditionally kept serving an entry we had removed.
    static func loadLocal() -> BenchmarkSuiteDocument? {
        let bundled = bundledURL
            .flatMap { try? Data(contentsOf: $0) }
            .flatMap(decode)
        let cached = cacheURL
            .flatMap { try? Data(contentsOf: $0) }
            .flatMap(decode)

        if let cached, isAtLeastAsNew(cached, as: bundled) { return cached }
        if cached != nil { purgeOutdatedCache() }
        return bundled
    }

    /// True when `candidate` may replace the suite this build ships. A
    /// missing bundled suite means anything is an improvement.
    static func isAtLeastAsNew(
        _ candidate: BenchmarkSuiteDocument,
        as bundled: BenchmarkSuiteDocument?
    ) -> Bool {
        guard let bundled else { return true }
        return candidate.suiteVersion >= bundled.suiteVersion
    }

    /// Drops a cache the bundled suite has overtaken, so the next refresh
    /// writes a fresh one instead of re-reading a file we just rejected.
    static func purgeOutdatedCache() {
        guard let cacheURL else { return }
        try? FileManager.default.removeItem(at: cacheURL)
    }

    /// Suite version compiled into this build, for the refresh floor.
    static var bundledSuiteVersion: Int {
        bundledURL
            .flatMap { try? Data(contentsOf: $0) }
            .flatMap(decode)?
            .suiteVersion ?? 0
    }

    static var bundledURL: URL? {
        // `Bundle.main` covers the app and app-hosted unit tests; the
        // fallback keeps the lookup working from the device test target.
        if let url = Bundle.main.url(forResource: "benchmarks", withExtension: "json") {
            return url
        }
        return Bundle.allBundles.compactMap {
            $0.url(forResource: "benchmarks", withExtension: "json")
        }.first
    }

    static func writeCache(_ data: Data) {
        guard let cacheURL else { return }
        try? data.write(to: cacheURL, options: .atomic)
    }
}

// MARK: - Snapshot

/// Lock-protected snapshot of the suite in effect. A benchmark run reads
/// through this synchronously: the network refresh is a nice-to-have, and a
/// run must never block on it.
nonisolated final class BenchmarkSuiteStorage: @unchecked Sendable {

    static let shared = BenchmarkSuiteStorage()

    private let lock = NSLock()
    private var storage: BenchmarkSuiteDocument?

    private init() {}

    /// Cached-or-bundled suite. Empty (rather than nil) when nothing could be
    /// read, so callers have one shape to handle.
    var current: BenchmarkSuiteDocument {
        lock.withLock {
            if let storage { return storage }
            let loaded = BenchmarkSuiteFile.loadLocal() ?? .empty
            storage = loaded
            return loaded
        }
    }

    func replace(with document: BenchmarkSuiteDocument) {
        guard !document.isEmpty else { return }
        lock.withLock { storage = document }
    }
}

// MARK: - Remote refresh

/// Best-effort updater for the task suite. Call `refresh()` from a `.task`
/// somewhere in the benchmark UI; a run that starts before it lands simply
/// uses the previous suite.
@MainActor
@Observable
final class RemoteBenchmarkSuite {

    static let shared = RemoteBenchmarkSuite()

    /// Mirrors `BenchmarkSuiteStorage.current` so SwiftUI redraws after a
    /// refresh lands.
    private(set) var suite: BenchmarkSuiteDocument
    private(set) var lastRefresh: Date?

    @ObservationIgnored
    private var isRefreshing = false

    private init() {
        suite = BenchmarkSuiteStorage.shared.current
    }

    /// Never throws and never surfaces an error: a stale suite is strictly
    /// better than no suite. Safe to call on every appearance.
    func refresh() async {
        // A refresh during a run would put the radio and a JSON decode in the
        // middle of a throughput measurement. Skipping one cycle costs
        // nothing; a shifted tok/s number that still looks publishable does.
        guard !BenchmarkRunGate.shared.isRunning else { return }
        guard !isRefreshing, let url = BenchmarkSuiteFile.remoteURL else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.cachePolicy = .reloadIgnoringLocalCacheData

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let fetched = BenchmarkSuiteFile.decode(data),
              // Same floor as `loadLocal`: the published file is allowed to
              // fix and extend the suite this build ships, never to shrink
              // it back to an older exam.
              fetched.suiteVersion >= BenchmarkSuiteFile.bundledSuiteVersion
        else { return }

        BenchmarkSuiteFile.writeCache(data)
        BenchmarkSuiteStorage.shared.replace(with: fetched)
        suite = fetched
        lastRefresh = Date()
    }
}
