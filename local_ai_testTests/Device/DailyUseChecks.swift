//
//  DailyUseChecks.swift
//
//  The automatic checks behind `DailyUseDeviceTests`, kept pure so they can
//  be pinned by unit tests without a device (`DailyUseChecksTests`).
//
//  These checks are deliberately shallow. They catch the obvious failures
//  (forgot the user's name, wrote nine bullets, invalid JSON, wrong total) so
//  a reader of the transcripts can sort runs quickly; judging the quality of
//  an answer is the grader's job, not theirs.
//
//  Where the benchmark already has a primitive (normalisation, number
//  scanning, JSON carving, reasoning-block stripping) it is reused from
//  `BenchmarkChecker` rather than reimplemented, so the two can never drift.
//

import Foundation
@testable import local_ai_test

enum DailyUseChecks {

    // MARK: - Reasoning blocks

    /// The text the checks grade: a leading `<think>...</think>` block is
    /// removed first (Qwen 3.5 reasons inside one). The transcript keeps the
    /// raw answer; only the checks look at this.
    static func gradedText(_ answer: String) -> String {
        BenchmarkChecker.stripLeadingReasoning(answer).answer
    }

    /// True when the answer opens with a reasoning block, closed or not.
    static func startsWithReasoning(_ answer: String) -> Bool {
        BenchmarkChecker.stripLeadingReasoning(answer).stripped
    }

    // MARK: - Context

    /// Whether the first user turn reached the model intact, for a
    /// multi-turn prompt. `nil` when there is no history yet or the engine
    /// reported neither its rendered prompt nor a cache report.
    ///
    /// The rendered prompt alone is not enough: the llama.cpp text engine
    /// renders the whole conversation and then cuts tokens from the front
    /// to fit its window, so the text still contains a turn the model never
    /// saw. The engine's cache report says what it actually did with the
    /// first user turn: kept, dropped with older exchanges, cut inside, or
    /// lost to a last-resort front token trim.
    static func firstTurnKept(
        turns: [ChatTurn],
        renderedText: String?,
        cache: PromptCacheReport?
    ) -> Bool? {
        // History means a user turn before the one being answered; a
        // system turn alone is not history.
        guard turns.filter({ $0.role == .user }).count > 1,
              let first = turns.first(where: { $0.role == .user }) else { return nil }
        if let cache, !cache.keptFirstUserTurn { return false }
        guard let renderedText else { return cache.map { $0.keptFirstUserTurn } }
        return renderedText.contains(String(first.content.prefix(48)))
    }

    // MARK: - Fact recall

    /// Which of `facts` appear in `answer`, case- and punctuation-insensitive
    /// ("LISBON!" matches "Lisbon"). The score is the number found.
    static func recalledFacts(in answer: String, facts: [String]) -> (score: Int, found: [String: Bool]) {
        let hay = " " + BenchmarkChecker.normalize(gradedText(answer)) + " "
        var found: [String: Bool] = [:]
        for fact in facts {
            let needle = BenchmarkChecker.normalize(fact)
            // Padded with spaces so "Ana" does not match inside "banana".
            found[fact] = !needle.isEmpty && hay.contains(" " + needle + " ")
        }
        return (found.values.filter { $0 }.count, found)
    }

    // MARK: - Bullets

    /// Lines that start like a list item: `-`, `*`, `+`, a bullet glyph, or
    /// a number followed by `.` or `)` ("1." / "2)"), after optional
    /// indentation. Headings, bold titles and prose lines do not count.
    static func bulletCount(in text: String) -> Int {
        let pattern = #"^\s*(?:[-*+•●▪‣–]|\d{1,2}[.)])\s+\S"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return 0 }
        return gradedText(text)
            .split(whereSeparator: { $0.isNewline })
            .filter { line in
                let s = String(line)
                return regex.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)) != nil
            }
            .count
    }

    // MARK: - Words

    /// Whitespace-separated words, the same rule `wc -w` and the benchmark
    /// use.
    static func wordCount(in text: String) -> Int {
        BenchmarkChecker.wordCount(in: gradedText(text))
    }

    // MARK: - JSON

    struct JSONKeyCheck: Equatable {
        /// A `{...}` block was found and decoded as an object.
        let parsed: Bool
        /// Required keys that are absent (exact, case-sensitive names).
        let missing: [String]
        /// Each present key's value, flattened to a lowercased string, so
        /// value checks do not have to care whether it is a string, an array
        /// or a number.
        let values: [String: String]

        var passed: Bool { parsed && missing.isEmpty }
    }

    /// Finds the first JSON object in the answer (code fences and prose
    /// around it are tolerated) and checks it carries every required key.
    static func jsonKeys(in answer: String, required: [String]) -> JSONKeyCheck {
        guard let object = BenchmarkChecker.jsonObject(in: gradedText(answer)) else {
            return JSONKeyCheck(parsed: false, missing: required, values: [:])
        }
        var values: [String: String] = [:]
        for (key, value) in object {
            values[key] = flatten(value).lowercased()
        }
        return JSONKeyCheck(
            parsed: true,
            missing: required.filter { object[$0] == nil },
            values: values
        )
    }

    private static func flatten(_ value: Any) -> String {
        if let string = value as? String { return string }
        if let array = value as? [Any] { return array.map(flatten).joined(separator: ", ") }
        if let dict = value as? [String: Any] {
            return dict.keys.sorted().map { "\($0): \(flatten(dict[$0]!))" }.joined(separator: ", ")
        }
        return "\(value)"
    }

    // MARK: - Numbers

    /// Every number in the answer, in order (`1,200` -> 1200, `$24.78` -> 24.78).
    static func numbers(in text: String) -> [Double] {
        BenchmarkChecker.numberMatches(in: gradedText(text)).map(\.value)
    }

    /// The number the model meant as its final answer (the benchmark's
    /// `####` / boxed / "answer is" / last-number chain).
    static func finalNumber(in text: String) -> Double? {
        BenchmarkChecker.extractAnswerNumber(in: gradedText(text))
    }

    /// True when `expected` appears anywhere in the answer.
    static func containsNumber(_ expected: Double, in text: String, tolerance: Double = 0.005) -> Bool {
        numbers(in: text).contains { abs($0 - expected) <= tolerance }
    }

    // MARK: - Clock times

    /// Every clock time in the answer, as minutes after midnight.
    ///
    ///   "11:00 AM", "11 am", "11 a.m." -> 660
    ///   "4:00 PM", "4pm"               -> 960
    ///   "16:00", "11:00" (no am/pm)    -> 960, 660 (24-hour reading)
    ///   "12 AM" -> 0, "12 PM" -> 720
    ///
    /// Bare numbers ("5 hours") are not times.
    static func clockTimes(in text: String) -> [Int] {
        let graded = gradedText(text)
        let ns = graded as NSString
        let full = NSRange(location: 0, length: ns.length)

        var found: [(location: Int, minutes: Int)] = []
        var claimed: [NSRange] = []

        // 12-hour times first, so "11:00 AM" is not also read as 24-hour 11:00.
        let meridiem = try! NSRegularExpression(
            pattern: #"(?i)\b(\d{1,2})(?::(\d{2}))?\s*([ap])\.?\s?m\b\.?"#
        )
        for match in meridiem.matches(in: graded, range: full) {
            guard let hour = Int(ns.substring(with: match.range(at: 1))), (1...12).contains(hour) else { continue }
            let minute = match.range(at: 2).location == NSNotFound
                ? 0 : Int(ns.substring(with: match.range(at: 2))) ?? 0
            guard minute < 60 else { continue }
            let isPM = ns.substring(with: match.range(at: 3)).lowercased() == "p"
            let hour24 = (hour % 12) + (isPM ? 12 : 0)
            found.append((match.range.location, hour24 * 60 + minute))
            claimed.append(match.range)
        }

        let twentyFour = try! NSRegularExpression(pattern: #"\b(\d{1,2}):(\d{2})\b"#)
        for match in twentyFour.matches(in: graded, range: full) {
            if claimed.contains(where: { NSIntersectionRange($0, match.range).length > 0 }) { continue }
            guard let hour = Int(ns.substring(with: match.range(at: 1))),
                  let minute = Int(ns.substring(with: match.range(at: 2))),
                  hour < 24, minute < 60 else { continue }
            found.append((match.range.location, hour * 60 + minute))
        }

        return found.sorted { $0.location < $1.location }.map(\.minutes)
    }

    /// Passes when the answer names `expected` and every other time it
    /// mentions is one of `allowedOthers` (typically the times given in the
    /// question). "11 AM" passes; "11 PM" or "9 AM" anywhere fails it.
    static func clockAnswerMatches(_ text: String, expected: Int, allowedOthers: Set<Int>) -> Bool {
        let times = clockTimes(in: text)
        return times.contains(expected)
            && times.allSatisfy { $0 == expected || allowedOthers.contains($0) }
    }

    // MARK: - Text

    /// Case- and punctuation-insensitive containment ("Maple Street
    /// Grocery." contains "maple street").
    static func mentions(_ needle: String, in text: String) -> Bool {
        let n = BenchmarkChecker.normalize(needle)
        return !n.isEmpty && BenchmarkChecker.normalize(gradedText(text)).contains(n)
    }

    /// Which of `needles` the answer mentions (same rule as `mentions`).
    static func mentioned(_ needles: [String], in text: String) -> [String] {
        let hay = " " + BenchmarkChecker.normalize(gradedText(text)) + " "
        return needles.filter { needle in
            let n = BenchmarkChecker.normalize(needle)
            return !n.isEmpty && hay.contains(" " + n + " ")
        }
    }

    private static func containsPhrase(_ phrases: [String], in text: String) -> String? {
        let hay = " " + BenchmarkChecker.normalize(gradedText(text)) + " "
        return phrases.first { hay.contains(" " + BenchmarkChecker.normalize($0) + " ") }
    }

    // MARK: - Language

    /// What the answer is written in, as an ISO 639-1 code, or nil when the
    /// text gives no clear signal. Script decides first (Japanese needs
    /// kana, Cyrillic is read as Russian, the only Cyrillic language the
    /// suite asks for); Latin-script text is scored on common function
    /// words plus a few letters only one candidate language uses. Only the
    /// languages the suite asks for are candidates: en, es, de, ja, ru (and
    /// zh, so a Chinese reply to a Japanese prompt is not read as Japanese).
    struct LanguageGuess: Equatable {
        let language: String?
        let detail: String
    }

    private static let functionWords: [String: Set<String>] = [
        // Words that are common in one candidate and absent or rare in the
        // others. Shared words ("es", "an", "so", "was", "will", "in", "no",
        // "a", "die") are left out on purpose.
        "en": ["the", "and", "of", "to", "is", "are", "you", "your", "that", "this", "with",
               "for", "it", "have", "has", "be", "i", "my", "at", "on", "what", "which", "can",
               "would", "there", "they", "we", "from", "or", "as", "but", "not", "if", "do",
               "does", "how", "when", "about", "more", "some", "here", "these", "very", "also",
               "every", "before", "because", "train", "morning"],
        "es": ["el", "la", "los", "las", "de", "del", "que", "y", "un", "una", "por", "con",
               "para", "se", "al", "lo", "como", "más", "pero", "su", "sus", "muy", "también",
               "está", "son", "este", "esta", "estos", "cada", "porque", "cuando", "puedes",
               "puede", "tiene", "hay", "yo", "mi", "ya", "sin", "sobre", "entre", "hacer",
               "libro", "libros", "mañana", "tren", "antes"],
        "de": ["der", "die", "das", "und", "ist", "nicht", "ich", "ein", "eine", "einen", "zu",
               "den", "dem", "mit", "auf", "für", "sich", "des", "von", "im", "wir", "sie",
               "auch", "aber", "wenn", "nach", "bei", "noch", "wird", "um", "zum", "zur", "oder",
               "sind", "kann", "können", "haben", "hat", "du", "dein", "deine", "mehr", "sehr",
               "jeden", "jede", "nur", "wie", "was", "an", "zug", "morgen", "bevor"]
    ]

    static func detectLanguage(_ text: String) -> LanguageGuess {
        let graded = gradedText(text)
        var latin = 0, cyrillic = 0, kana = 0, han = 0, other = 0
        for scalar in graded.unicodeScalars where CharacterSet.letters.contains(scalar) {
            switch scalar.value {
            case 0x41...0x5A, 0x61...0x7A, 0xC0...0x24F: latin += 1
            case 0x400...0x4FF: cyrillic += 1
            case 0x3040...0x30FF, 0x31F0...0x31FF, 0xFF66...0xFF9F: kana += 1
            case 0x3400...0x4DBF, 0x4E00...0x9FFF: han += 1
            default: other += 1
            }
        }
        let letters = latin + cyrillic + kana + han + other
        guard letters > 0 else { return LanguageGuess(language: nil, detail: "no letters") }
        func share(_ n: Int) -> Int { n * 100 / letters }
        let scripts = "latin \(share(latin))%, cyrillic \(share(cyrillic))%, "
            + "kana \(share(kana))%, han \(share(han))%"

        if Double(kana + han) / Double(letters) >= 0.3 {
            // Japanese prose always carries kana; Chinese has none.
            let language = kana > 0 && Double(kana) / Double(kana + han) >= 0.1 ? "ja" : "zh"
            return LanguageGuess(language: language, detail: scripts)
        }
        if Double(cyrillic) / Double(letters) >= 0.5 {
            return LanguageGuess(language: "ru", detail: scripts)
        }
        guard Double(latin) / Double(letters) >= 0.5 else {
            return LanguageGuess(language: nil, detail: scripts)
        }

        let words = graded.lowercased()
            .split(whereSeparator: { !$0.isLetter })
            .map(String.init)
        var scores: [String: Int] = [:]
        for (language, list) in functionWords {
            scores[language] = words.filter { list.contains($0) }.count
        }
        // Letters only one of the three Latin candidates uses.
        let lower = graded.lowercased()
        scores["es", default: 0] += lower.filter { "ñ¿¡".contains($0) }.count * 2
        scores["de", default: 0] += lower.filter { "äöüß".contains($0) }.count * 2

        let ranked = scores.sorted { $0.value != $1.value ? $0.value > $1.value : $0.key < $1.key }
        let detail = scripts + "; words " + ranked.map { "\($0.key) \($0.value)" }.joined(separator: ", ")
        guard let best = ranked.first, best.value >= 2,
              ranked.count < 2 || best.value > ranked[1].value else {
            return LanguageGuess(language: nil, detail: detail)
        }
        return LanguageGuess(language: best.key, detail: detail)
    }

    // MARK: - Formatting

    struct MarkdownTable: Equatable {
        let headers: [String]
        let rows: [[String]]
    }

    private static func tableCells(_ line: String) -> [String] {
        var cells = line.trimmingCharacters(in: .whitespaces)
            .split(separator: "|", omittingEmptySubsequences: false)
            .map { $0.trimmingCharacters(in: .whitespaces) }
        if cells.first == "" { cells.removeFirst() }
        if cells.last == "" { cells.removeLast() }
        return cells
    }

    /// The first markdown table in the answer: a header row, a `---`
    /// separator row, then every following row that still has pipes.
    static func markdownTable(in text: String) -> MarkdownTable? {
        let lines = gradedText(text).components(separatedBy: .newlines)
        let separator = try! NSRegularExpression(pattern: #"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$"#)
        for i in 0..<max(0, lines.count - 1) {
            let header = lines[i]
            let next = lines[i + 1]
            guard header.filter({ $0 == "|" }).count >= 1,
                  separator.firstMatch(in: next, range: NSRange(next.startIndex..., in: next)) != nil
            else { continue }
            var rows: [[String]] = []
            for line in lines[(i + 2)...] {
                guard line.contains("|") else { break }
                rows.append(tableCells(line))
            }
            return MarkdownTable(headers: tableCells(header), rows: rows)
        }
        return nil
    }

    /// Required columns the table's header row lacks (case, bold and
    /// punctuation ignored; "Cost (EUR)" counts as "Cost").
    static func missingColumns(_ required: [String], in table: MarkdownTable) -> [String] {
        let headers = table.headers.map { " " + BenchmarkChecker.normalize($0) + " " }
        return required.filter { column in
            let needle = " " + BenchmarkChecker.normalize(column) + " "
            return !headers.contains { $0.contains(needle) }
        }
    }

    /// Sentences in the answer: runs of text ended by `.`, `!` or `?`
    /// followed by whitespace, or by the end of the text. Line breaks
    /// between non-empty lines count as a break too.
    static func sentenceCount(in text: String) -> Int {
        let graded = gradedText(text)
        let regex = try! NSRegularExpression(pattern: #"[.!?]+["')\]]*(?=\s|$)"#)
        var count = 0
        for line in graded.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { continue }
            let ns = trimmed as NSString
            var pieces = 0
            var start = 0
            for match in regex.matches(in: trimmed, range: NSRange(location: 0, length: ns.length)) {
                let end = match.range.location + match.range.length
                if !ns.substring(with: NSRange(location: start, length: end - start))
                    .trimmingCharacters(in: .whitespaces).isEmpty {
                    pieces += 1
                }
                start = end
            }
            if !ns.substring(from: start).trimmingCharacters(in: .whitespaces)
                .filter({ $0.isLetter || $0.isNumber }).isEmpty {
                pieces += 1
            }
            count += pieces
        }
        return count
    }

    /// The numbers of numbered-list lines, in order: "1. ...", "2) ...",
    /// "**Step 3:** ...", "### 4. ...".
    static func numberedSteps(in text: String) -> [Int] {
        let regex = try! NSRegularExpression(
            pattern: #"(?i)^\s*(?:#{1,6}\s*)?(?:\*\*|__)?\s*(?:step\s+)?(\d{1,2})\s*(?:\*\*|__)?\s*[.):]"#
        )
        return gradedText(text).components(separatedBy: .newlines).compactMap { line in
            let ns = line as NSString
            guard let match = regex.firstMatch(in: line, range: NSRange(location: 0, length: ns.length))
            else { return nil }
            return Int(ns.substring(with: match.range(at: 1)))
        }
    }

    /// At least `minimum` numbered steps, counting 1, 2, 3, ... with no gap
    /// or restart.
    static func stepsAreSequential(_ steps: [Int], minimum: Int = 3) -> Bool {
        steps.count >= minimum && steps == Array(1...steps.count)
    }

    // MARK: - Money

    enum Currency {
        case euro, dollar

        var markers: String {
            switch self {
            case .euro: return #"(?:€|\bEUR\b|\beuros?\b)"#
            case .dollar: return #"(?:\$|\bUSD\b|\bdollars?\b)"#
            }
        }
    }

    /// Every amount written with the currency's sign or name, either side
    /// of the number ("€25", "25 euros", "EUR 25"), both ends of a range
    /// ("€20-30", "20 to 30 euros") included.
    static func amounts(in text: String, currency: Currency) -> [Double] {
        let graded = gradedText(text)
        let ns = graded as NSString
        let number = #"(\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+(?:[.,]\d{1,2})?)"#
        let range = #"(?:\s?(?:-|–|to)\s?"# + currency.markers + #"?\s?"# + number + ")?"
        let prefix = try! NSRegularExpression(pattern: "(?i)" + currency.markers + #"\s?"# + number + range)
        let suffix = try! NSRegularExpression(
            pattern: "(?i)" + number + #"(?:\s?(?:-|–|to)\s?"# + number + #")?\s?"# + currency.markers
        )
        func value(_ raw: String) -> Double? {
            if raw.contains(",") && !raw.contains(".") {
                let parts = raw.split(separator: ",")
                // "1,450" is thousands; "12,50" is a decimal comma.
                if parts.count == 2, parts[1].count <= 2 { return Double(raw.replacingOccurrences(of: ",", with: ".")) }
            }
            return Double(raw.replacingOccurrences(of: ",", with: ""))
        }
        var found: [(Int, Double)] = []
        var claimed: [NSRange] = []
        for regex in [prefix, suffix] {
            for match in regex.matches(in: graded, range: NSRange(location: 0, length: ns.length)) {
                for group in 1..<match.numberOfRanges {
                    let r = match.range(at: group)
                    guard r.location != NSNotFound,
                          !claimed.contains(where: { NSIntersectionRange($0, r).length > 0 }),
                          let v = value(ns.substring(with: r)) else { continue }
                    claimed.append(r)
                    found.append((r.location, v))
                }
            }
        }
        return found.sorted { $0.0 < $1.0 }.map(\.1)
    }

    // MARK: - Diet

    /// Hyphens normalise to spaces, so "plant-based" also matches "plant based".
    static let vegetarianWords = ["vegetarian", "veggie", "vegan", "plant-based", "meat-free", "meatless"]

    static let meatAndFishWords = ["meat", "chicken", "beef", "pork", "ham", "bacon", "sausage",
                                   "sausages", "steak", "lamb", "veal", "francesinha", "bifana",
                                   "bifanas", "sardine", "sardines", "bacalhau", "cod", "fish",
                                   "seafood", "octopus", "tuna", "shrimp", "prawns", "chorizo",
                                   "chouriço", "alheira", "tripe", "tripas", "salami", "prosciutto"]

    /// Meat or fish dishes the answer suggests. A mention right after a
    /// qualifier that rules the meat out ("vegetarian francesinha", "no
    /// meat", "skip the bacalhau", "instead of fish") is not a suggestion.
    static func meatSuggestions(in text: String) -> [String] {
        let words = BenchmarkChecker.normalize(gradedText(text)).split(separator: " ").map(String.init)
        let meat = Set(meatAndFishWords.map { BenchmarkChecker.normalize($0) })
        let qualifiers: Set<String> = ["vegetarian", "veggie", "vegan", "no", "without", "avoid",
                                       "skip", "instead", "free", "not", "except", "replace",
                                       "replaced", "swap", "plant", "meatless", "non"]
        var found: [String] = []
        for (i, word) in words.enumerated() where meat.contains(word) {
            let before = words[max(0, i - 3)..<i]
            let after = words[(i + 1)..<min(words.count, i + 2)]
            // "meat-free", "fish-free" normalise to "meat free".
            if before.contains(where: qualifiers.contains) || after.contains("free") { continue }
            found.append(word)
        }
        return found
    }

    // MARK: - Grounding

    private static let notStatedPhrases = [
        "not mention", "no mention", "doesn't mention", "does not mention", "isn't mentioned",
        "is not mentioned", "not mentioned", "not specified", "doesn't specify", "does not specify",
        "not stated", "doesn't state", "does not state", "doesn't say", "does not say",
        "no information", "not covered", "doesn't cover", "does not cover", "not addressed",
        "doesn't address", "does not address", "not provided", "not in the lease",
        "not in the text", "not in the document", "doesn't contain", "does not contain",
        "no reference", "nothing about", "no details", "silent on", "doesn't include",
        "does not include", "not included in the lease", "doesn't provide", "does not provide",
        "not explicitly", "no clause", "not say", "not specify", "not state", "not contain",
        "not include"
    ]

    /// The answer says the document does not answer the question.
    static func saysNotStated(_ text: String) -> Bool {
        containsPhrase(notStatedPhrases, in: text) != nil
    }

    private static let unknownPhrases = [
        "not familiar", "don't have information", "do not have information",
        "don't have any information", "do not have any information", "couldn't find",
        "could not find", "can't find", "cannot find", "not aware", "no information",
        "doesn't appear to exist", "does not appear to exist", "unable to find", "i don't know",
        "i do not know", "no record", "not able to find", "may not exist", "might not exist",
        "does not exist", "doesn't exist", "fictional", "not a real", "unable to verify",
        "cannot verify", "can't verify", "not widely known", "no widely known",
        "no reliable information", "not recognize", "don't recognize", "do not recognize",
        "no knowledge", "not find any", "no publicly available", "not publicly available",
        "not something i can confirm", "can't confirm", "cannot confirm", "unable to confirm",
        "no such", "not aware of any", "doesn't seem to exist", "does not seem to exist",
        "not in my training", "beyond my knowledge", "my knowledge cutoff", "my training data"
    ]

    /// The answer admits it does not know the thing asked about.
    static func admitsNotKnowing(_ text: String) -> Bool {
        containsPhrase(unknownPhrases, in: text) != nil
    }

    // MARK: - Dates and days

    private static let monthNames = ["january", "february", "march", "april", "may", "june", "july",
                                     "august", "september", "october", "november", "december"]

    /// True when the answer names the given day of the month in any common
    /// form: "April 1, 2027", "1 April", "April 1st", "1st of April",
    /// "2027-04-01", "01/04/2027", "04/01/2027".
    static func mentionsDate(_ text: String, month: Int, day: Int) -> Bool {
        guard (1...12).contains(month) else { return false }
        let hay = BenchmarkChecker.normalize(gradedText(text))
        let name = monthNames[month - 1]
        let names = "(?:\(name)|\(name.prefix(3)))"
        let d = "0?\(day)(?:st|nd|rd|th)?"
        let m = "0?\(month)"
        let patterns = [
            "\\b\(names)\\s+\(d)\\b",
            "\\b\(d)\\s+(?:of\\s+)?\(names)\\b",
            "\\b\\d{4}\\s+\(m)\\s+0?\(day)\\b",
            "\\b0?\(day)\\s+\(m)\\s+\\d{2,4}\\b",
            "\\b\(m)\\s+0?\(day)\\s+\\d{2,4}\\b"
        ]
        return patterns.contains { pattern in
            hay.range(of: pattern, options: .regularExpression) != nil
        }
    }

    private static let dayShiftPhrases = ["previous day", "day before", "yesterday", "next day",
                                          "following day", "tomorrow", "prior day", "night before"]

    /// The answer moves the event to another calendar day ("11:00 AM
    /// (previous day)"), unless it says "same day".
    static func mentionsDayShift(_ text: String) -> Bool {
        containsPhrase(dayShiftPhrases, in: text) != nil
            && containsPhrase(["same day"], in: text) == nil
    }

    // MARK: - Code

    /// The bodies of fenced code blocks, or the whole answer when there
    /// are none.
    static func codeBlocks(in text: String) -> [String] {
        let graded = gradedText(text)
        let regex = try! NSRegularExpression(pattern: "```[^\\n]*\\n([\\s\\S]*?)```")
        let ns = graded as NSString
        let blocks = regex.matches(in: graded, range: NSRange(location: 0, length: ns.length))
            .map { ns.substring(with: $0.range(at: 1)) }
        return blocks.isEmpty ? [graded] : blocks
    }

    struct CodeCheck: Equatable {
        let passed: Bool
        let reason: String
    }

    /// A static look at the order-preserving dedupe function, since the
    /// code is not executed. Fails `list(set(x))` (loses order) and
    /// `str(item)` keys (merges 1 and "1"); passes an insertion-ordered
    /// structure (`dict.fromkeys`, `OrderedDict`) or a seen-set loop that
    /// appends to a list.
    static func dedupeKeepsOrder(_ answer: String) -> CodeCheck {
        let code = codeBlocks(in: answer).joined(separator: "\n")
        func has(_ pattern: String) -> Bool {
            code.range(of: pattern, options: .regularExpression) != nil
        }
        guard has(#"\bdef\s+\w+\s*\("#) else { return CodeCheck(passed: false, reason: "no function") }
        if has(#"(?:list|sorted)\s*\(\s*set\s*\("#) {
            return CodeCheck(passed: false, reason: "list(set(...)) / sorted(set(...)) does not keep the original order")
        }
        if has(#"str\s*\(\s*\w+\s*\)\s*(?:not\s+)?in\b"#) || has(#"\.add\s*\(\s*str\s*\("#)
            || has(#"\[\s*str\s*\(\s*\w+\s*\)\s*\]"#) {
            return CodeCheck(passed: false, reason: "str(item) keys treat 1 and \"1\" as duplicates")
        }
        if has(#"dict\s*\.\s*fromkeys\s*\("#) || has(#"\bOrderedDict\b"#) {
            return CodeCheck(passed: true, reason: "insertion-ordered dict")
        }
        if has(#"\.append\s*\("#) && (has(#"\bnot\s+in\b"#) || has(#"\bin\s+seen\b"#)) {
            return CodeCheck(passed: true, reason: "seen-check loop appending to a list")
        }
        if has(#"\bin\s+seen\b"#) && has(#"seen\s*\.\s*add\s*\("#) {
            return CodeCheck(passed: true, reason: "comprehension with a seen set")
        }
        return CodeCheck(passed: false, reason: "no order-preserving pattern found")
    }

    // MARK: - Failures

    /// Coarse kind for an engine error message, for the step's `outcome`.
    static func errorKind(_ message: String) -> String {
        let lower = message.lowercased()
        // Apple Intelligence: the framework's safety filter and its
        // supported-language list, which a small local model does not have.
        if lower.contains("guardrail") || lower.contains("unsafe") { return "guardrail" }
        if lower.contains("unsupported language") || lower.contains("unsupportedlanguage")
            || lower.contains("locale") { return "unsupported_language" }
        if lower.contains("context") { return "context_full" }
        if lower.contains("memory") { return "out_of_memory" }
        if lower.contains("cancel") { return "cancelled" }
        if lower.contains("not loaded") { return "model_not_loaded" }
        return "engine_error"
    }
}
