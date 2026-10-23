//
//  MiniYAML.swift
//  local_ai_test
//
//  A deliberately small YAML reader, written for exactly one job: reading a
//  promptfoo config off a phone.
//
//  There is no YAML parser in this project and adding a full one would drag
//  in a dependency (and a spec) far larger than the feature that needs it.
//  So this covers the subset a `promptfooconfig.yaml` actually uses —
//  mappings, block sequences, quoted and block scalars, flow sequences of
//  scalars, comments — and REFUSES everything else by name.
//
//  Refusing is the whole design. A benchmark suite decides what a model's
//  score means, so a parser that quietly mis-reads an anchor, a flow mapping
//  or a tab-indented block would hand the user a number that looks right and
//  is not. Every construct outside the subset raises a `YAMLError` carrying
//  the line number, and the import stops there.
//
//  Not supported, on purpose (each throws): tabs in indentation, anchors and
//  aliases (`&`/`*`), tags (`!!str`), merge keys (`<<`), flow mappings
//  (`{a: 1}`), nested flow collections, more than one document, explicit
//  block-scalar indent indicators (`|2`), duplicate keys in one mapping.
//
//  Scalars are NOT type-inferred: `true`, `42` and `hello` all arrive as
//  strings, and only `~`/`null` becomes `.null`. Callers know what shape
//  they want and convert explicitly, which keeps YAML's famous "Norway
//  problem" out of a scoring path.
//

import Foundation

// MARK: - Value

/// A parsed YAML node. Scalars stay strings — see the file header.
nonisolated indirect enum YAMLValue: Equatable, Sendable {
    case scalar(String)
    case null
    case sequence([YAMLValue])
    case mapping([YAMLPair])
}

/// One mapping entry. An ordered array rather than a dictionary because the
/// import report quotes tests in file order, and a dictionary would shuffle
/// them.
nonisolated struct YAMLPair: Equatable, Sendable {
    let key: String
    let value: YAMLValue
}

extension YAMLValue {

    var stringValue: String? {
        if case .scalar(let text) = self { return text }
        return nil
    }

    var sequenceValue: [YAMLValue]? {
        if case .sequence(let items) = self { return items }
        return nil
    }

    var mappingValue: [YAMLPair]? {
        if case .mapping(let pairs) = self { return pairs }
        return nil
    }

    var isNull: Bool { self == .null }

    /// Value for a mapping key, or `nil` when this is not a mapping or the
    /// key is absent.
    subscript(key: String) -> YAMLValue? {
        mappingValue?.first { $0.key == key }?.value
    }

    /// A scalar as a one-element list, a sequence as itself. promptfoo lets
    /// several fields be "one thing or a list of things", and every caller
    /// here wants the list.
    var asSequence: [YAMLValue] {
        switch self {
        case .sequence(let items): return items
        case .null:                return []
        default:                   return [self]
        }
    }
}

// MARK: - Error

/// Where the parser gave up and why. The line number is 1-based so it can be
/// shown to a user staring at the same file in an editor.
nonisolated struct YAMLError: Error, LocalizedError, Equatable, Sendable {
    let line: Int
    let reason: String

    var errorDescription: String? { "YAML line \(line): \(reason)" }
}

// MARK: - Parser

nonisolated enum MiniYAML {

    /// Parses a whole document. Throws `YAMLError` for anything outside the
    /// supported subset; never returns a partially understood tree.
    static func parse(_ text: String) throws -> YAMLValue {
        var parser = Parser(text: text)
        return try parser.parseDocument()
    }

    // MARK: Lines

    fileprivate struct RawLine {
        let number: Int
        /// Leading spaces. Tabs are rejected before a line ever gets here.
        let indent: Int
        /// Verbatim text, used for block-scalar bodies where `#` is content.
        let raw: String
        /// Comment-stripped and trailing-trimmed, used for structure.
        let content: String
        let isBlank: Bool
    }

    fileprivate struct Parser {

        private var lines: [RawLine] = []
        private var index = 0

        init(text: String) {
            let split = text
                .replacingOccurrences(of: "\r\n", with: "\n")
                .replacingOccurrences(of: "\r", with: "\n")
                .components(separatedBy: "\n")
            lines = split.enumerated().map { offset, raw in
                let indent = raw.prefix { $0 == " " }.count
                let content = Self.stripComment(raw)
                    .trimmingCharacters(in: .whitespaces)
                return RawLine(
                    number: offset + 1,
                    indent: indent,
                    raw: raw,
                    content: content,
                    isBlank: content.isEmpty
                )
            }
        }

        // MARK: Document

        mutating func parseDocument() throws -> YAMLValue {
            try rejectTabs()
            guard var first = peek() else { return .null }

            // A single leading "---" is idiomatic and harmless. Anything
            // beyond it means a multi-document stream, which has no sensible
            // meaning for "import this suite".
            if first.content == "---" {
                index += 1
                guard let next = peek() else { return .null }
                first = next
            }
            for line in lines where !line.isBlank {
                if line.content == "---" || line.content == "..." {
                    throw error(line, "multiple YAML documents are not supported")
                }
            }

            let value = try parseNode(indent: first.indent)
            if let extra = peek() {
                throw error(extra, "unexpected content after the document")
            }
            return value
        }

        /// Indentation with a tab in it is the single most common way a
        /// hand-edited YAML file silently changes shape, so it is refused
        /// before parsing rather than counted as some number of columns.
        private func rejectTabs() throws {
            for line in lines where !line.isBlank {
                let leading = line.raw.prefix { $0 == " " || $0 == "\t" }
                if leading.contains("\t") {
                    throw error(line, "tabs cannot be used for indentation")
                }
            }
        }

        // MARK: Cursor

        private var current: RawLine? { index < lines.count ? lines[index] : nil }

        /// Next structural line, skipping blanks and comment-only lines.
        private mutating func peek() -> RawLine? {
            while index < lines.count, lines[index].isBlank { index += 1 }
            return current
        }

        private func error(_ line: RawLine, _ reason: String) -> YAMLError {
            YAMLError(line: line.number, reason: reason)
        }

        // MARK: Nodes

        private mutating func parseNode(indent: Int) throws -> YAMLValue {
            guard let line = peek() else { return .null }
            if Self.isSequenceItem(line.content) {
                return try parseSequence(indent: indent)
            }
            return try parseMapping(indent: indent)
        }

        private mutating func parseMapping(indent: Int) throws -> YAMLValue {
            var pairs: [YAMLPair] = []
            var seen: Set<String> = []

            while let line = peek() {
                if line.indent < indent { break }
                if line.indent > indent {
                    throw error(line, "unexpected indentation")
                }
                if Self.isSequenceItem(line.content) { break }

                let (key, rest) = try splitKey(line)
                guard seen.insert(key).inserted else {
                    throw error(line, "duplicate key '\(key)'")
                }
                index += 1
                let value = try parseValue(after: rest, keyLine: line, parentIndent: indent)
                pairs.append(YAMLPair(key: key, value: value))
            }
            return .mapping(pairs)
        }

        private mutating func parseSequence(indent: Int) throws -> YAMLValue {
            var items: [YAMLValue] = []

            while let line = peek(), line.indent == indent, Self.isSequenceItem(line.content) {
                let afterDash = line.content.dropFirst()
                let gap = afterDash.prefix { $0 == " " }.count
                let rest = String(afterDash.dropFirst(gap))

                if rest.isEmpty {
                    index += 1
                    if let next = peek(), next.indent > indent {
                        items.append(try parseNode(indent: next.indent))
                    } else {
                        items.append(.null)
                    }
                } else if Self.isSequenceItem(rest) {
                    throw error(line, "nested inline sequences are not supported")
                } else if Self.startsMapping(rest) {
                    // "- type: equals" opens a mapping whose keys line up
                    // with the column the text starts at, not with the dash.
                    // Rewriting the line at that column lets the ordinary
                    // mapping parser pick up the following keys unchanged.
                    let column = line.indent + 1 + gap
                    lines[index] = RawLine(
                        number: line.number,
                        indent: column,
                        raw: line.raw,
                        content: rest,
                        isBlank: false
                    )
                    items.append(try parseNode(indent: column))
                } else {
                    index += 1
                    items.append(try parseInline(rest, line: line))
                }
            }
            return .sequence(items)
        }

        /// The value of a `key:` — either inline on the same line, or the
        /// indented block that follows it.
        private mutating func parseValue(
            after rest: String,
            keyLine: RawLine,
            parentIndent: Int
        ) throws -> YAMLValue {
            guard rest.isEmpty else {
                return try parseInline(rest, line: keyLine)
            }
            guard let next = peek() else { return .null }
            if next.indent > parentIndent {
                return try parseNode(indent: next.indent)
            }
            // A block sequence may sit at its key's own indentation:
            //     tests:
            //     - vars: {...}
            if next.indent == parentIndent, Self.isSequenceItem(next.content) {
                return try parseSequence(indent: parentIndent)
            }
            return .null
        }

        /// A value written on one line, after the key or after the dash. The
        /// cursor has already moved past `line`.
        private mutating func parseInline(_ text: String, line: RawLine) throws -> YAMLValue {
            guard let first = text.first else { return .null }
            switch first {
            case "|", ">":
                return try parseBlockScalar(header: text, line: line, parentIndent: line.indent)
            case "[":
                return try parseFlowSequence(text, line: line)
            case "{":
                throw error(line, "flow mappings ({...}) are not supported")
            case "&", "*":
                throw error(line, "anchors and aliases are not supported")
            case "!":
                throw error(line, "YAML tags are not supported")
            default:
                return try Self.parseScalar(text, line: line)
            }
        }

        // MARK: Block scalars

        private mutating func parseBlockScalar(
            header: String,
            line: RawLine,
            parentIndent: Int
        ) throws -> YAMLValue {
            let style = header.first
            let indicator = String(header.dropFirst())
            guard ["", "-", "+"].contains(indicator) else {
                throw error(line, "unsupported block scalar indicator '\(header)'")
            }

            // Body lines are read RAW: inside a block scalar a '#' is text,
            // not a comment.
            var body: [RawLine] = []
            while index < lines.count {
                let candidate = lines[index]
                let isWhitespaceOnly = candidate.raw.allSatisfy { $0 == " " }
                guard isWhitespaceOnly || candidate.indent > parentIndent else { break }
                body.append(candidate)
                index += 1
            }
            // Trailing blank lines are handed back to the outer parser: only
            // the "+" (keep) indicator cares how many there were, and every
            // other reader skips blanks anyway.
            var trailingBlanks = 0
            while let last = body.last, last.raw.allSatisfy({ $0 == " " }) {
                body.removeLast()
                index -= 1
                trailingBlanks += 1
            }
            // Indentation is set by the first line that has content: a block
            // may legally open with blank lines, which stay as newlines.
            guard let firstContent = body.first(
                where: { !$0.raw.allSatisfy { $0 == " " } }
            ) else { return .scalar("") }
            let blockIndent = firstContent.indent

            var texts: [String] = []
            for entry in body {
                if entry.raw.allSatisfy({ $0 == " " }) {
                    texts.append("")
                    continue
                }
                guard entry.indent >= blockIndent else {
                    throw error(entry, "inconsistent indentation inside a block scalar")
                }
                if style == ">", entry.indent > blockIndent {
                    throw error(entry, "extra indentation inside a folded block scalar is not supported")
                }
                texts.append(String(entry.raw.dropFirst(blockIndent)))
            }
            let joined: String
            if style == "|" {
                joined = texts.joined(separator: "\n")
            } else {
                joined = Self.fold(texts)
            }

            switch indicator {
            case "-":
                return .scalar(joined)
            case "+":
                return .scalar(joined.isEmpty
                    ? String(repeating: "\n", count: trailingBlanks)
                    : joined + String(repeating: "\n", count: trailingBlanks + 1))
            default:
                // Clip: exactly one trailing newline when there was content.
                return .scalar(joined.isEmpty ? "" : joined + "\n")
            }
        }

        /// Folded style: a break between two non-empty lines becomes a space,
        /// a blank line becomes a newline.
        private static func fold(_ texts: [String]) -> String {
            var result = ""
            var previousWasBlank = false
            for (offset, text) in texts.enumerated() {
                if text.isEmpty {
                    result += "\n"
                    previousWasBlank = true
                    continue
                }
                if offset > 0, !previousWasBlank { result += " " }
                result += text
                previousWasBlank = false
            }
            return result
        }

        // MARK: Flow sequences

        private mutating func parseFlowSequence(_ text: String, line: RawLine) throws -> YAMLValue {
            var elements: [String] = []
            var buffer = ""
            var quote: Character?
            var closed = false
            var cursor = text.index(after: text.startIndex)

            while cursor < text.endIndex {
                let character = text[cursor]
                if let open = quote {
                    buffer.append(character)
                    if character == open { quote = nil }
                    cursor = text.index(after: cursor)
                    continue
                }
                switch character {
                case "\"", "'":
                    quote = character
                    buffer.append(character)
                case "[", "{":
                    throw error(line, "nested flow collections are not supported")
                case "]":
                    closed = true
                case ",":
                    elements.append(buffer)
                    buffer = ""
                default:
                    buffer.append(character)
                }
                if closed { cursor = text.index(after: cursor); break }
                cursor = text.index(after: cursor)
            }
            guard closed else { throw error(line, "unclosed flow sequence") }
            let tail = String(text[cursor...]).trimmingCharacters(in: .whitespaces)
            guard tail.isEmpty else {
                throw error(line, "unexpected text after a flow sequence")
            }
            elements.append(buffer)

            let trimmed = elements.map { $0.trimmingCharacters(in: .whitespaces) }
            if trimmed.count == 1, trimmed[0].isEmpty { return .sequence([]) }
            var values: [YAMLValue] = []
            for element in trimmed {
                guard !element.isEmpty else {
                    throw error(line, "empty element in a flow sequence")
                }
                values.append(try Self.parseScalar(element, line: line))
            }
            return .sequence(values)
        }

        // MARK: Keys and scalars

        /// Splits `key: value`. The colon must be followed by a space or end
        /// of line, so "openai:gpt-4o" and "https://x" stay scalars.
        private func splitKey(_ line: RawLine) throws -> (String, String) {
            let content = line.content
            if let first = content.first, first == "\"" || first == "'" {
                let (key, after) = try Self.readQuoted(content, line: line)
                let rest = after.drop { $0 == " " }
                guard rest.first == ":" else {
                    throw error(line, "expected ':' after a quoted key")
                }
                return (key, String(rest.dropFirst()).trimmingCharacters(in: .whitespaces))
            }

            var quote: Character?
            var cursor = content.startIndex
            while cursor < content.endIndex {
                let character = content[cursor]
                if let open = quote {
                    if character == open { quote = nil }
                } else if character == "\"" || character == "'" {
                    quote = character
                } else if character == ":" {
                    let next = content.index(after: cursor)
                    if next == content.endIndex || content[next] == " " {
                        let key = String(content[content.startIndex..<cursor])
                            .trimmingCharacters(in: .whitespaces)
                        guard !key.isEmpty else { throw error(line, "empty key") }
                        guard !key.hasPrefix("<<") else {
                            throw error(line, "merge keys (<<) are not supported")
                        }
                        let rest = String(content[next...]).trimmingCharacters(in: .whitespaces)
                        return (key, rest)
                    }
                }
                cursor = content.index(after: cursor)
            }
            throw error(line, "expected 'key: value'")
        }

        /// True when a sequence item's text opens a mapping rather than being
        /// a plain scalar. Mirrors `splitKey` without throwing.
        private static func startsMapping(_ text: String) -> Bool {
            var quote: Character?
            var cursor = text.startIndex
            while cursor < text.endIndex {
                let character = text[cursor]
                if let open = quote {
                    if character == open { quote = nil }
                } else if character == "\"" || character == "'" {
                    quote = character
                } else if character == ":" {
                    let next = text.index(after: cursor)
                    if next == text.endIndex || text[next] == " " { return true }
                }
                cursor = text.index(after: cursor)
            }
            return false
        }

        private static func isSequenceItem(_ content: String) -> Bool {
            content == "-" || content.hasPrefix("- ")
        }

        static func parseScalar(_ text: String, line: RawLine) throws -> YAMLValue {
            guard let first = text.first else { return .null }
            if first == "\"" || first == "'" {
                let (value, after) = try readQuoted(text, line: line)
                let tail = after.trimmingCharacters(in: .whitespaces)
                guard tail.isEmpty else {
                    throw YAMLError(line: line.number, reason: "unexpected text after a quoted scalar")
                }
                return .scalar(value)
            }
            if ["~", "null", "Null", "NULL"].contains(text) { return .null }
            if first == "&" || first == "*" {
                throw YAMLError(line: line.number, reason: "anchors and aliases are not supported")
            }
            return .scalar(text)
        }

        /// Reads one quoted token and returns it with whatever followed it.
        private static func readQuoted(
            _ text: String, line: RawLine
        ) throws -> (String, Substring) {
            let quote = text[text.startIndex]
            var value = ""
            var cursor = text.index(after: text.startIndex)

            while cursor < text.endIndex {
                let character = text[cursor]
                if quote == "'" {
                    if character == "'" {
                        let next = text.index(after: cursor)
                        // '' inside a single-quoted scalar is one quote.
                        if next < text.endIndex, text[next] == "'" {
                            value.append("'")
                            cursor = text.index(after: next)
                            continue
                        }
                        return (value, text[text.index(after: cursor)...])
                    }
                    value.append(character)
                    cursor = text.index(after: cursor)
                    continue
                }
                if character == "\\" {
                    let next = text.index(after: cursor)
                    guard next < text.endIndex else {
                        throw YAMLError(line: line.number, reason: "unterminated escape sequence")
                    }
                    let escaped = text[next]
                    switch escaped {
                    case "n":  value.append("\n")
                    case "t":  value.append("\t")
                    case "r":  value.append("\r")
                    case "0":  value.append("\0")
                    case "\"": value.append("\"")
                    case "\\": value.append("\\")
                    case "/":  value.append("/")
                    case "u":
                        let start = text.index(next, offsetBy: 1)
                        guard let end = text.index(
                            start, offsetBy: 4, limitedBy: text.endIndex
                        ), let code = UInt32(text[start..<end], radix: 16),
                           let scalar = Unicode.Scalar(code) else {
                            throw YAMLError(line: line.number, reason: "invalid \\u escape")
                        }
                        value.append(Character(scalar))
                        cursor = end
                        continue
                    default:
                        throw YAMLError(
                            line: line.number,
                            reason: "unsupported escape sequence '\\\(escaped)'"
                        )
                    }
                    cursor = text.index(after: next)
                    continue
                }
                if character == quote {
                    return (value, text[text.index(after: cursor)...])
                }
                value.append(character)
                cursor = text.index(after: cursor)
            }
            throw YAMLError(line: line.number, reason: "unterminated quoted string")
        }

        // MARK: Comments

        /// Drops a trailing `#` comment. A `#` only starts one at the start of
        /// a line or after whitespace, and never inside quotes — so
        /// "colour: #ff0000" keeps its value.
        static func stripComment(_ raw: String) -> String {
            var quote: Character?
            var previous: Character?
            var cursor = raw.startIndex

            while cursor < raw.endIndex {
                let character = raw[cursor]
                if let open = quote {
                    if character == open { quote = nil }
                } else if character == "\"" || character == "'" {
                    quote = character
                } else if character == "#" {
                    if previous == nil || previous == " " || previous == "\t" {
                        return String(raw[raw.startIndex..<cursor])
                    }
                }
                previous = character
                cursor = raw.index(after: cursor)
            }
            return raw
        }
    }
}
