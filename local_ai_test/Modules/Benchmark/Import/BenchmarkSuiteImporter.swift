//
//  BenchmarkSuiteImporter.swift
//  local_ai_test
//
//  "Bring your own benchmark suite": a file from Files or a pasted URL turns
//  into a suite this app can run, or into a refusal that says why.
//
//  WHY THIS SHAPE. The two formats people actually write evals in are
//  EleutherAI's lm-evaluation-harness and promptfoo. Both are CLIs — one
//  Python, one Node — and neither can run on a phone. So the app does not
//  pretend to host their runtimes: it accepts their FILES and maps the part
//  that is a deterministic string check onto its own checkers. Everything
//  that would need to execute code or ask a judge model is refused at the
//  door, by name.
//
//  Two formats in:
//    • this app's own `benchmarks.json` schema, which is also what the
//      exporter and the bundled suite use;
//    • a promptfoo `promptfooconfig.yaml` — see `PromptfooImport`.
//
//  THREE RULES THE IMPORT KEEPS.
//
//  1. Never grade with a judge. `llm-rubric`, `model-graded-*`, `g-eval` and
//     friends make a score depend on another model's mood, its quantization
//     and the phone it ran on. The whole point of this benchmark is that the
//     same answers always produce the same score, so an import carrying one
//     is refused outright rather than partially honoured.
//  2. Never execute anything. `javascript` and `python` assertions are code;
//     running downloaded code is not a thing this app does.
//  3. Never partially import in silence. Every test that could not be mapped
//     is listed with a reason BEFORE anything runs. A suite that quietly
//     dropped a third of its tests would produce a real-looking score for an
//     exam nobody wrote.
//

import Foundation

// MARK: - Format

nonisolated enum SuiteImportFormat: String, Codable, Sendable, Hashable {
    /// The app's own `benchmarks.json` shape.
    case appJSON
    /// A promptfoo `promptfooconfig.yaml`.
    case promptfooYAML

    var displayName: String {
        switch self {
        case .appJSON:       return "Pocket AI Lab JSON"
        case .promptfooYAML: return "promptfoo YAML"
        }
    }
}

// MARK: - Limits

/// Guardrails a pasted file cannot argue with.
nonisolated enum SuiteImportLimits {

    /// Bytes accepted from a file or a URL. A suite is text; a megabyte of it
    /// is already thousands of tasks.
    static let maxBytes = 1_048_576

    /// Hard ceiling on tasks in one suite.
    ///
    /// The cap exists because of what a task COSTS, not because of what it
    /// weighs: on a small phone a single 96-token answer takes roughly ten
    /// seconds, so two hundred tasks is already the better part of an hour.
    /// A file with a thousand tests would start a six-hour run from a single
    /// tap, and no progress bar makes that acceptable.
    static let maxTasks = 200

    /// Per-task generation ceiling for an imported task. Imported suites do
    /// not get to set this freely: a task asking for 4096 tokens turns one
    /// chatty model into an afternoon. Anything higher is clamped and the
    /// clamp is reported.
    static let maxTokensPerTask = 256

    /// Token cap given to an imported task that does not ask for one. Matches
    /// `BenchmarkTask`'s own default.
    static let defaultTokensPerTask = 96

    /// Seconds per task used for the "this will take a while" estimate — a
    /// deliberately pessimistic small-phone figure, because the estimate
    /// exists to warn, not to flatter.
    static let estimatedSecondsPerTask: Double = 12

    /// Tasks above which the import result carries a runtime warning.
    static let runtimeWarningTaskCount = 40
}

// MARK: - Result

/// One test that did not become a task, and why. Shown before the suite is
/// saved — see rule 3 in the file header.
nonisolated struct SkippedSuiteEntry: Sendable, Hashable, Identifiable {
    /// "test 4" or the test's own description.
    let name: String
    let reason: String

    var id: String { name + "|" + reason }
}

/// What an import produced. Nothing has been written to disk yet.
nonisolated struct SuiteImportResult: Sendable, Hashable {
    let format: SuiteImportFormat
    let suite: StoredBenchmarkSuite
    let skipped: [SkippedSuiteEntry]
    /// Things the user should know that are not refusals: clamped token
    /// caps, an expected runtime, an assertion list folded into one task.
    let notes: [String]

    var recognizedTaskCount: Int { suite.taskCount }

    var summaryLine: String {
        let tasks = "\(recognizedTaskCount) task\(recognizedTaskCount == 1 ? "" : "s") recognised"
        guard !skipped.isEmpty else { return tasks }
        return tasks + ", \(skipped.count) skipped"
    }
}

// MARK: - Errors

/// Every way an import refuses. Each case names the thing it refused, because
/// "import failed" tells the user nothing about which line to fix.
nonisolated enum SuiteImportError: Error, LocalizedError, Equatable, Sendable {

    case empty
    case tooLarge(bytes: Int)
    case notText
    case unrecognizedFormat
    case malformedJSON(String)
    case malformedYAML(line: Int, reason: String)
    case unsupportedSchemaVersion(Int)
    /// An assertion that would need a judge model. Carries its promptfoo type.
    case judgeAssertion(String)
    /// An assertion that would need to execute code. Carries its type.
    case executableAssertion(String)
    case duplicateTaskID(String)
    case tooManyTasks(count: Int)
    case noRunnableTasks
    case missingTests
    case missingPrompt
    case multiplePrompts(Int)
    case unsupportedPrompt(String)
    /// The file points at something this app cannot fetch (`file://…`, a CSV
    /// of tests, a prompt in another file).
    case externalReference(String)
    case networkBlockedDuringRun
    case downloadFailed(String)
    case storageUnavailable

    var errorDescription: String? {
        switch self {
        case .empty:
            return "The file is empty."
        case .tooLarge(let bytes):
            return "The file is \(bytes / 1024) KB. A suite may be at most "
                + "\(SuiteImportLimits.maxBytes / 1024) KB."
        case .notText:
            return "The file is not text. A suite must be a JSON or YAML document."
        case .unrecognizedFormat:
            return "This is neither a Pocket AI Lab suite (a JSON object) nor a "
                + "promptfoo config (a YAML mapping)."
        case .malformedJSON(let detail):
            return "The JSON could not be read: \(detail)"
        case .malformedYAML(let line, let reason):
            return "The YAML could not be read - line \(line): \(reason)"
        case .unsupportedSchemaVersion(let version):
            return "This suite declares schemaVersion \(version). This build reads "
                + "\(BenchmarkSuiteDocument.minimumSchemaVersion) to "
                + "\(BenchmarkSuiteDocument.supportedSchemaVersion)."
        case .judgeAssertion(let type):
            return "The assertion '\(type)' is graded by another model. Scores here "
                + "are deterministic, so a suite that needs a judge is not imported."
        case .executableAssertion(let type):
            return "The assertion '\(type)' runs code. This app does not execute "
                + "code from an imported file, so the suite is not imported."
        case .duplicateTaskID(let id):
            return "Two tasks share the id '\(id)'. Task ids key the results, so "
                + "they have to be unique."
        case .tooManyTasks(let count):
            return "The suite has \(count) tasks. At most \(SuiteImportLimits.maxTasks) "
                + "can be imported - beyond that a single run takes hours."
        case .noRunnableTasks:
            return "Nothing in this file could be run as a task."
        case .missingTests:
            return "No 'tests' were found. A promptfoo config needs a tests list."
        case .missingPrompt:
            return "No prompt was found. A promptfoo config needs a 'prompts' entry, "
                + "or a 'prompt' on each test."
        case .multiplePrompts(let count):
            return "The config defines \(count) prompts. Only one prompt template can "
                + "be imported - running every prompt against every test would score "
                + "something other than what the file describes."
        case .unsupportedPrompt(let detail):
            return "This prompt cannot be used on device: \(detail)"
        case .externalReference(let reference):
            return "The file points at '\(reference)'. An imported suite has to be "
                + "self-contained - nothing outside it can be read on device."
        case .networkBlockedDuringRun:
            return "A benchmark is running. Downloading a suite now would change the "
                + "numbers it is measuring - try again when the run finishes."
        case .downloadFailed(let detail):
            return "The suite could not be downloaded: \(detail)"
        case .storageUnavailable:
            return "The suite could not be saved on this device."
        }
    }
}

// MARK: - Importer

nonisolated enum BenchmarkSuiteImporter {

    // MARK: Entry points

    /// Reads a suite out of raw bytes.
    ///
    /// - Parameters:
    ///   - data: file contents. UTF-8 (or Latin-1 as a fallback) text.
    ///   - source: filename or URL, kept in the stored suite and shown in the
    ///     list.
    ///   - suiteID: injectable so tests are deterministic; production callers
    ///     take the default.
    static func makeSuite(
        data: Data,
        source: String,
        suiteID: String = CustomBenchmarkSuiteFile.newID()
    ) throws -> SuiteImportResult {
        guard !data.isEmpty else { throw SuiteImportError.empty }
        guard data.count <= SuiteImportLimits.maxBytes else {
            throw SuiteImportError.tooLarge(bytes: data.count)
        }
        guard let text = String(data: data, encoding: .utf8)
            ?? String(data: data, encoding: .isoLatin1) else {
            throw SuiteImportError.notText
        }
        return try makeSuite(text: text, source: source, suiteID: suiteID)
    }

    static func makeSuite(
        text: String,
        source: String,
        suiteID: String = CustomBenchmarkSuiteFile.newID()
    ) throws -> SuiteImportResult {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw SuiteImportError.empty }

        // Format is decided by the first character rather than by the file
        // extension: files arrive from Files.app, from a share sheet and from
        // a URL, and the extension is the least reliable of the three.
        if trimmed.hasPrefix("{") {
            return try importAppJSON(text: trimmed, source: source, suiteID: suiteID)
        }
        if trimmed.hasPrefix("[") {
            throw SuiteImportError.unrecognizedFormat
        }
        return try importPromptfoo(text: text, source: source, suiteID: suiteID)
    }

    /// Fetches a suite from a URL and reads it. Refused outright while a
    /// benchmark is being measured — see `BenchmarkRunGate`.
    static func makeSuite(
        from url: URL,
        suiteID: String = CustomBenchmarkSuiteFile.newID()
    ) async throws -> SuiteImportResult {
        guard !BenchmarkRunGate.shared.isRunning else {
            throw SuiteImportError.networkBlockedDuringRun
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 20
        request.cachePolicy = .reloadIgnoringLocalCacheData

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw SuiteImportError.downloadFailed(error.localizedDescription)
        }
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw SuiteImportError.downloadFailed("HTTP \(http.statusCode)")
        }
        return try makeSuite(
            data: data,
            source: url.absoluteString,
            suiteID: suiteID
        )
    }

    /// Turns a pasted string into a URL the way the model importer does:
    /// bare hosts get https, anything without a host is rejected.
    static func parseURL(from text: String) -> URL? {
        var trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if !trimmed.lowercased().hasPrefix("http://"),
           !trimmed.lowercased().hasPrefix("https://") {
            trimmed = "https://" + trimmed
        }
        guard let url = URL(string: trimmed), let host = url.host, host.contains(".") else {
            return nil
        }
        return url
    }

    // MARK: The app's own schema

    private static func importAppJSON(
        text: String,
        source: String,
        suiteID: String
    ) throws -> SuiteImportResult {
        guard let data = text.data(using: .utf8) else { throw SuiteImportError.notText }

        let decoded: BenchmarkSuiteDocument
        do {
            decoded = try JSONDecoder().decode(BenchmarkSuiteDocument.self, from: data)
        } catch {
            // A `kind` this build has never heard of lands here too, which is
            // the same forward-compatibility rule the remote suite follows:
            // a partially understood document is not run.
            throw SuiteImportError.malformedJSON(shortDecodingMessage(error))
        }
        guard BenchmarkSuiteDocument.canRun(schemaVersion: decoded.schemaVersion) else {
            throw SuiteImportError.unsupportedSchemaVersion(decoded.schemaVersion)
        }

        var notes: [String] = []
        let tasks = try finalize(tasks: decoded.tasks, notes: &notes)

        let name = decoded.name?.trimmingCharacters(in: .whitespacesAndNewlines)
        let document = BenchmarkSuiteDocument(
            schemaVersion: BenchmarkSuiteDocument.supportedSchemaVersion,
            suiteVersion: decoded.suiteVersion,
            tasks: tasks,
            id: suiteID,
            name: displayName(preferred: name, source: source),
            // Kept so the imported copy runs in the same order as the file
            // it came from would.
            samplingSeed: decoded.samplingSeed
        )
        return SuiteImportResult(
            format: .appJSON,
            suite: StoredBenchmarkSuite(
                id: suiteID,
                name: document.name ?? "Imported suite",
                source: source,
                format: .appJSON,
                document: document
            ),
            skipped: [],
            notes: notes
        )
    }

    // MARK: promptfoo

    private static func importPromptfoo(
        text: String,
        source: String,
        suiteID: String
    ) throws -> SuiteImportResult {
        let root: YAMLValue
        do {
            root = try MiniYAML.parse(text)
        } catch let error as YAMLError {
            throw SuiteImportError.malformedYAML(line: error.line, reason: error.reason)
        }
        guard root.mappingValue != nil else { throw SuiteImportError.unrecognizedFormat }

        let mapped = try PromptfooImport.map(root)
        var notes = mapped.notes
        let tasks = try finalize(tasks: mapped.tasks, notes: &notes)

        let document = BenchmarkSuiteDocument(
            schemaVersion: BenchmarkSuiteDocument.supportedSchemaVersion,
            // promptfoo has no notion of a suite version. 1 is honest: this
            // is the first (and only) version of this particular file, and
            // the suite ID is what actually keeps its scores apart from ours.
            suiteVersion: 1,
            tasks: tasks,
            id: suiteID,
            name: displayName(preferred: mapped.name, source: source)
        )
        return SuiteImportResult(
            format: .promptfooYAML,
            suite: StoredBenchmarkSuite(
                id: suiteID,
                name: document.name ?? "Imported suite",
                source: source,
                format: .promptfooYAML,
                document: document
            ),
            skipped: mapped.skipped,
            notes: notes
        )
    }

    // MARK: Shared checks

    /// The guardrails every format passes through: unique ids, the size cap,
    /// the per-task token cap, and the runtime warning.
    private static func finalize(
        tasks: [BenchmarkTask],
        notes: inout [String]
    ) throws -> [BenchmarkTask] {
        guard !tasks.isEmpty else { throw SuiteImportError.noRunnableTasks }
        guard tasks.count <= SuiteImportLimits.maxTasks else {
            throw SuiteImportError.tooManyTasks(count: tasks.count)
        }

        var seen: Set<String> = []
        for task in tasks {
            guard seen.insert(task.id).inserted else {
                throw SuiteImportError.duplicateTaskID(task.id)
            }
        }

        var clamped = 0
        let capped = tasks.map { task -> BenchmarkTask in
            guard task.maxTokens > SuiteImportLimits.maxTokensPerTask else { return task }
            clamped += 1
            return task.withMaxTokens(SuiteImportLimits.maxTokensPerTask)
        }
        if clamped > 0 {
            notes.append("\(clamped) task\(clamped == 1 ? "" : "s") asked for more than "
                + "\(SuiteImportLimits.maxTokensPerTask) tokens and were capped there.")
        }
        if capped.count >= SuiteImportLimits.runtimeWarningTaskCount {
            let minutes = Int(
                (Double(capped.count) * SuiteImportLimits.estimatedSecondsPerTask / 60).rounded()
            )
            notes.append("A full run of \(capped.count) tasks takes roughly \(minutes) "
                + "minutes on a small model, longer on a large one.")
        }
        return capped
    }

    /// Suite name: what the file called itself, else the file it came from.
    private static func displayName(preferred: String?, source: String) -> String {
        if let preferred, !preferred.isEmpty { return String(preferred.prefix(80)) }
        let last = source.split(separator: "/").last.map(String.init) ?? source
        let stripped = last.split(separator: "?").first.map(String.init) ?? last
        return stripped.isEmpty ? "Imported suite" : String(stripped.prefix(80))
    }

    /// `DecodingError` prints a whole coding path by default, which is noise
    /// in an alert. This keeps the part that names the field.
    private static func shortDecodingMessage(_ error: any Error) -> String {
        guard let decoding = error as? DecodingError else { return error.localizedDescription }
        switch decoding {
        case .keyNotFound(let key, let context):
            return "missing '\(key.stringValue)'\(pathSuffix(context))"
        case .typeMismatch(_, let context):
            return "wrong type\(pathSuffix(context))"
        case .valueNotFound(_, let context):
            return "missing value\(pathSuffix(context))"
        case .dataCorrupted(let context):
            return context.debugDescription.isEmpty
                ? "the document is not valid JSON"
                : context.debugDescription
        @unknown default:
            return error.localizedDescription
        }
    }

    private static func pathSuffix(_ context: DecodingError.Context) -> String {
        let path = context.codingPath.map(\.stringValue).filter { !$0.isEmpty }
        return path.isEmpty ? "" : " at \(path.joined(separator: "."))"
    }
}
