//
//  ProblemReport.swift
//  local_ai_test
//
//  "Report a problem": the user describes what happened, sees every line of
//  diagnostics that goes with it, and hands the result to their own mail app.
//  Nothing goes through our servers and nothing is sent without the user
//  pressing Send in Mail.
//
//  Everything here is a pure function of its inputs (no clocks, no device
//  reads, no singletons), so the unit tests pin exactly what a report
//  contains. ProblemReportContext+Device.swift collects the readings.
//
//  Deliberately absent: the analytics installation id (a report must never be
//  joinable to the measurement records) and any conversation content, unless
//  the user pastes it into the description.
//

import Foundation

nonisolated enum ProblemReportCategory: String, CaseIterable, Identifiable, Sendable {
    case chat, download, `import`, benchmark, voice, other

    var id: String { rawValue }

    var label: String {
        switch self {
        case .chat: return "Chat"
        case .download: return "Download"
        case .import: return "Import"
        case .benchmark: return "Benchmark"
        case .voice: return "Voice"
        case .other: return "Other"
        }
    }
}

/// The readings a report can carry, collected once when the screen opens.
nonisolated struct ProblemReportContext: Equatable, Sendable {

    struct Model: Equatable, Sendable {
        /// Catalog id, or the id the import flow gave the model.
        var id: String
        var name: String
        var isImported: Bool
        /// Hugging Face repository of an imported model ("" for a local file).
        var repo: String
        /// The weights file of an imported model, when it has a single one.
        var file: String?
        /// ModelBackend raw value ("llamaCpp").
        var backend: String
    }

    struct Failure: Equatable, Sendable {
        var summary: String
        var detail: String
        var phase: String
        var type: String
        var backend: String
        var modelName: String
        var modelId: String
    }

    var appVersion: String
    var appBuild: String
    /// Debug (or simulator) build. Reports from those are still sendable -
    /// they are written and sent by a person on purpose, unlike measurements,
    /// so there is no noise to keep out - but they say so in the subject and
    /// the diagnostics, so the owner can tell a developer's report apart.
    var debugBuild: Bool
    var osVersion: String
    var deviceIdentifier: String
    var deviceName: String
    var ramBytes: UInt64
    var freeDiskBytes: Int64
    var model: Model?
    var lastError: Failure?
    /// "nominal", "fair", "serious", "critical" or "unknown".
    var thermalState: String
    var lowPowerMode: Bool
    var locale: String
}

/// What the mail composer (or the fallbacks) gets.
nonisolated struct ProblemReportMail: Equatable, Sendable {
    var recipient: String
    var subject: String
    /// Description, then the diagnostics as plain readable text.
    var body: String
    /// The same diagnostics with the error detail untrimmed, attached as a
    /// file when the composer is available so a long error survives.
    var attachmentText: String
    static let attachmentName = "diagnostics.txt"
}

nonisolated enum ProblemReportLimits {
    static let descriptionMax = 4000
    /// Error detail inside the mail body; the attachment carries up to
    /// `attachmentErrorDetailMax`.
    static let bodyErrorDetailMax = 1500
    static let attachmentErrorDetailMax = 20_000
    static let shortFieldMax = 200
    /// Longest mailto: URL the fallback builds. Mail on iOS takes far more,
    /// but third-party mail apps and handlers truncate or refuse long links.
    static let mailtoMax = 4000
}

nonisolated enum ProblemReportBuilder {

    static let placeholderModel = "<model>"
    static let conversationNote = "No conversation content is included unless you pasted it above."

    /// The description as it will be sent: control characters out (newlines
    /// and tabs kept), outer whitespace trimmed, capped at
    /// `ProblemReportLimits.descriptionMax` characters.
    static func cleanDescription(_ text: String) -> String {
        cap(clean(text), ProblemReportLimits.descriptionMax)
    }

    static func canSend(description: String) -> Bool {
        !cleanDescription(description).isEmpty
    }

    /// One labelled line per reading, in the order the screen and the mail
    /// show them. The screen's "Will be sent" list is exactly these rows.
    static func rows(context: ProblemReportContext, category: ProblemReportCategory?,
                     includeModel: Bool) -> [(label: String, value: String)] {
        var rows: [(String, String)] = []
        rows.append(("Category", category?.label ?? "Not specified"))
        let build = context.appBuild.isEmpty ? context.appVersion : "\(context.appVersion) (\(context.appBuild))"
        rows.append(("App", build + (context.debugBuild ? " Debug" : "")))
        rows.append(("iOS", context.osVersion))
        let device = context.deviceName.isEmpty || context.deviceName == context.deviceIdentifier
            ? context.deviceIdentifier : "\(context.deviceName) (\(context.deviceIdentifier))"
        rows.append(("Device", device))
        rows.append(("RAM", gigabytes(Double(context.ramBytes))))
        rows.append(("Free disk", gigabytes(Double(max(context.freeDiskBytes, 0)))))
        if let model = context.model {
            rows.append(("Model", includeModel ? modelLabel(model) : "Not included"))
            rows.append(("Backend", model.backend))
        } else {
            rows.append(("Model", "None selected"))
            if let backend = context.lastError?.backend, !backend.isEmpty { rows.append(("Backend", backend)) }
        }
        rows.append(("Thermal state", context.thermalState))
        rows.append(("Low Power Mode", context.lowPowerMode ? "On" : "Off"))
        rows.append(("Locale", context.locale))
        if let failure = context.lastError {
            rows.append(("Last error", redact(single(failure.summary), context: context, includeModel: includeModel)))
            rows.append(("Error phase", failure.phase))
            if !failure.type.isEmpty { rows.append(("Error type", single(failure.type))) }
        }
        return rows.map { (label: $0.0, value: cap(single($0.1), ProblemReportLimits.shortFieldMax)) }
    }

    /// "unsloth/Qwen3-4B-GGUF (catalog)", or the repository and file of an
    /// imported model so the user can see exactly what goes out.
    static func modelLabel(_ model: ProblemReportContext.Model) -> String {
        guard model.isImported else { return "\(model.id) (catalog)" }
        var parts: [String] = []
        if !model.repo.isEmpty { parts.append(model.repo) }
        if let file = model.file, !file.isEmpty { parts.append(file) }
        if parts.isEmpty { parts.append(model.name.isEmpty ? model.id : model.name) }
        return parts.joined(separator: "/") + " (imported)"
    }

    /// The error detail as it appears in the mail (`limit` characters at most).
    static func errorDetail(context: ProblemReportContext, includeModel: Bool, limit: Int) -> String? {
        guard let failure = context.lastError else { return nil }
        let detail = redact(clean(failure.detail), context: context, includeModel: includeModel)
        guard !detail.isEmpty else { return nil }
        return detail.count > limit ? String(detail.prefix(limit)) + "\n[... trimmed, full text in diagnostics.txt]" : detail
    }

    static func subject(context: ProblemReportContext, category: ProblemReportCategory?) -> String {
        let version = context.appVersion + (context.debugBuild ? " Debug" : "")
        return "Pocket AI Lab problem: \(category?.label ?? "General") (\(version))"
    }

    static func diagnostics(context: ProblemReportContext, category: ProblemReportCategory?,
                            includeModel: Bool, errorDetailLimit: Int) -> String {
        let rows = rows(context: context, category: category, includeModel: includeModel)
        let width = (rows.map(\.label.count).max() ?? 0) + 2
        var lines = rows.map { row in
            (row.label + ":").padding(toLength: width, withPad: " ", startingAt: 0) + row.value
        }
        if let detail = errorDetail(context: context, includeModel: includeModel, limit: errorDetailLimit) {
            lines += ["", "Error detail:", detail]
        }
        return lines.joined(separator: "\n")
    }

    static func mail(description: String, category: ProblemReportCategory?, includeModel: Bool,
                     context: ProblemReportContext) -> ProblemReportMail {
        let text = cleanDescription(description)
        let body = [
            text,
            "",
            "--- Diagnostics ---",
            diagnostics(context: context, category: category, includeModel: includeModel,
                        errorDetailLimit: ProblemReportLimits.bodyErrorDetailMax),
            "",
            conversationNote,
        ].joined(separator: "\n")
        let attachment = [
            subject(context: context, category: category),
            "",
            diagnostics(context: context, category: category, includeModel: includeModel,
                        errorDetailLimit: ProblemReportLimits.attachmentErrorDetailMax),
        ].joined(separator: "\n")
        return ProblemReportMail(recipient: SupportContact.email,
                                 subject: subject(context: context, category: category),
                                 body: body, attachmentText: attachment + "\n")
    }

    /// The whole report as one text, for "Copy report".
    static func clipboardText(_ mail: ProblemReportMail) -> String {
        "To: \(mail.recipient)\nSubject: \(mail.subject)\n\n\(mail.body)\n"
    }

    // MARK: - mailto fallback

    static let mailtoTrimNote = "[Trimmed: the report was too long for a mail link. "
        + "Use Copy report in the app and paste the rest here.]"

    /// RFC 6068: everything but unreserved characters is percent-encoded, and
    /// line breaks are sent as CRLF.
    static func mailtoEncode(_ text: String) -> String {
        let unreserved = CharacterSet(charactersIn:
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        let crlf = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\n", with: "\r\n")
        return crlf.addingPercentEncoding(withAllowedCharacters: unreserved) ?? ""
    }

    /// A mailto: URL with the same subject and body, at most `maxLength`
    /// characters. A body that does not fit is cut on a character boundary
    /// (never inside a percent escape) and ends with `mailtoTrimNote`.
    static func mailtoURL(_ mail: ProblemReportMail, maxLength: Int = ProblemReportLimits.mailtoMax) -> URL? {
        let head = "mailto:\(mail.recipient)?subject=\(mailtoEncode(mail.subject))&body="
        func url(_ body: String) -> String { head + mailtoEncode(body) }
        let full = url(mail.body)
        if full.count <= maxLength { return URL(string: full) }
        let chars = Array(mail.body)
        var low = 0, high = chars.count
        while low < high {
            let mid = (low + high + 1) / 2
            if url(trimmed(chars, mid)).count <= maxLength { low = mid } else { high = mid - 1 }
        }
        return URL(string: url(trimmed(chars, low)))
    }

    private static func trimmed(_ chars: [Character], _ count: Int) -> String {
        String(chars.prefix(count)).trimmingCharacters(in: .whitespacesAndNewlines) + "\n\n" + mailtoTrimNote
    }

    // MARK: - Text helpers

    /// Control characters and bidi overrides out (newline and tab kept),
    /// CRLF folded to LF, outer whitespace trimmed.
    static func clean(_ text: String) -> String {
        let normalized = text.replacingOccurrences(of: "\r\n", with: "\n").replacingOccurrences(of: "\r", with: "\n")
        var scalars = String.UnicodeScalarView()
        for scalar in normalized.unicodeScalars {
            let v = scalar.value
            let control = (v < 0x20 && v != 0x0A && v != 0x09) || (0x7F...0x9F).contains(v)
            let bidi = (0x202A...0x202E).contains(v) || (0x2066...0x2069).contains(v)
            if !control && !bidi { scalars.append(scalar) }
        }
        return String(scalars).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    static func single(_ text: String) -> String {
        clean(text).components(separatedBy: .newlines).joined(separator: " ")
            .replacingOccurrences(of: "\t", with: " ")
    }

    static func cap(_ text: String, _ limit: Int) -> String {
        text.count > limit ? String(text.prefix(limit)) : text
    }

    /// With "Include model name" off, the model's id, name, repository and
    /// file never appear, not even inside the error text.
    static func redact(_ text: String, context: ProblemReportContext, includeModel: Bool) -> String {
        guard !includeModel else { return text }
        var needles: [String] = []
        if let m = context.model { needles += [m.id, m.name, m.repo, m.file ?? ""] }
        if let f = context.lastError { needles += [f.modelId, f.modelName] }
        var out = text
        // Longest first, so "org/repo" goes before "repo" leaves "org/" behind.
        for needle in Set(needles.filter { $0.count >= 3 }).sorted(by: { $0.count > $1.count }) {
            out = out.replacingOccurrences(of: needle, with: placeholderModel, options: .caseInsensitive)
        }
        return out
    }

    private static func gigabytes(_ bytes: Double) -> String {
        String(format: "%.1f GB", bytes / 1_073_741_824)
    }
}
