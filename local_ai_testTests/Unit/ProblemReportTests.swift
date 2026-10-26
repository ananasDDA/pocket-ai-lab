//
//  ProblemReportTests.swift
//
//  Pins what "Report a problem" puts into the mail: the readings shown on
//  the screen are exactly the ones in the body, the model toggle removes the
//  model everywhere (error text included), no installation id and no
//  conversation, length caps, the subject, and the mailto: fallback's
//  encoding and trimming.
//

import Foundation
import Testing
@testable import local_ai_test

struct ProblemReportTests {

    static let catalogModel = ProblemReportContext.Model(
        id: "unsloth/Qwen3.5-2B-GGUF", name: "Qwen3.5 2B", isImported: false, repo: "", file: nil,
        backend: "llamaCpp")
    static let importedModel = ProblemReportContext.Model(
        id: "import-7c1e", name: "My Secret Finetune", isImported: true, repo: "someone/secret-finetune-GGUF",
        file: "secret-finetune-Q4_K_M.gguf", backend: "llamaCpp")
    static let failure = ProblemReportContext.Failure(
        summary: "Generation failed", detail: "llama_decode returned -1\nwhile running My Secret Finetune",
        phase: "generate", type: "LlamaError", backend: "llama.cpp", modelName: "My Secret Finetune",
        modelId: "import-7c1e")

    static func context(model: ProblemReportContext.Model? = catalogModel,
                        failure: ProblemReportContext.Failure? = nil,
                        debug: Bool = false) -> ProblemReportContext {
        ProblemReportContext(
            appVersion: "1.2", appBuild: "45", debugBuild: debug, osVersion: "26.0.1",
            deviceIdentifier: "iPhone18,4", deviceName: "iPhone Air", ramBytes: 12 * 1_073_741_824,
            freeDiskBytes: Int64(41.3 * 1_073_741_824), model: model, lastError: failure,
            thermalState: "fair", lowPowerMode: false, locale: "en_GB")
    }

    @Test func theMailHasTheTextThenTheDiagnostics() {
        let mail = ProblemReportBuilder.mail(description: "  It stops answering.\r\nThen freezes. ",
                                             category: .chat, includeModel:          true, context: Self.context())
        #expect(mail.recipient == SupportContact.email)
        #expect(mail.subject == "Pocket AI Lab problem: Chat (1.2)")
        #expect(mail.body == """
        It stops answering.
        Then freezes.

        --- Diagnostics ---
        Category:       Chat
        App:            1.2 (45)
        iOS:            26.0.1
        Device:         iPhone Air (iPhone18,4)
        RAM:            12.0 GB
        Free disk:      41.3 GB
        Model:          unsloth/Qwen3.5-2B-GGUF (catalog)
        Backend:        llamaCpp
        Thermal state:  fair
        Low Power Mode: Off
        Locale:         en_GB

        No conversation content is included unless you pasted it above.
        """)
        #expect(ProblemReportMail.attachmentName == "diagnostics.txt")
        #expect(mail.attachmentText.hasPrefix("Pocket AI Lab problem: Chat (1.2)\n\nCategory:"))
    }

    @Test func everyRowOnTheScreenIsInTheMail() {
        let ctx = Self.context(model: Self.importedModel, failure: Self.failure)
        for include in [true, false] {
            let mail = ProblemReportBuilder.mail(description: "x", category: nil, includeModel:          include, context: ctx)
            for row in ProblemReportBuilder.rows(context: ctx, category: nil, includeModel:          include) {
                #expect(mail.body.contains(row.value), "\(row.label)")
                #expect(mail.attachmentText.contains(row.value), "\(row.label)")
            }
        }
    }

    @Test func anImportedModelShowsItsRepoAndFile() {
        let rows = ProblemReportBuilder.rows(context: Self.context(model: Self.importedModel), category: nil,
                                             includeModel:          true)
        #expect(rows.first { $0.label == "Model" }?.value
                == "someone/secret-finetune-GGUF/secret-finetune-Q4_K_M.gguf (imported)")
    }

    @Test func theToggleRemovesTheModelEverywhere() {
        let ctx = Self.context(model: Self.importedModel, failure: Self.failure)
        let mail = ProblemReportBuilder.mail(description: "x", category: nil, includeModel:          false, context: ctx)
        for text in [mail.body, mail.attachmentText, mail.subject] {
            for secret in ["secret-finetune", "Secret Finetune", "import-7c1e", "someone/"] {
                #expect(!text.contains(secret), "\(secret)")
            }
        }
        #expect(mail.body.contains("Model:          Not included"))
        #expect(mail.body.contains("Backend:        llamaCpp"), "the engine still goes")
        #expect(mail.body.contains("while running <model>"))
        let on = ProblemReportBuilder.mail(description: "x", category: nil, includeModel:          true, context: ctx)
        #expect(on.body.contains("while running My Secret Finetune"))
    }

    @Test func theErrorGoesWithItsDetail() {
        let mail = ProblemReportBuilder.mail(description: "x", category: nil, includeModel:          true,
                                             context: Self.context(failure: Self.failure))
        #expect(mail.body.contains("Last error:     Generation failed"))
        #expect(mail.body.contains("Error phase:    generate"))
        #expect(mail.body.contains("Error type:     LlamaError"))
        #expect(mail.body.contains("Error detail:\nllama_decode returned -1"))
        #expect(mail.subject == "Pocket AI Lab problem: General (1.2)")
    }

    @Test func aLongErrorIsTrimmedInTheBodyButNotInTheAttachment() {
        var failure = Self.failure
        failure.detail = String(repeating: "e", count: 5000)
        let mail = ProblemReportBuilder.mail(description: "x", category: nil, includeModel:          true,
                                             context: Self.context(failure: failure))
        #expect(mail.body.contains("[... trimmed, full text in diagnostics.txt]"))
        #expect(!mail.body.contains(String(repeating: "e", count: ProblemReportLimits.bodyErrorDetailMax + 1)))
        #expect(mail.attachmentText.contains(String(repeating: "e", count: 5000)))
    }

    @Test func noInstallationIdAndNoConversation() {
        let mail = ProblemReportBuilder.mail(description: "x", category: .chat, includeModel:          true,
                                             context: Self.context(model: Self.importedModel, failure: Self.failure))
        let all = (mail.body + mail.attachmentText + mail.subject).lowercased()
        #expect(!all.contains("installation"))
        #expect(!all.contains("timezone"))
        #if canImport(UIKit)
        #expect(!all.contains(PALAnalyticsDevice.installationId.lowercased()))
        #endif
        // The builder has no input for chat content: only the description is user text.
        #expect(mail.body.hasPrefix("x\n\n--- Diagnostics ---"))
    }

    @Test func debugBuildsAreSendableButMarked() {
        let mail = ProblemReportBuilder.mail(description: "x", category: .voice, includeModel:          true,
                                             context: Self.context(debug: true))
        #expect(mail.subject == "Pocket AI Lab problem: Voice (1.2 Debug)")
        #expect(mail.body.contains("App:            1.2 (45) Debug"))
    }

    @Test func descriptionIsRequiredCleanedAndCapped() {
        #expect(!ProblemReportBuilder.canSend(description: ""))
        #expect(!ProblemReportBuilder.canSend(description: " \n\t "))
        #expect(!ProblemReportBuilder.canSend(description: "\u{0000}\u{001B}\u{202E}"))
        #expect(ProblemReportBuilder.canSend(description: "a"))
        #expect(ProblemReportBuilder.cleanDescription("a\u{0000}b\u{202E}c\u{0007}\nd\te") == "abc\nd\te")
        let long = String(repeating: "й", count: ProblemReportLimits.descriptionMax + 50)
        #expect(ProblemReportBuilder.cleanDescription(long).count == ProblemReportLimits.descriptionMax)
    }

    @Test func noModelSelected() {
        let rows = ProblemReportBuilder.rows(context: Self.context(model: nil, failure: Self.failure),
                                             category: nil, includeModel:          true)
        #expect(rows.first { $0.label == "Model" }?.value == "None selected")
        #expect(rows.first { $0.label == "Backend" }?.value == "llama.cpp")
    }

    // MARK: - mailto fallback

    @Test func mailtoEncodesEverythingButUnreserved() {
        #expect(ProblemReportBuilder.mailtoEncode("a b&c=d?e#f+g/h%i\nj") == "a%20b%26c%3Dd%3Fe%23f%2Bg%2Fh%25i%0D%0Aj")
        #expect(ProblemReportBuilder.mailtoEncode("é") == "%C3%A9")
        #expect(ProblemReportBuilder.mailtoEncode("x\r\ny") == "x%0D%0Ay")
    }

    @Test func aShortReportFitsWhole() throws {
        let mail = ProblemReportBuilder.mail(description: "It froze & quit?", category: .chat, includeModel:          true,
                                             context: Self.context())
        let url = try #require(ProblemReportBuilder.mailtoURL(mail))
        let parts = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
        #expect(parts.scheme == "mailto")
        #expect(parts.path == SupportContact.email)
        let items = Dictionary(uniqueKeysWithValues: (parts.queryItems ?? []).map { ($0.name, $0.value ?? "") })
        #expect(items["subject"] == mail.subject)
        #expect(items["body"] == mail.body.replacingOccurrences(of: "\n", with: "\r\n"))
        #expect(!url.absoluteString.contains(ProblemReportBuilder.mailtoEncode("[Trimmed")))
    }

    @Test func aLongReportIsTrimmedWithANote() throws {
        let text = String(repeating: "Привет & 你好 ", count: 400)
        let mail = ProblemReportBuilder.mail(description: text, category: .chat, includeModel:          true,
                                             context: Self.context(failure: Self.failure))
        for limit in [1000, 2000, ProblemReportLimits.mailtoMax] {
            let url = try #require(ProblemReportBuilder.mailtoURL(mail, maxLength: limit))
            #expect(url.absoluteString.count <= limit)
            #expect(url.absoluteString.count > limit - 40, "uses the room it has")
            let parts = try #require(URLComponents(url: url, resolvingAgainstBaseURL: false))
            let body = try #require(parts.queryItems?.first { $0.name == "body" }?.value)
            #expect(body.hasSuffix(ProblemReportBuilder.mailtoTrimNote))
            #expect(body.hasPrefix("Привет & 你好"))
            #expect(parts.queryItems?.first { $0.name == "subject" }?.value == mail.subject)
        }
    }

    @Test func copyReportIsTheWholeMail() {
        let mail = ProblemReportBuilder.mail(description: "x", category: nil, includeModel:          true, context: Self.context())
        let text = ProblemReportBuilder.clipboardText(mail)
        #expect(text.hasPrefix("To: \(SupportContact.email)\nSubject: \(mail.subject)\n\nx\n"))
        #expect(text.contains(mail.body))
    }
}
