//
//  ReportProblemView.swift
//  local_ai_test
//
//  "Report a problem": the user writes what happened, sees every line of
//  diagnostics that goes with it, and continues in their own mail app, where
//  they review the message and send it themselves. Without a mail account
//  the same report opens as a mailto: link, and "Copy report" is always
//  there as the last resort. Nothing goes through our servers.
//
//  Reached from Testing, from the error details sheet (with that error) and
//  from the chat's options menu (with the selected model).
//

import SwiftUI
import MessageUI

struct ReportProblemView: View {
    let context: ProblemReportContext
    @State private var text = ""
    @State private var category: ProblemReportCategory?
    @State private var includeModel = true
    @State private var showDiagnostics = false
    @State private var showComposer = false
    @State private var copied = false
    @State private var notice: String?
    @State private var showNoMailApp = false
    @Environment(\.dismiss) private var dismiss
    @Environment(\.openURL) private var openURL

    init(context: ProblemReportContext, category: ProblemReportCategory? = nil) {
        self.context = context
        _category = State(initialValue: category)
    }

    private var mail: ProblemReportMail {
        ProblemReportBuilder.mail(description: text, category: category, includeModel: includeModel,
                                  context: context)
    }

    private var rows: [(label: String, value: String)] {
        ProblemReportBuilder.rows(context: context, category: category, includeModel: includeModel)
    }

    private var canSend: Bool { ProblemReportBuilder.canSend(description: text) }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $text)
                        .font(.callout)
                        .frame(minHeight: 140)
                        .overlay(alignment: .topLeading) {
                            if text.isEmpty {
                                Text("What happened, and what did you expect instead?")
                                    .font(.callout)
                                    .foregroundStyle(.tertiary)
                                    .padding(.top, 8)
                                    .padding(.leading, 5)
                                    .allowsHitTesting(false)
                            }
                        }
                        .onChange(of: text) { _, new in
                            if new.count > ProblemReportLimits.descriptionMax {
                                text = String(new.prefix(ProblemReportLimits.descriptionMax))
                            }
                        }
                } header: {
                    Text("What happened")
                } footer: {
                    Text("\(text.count) / \(ProblemReportLimits.descriptionMax). "
                         + "No conversation content is included unless you paste it here.")
                }

                Section("What were you doing") {
                    Picker("Area", selection: $category) {
                        Text("Not specified").tag(ProblemReportCategory?.none)
                        ForEach(ProblemReportCategory.allCases) { item in
                            Text(item.label).tag(ProblemReportCategory?.some(item))
                        }
                    }
                }

                Section {
                    if context.model != nil {
                        Toggle("Include model name", isOn: $includeModel)
                    }
                    DisclosureGroup(isExpanded: $showDiagnostics) {
                        ForEach(rows, id: \.label) { row in
                            LabeledContent(row.label) {
                                Text(row.value)
                                    .font(.system(.caption, design: .monospaced))
                                    .multilineTextAlignment(.trailing)
                                    .textSelection(.enabled)
                            }
                        }
                        if let detail = ProblemReportBuilder.errorDetail(
                            context: context, includeModel: includeModel,
                            limit: ProblemReportLimits.attachmentErrorDetailMax) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Error detail")
                                Text(detail)
                                    .font(.system(.caption2, design: .monospaced))
                                    .foregroundStyle(.secondary)
                                    .textSelection(.enabled)
                            }
                        }
                    } label: {
                        Text("Diagnostics (\(rows.count) items)")
                    }
                } header: {
                    Text("Will be sent")
                } footer: {
                    Text("Exactly this is added below your text, plus a diagnostics.txt file with the same "
                         + "lines. Mail opens with everything filled in, so you can read it, change it, "
                         + "and send it from your own address to \(SupportContact.email).")
                }

                Section {
                    Button {
                        continueInMail()
                    } label: {
                        Label("Continue in Mail", systemImage: "envelope")
                    }
                    .disabled(!canSend)

                    Button {
                        copyReport()
                    } label: {
                        Label(copied ? "Copied" : "Copy report", systemImage: copied ? "checkmark" : "doc.on.doc")
                    }
                    .disabled(!canSend)
                } footer: {
                    VStack(alignment: .leading, spacing: 6) {
                        if let notice {
                            Text(notice).foregroundStyle(.primary)
                        }
                        Text(LocalizedStringKey("Sending a report is optional and only happens when you send "
                            + "the mail. [Privacy policy](\(SupportContact.privacyURL.absoluteString))"))
                    }
                }
            }
            .navigationTitle("Report a problem")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
            }
            .sheet(isPresented: $showComposer) {
                MailComposeView(mail: mail) { result in
                    showComposer = false
                    handle(result)
                }
                .ignoresSafeArea()
            }
            .alert("No mail app available", isPresented: $showNoMailApp) {
                Button("Copy report") { copyReport() }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Copy the report and send it to \(SupportContact.email) from any app.")
            }
        }
    }

    private func continueInMail() {
        notice = nil
        if MFMailComposeViewController.canSendMail() {
            showComposer = true
            return
        }
        guard let url = ProblemReportBuilder.mailtoURL(mail) else {
            showNoMailApp = true
            return
        }
        openURL(url) { accepted in
            if !accepted { showNoMailApp = true }
        }
    }

    private func copyReport() {
        UIPasteboard.general.string = ProblemReportBuilder.clipboardText(mail)
        withAnimation { copied = true }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
            withAnimation { copied = false }
        }
    }

    private func handle(_ result: MFMailComposeResult) {
        switch result {
        case .sent:
            notice = "Thanks. Mail is sending your report."
        case .saved:
            notice = "Saved as a draft in Mail."
        case .failed:
            notice = "Mail could not send the report. Try again, or use Copy report."
        case .cancelled:
            break
        @unknown default:
            break
        }
    }
}

// MARK: - Mail composer (UIKit bridge)

private struct MailComposeView: UIViewControllerRepresentable {
    let mail: ProblemReportMail
    let onFinish: (MFMailComposeResult) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(onFinish: onFinish) }

    func makeUIViewController(context: Context) -> MFMailComposeViewController {
        let controller = MFMailComposeViewController()
        controller.mailComposeDelegate = context.coordinator
        controller.setToRecipients([mail.recipient])
        controller.setSubject(mail.subject)
        controller.setMessageBody(mail.body, isHTML: false)
        controller.addAttachmentData(Data(mail.attachmentText.utf8), mimeType: "text/plain",
                                     fileName: ProblemReportMail.attachmentName)
        return controller
    }

    func updateUIViewController(_ controller: MFMailComposeViewController, context: Context) {}

    final class Coordinator: NSObject, MFMailComposeViewControllerDelegate {
        let onFinish: (MFMailComposeResult) -> Void
        init(onFinish: @escaping (MFMailComposeResult) -> Void) { self.onFinish = onFinish }

        func mailComposeController(_ controller: MFMailComposeViewController,
                                   didFinishWith result: MFMailComposeResult, error: Error?) {
            onFinish(result)
        }
    }
}
