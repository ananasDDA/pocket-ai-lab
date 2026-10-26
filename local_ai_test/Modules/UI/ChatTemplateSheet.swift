//
//  ChatTemplateSheet.swift
//  local_ai_test
//
//  Two windows onto what the model is really being asked.
//
//  `ChatTemplateSheet` - which template the app is using for this model,
//  its text, and a place to paste your own.
//  `PromptInspectorSheet` - the exact transcript one message produced.
//
//  Both exist because the chat format is invisible and decides answer
//  quality: an imported model formatted by the wrong family answers worse
//  for a reason nothing on screen used to explain.
//

import SwiftUI

struct ChatTemplateSheet: View {

    let model: AIModel?

    @Environment(\.dismiss) private var dismiss
    @State private var embedded: String?
    @State private var draft: String = ""
    @State private var loaded = false
    @State private var rejected = false
    @State private var savedNotice = false

    private var store: ChatTemplateStore { .shared }

    private var source: ChatTemplateSource {
        guard let model else { return .fallback(.plain) }
        if model.backend == .appleIntelligence { return .session }
        return ChatTemplateResolver.source(modelID: model.id, embedded: embedded)
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("In use", value: source.label)
                    if let model {
                        LabeledContent("Model", value: model.name)
                    }
                } footer: {
                    Text(explanation)
                }

                if let embedded {
                    Section("Template in the model file") {
                        Text(embedded)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                    }
                } else if loaded {
                    Section {
                        Label(
                            "This model file carries no chat template.",
                            systemImage: "exclamationmark.triangle"
                        )
                        .font(.callout)
                    }
                }

                Section {
                    TextEditor(text: $draft)
                        .font(.system(.caption, design: .monospaced))
                        .frame(minHeight: 160)
                    if rejected {
                        Label(
                            "llama.cpp does not implement this template, so it "
                            + "was not saved. Paste the template from the model's "
                            + "Hugging Face page, or leave this empty to use the "
                            + "model's own.",
                            systemImage: "xmark.octagon"
                        )
                        .font(.caption)
                        .foregroundStyle(.red)
                    }
                    if savedNotice {
                        Label("Saved. It applies to your next message.", systemImage: "checkmark.circle")
                            .font(.caption)
                            .foregroundStyle(.green)
                    }
                    HStack {
                        Button("Use this template") { save() }
                            .disabled(model == nil || draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        Spacer()
                        Button("Clear", role: .destructive) { clear() }
                            .disabled(model.map { !store.hasCustom(for: $0.id) } ?? true)
                    }
                } header: {
                    Text("Custom template")
                } footer: {
                    Text("Takes priority over the model's own. llama.cpp renders "
                         + "the formats it implements natively, so a template it "
                         + "does not recognise is refused rather than silently "
                         + "producing a transcript nobody checked.")
                }
            }
            .navigationTitle("Chat template")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .task {
                embedded = await InferenceManager.shared.currentChatTemplate()
                if let model { draft = store.custom(for: model.id) ?? "" }
                loaded = true
            }
        }
    }

    private var explanation: String {
        switch source {
        case .model:
            return "The transcript is built with the template stored in the "
                + "model file - the format this model was trained on."
        case .custom:
            return "The transcript is built with the template you set for this model."
        case .fallback(let family):
            return "This model file carries no template llama.cpp can use, so "
                + "the app formats the conversation as \(family.rawValue). That is a "
                + "guess from the model's name and it can cost answer quality; "
                + "paste the model's own template below to be sure."
        case .session:
            return "Apple's framework formats the conversation itself. The "
                + "system prompt becomes the session's instructions; a custom "
                + "template does not apply."
        }
    }

    private func save() {
        guard let model else { return }
        let candidate = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard LlamaChatTemplate.isSupported(candidate) else {
            rejected = true
            savedNotice = false
            return
        }
        store.setCustom(candidate, for: model.id)
        rejected = false
        savedNotice = true
    }

    private func clear() {
        guard let model else { return }
        store.setCustom(nil, for: model.id)
        draft = ""
        rejected = false
        savedNotice = false
    }
}

// MARK: - System prompt

/// The instructions every chat starts with (`SystemPromptStore`): read them,
/// change them, turn them off, or go back to the app's default. One setting
/// for all models.
struct SystemPromptSheet: View {

    @Environment(\.dismiss) private var dismiss
    @State private var draft: String = ""
    @State private var savedNotice = false

    private var store: SystemPromptStore { .shared }

    private var trimmedDraft: String {
        draft.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var status: String {
        if store.isOff { return "Off" }
        return store.isDefault ? "Default" : "Custom"
    }

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    LabeledContent("In use", value: status)
                } footer: {
                    Text("Sent as the first message of every chat, with every model. "
                         + "Models without a system role get it at the start of "
                         + "your first message instead. Benchmarks never use it.")
                }

                Section {
                    TextEditor(text: $draft)
                        .font(.callout)
                        .frame(minHeight: 140)
                        .onChange(of: draft) { savedNotice = false }
                    if savedNotice {
                        Label("Saved. It applies to your next message.", systemImage: "checkmark.circle")
                            .font(.caption)
                            .foregroundStyle(.green)
                    }
                    HStack {
                        Button("Save") { save() }
                            .disabled(trimmedDraft == store.editableText)
                        Spacer()
                        Button("Reset to default") { reset() }
                            .disabled(store.isDefault && trimmedDraft == SystemPrompt.defaultText)
                    }
                } header: {
                    Text("System prompt")
                } footer: {
                    Text("Leave it empty and save to send no system prompt at all.")
                }

                Section("App default") {
                    Text(SystemPrompt.defaultText)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }
            }
            .navigationTitle("System prompt")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .onAppear { draft = store.editableText }
        }
    }

    private func save() {
        store.set(draft)
        draft = store.editableText
        savedNotice = true
    }

    private func reset() {
        store.resetToDefault()
        draft = store.editableText
        savedNotice = true
    }
}

// MARK: - Prompt inspector

struct PromptInspectorSheet: View {

    let prompt: RenderedPrompt
    /// What was attached to the message, shown above the transcript: the
    /// transcript only carries a placeholder where an image went.
    var attachments: [ChatAttachment] = []

    @Environment(\.dismiss) private var dismiss
    @State private var copied = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    if !attachments.isEmpty {
                        attachmentsSection
                    }
                    HStack(spacing: 8) {
                        Label(prompt.source.label, systemImage: prompt.source.isAuthoritative
                              ? "checkmark.seal"
                              : "questionmark.circle")
                            .font(.caption)
                            .foregroundStyle(prompt.source.isAuthoritative ? .green : .orange)
                        Spacer()
                        Text("\(prompt.text.count) chars")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }

                    Text(prompt.source == .session
                         ? "What went to Apple's framework: the session's "
                           + "instructions (the system prompt) and the new message. "
                           + "The session keeps the earlier turns itself."
                         : "Everything below went to the model as one string: the "
                           + "system prompt, the whole conversation so far, the chat "
                           + "markers, and the image placeholders.")
                        .font(.caption)
                        .foregroundStyle(.secondary)

                    Text(prompt.text)
                        .font(.system(.footnote, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(10)
                        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
                }
                .padding(16)
            }
            .navigationTitle("Prompt sent")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button(copied ? "Copied" : "Copy") {
                        UIPasteboard.general.string = prompt.text
                        copied = true
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    /// Images large enough to read, in the order they were attached; audio
    /// and video keep the compact chips the chat bubble uses.
    @ViewBuilder
    private var attachmentsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(attachments.count == 1 ? "Attachment" : "Attachments (\(attachments.count))")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(Array(attachments.enumerated()), id: \.offset) { _, attachment in
                if case .image(let data) = attachment, let image = UIImage(data: data) {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                        .frame(maxWidth: .infinity, maxHeight: 280)
                        .clipShape(RoundedRectangle(cornerRadius: 12))
                } else {
                    AttachmentPreviewStrip(attachments: [attachment])
                }
            }
        }
    }
}
