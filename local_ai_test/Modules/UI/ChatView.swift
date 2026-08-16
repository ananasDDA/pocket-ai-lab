//
//  ChatView.swift
//  local_ai_test
//
//  Multimodal chat UI. Shows attachment bar gated on the loaded model's
//  capabilities; renders image / audio / video attachments inside bubbles;
//  embeds a live input-level waveform while the mic is active; offers a
//  "Speak response" toggle that pipes assistant text through Kokoro TTS.
//

import SwiftUI
import PhotosUI
import UniformTypeIdentifiers

struct ChatView: View {
    @Binding var selectedTab: Int
    @Binding var focusedModelID: String?
    @State private var vm = ChatViewModel()
    @State private var showModelPicker = false
    @State private var showPhotoPicker = false
    @State private var photoItem: PhotosPickerItem?
    @State private var showVideoPicker = false
    @State private var videoItem: PhotosPickerItem?
    @State private var showAttachmentFileImporter = false
    @State private var showDiagnostics = false
    @State private var showErrorDetail = false
    @State private var showSpeakInfo = false
    @State private var showKokoroMissing = false
    @State private var showLiveMetrics = false
    @Bindable private var liveMetrics = LiveDeviceMetrics.shared
    @AppStorage("chatDisplayStyle") private var chatStyleRaw: String = ChatDisplayStyle.bubble.rawValue
    @Bindable private var recorder = AudioRecorder.shared

    private var chatStyle: ChatDisplayStyle {
        get { ChatDisplayStyle(rawValue: chatStyleRaw) ?? .bubble }
    }

    private func setChatStyle(_ style: ChatDisplayStyle) {
        chatStyleRaw = style.rawValue
    }

    var body: some View {
        // The metrics panel is a LAYOUT sibling of the NavigationStack, not an
        // overlay: when it appears it takes real vertical space and pushes the
        // entire chat screen — nav bar, messages, input bar — downwards.
        VStack(spacing: 0) {
            if showLiveMetrics {
                LiveMetricsPanel(metrics: liveMetrics) {
                    withAnimation(.spring(response: 0.35, dampingFraction: 0.85)) {
                        showLiveMetrics = false
                    }
                }
                .transition(.move(edge: .top).combined(with: .opacity))
                .zIndex(10)
            }

            NavigationStack {
            VStack(spacing: 0) {
                if case .error(let message) = vm.inferenceState {
                    errorBanner(message)
                }
                messageList
                if vm.currentCapabilities.contains(.audioIn) {
                    voiceAttachmentBar
                }
                if !vm.stagedAttachments.isEmpty || vm.pendingStagedAttachment != .none {
                    stagedAttachmentRow
                }
                if recorder.isRecording {
                    recordingBanner
                }
                Divider()
                inputBar
            }
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) { liveMetricsButton }
                ToolbarItem(placement: .principal) { modelSelectorButton }
                ToolbarItem(placement: .primaryAction) { chatOptionsMenu }
            }
            }
        }
        .onChange(of: showLiveMetrics) { _, isVisible in
            if isVisible {
                liveMetrics.start()
            } else {
                liveMetrics.stop()
            }
        }
        .task { await vm.prepareCurrentModel() }
        .alert("Kokoro TTS not installed", isPresented: $showKokoroMissing) {
            Button("Open Models") {
                focusedModelID = ChatViewModel.kokoroCatalogID
                selectedTab = 0
            }
            Button("Cancel", role: .cancel) { }
        } message: {
            Text("""
            Speaking responses requires the Kokoro-82M Core ML model (~200 MB). \
            Install it once and this toggle will turn on automatically.
            """)
        }
        .sheet(isPresented: $showSpeakInfo) {
            SpeakResponseInfoSheet(isPresented: $showSpeakInfo)
                .presentationDetents([.medium, .large])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showModelPicker) {
            ModelPickerSheet(selectedTab: $selectedTab, isPresented: $showModelPicker, vm: vm)
                .presentationDetents([.medium])
                .presentationDragIndicator(.visible)
        }
        .sheet(isPresented: $showDiagnostics) {
            DiagnosticsView()
        }
        .sheet(isPresented: $showErrorDetail) {
            if let failure = InferenceManager.shared.lastError {
                ErrorDetailSheet(failure: failure)
            }
        }
        .photosPicker(isPresented: $showPhotoPicker, selection: $photoItem, matching: .images)
        .onChange(of: photoItem) { _, newItem in
            guard let newItem else { return }
            Task {
                await MainActor.run { vm.pendingStagedAttachment = .loadingPhoto }
                let data = try? await newItem.loadTransferable(type: Data.self)
                await MainActor.run {
                    vm.pendingStagedAttachment = .none
                    if let data { vm.attachImage(data) }
                    photoItem = nil
                }
            }
        }
        .photosPicker(isPresented: $showVideoPicker, selection: $videoItem, matching: .videos)
        .onChange(of: videoItem) { _, newItem in
            guard let newItem else { return }
            Task {
                await MainActor.run { vm.pendingStagedAttachment = .preparingVideo }
                let movie = try? await newItem.loadTransferable(type: MovieTransferable.self)
                var extract: VideoFrameExtractor.ExtractResult?
                if let movie {
                    extract = try? await VideoFrameExtractor.extract(from: movie.url)
                }
                await MainActor.run {
                    vm.pendingStagedAttachment = .none
                    if let extract { vm.attachVideo(extract) }
                    videoItem = nil
                }
            }
        }
        .fileImporter(
            isPresented: $showAttachmentFileImporter,
            allowedContentTypes: attachmentFileImporterTypes,
            allowsMultipleSelection: false
        ) { result in
            handleAttachmentFileImport(result)
        }
    }

    /// Types allowed in the Files picker next to the paperclip.
    private var attachmentFileImporterTypes: [UTType] {
        var types: [UTType] = []
        if vm.currentCapabilities.contains(.imageIn) {
            types.append(.image)
        }
        if vm.currentCapabilities.contains(.videoIn) {
            types.append(contentsOf: [.movie, .video])
        }
        return types
    }

    private var canAttachMediaInComposer: Bool {
        vm.currentCapabilities.contains(.imageIn) || vm.currentCapabilities.contains(.videoIn)
    }

    private func handleAttachmentFileImport(_ result: Result<[URL], Error>) {
        guard case .success(let urls) = result, let url = urls.first else { return }
        Task {
            let accessed = url.startAccessingSecurityScopedResource()
            defer {
                if accessed { url.stopAccessingSecurityScopedResource() }
            }
            let caps = await MainActor.run { vm.currentCapabilities }
            let ct = (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType)
            let looksLikeVideo =
                (ct?.conforms(to: .movie) == true)
                || (ct?.conforms(to: .video) == true)
                || {
                    let ext = url.pathExtension.lowercased()
                    return ["mov", "mp4", "m4v", "avi", "mkv"].contains(ext)
                }()
            if looksLikeVideo {
                guard caps.contains(.videoIn) else { return }
                await MainActor.run { vm.pendingStagedAttachment = .preparingVideo }
                let extract = try? await VideoFrameExtractor.extract(from: url)
                await MainActor.run {
                    vm.pendingStagedAttachment = .none
                    if let extract { vm.attachVideo(extract) }
                }
            } else {
                await MainActor.run { vm.pendingStagedAttachment = .importingFile }
                let data = try? Data(contentsOf: url)
                await MainActor.run {
                    vm.pendingStagedAttachment = .none
                    if let data { vm.attachImage(data) }
                }
            }
        }
    }

    // MARK: - Options menu

    /// Top-right ellipsis menu. Order is deliberate:
    ///  1. Speak response (with an "About" help pane)
    ///  2. Chat style (bubbles vs markdown document)
    ///  3. Testing (self-tests & benchmarks — formerly "Diagnostics")
    ///  4. Clear chat — destructive, always last.
    private var chatOptionsMenu: some View {
        Menu {
            // Hidden while no TTS model exists in the catalog: the Kokoro
            // engine is not production-ready and its catalog entry is pulled
            // for 1.0. When a working entry ships via the remote catalog,
            // this section comes back on its own — no app update needed.
            if ModelCatalog.all.contains(where: { $0.backend == .coreMLKokoro }) {
                Section {
                    Button {
                        toggleSpeakResponses()
                    } label: {
                        Label(
                            vm.speakResponses ? "Speak response · On" : "Speak response · Off",
                            systemImage: vm.speakResponses
                                ? "speaker.wave.2.fill"
                                : "speaker.slash"
                        )
                    }
                    Button {
                        showSpeakInfo = true
                    } label: {
                        Label("About Speak response", systemImage: "questionmark.circle")
                    }
                }
            }

            Section("Chat style") {
                ForEach(ChatDisplayStyle.allCases) { style in
                    Button {
                        setChatStyle(style)
                    } label: {
                        Label(
                            style.label,
                            systemImage: chatStyle == style ? "checkmark" : style.systemImage
                        )
                    }
                }
            }

            Section {
                Button {
                    showDiagnostics = true
                } label: {
                    Label("Testing", systemImage: "testtube.2")
                }
            }

            // Destructive action is always last so it can't be hit by accident.
            Section {
                Button(role: .destructive) {
                    vm.clearChat()
                } label: {
                    Label("Clear chat", systemImage: "trash")
                }
            }
        } label: {
            Image(systemName: "ellipsis.circle")
        }
    }

    // MARK: - Live metrics toolbar button

    /// Left-side toolbar button that pulls down the live-metrics overlay.
    /// Visually symmetric to the right-side ellipsis menu.
    private var liveMetricsButton: some View {
        Button {
            withAnimation(.spring(response: 0.38, dampingFraction: 0.85)) {
                showLiveMetrics.toggle()
            }
        } label: {
            Image(systemName: "waveform.path.ecg")
                .symbolRenderingMode(.hierarchical)
                .foregroundStyle(showLiveMetrics ? Color.accentColor : Color.primary)
        }
        .accessibilityLabel(showLiveMetrics ? "Hide live metrics" : "Show live metrics")
    }

    private func toggleSpeakResponses() {
        let next = !vm.speakResponses
        if !vm.trySetSpeakResponses(next) {
            showKokoroMissing = true
        }
    }

    // MARK: - Error banner

    /// Slim tappable strip under the toolbar. Lives OUTSIDE the model
    /// selector so an error never blocks switching to another model.
    private func errorBanner(_ message: String) -> some View {
        Button {
            if InferenceManager.shared.lastError != nil {
                showErrorDetail = true
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: "exclamationmark.triangle.fill")
                Text(message)
                    .lineLimit(1)
                Spacer(minLength: 4)
                Image(systemName: "info.circle")
            }
            .font(.caption)
            .foregroundStyle(.red)
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color.red.opacity(0.1))
        }
        .buttonStyle(.plain)
    }

    // MARK: - Model selector (toolbar center)

    private var modelSelectorButton: some View {
        Button {
            showModelPicker = true
        } label: {
            VStack(spacing: 1) {
                HStack(spacing: 4) {
                    Text(vm.selectedModel?.name ?? "No model selected")
                        .font(.headline)
                    Image(systemName: "chevron.down")
                        .font(.caption).bold()
                }
                .foregroundStyle(.primary)

                inferenceStateLabel
            }
        }
    }

    @ViewBuilder
    private var inferenceStateLabel: some View {
        switch vm.inferenceState {
        case .loading:
            HStack(spacing: 4) {
                ProgressView().scaleEffect(0.5)
                Text("Loading model...")
            }
            .font(.caption2)
            .foregroundStyle(.orange)
        case .compiling(let p):
            HStack(spacing: 4) {
                ProgressView().scaleEffect(0.5)
                Text("Compiling Core ML model... \(Int(p * 100))%")
            }
            .font(.caption2)
            .foregroundStyle(.orange)
        case .loadingWeights(let p):
            HStack(spacing: 4) {
                ProgressView().scaleEffect(0.5)
                Text("Loading weights \(Int(p * 100))%")
            }
            .font(.caption2)
            .foregroundStyle(.orange)
        case .error:
            // Plain text, NOT a button: this label lives inside the model
            // selector button, and a nested button used to swallow the tap —
            // making it impossible to open the picker while an error was
            // showing. Details live in the banner below the toolbar.
            Text("Model failed to load")
                .font(.caption2)
                .foregroundStyle(.red)
        case .generating:
            Text("Generating...")
                .font(.caption2)
                .foregroundStyle(.blue)
        case .ready:
            Text("Ready")
                .font(.caption2)
                .foregroundStyle(.green)
        case .idle:
            EmptyView()
        }
    }

    // MARK: - Voice bar (photo / video attach moved to paperclip in the input bar)

    private var voiceAttachmentBar: some View {
        HStack(spacing: 16) {
            AttachmentButton(systemImage: recorder.isRecording ? "mic.fill" : "mic", label: recorder.isRecording ? "Stop" : "Voice") {
                Task { await toggleRecording() }
            }
            .foregroundStyle(recorder.isRecording ? Color.red : Color.accentColor)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.top, 6)
    }

    private var stagedAttachmentRow: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Array(vm.stagedAttachments.enumerated()), id: \.offset) { idx, attachment in
                    AttachmentChip(attachment: attachment) {
                        vm.removeAttachment(at: idx)
                    }
                }
                if vm.pendingStagedAttachment != .none {
                    PendingStagedAttachmentChip(kind: vm.pendingStagedAttachment)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 6)
        }
    }

    private var recordingBanner: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(Color.red)
                .frame(width: 8, height: 8)
                .opacity(0.9)
                .scaleEffect(recorder.level > 0.01 ? 1.0 + Double(recorder.level) * 0.6 : 1.0)
                .animation(.easeInOut(duration: 0.12), value: recorder.level)
            Text("Recording…")
                .font(.caption)
                .foregroundStyle(.red)
            WaveformBar(level: recorder.level)
            Spacer()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 4)
    }

    private func toggleRecording() async {
        if recorder.isRecording {
            let pcm = recorder.stop()
            if !pcm.isEmpty {
                vm.attachAudio(pcm: pcm, sampleRate: 16_000)
            }
        } else {
            try? await recorder.record()
        }
    }

    // MARK: - Message list

    private var messageList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(spacing: chatStyle == .document ? 20 : 12) {
                    if vm.messages.isEmpty {
                        emptyState
                    } else {
                        ForEach(vm.messages) { msg in
                            MessageRowView(message: msg, style: chatStyle)
                                .id(msg.id)
                        }
                        if vm.isGenerating, let last = vm.messages.last, last.text.isEmpty {
                            TypingIndicator(style: chatStyle).id("typing")
                        }
                    }
                }
                .padding(chatStyle == .document ? EdgeInsets(top: 16, leading: 20, bottom: 16, trailing: 20)
                                                 : EdgeInsets(top: 16, leading: 16, bottom: 16, trailing: 16))
            }
            .scrollDismissesKeyboard(.interactively)
            .onChange(of: vm.messages.count) {
                withAnimation {
                    if let lastId = vm.messages.last?.id {
                        proxy.scrollTo(lastId, anchor: .bottom)
                    }
                }
            }
            .onChange(of: vm.isGenerating) {
                if vm.isGenerating {
                    withAnimation { proxy.scrollTo("typing", anchor: .bottom) }
                }
            }
        }
    }

    private var emptyState: some View {
        VStack(spacing: 10) {
            Image(systemName: "bubble.left.and.bubble.right")
                .font(.system(size: 44))
                .foregroundStyle(.secondary.opacity(0.4))
            if let model = vm.selectedModel {
                Text("Start a conversation").font(.headline)
                Text("Ask anything — \(model.name) is ready.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            } else {
                Text("No model installed").font(.headline)
                Text("Go to the Models tab to find and install a model.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                Button("Go to Models") { selectedTab = 0 }
                    .buttonStyle(.borderedProminent)
                    .padding(.top, 4)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.top, 80)
    }

    // MARK: - Input bar

    private var inputBar: some View {
        HStack(spacing: 10) {
            if canAttachMediaInComposer {
                Menu {
                    if vm.currentCapabilities.contains(.imageIn) {
                        Button {
                            showPhotoPicker = true
                        } label: {
                            Label("Photo", systemImage: "photo")
                        }
                    }
                    if vm.currentCapabilities.contains(.videoIn) {
                        Button {
                            showVideoPicker = true
                        } label: {
                            Label("Video", systemImage: "film")
                        }
                    }
                    Button {
                        showAttachmentFileImporter = true
                    } label: {
                        Label("File", systemImage: "doc")
                    }
                } label: {
                    Image(systemName: "paperclip.circle.fill")
                        .font(.system(size: 32))
                        .symbolRenderingMode(.hierarchical)
                        .foregroundStyle(
                            vm.pendingStagedAttachment != .none ? Color.secondary : Color.accentColor
                        )
                }
                .accessibilityLabel("Attach")
                .disabled(vm.pendingStagedAttachment != .none)
            }

            TextField("Message", text: $vm.inputText, axis: .vertical)
                .lineLimit(1...5)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
                .onSubmit {
                    if !vm.isGenerating { vm.send() }
                }

            Button {
                if vm.isGenerating {
                    vm.cancelGeneration()
                } else {
                    vm.send()
                }
            } label: {
                Image(systemName: vm.isGenerating ? "stop.circle.fill" : "arrow.up.circle.fill")
                    .font(.system(size: 32))
                    .foregroundStyle(
                        vm.isGenerating
                            ? Color.red
                            : ((vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                && vm.stagedAttachments.isEmpty)
                                || vm.pendingStagedAttachment != .none
                                ? Color.secondary
                                : Color.blue)
                    )
            }
            .disabled(
                !vm.isGenerating
                    && (vm.pendingStagedAttachment != .none
                        || (vm.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            && vm.stagedAttachments.isEmpty))
            )
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(.bar)
    }
}

// MARK: - Helper Views

private struct AttachmentButton: View {
    let systemImage: String
    let label: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(spacing: 2) {
                Image(systemName: systemImage)
                    .font(.title3)
                Text(label)
                    .font(.caption2)
            }
        }
    }
}

private struct AttachmentChip: View {
    let attachment: ChatAttachment
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 6) {
            icon
            Text(label)
                .font(.caption)
            Button(action: onRemove) {
                Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.regularMaterial, in: Capsule())
    }

    @ViewBuilder
    private var icon: some View {
        switch attachment {
        case .image: Image(systemName: "photo").foregroundStyle(.blue)
        case .audio: Image(systemName: "waveform").foregroundStyle(.green)
        case .videoFrames: Image(systemName: "video").foregroundStyle(.orange)
        }
    }

    private var label: String {
        switch attachment {
        case .image(let d):
            return "Image (\(byteLabel(d.count)))"
        case .audio(let d, let sr, _):
            let secs = Double(d.count / MemoryLayout<Float>.size) / Double(sr)
            return String(format: "Audio %.1fs", secs)
        case .videoFrames(let frames, let duration):
            return "Video \(frames.count) frames · \(Int(duration))s"
        }
    }

    private func byteLabel(_ bytes: Int) -> String {
        if bytes > 1_000_000 { return String(format: "%.1f MB", Double(bytes) / 1_000_000) }
        return "\(bytes / 1000) KB"
    }
}

/// Placeholder in the staged-attachment strip while async work runs.
private struct PendingStagedAttachmentChip: View {
    let kind: PendingStagedAttachmentKind

    var body: some View {
        HStack(spacing: 8) {
            ProgressView()
                .scaleEffect(0.85)
            Text(caption)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .combine)
        .accessibilityLabel(caption)
    }

    private var caption: String {
        switch kind {
        case .none: return ""
        case .loadingPhoto: return "Loading photo…"
        case .preparingVideo: return "Preparing video…"
        case .importingFile: return "Importing file…"
        }
    }
}

private struct WaveformBar: View {
    let level: Float

    var body: some View {
        HStack(spacing: 2) {
            ForEach(0..<12, id: \.self) { i in
                let threshold = Float(i) / 12.0
                RoundedRectangle(cornerRadius: 1)
                    .fill(level > threshold ? Color.accentColor : Color.secondary.opacity(0.2))
                    .frame(width: 3, height: CGFloat(6 + i * 2))
            }
        }
    }
}

private struct MessageRowView: View {
    let message: ChatMessage
    let style: ChatDisplayStyle

    var body: some View {
        switch style {
        case .bubble:    bubbleLayout
        case .document:  documentLayout
        }
    }

    // MARK: - Bubble layout (iMessage-style)

    private var bubbleLayout: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 48) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                if !message.attachments.isEmpty {
                    AttachmentPreviewStrip(attachments: message.attachments, alignTrailing: message.role == .user)
                }
                if !message.text.isEmpty {
                    // `Text(.init(...))` parses inline Markdown (bold, italic,
                    // code, links). Good enough for a bubble; block-level
                    // Markdown is handled by the document style instead.
                    Text(.init(message.text))
                        .textSelection(.enabled)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            message.role == .user ? Color.blue : Color(.secondarySystemBackground),
                            in: RoundedRectangle(cornerRadius: 18)
                        )
                        .foregroundStyle(message.role == .user ? .white : .primary)
                }
            }
            if message.role == .assistant { Spacer(minLength: 48) }
        }
        .frame(maxWidth: .infinity, alignment: message.role == .user ? .trailing : .leading)
    }

    // MARK: - Document layout

    @ViewBuilder
    private var documentLayout: some View {
        switch message.role {
        case .user:
            // User turns still look like a compact bubble, anchored right,
            // so the turn boundary is obvious at a glance.
            HStack(alignment: .top) {
                Spacer(minLength: 32)
                VStack(alignment: .trailing, spacing: 6) {
                    if !message.attachments.isEmpty {
                        AttachmentPreviewStrip(attachments: message.attachments, alignTrailing: true)
                    }
                    if !message.text.isEmpty {
                        Text(.init(message.text))
                            .textSelection(.enabled)
                            .padding(.horizontal, 14)
                            .padding(.vertical, 10)
                            .background(Color.blue, in: RoundedRectangle(cornerRadius: 18))
                            .foregroundStyle(.white)
                    }
                }
            }

        case .assistant:
            // Assistant replies take the full column width without a bubble
            // or header chrome — user/assistant turns are already visually
            // distinct (user is a right-aligned blue bubble). Content is
            // rendered as block-level Markdown so headings, lists, and
            // fenced code blocks read like a document.
            VStack(alignment: .leading, spacing: 10) {
                if !message.attachments.isEmpty {
                    AttachmentPreviewStrip(attachments: message.attachments, alignTrailing: false)
                }
                if !message.text.isEmpty {
                    MarkdownText(raw: message.text)
                        .textSelection(.enabled)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

/// Lightweight block-level Markdown renderer. Splits on fenced ``` code
/// blocks and renders each remaining paragraph via
/// `AttributedString(markdown:)` so bold / italic / inline code / links /
/// lists all work without pulling in a third-party dependency.
private struct MarkdownText: View {
    let raw: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, block in
                switch block {
                case .code(let content, let language):
                    CodeBlockView(content: content, language: language)
                case .paragraph(let content):
                    Text(attributed(content))
                        .fixedSize(horizontal: false, vertical: true)
                        .foregroundStyle(.primary)
                }
            }
        }
    }

    private enum Block {
        case paragraph(String)
        case code(String, language: String?)
    }

    private var blocks: [Block] {
        var result: [Block] = []
        var iterator = raw.components(separatedBy: "\n").makeIterator()
        var paragraphBuffer: [String] = []

        func flushParagraph() {
            let joined = paragraphBuffer.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            paragraphBuffer.removeAll()
            if !joined.isEmpty { result.append(.paragraph(joined)) }
        }

        while let line = iterator.next() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                flushParagraph()
                let language = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                var codeLines: [String] = []
                while let codeLine = iterator.next() {
                    if codeLine.trimmingCharacters(in: .whitespaces) == "```" { break }
                    codeLines.append(codeLine)
                }
                result.append(.code(codeLines.joined(separator: "\n"), language: language.isEmpty ? nil : language))
            } else {
                paragraphBuffer.append(line)
            }
        }
        flushParagraph()
        return result
    }

    private func attributed(_ text: String) -> AttributedString {
        // `.full` accepts block-ish syntax (lists, headings) best-effort.
        // On failure we fall back to plain text rather than crashing mid-stream.
        if let attr = try? AttributedString(
            markdown: text,
            options: .init(
                interpretedSyntax: .inlineOnlyPreservingWhitespace,
                failurePolicy: .returnPartiallyParsedIfPossible
            )
        ) {
            return attr
        }
        return AttributedString(text)
    }
}

private struct CodeBlockView: View {
    let content: String
    let language: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let language, !language.isEmpty {
                Text(language.uppercased())
                    .font(.caption2).bold()
                    .foregroundStyle(.secondary)
            }
            ScrollView(.horizontal, showsIndicators: false) {
                Text(content)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(10)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10))
        }
    }
}

private struct AttachmentPreviewStrip: View {
    let attachments: [ChatAttachment]
    /// User-sent media should hug the right edge like the blue bubble; a bare
    /// `ScrollView` expands to full width and lays out its `HStack` from the
    /// leading edge, which reads as "incoming" rather than "outgoing".
    var alignTrailing: Bool = false

    var body: some View {
        Group {
            if alignTrailing {
                HStack(spacing: 0) {
                    Spacer(minLength: 0)
                    stripScroll
                        .fixedSize(horizontal: true, vertical: false)
                }
            } else {
                stripScroll
            }
        }
    }

    private var stripScroll: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(attachments.enumerated()), id: \.offset) { _, attachment in
                    preview(for: attachment)
                }
            }
        }
    }

    @ViewBuilder
    private func preview(for attachment: ChatAttachment) -> some View {
        switch attachment {
        case .image(let data):
            if let ui = UIImage(data: data) {
                Image(uiImage: ui)
                    .resizable()
                    .scaledToFill()
                    .frame(width: 80, height: 80)
                    .clipShape(RoundedRectangle(cornerRadius: 10))
            }
        case .audio(let data, let sr, _):
            let secs = Double(data.count / MemoryLayout<Float>.size) / Double(sr)
            HStack(spacing: 6) {
                Image(systemName: "waveform.circle.fill").font(.title2)
                Text(String(format: "%.1fs", secs)).font(.caption)
            }
            .padding(10)
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
        case .videoFrames(let frames, _):
            HStack(spacing: 2) {
                ForEach(Array(frames.prefix(4).enumerated()), id: \.offset) { _, frame in
                    if let ui = UIImage(data: frame) {
                        Image(uiImage: ui)
                            .resizable()
                            .scaledToFill()
                            .frame(width: 40, height: 40)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                    }
                }
            }
        }
    }
}

private struct TypingIndicator: View {
    var style: ChatDisplayStyle = .bubble
    @State private var phase = 0

    var body: some View {
        let dots = HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { i in
                Circle()
                    .frame(width: 7, height: 7)
                    .foregroundStyle(.secondary)
                    .scaleEffect(phase == i ? 1.3 : 1.0)
            }
        }

        return Group {
            switch style {
            case .bubble:
                dots
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18))
                    .frame(maxWidth: .infinity, alignment: .leading)
            case .document:
                dots
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.4).repeatForever().delay(0)) {
                phase = (phase + 1) % 3
            }
        }
    }
}

// MARK: - Movie transferable for PhotosPicker

private struct MovieTransferable: Transferable {
    let url: URL

    static var transferRepresentation: some TransferRepresentation {
        FileRepresentation(contentType: .movie) { movie in
            SentTransferredFile(movie.url)
        } importing: { received in
            let copy = FileManager.default.temporaryDirectory
                .appendingPathComponent(UUID().uuidString + ".mov")
            try FileManager.default.copyItem(at: received.file, to: copy)
            return MovieTransferable(url: copy)
        }
    }
}

// MARK: - Speak Response Info Sheet

private struct SpeakResponseInfoSheet: View {
    @Binding var isPresented: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    Label("Speak response", systemImage: "speaker.wave.2.fill")
                        .font(.title2).bold()
                        .foregroundStyle(.blue)

                    Text("""
                    When enabled, the assistant's reply is synthesised to speech \
                    in parallel with token generation — you start hearing the \
                    answer as soon as the first sentence is ready.
                    """)

                    sectionHeader("How it works")
                    bullet("The active LLM streams text tokens as usual.")
                    bullet("Each completed sentence is passed to the on-device Kokoro-82M Core ML voice.")
                    bullet("Kokoro renders a short PCM chunk which `AudioPlayer` queues and plays back seamlessly.")
                    bullet("Nothing leaves the device — the whole pipeline is local.")

                    sectionHeader("Requirements")
                    bullet("The Kokoro-82M Core ML model must be installed (~200 MB) — available on the Models tab.")
                    bullet("Works with any text LLM. Does not interfere with vision or audio-in pipelines.")

                    sectionHeader("Tips")
                    bullet("Turn it off if you want plain text streaming without audio latency.")
                    bullet("Use Clear chat to reset both the transcript and the LLM's KV cache.")
                }
                .padding(20)
            }
            .navigationTitle("About Speak response")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { isPresented = false }
                }
            }
        }
    }

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.headline)
            .padding(.top, 4)
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•").foregroundStyle(.secondary)
            Text(text)
        }
    }
}

// MARK: - Model Picker Sheet

private struct ModelPickerSheet: View {
    @Binding var selectedTab: Int
    @Binding var isPresented: Bool
    let vm: ChatViewModel
    @State private var unavailableBackend: ModelBackend?

    var body: some View {
        NavigationStack {
            Group {
                if vm.installedModels.isEmpty {
                    noModelsView
                } else {
                    modelList
                }
            }
            .navigationTitle("Select Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { isPresented = false }
                }
            }
            .alert(item: $unavailableBackend) { backend in
                Alert(
                    title: Text("\(backend.displayName) not available"),
                    message: Text(backend.unavailableReason
                        ?? "This backend is not linked into the current build."),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
    }

    private var modelList: some View {
        List {
            Section("Installed") {
                ForEach(vm.installedModels) { model in
                    Button {
                        if model.backend.isLinked {
                            vm.selectModel(model)
                            isPresented = false
                        } else {
                            unavailableBackend = model.backend
                        }
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack(spacing: 6) {
                                    Text(model.name)
                                        .foregroundStyle(model.backend.isLinked ? .primary : .secondary)
                                    CapabilityIcons(capabilities: model.capabilities)
                                }
                                Text(model.parameterSize + " · " + model.quantization + " · " + model.backend.displayName)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                if let badge = model.backend.unavailableBadge {
                                    HStack(spacing: 4) {
                                        Image(systemName: "exclamationmark.triangle.fill")
                                            .imageScale(.small)
                                        Text(badge)
                                    }
                                    .font(.caption2)
                                    .foregroundStyle(.orange)
                                }
                            }
                            Spacer()
                            if vm.selectedModel?.id == model.id {
                                Image(systemName: "checkmark").foregroundStyle(.blue)
                            }
                        }
                    }
                }
            }

            Section {
                Button {
                    isPresented = false
                    selectedTab = 0
                } label: {
                    Label("Find & download more models", systemImage: "arrow.down.circle")
                }
            }
        }
    }

    private var noModelsView: some View {
        VStack(spacing: 14) {
            Image(systemName: "cpu")
                .font(.system(size: 44))
                .foregroundStyle(.secondary.opacity(0.4))
            Text("No models installed").font(.headline)
            Text("Browse the Models tab to find the best model for your device.")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
            Button("Browse Models") {
                isPresented = false
                selectedTab = 0
            }
            .buttonStyle(.borderedProminent)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

struct CapabilityIcons: View {
    let capabilities: ModelCapabilities

    var body: some View {
        HStack(spacing: 2) {
            if capabilities.contains(.imageIn) { Image(systemName: "photo").imageScale(.small).foregroundStyle(.blue) }
            if capabilities.contains(.videoIn) { Image(systemName: "video").imageScale(.small).foregroundStyle(.orange) }
            if capabilities.contains(.audioIn) { Image(systemName: "mic").imageScale(.small).foregroundStyle(.green) }
            if capabilities.contains(.audioOut) { Image(systemName: "speaker.wave.2").imageScale(.small).foregroundStyle(.purple) }
        }
    }
}
