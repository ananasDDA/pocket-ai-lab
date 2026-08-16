//
//  ModelCardView.swift
//  local_ai_test
//

import SwiftUI

struct ModelCardView: View {
    let model: AIModel
    let isRecommended: Bool
    /// When true the card is wrapped in a pulsing blue border. Driven by
    /// `SetupView` after a deep link from the chat "install Kokoro" prompt.
    var isHighlighted: Bool = false
    @State private var showDeleteConfirmation = false
    @State private var highlightPulse = false
    @State private var webLink: WebLink?
    @State private var isResolvingDemo = false

    private var downloader: ModelDownloader { ModelDownloader.shared }
    private var downloadState: ModelDownloadState { downloader.state(for: model) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            // Header, styled after the huggingface.co model cards.
            HFModelHeader(
                repo: model.huggingFaceRepo,
                displayName: model.name,
                parameterSize: model.parameterSize
            )

            HStack(alignment: .center, spacing: 6) {
                if isRecommended {
                    Text("Recommended")
                        .font(.caption2).bold()
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.blue, in: Capsule())
                }
                if model.source == .imported {
                    Text("Imported")
                        .font(.caption2).bold()
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(.purple, in: Capsule())
                }
                Text(model.family + " · " + model.parameterSize + " · " + model.quantization)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                Spacer(minLength: 6)
                qualityBadge
            }

            Divider()

            // Stats
            HStack(spacing: 16) {
                stat(icon: "memorychip", value: ramLabel, label: "RAM")
                stat(icon: "internaldrive", value: diskLabel, label: "Disk")
                stat(icon: "text.alignleft", value: contextFormatted, label: "Context")
                stat(icon: "cpu", value: backendLabel, label: "Engine")
            }

            // Download control
            if model.backend == .appleIntelligence {
                Label("Built-in · Always available", systemImage: "checkmark.seal.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                downloadControl
            }

            tryOnlineControl
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 14)
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: Self.cornerRadius))
        .overlay(
            RoundedRectangle(cornerRadius: Self.cornerRadius)
                .stroke(highlightStrokeColor, lineWidth: highlightStrokeWidth)
        )
        .shadow(color: isHighlighted ? Color.blue.opacity(highlightPulse ? 0.55 : 0.0) : .clear,
                radius: highlightPulse ? 14 : 0)
        .animation(.easeInOut(duration: 0.9).repeatForever(autoreverses: true), value: highlightPulse)
        .onChange(of: isHighlighted) { _, newValue in
            highlightPulse = newValue
        }
        .onAppear {
            if isHighlighted { highlightPulse = true }
        }
        .sheet(item: $webLink) { SafariView(url: $0.url) }
    }

    // MARK: - Try online

    /// Lets the user hear the model out before spending gigabytes on it. The
    /// Spaces lookup happens on tap rather than on appear — one request per
    /// card on every scroll would be wasteful.
    @ViewBuilder
    private var tryOnlineControl: some View {
        if !model.huggingFaceRepo.isEmpty {
            Button {
                Task { await openDemo() }
            } label: {
                HStack(spacing: 6) {
                    if isResolvingDemo {
                        ProgressView().controlSize(.mini)
                    } else {
                        Image(systemName: "safari")
                    }
                    Text("Try online")
                }
                .font(.caption)
            }
            .buttonStyle(.plain)
            .foregroundStyle(.blue)
            .disabled(isResolvingDemo)
        }
    }

    private func openDemo() async {
        isResolvingDemo = true
        defer { isResolvingDemo = false }
        let spaces = await HuggingFaceAPI.fetchSpaces(repo: model.huggingFaceRepo)
        webLink = HuggingFaceAPI
            .onlineDemoURL(repo: model.huggingFaceRepo, spaces: spaces)
            .map(WebLink.init)
    }

    private static let cornerRadius: CGFloat = 12

    /// Hairline by default, blue while recommended, brighter blue while the
    /// card is the target of a deep link.
    private var highlightStrokeColor: Color {
        if isHighlighted { return Color.blue.opacity(0.9) }
        if isRecommended { return Color.blue.opacity(0.4) }
        return Color.primary.opacity(0.08)
    }

    private var highlightStrokeWidth: CGFloat {
        if isHighlighted { return 2.5 }
        return isRecommended ? 1.5 : 1
    }

    // MARK: - Download control

    @ViewBuilder
    private var downloadControl: some View {
        switch downloadState {
        case .idle:
            Button {
                downloader.download(model)
            } label: {
                Label("Install", systemImage: "arrow.down.circle.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)

        case .fetchingManifest:
            HStack(spacing: 8) {
                ProgressView().scaleEffect(0.8)
                Text("Preparing...")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

        case .downloading(let progress, let downloaded, let total):
            VStack(alignment: .leading, spacing: 6) {
                // Animated progress bar
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.secondary.opacity(0.2))
                        RoundedRectangle(cornerRadius: 4)
                            .fill(Color.blue)
                            .frame(width: geo.size.width * max(progress, 0.01))
                            .animation(.easeInOut(duration: 0.3), value: progress)
                    }
                }
                .frame(height: 8)

                HStack {
                    Text(formatBytes(downloaded) + " / " + formatBytes(total))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("(\(Int(progress * 100))%)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Button("Cancel") {
                        downloader.cancel(model)
                    }
                    .font(.caption)
                    .foregroundStyle(.red)
                }
            }

        case .compiling(let progress):
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.8)
                    Text("Compiling on-device… \(Int(progress * 100))%")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
                Text("First-time setup takes 1–3 minutes")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

        case .installed:
            HStack {
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(.subheadline)
                Spacer()
                Button("Remove") {
                    showDeleteConfirmation = true
                }
                .font(.caption)
                .foregroundStyle(.red)
            }
            .alert("Remove Model?", isPresented: $showDeleteConfirmation) {
                Button("Cancel", role: .cancel) {}
                Button("Remove", role: .destructive) {
                    downloader.delete(model)
                }
            } message: {
                Text("This will delete \(model.name) (\(diskLabel)) from your device. You can reinstall it later.")
            }

        case .error(let msg):
            VStack(alignment: .leading, spacing: 6) {
                Label(msg, systemImage: "exclamationmark.circle.fill")
                    .foregroundStyle(.red)
                    .font(.caption)
                    .lineLimit(2)
                Button {
                    downloader.download(model)
                } label: {
                    Label("Retry", systemImage: "arrow.clockwise")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.small)
            }
        }
    }

    // MARK: - Helpers

    private var qualityBadge: some View {
        HStack(spacing: 2) {
            ForEach(0..<3) { i in
                Circle()
                    .frame(width: 7, height: 7)
                    .foregroundStyle(i < model.quality.rawValue ? qualityColor : Color.secondary.opacity(0.3))
            }
        }
    }

    private var qualityColor: Color {
        switch model.quality {
        case .good: return .yellow
        case .great: return .orange
        case .excellent: return .green
        }
    }

    private var ramLabel: String {
        model.backend == .appleIntelligence ? "—" : String(format: "%.1f GB", model.ramRequiredGB)
    }

    private var diskLabel: String {
        model.backend == .appleIntelligence ? "—" : String(format: "%.1f GB", model.diskSizeGB)
    }

    private var backendLabel: String {
        switch model.backend {
        case .appleIntelligence: return "Apple AI"
        case .mlx:               return "MLX"
        case .mlxVision:         return "MLX-VLM"
        case .mlxAudio:          return "MLX-Audio"
        case .coreML:            return "Core ML"
        case .coreMLWhisper:     return "Whisper"
        case .coreMLKokoro:      return "Kokoro"
        case .llamaCpp:          return "llama.cpp"
        case .llamaCppVision:    return "llama.cpp+VLM"
        case .whisperCpp:        return "whisper.cpp"
        }
    }

    private var contextFormatted: String {
        model.contextLength >= 1000 ? "\(model.contextLength / 1000)K" : "\(model.contextLength)"
    }

    private func stat(icon: String, value: String, label: String) -> some View {
        VStack(spacing: 2) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption).bold()
            Text(label)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let gb = Double(bytes) / 1_073_741_824
        if gb >= 1 { return String(format: "%.1f GB", gb) }
        let mb = Double(bytes) / 1_048_576
        return String(format: "%.0f MB", mb)
    }
}
