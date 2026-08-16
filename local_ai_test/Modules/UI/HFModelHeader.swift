//
//  HFModelHeader.swift
//  local_ai_test
//
//  A 1:1 take on the huggingface.co/models mini-card header, reused by model
//  cards and by the import sheet: a small square avatar inline with the
//  monospaced `owner/repo`, then one bullet-separated line of metadata.
//

import SwiftUI

struct HFModelHeader: View {

    /// `owner/repo`. Empty for models that have no HF origin (Apple
    /// Intelligence), in which case the header degrades to a plain title.
    let repo: String
    /// Shown instead of `owner/repo` when there is no repo.
    let displayName: String
    /// Rendered in the metadata line between the task and the update date.
    /// HF shows it there for repos that declare it; we always know it.
    var parameterSize: String?
    /// The import sheet renders its own authoritative counters, so it turns
    /// this off rather than showing two copies of the same numbers.
    var showsMetrics: Bool = true

    @State private var webLink: WebLink?

    private static let avatarSize: CGFloat = 20

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter
    }()

    private var metadata: HFModelMetadata? {
        repo.isEmpty ? nil : HFMetadataStore.shared.metadata(for: repo)
    }

    private var owner: String { HFMetadataStore.owner(of: repo) ?? repo }

    var body: some View {
        if repo.isEmpty {
            Text(displayName)
                .font(.headline)
        } else {
            VStack(alignment: .leading, spacing: 5) {
                // Its own hit area: the card below is full of buttons and a
                // stray tap must not start a download.
                Button {
                    webLink = URL(string: "\(HuggingFaceAPI.webHost)/\(repo)").map(WebLink.init)
                } label: {
                    HStack(spacing: 7) {
                        avatar
                        Text(repo)
                            .font(.system(.subheadline, design: .monospaced))
                            .foregroundStyle(.primary)
                            .lineLimit(2)
                            .multilineTextAlignment(.leading)
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .accessibilityHint("Opens the model page on Hugging Face")

                if showsMetrics {
                    metadataLine
                }
            }
            .animation(.default, value: metadata)
            .task { HFMetadataStore.shared.ensureLoaded(repo: repo) }
            .sheet(item: $webLink) { SafariView(url: $0.url) }
        }
    }

    // MARK: - Avatar

    @ViewBuilder
    private var avatar: some View {
        if let url = metadata?.avatarURL {
            AsyncImage(url: url) { image in
                image.resizable().scaledToFill()
            } placeholder: {
                avatarPlaceholder
            }
            .frame(width: Self.avatarSize, height: Self.avatarSize)
            .clipShape(RoundedRectangle(cornerRadius: 4))
        } else {
            avatarPlaceholder
                .frame(width: Self.avatarSize, height: Self.avatarSize)
        }
    }

    private var avatarPlaceholder: some View {
        RoundedRectangle(cornerRadius: 4)
            .fill(Color.secondary.opacity(0.2))
            .overlay(
                Text(ownerInitial)
                    .font(.system(size: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
            )
    }

    private var ownerInitial: String {
        owner.first.map { String($0).uppercased() } ?? "?"
    }

    // MARK: - Metadata line

    private struct MetaItem: Identifiable {
        let id: String
        let symbol: String?
        let text: String
    }

    /// Same order HF uses: task, size, freshness, downloads, likes. Anything
    /// we have no value for is simply left out — plenty of real cards show
    /// only the last three.
    private var metaItems: [MetaItem] {
        var items: [MetaItem] = []
        if let tag = metadata?.pipelineTag {
            items.append(MetaItem(id: "task",
                                  symbol: Self.pipelineSymbol(tag),
                                  text: Self.pipelineLabel(tag)))
        }
        if let parameterSize, !parameterSize.isEmpty {
            items.append(MetaItem(id: "params", symbol: "cpu", text: parameterSize))
        }
        if let updated = metadata?.lastModified {
            items.append(MetaItem(
                id: "updated",
                symbol: nil,
                text: "Updated " + Self.relativeFormatter.localizedString(
                    for: updated, relativeTo: Date())
            ))
        }
        if let downloads = metadata?.downloads, downloads > 0 {
            items.append(MetaItem(id: "downloads",
                                  symbol: "arrow.down.to.line",
                                  text: HFMetadataStore.compactCount(downloads)))
        }
        if let likes = metadata?.likes, likes > 0 {
            items.append(MetaItem(id: "likes",
                                  symbol: "heart",
                                  text: HFMetadataStore.compactCount(likes)))
        }
        return items
    }

    /// Hidden entirely until something loads — a row of zeroes reads as "this
    /// model has no downloads", which is not what an empty cache means.
    @ViewBuilder
    private var metadataLine: some View {
        let items = metaItems
        if !items.isEmpty {
            HStack(spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
                    if index > 0 {
                        Text("•")
                    }
                    if let symbol = item.symbol {
                        Image(systemName: symbol)
                    }
                    Text(item.text)
                }
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .lineLimit(1)
        }
    }

    /// `text-generation` → `Text Generation`, the way HF renders it.
    nonisolated static func pipelineLabel(_ tag: String) -> String {
        tag.replacingOccurrences(of: "-", with: " ")
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    nonisolated static func pipelineSymbol(_ tag: String) -> String {
        switch tag {
        case "text-generation", "text2text-generation", "fill-mask", "summarization":
            return "text.alignleft"
        case "image-text-to-text", "image-to-text", "visual-question-answering",
             "image-classification", "object-detection", "text-to-image":
            return "photo.on.rectangle"
        case "automatic-speech-recognition", "audio-classification", "voice-activity-detection":
            return "waveform"
        case "text-to-speech", "text-to-audio":
            return "speaker.wave.2"
        default:
            return "sparkles"
        }
    }
}

/// The one-line "where these come from" note above the model list.
struct HFSourceBanner: View {
    var body: some View {
        HStack(spacing: 6) {
            Image("HuggingFaceLogo")
                .resizable()
                .scaledToFit()
                .frame(height: 18)
                .accessibilityHidden(true)
            Text("Models from Hugging Face")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}
