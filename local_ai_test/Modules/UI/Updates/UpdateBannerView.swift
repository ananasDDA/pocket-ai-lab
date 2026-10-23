//
//  UpdateBannerView.swift
//  local_ai_test
//
//  The "there is a newer build" bar pinned under the Models tab title. Styled
//  after huggingface.co's hero/announcement bars rather than after a system
//  alert: one rounded card, a diagonal brand gradient (HF yellow -> orange ->
//  pink), near-black text on top of it, and a black filled capsule for the
//  action. It is deliberately NOT a warning colour - the mandatory variant
//  only shifts the ramp hotter instead of turning into a red banner.
//
//  Two shapes, one card:
//
//    * EXPANDED - version pill, headline, "What's new" list, the two buttons.
//    * COLLAPSED - a 44pt single-line strip ("1.0.1 is available") that the
//      user gets after tapping "Later", and that expands back on tap. "Later"
//      never removes the bar; only actually updating does (see
//      `UpdateChecker.decide`).
//

import SwiftUI

// MARK: - Brand palette

/// Hugging Face's brand ramp. Kept here rather than in an asset catalogue so
/// the exact hex values stay readable next to the view that uses them, and
/// shared with `WhatsNewSheet` so "here is what you are about to get" and
/// "here is what you just got" are visibly the same object.
enum HFBrand {

    /// #FFD21E - the Hugging Face yellow.
    static let yellow = Color(red: 1.0, green: 0.824, blue: 0.118)
    /// #FF9D00 - the warm orange the site pairs with it.
    static let orange = Color(red: 1.0, green: 0.616, blue: 0.0)
    /// #FF6FA3 - the pink their hero gradients fade into.
    static let pink = Color(red: 1.0, green: 0.435, blue: 0.639)
    /// #D6336C - only used for the "update required" ramp, which runs hotter
    /// so the two states are distinguishable without reading the headline.
    static let magenta = Color(red: 0.839, green: 0.200, blue: 0.424)

    /// #1B1B1F. The gradient is bright in BOTH colour schemes, so the text on
    /// it is a fixed near-black in both - flipping it to white in dark mode
    /// would put white on yellow, which is the one combination that fails.
    static let ink = Color(red: 0.106, green: 0.106, blue: 0.122)

    static func heroGradient(mandatory: Bool) -> LinearGradient {
        LinearGradient(
            stops: mandatory
                ? [.init(color: orange, location: 0.0),
                   .init(color: pink, location: 0.55),
                   .init(color: magenta, location: 1.0)]
                : [.init(color: yellow, location: 0.0),
                   .init(color: orange, location: 0.48),
                   .init(color: pink, location: 1.0)],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }
}

// MARK: - Banner

struct UpdateBannerView: View {

    let release: ReleaseInfo
    /// Installed build is below `minimumSupportedVersion`: hotter ramp, no
    /// "Later", and the strip can never be collapsed.
    let isMandatory: Bool
    /// Drawn as the compact strip instead of the full card. Ignored when
    /// `isMandatory` - a required update always shows its reasons.
    let isCollapsed: Bool
    /// "Later" - collapses, never hides.
    let onCollapse: () -> Void
    /// Tapping the compact strip.
    let onExpand: () -> Void

    @Environment(\.openURL) private var openURL
    @Environment(\.colorScheme) private var colorScheme

    private static let cornerRadius: CGFloat = 14

    /// The card is PINNED now, so it eats screen height the model list never
    /// gets back. Four bullets is where a summary stops being a summary; the
    /// rest live on the App Store page.
    private static let maxVisibleNotes = 4

    private var showsCompact: Bool { isCollapsed && !isMandatory }

    var body: some View {
        Group {
            if showsCompact {
                compactStrip
            } else {
                expandedCard
            }
        }
        .animation(.spring(response: 0.38, dampingFraction: 0.86), value: showsCompact)
    }

    // MARK: - Card chrome

    private var shape: RoundedRectangle {
        RoundedRectangle(cornerRadius: Self.cornerRadius, style: .continuous)
    }

    /// A bright gradient sitting on a dark page glares, so dark mode composites
    /// 12% black over it: still light enough for near-black text (about 8:1),
    /// dim enough not to be the brightest thing on screen.
    private var cardBackground: some View {
        shape
            .fill(HFBrand.heroGradient(mandatory: isMandatory))
            .overlay(shape.fill(Color.black.opacity(colorScheme == .dark ? 0.12 : 0)))
            .overlay(
                shape.stroke(
                    colorScheme == .dark
                        ? Color.white.opacity(0.18)
                        : HFBrand.ink.opacity(0.10),
                    lineWidth: 1
                )
            )
            .shadow(color: .black.opacity(colorScheme == .dark ? 0.35 : 0.10),
                    radius: 7, y: 3)
    }

    // MARK: - Compact strip

    private var compactStrip: some View {
        Button(action: onExpand) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .font(.subheadline)
                    .accessibilityHidden(true)

                Text("\(release.version) is available")
                    .font(.subheadline.weight(.semibold))
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)

                Spacer(minLength: 8)

                Image(systemName: "chevron.down")
                    .font(.caption.weight(.bold))
                    .accessibilityHidden(true)
            }
            .foregroundStyle(HFBrand.ink)
            .padding(.horizontal, 14)
            .frame(maxWidth: .infinity, alignment: .leading)
            .frame(minHeight: 44)
            .background(cardBackground)
            .contentShape(shape)
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Version \(release.version) is available")
        .accessibilityHint("Shows the update details")
        .transition(.opacity.combined(with: .scale(scale: 0.98, anchor: .top)))
    }

    // MARK: - Expanded card

    private var expandedCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            header

            if !release.notes.isEmpty {
                notesList
            }

            actions
        }
        .foregroundStyle(HFBrand.ink)
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(cardBackground)
        .transition(.opacity.combined(with: .scale(scale: 0.98, anchor: .top)))
    }

    // MARK: - Header

    private var headline: String { isMandatory ? "Update required" : "Update available" }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            // Title and version pill share a line while they fit, then stack.
            // No `Spacer` in either candidate - a flexible child would make
            // the row claim to fit at every size and the fallback would never
            // run.
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    titleLine
                    versionPill
                }
                VStack(alignment: .leading, spacing: 6) {
                    titleLine
                    versionPill
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if let date = release.releaseDateLabel {
                Text(date)
                    .font(.caption)
                    .foregroundStyle(HFBrand.ink.opacity(0.62))
            }

            if isMandatory {
                Text("This build has a known problem. Please update.")
                    .font(.caption)
                    .foregroundStyle(HFBrand.ink.opacity(0.72))
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        // One announcement - "Update available, 1.0.1, 5 Sep 2026" - instead
        // of four separate swipes through fragments.
        .accessibilityElement(children: .combine)
    }

    private var titleLine: some View {
        HStack(spacing: 6) {
            Image(systemName: isMandatory
                  ? "exclamationmark.arrow.trianglehead.2.clockwise.rotate.90"
                  : "sparkles")
                .font(.subheadline.weight(.semibold))
                .accessibilityHidden(true)

            Text(headline)
                .font(.headline.weight(.semibold))
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    /// Monospaced in a capsule, the way `HFModelHeader` renders `owner/repo`:
    /// the version is an identifier, not prose.
    private var versionPill: some View {
        Text(release.version)
            .font(.system(.caption, design: .monospaced).weight(.semibold))
            .foregroundStyle(HFBrand.ink)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Color.white.opacity(0.45), in: Capsule())
            .overlay(Capsule().stroke(HFBrand.ink.opacity(0.16), lineWidth: 1))
    }

    // MARK: - Notes

    private var notesList: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("What's new")
                .font(.caption.weight(.semibold))
                .foregroundStyle(HFBrand.ink.opacity(0.62))

            ForEach(Array(release.notes.prefix(Self.maxVisibleNotes).enumerated()),
                    id: \.offset) { index, note in
                Label {
                    Text(note)
                        .font(.subheadline)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: index == 0 ? "sparkles" : "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(HFBrand.ink.opacity(0.7))
                        .accessibilityHidden(true)
                }
            }

            if release.notes.count > Self.maxVisibleNotes {
                Text("and more in the App Store notes")
                    .font(.caption)
                    .foregroundStyle(HFBrand.ink.opacity(0.62))
            }
        }
    }

    // MARK: - Actions

    private var actions: some View {
        // Side by side while both buttons fit; stacked once an accessibility
        // text size makes "Update in App Store" alone wider than the card.
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 8) {
                updateButton
                laterButton
            }
            VStack(alignment: .leading, spacing: 4) {
                updateButton
                laterButton
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// HF's own button style: a near-black filled capsule with white text,
    /// which reads on every stop of the gradient. `borderedProminent` in iOS
    /// blue would be the one colour on screen that belongs to nothing.
    private var updateButton: some View {
        Button {
            openURL(release.storeLink)
        } label: {
            HStack(spacing: 6) {
                Text("Update in App Store")
                    .font(.subheadline.weight(.semibold))
                Image(systemName: "arrow.up.forward")
                    .font(.caption.weight(.bold))
                    .accessibilityHidden(true)
            }
            .foregroundStyle(.white)
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .frame(minHeight: 44)
            .background(HFBrand.ink, in: Capsule())
            .contentShape(Capsule())
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Update in App Store")
    }

    @ViewBuilder
    private var laterButton: some View {
        if !isMandatory {
            // A bare plain-style button is only as tall as its label, which
            // left "Later" well under the 44pt minimum touch target.
            Button("Later", action: onCollapse)
                .font(.subheadline.weight(.medium))
                .buttonStyle(.plain)
                .foregroundStyle(HFBrand.ink.opacity(0.72))
                .padding(.horizontal, 12)
                .frame(minHeight: 44)
                .contentShape(Rectangle())
                .accessibilityHint("Collapses this banner to a single line")
        }
    }
}

#Preview("Available") {
    UpdateBannerView(
        release: ReleaseInfo(
            version: "1.0.1",
            build: 2,
            published: true,
            releasedAt: "2026-09-05",
            minimumSupportedVersion: "1.0.1",
            notes: [
                "llama.cpp models work again - the engine is back",
                "Core ML text models load and answer",
                "No more crash on long prompts with GGUF models"
            ]
        ),
        isMandatory: false,
        isCollapsed: false,
        onCollapse: {},
        onExpand: {}
    )
    .padding()
}

#Preview("Collapsed") {
    UpdateBannerView(
        release: ReleaseInfo(
            version: "1.0.1",
            build: 2,
            published: true,
            releasedAt: "2026-09-05",
            minimumSupportedVersion: nil,
            notes: ["llama.cpp models work again - the engine is back"]
        ),
        isMandatory: false,
        isCollapsed: true,
        onCollapse: {},
        onExpand: {}
    )
    .padding()
}

#Preview("Required") {
    UpdateBannerView(
        release: ReleaseInfo(
            version: "1.0.1",
            build: 2,
            published: true,
            releasedAt: "2026-09-05",
            minimumSupportedVersion: "1.0.1",
            notes: ["llama.cpp models work again - the engine is back"]
        ),
        isMandatory: true,
        isCollapsed: true,
        onCollapse: {},
        onExpand: {}
    )
    .padding()
}
