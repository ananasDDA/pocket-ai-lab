//
//  WhatsNewSheet.swift
//  local_ai_test
//
//  Shown once, on the first launch after an update, from the notes compiled
//  into this build. Same visual language as `UpdateBannerView` — the same HF
//  brand gradient, the same bullet list, the same near-black capsule button —
//  so "here is what you are about to get" and "here is what you just got"
//  look like the same thing said twice.
//

import SwiftUI

struct WhatsNewSheet: View {

    let release: ReleaseInfo
    let onContinue: () -> Void

    @Environment(\.colorScheme) private var colorScheme

    private static let cornerRadius: CGFloat = 14

    var body: some View {
        VStack(spacing: 0) {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    heading
                    notesCard
                }
                .padding(16)
            }

            Divider()

            Button(action: onContinue) {
                Text("Continue")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .frame(minHeight: 50)
                    .background(HFBrand.ink, in: Capsule())
                    .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .padding(16)
        }
        // No drag indicator: `interactiveDismissDisabled()` is set on this
        // sheet, so the grabber would advertise a gesture that does nothing.
    }

    private var shape: RoundedRectangle {
        RoundedRectangle(cornerRadius: Self.cornerRadius, style: .continuous)
    }

    // MARK: - Heading

    /// The same hero gradient the pinned banner uses, at the top of the sheet
    /// instead of a bare blue glyph.
    private var heading: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: "sparkles")
                .font(.largeTitle)
                .accessibilityHidden(true)

            Text("What's new in \(release.version)")
                .font(.title2).bold()
                .fixedSize(horizontal: false, vertical: true)

            if let date = release.releaseDateLabel {
                Text(date)
                    .font(.caption)
                    .foregroundStyle(HFBrand.ink.opacity(0.62))
            }
        }
        .foregroundStyle(HFBrand.ink)
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            shape
                .fill(HFBrand.heroGradient(mandatory: false))
                .overlay(shape.fill(Color.black.opacity(colorScheme == .dark ? 0.12 : 0)))
                .overlay(shape.stroke(colorScheme == .dark
                                      ? Color.white.opacity(0.18)
                                      : HFBrand.ink.opacity(0.10),
                                      lineWidth: 1))
        )
        .accessibilityElement(children: .combine)
    }

    // MARK: - Notes

    private var notesCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            ForEach(Array(release.notes.enumerated()), id: \.offset) { index, note in
                Label {
                    Text(note)
                        .font(.subheadline)
                        .fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: index == 0 ? "sparkles" : "checkmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(HFBrand.orange)
                        .accessibilityHidden(true)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(.background.secondary, in: shape)
        .overlay(shape.stroke(HFBrand.orange.opacity(0.28), lineWidth: 1))
    }
}

#Preview {
    WhatsNewSheet(
        release: ReleaseInfo(
            version: "1.0.1",
            build: 2,
            published: false,
            releasedAt: "2026-09-05",
            minimumSupportedVersion: "1.0.1",
            notes: [
                "llama.cpp models work again - the engine is back",
                "Core ML text models load and answer",
                "No more crash on long prompts with GGUF models",
                "Import your own GGUF from the Files app",
                "Memory budgets tuned for 6 GB and 8 GB iPhones",
                "New Model Benchmark with a shareable report card",
                "Build test measures the whole catalog in one run"
            ]
        ),
        onContinue: {}
    )
}
