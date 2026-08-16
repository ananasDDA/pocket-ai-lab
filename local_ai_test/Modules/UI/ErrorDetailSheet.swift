//
//  ErrorDetailSheet.swift
//  local_ai_test
//
//  Full-screen sheet that displays the last engine failure in a
//  copyable, scrollable monospaced view. Reached by tapping the red
//  status label in the toolbar or from Diagnostics.
//

import SwiftUI

struct ErrorDetailSheet: View {
    let failure: EngineFailure
    @Environment(\.dismiss) private var dismiss
    @State private var copied = false
    @State private var showShare = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header

                    metadataGrid

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Details")
                            .font(.caption.smallCaps())
                            .foregroundStyle(.secondary)
                        Text(failure.detail)
                            .font(.system(.callout, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color(.secondarySystemBackground))
                            )
                    }

                    VStack(alignment: .leading, spacing: 6) {
                        Text("Copy-ready report")
                            .font(.caption.smallCaps())
                            .foregroundStyle(.secondary)
                        Text(failure.exportText)
                            .font(.system(.caption2, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .background(
                                RoundedRectangle(cornerRadius: 10)
                                    .fill(Color(.secondarySystemBackground))
                            )
                    }
                }
                .padding(16)
            }
            .navigationTitle("Error details")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { dismiss() }
                }
                ToolbarItemGroup(placement: .primaryAction) {
                    Button {
                        UIPasteboard.general.string = failure.exportText
                        withAnimation { copied = true }
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                            withAnimation { copied = false }
                        }
                    } label: {
                        Image(systemName: copied ? "checkmark" : "doc.on.doc")
                    }
                    .accessibilityLabel("Copy report")

                    Button {
                        showShare = true
                    } label: {
                        Image(systemName: "square.and.arrow.up")
                    }
                    .accessibilityLabel("Share report")
                }
            }
            .overlay(alignment: .bottom) {
                if copied {
                    Text("Copied")
                        .font(.caption).bold()
                        .padding(.horizontal, 14).padding(.vertical, 8)
                        .background(.thinMaterial, in: Capsule())
                        .padding(.bottom, 24)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .sheet(isPresented: $showShare) {
                ShareSheet(items: [failure.exportText])
            }
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.title)
                .foregroundStyle(.orange)
            VStack(alignment: .leading, spacing: 4) {
                Text(failure.summary)
                    .font(.headline)
                Text("\(failure.modelName) · \(failure.backend)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
        }
    }

    private var metadataGrid: some View {
        Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 6) {
            row("Phase",   failure.phase.rawValue)
            row("Backend", failure.backend)
            row("Model",   failure.modelName)
            row("ID",      failure.modelId)
            row("Type",    failure.underlyingType)
            row("Time",    ISO8601DateFormatter().string(from: failure.timestamp))
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(Color(.secondarySystemBackground))
        )
    }

    private func row(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label)
                .font(.caption.smallCaps())
                .foregroundStyle(.secondary)
                .gridColumnAlignment(.leading)
            Text(value)
                .font(.system(.caption, design: .monospaced))
                .textSelection(.enabled)
        }
    }
}

// MARK: - Share sheet (UIKit bridge)

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}
