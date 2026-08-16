//
//  SetupView.swift
//  local_ai_test
//

import SwiftUI

struct SetupView: View {
    @State private var vm = SetupViewModel()
    @State private var backendFilter: BackendFamily? = nil
    @State private var modalityFilter: ModalityFilter = .all
    /// Temporarily non-nil after a deep link arrived (e.g. "install Kokoro"
    /// from chat). Drives the scroll target and the card's glow overlay,
    /// then the view auto-clears it after the pulse animation finishes.
    @State private var highlightedModelID: String?
    @State private var showImportSheet = false

    private var store: InstalledModelsStore { InstalledModelsStore.shared }

    /// Deep-link channel from `ContentView`. When this flips to a non-nil
    /// id while the Models tab is visible, `SetupView` scrolls to the
    /// matching card and pulses its border. The binding is cleared
    /// immediately so the focus is consumed exactly once.
    @Binding var focusedModelID: String?

    var body: some View {
        NavigationStack {
            Group {
                switch vm.state {
                case .scanning:
                    scanningView
                case .ready(let device, let result):
                    readyView(device: device, result: result)
                case .error(let msg):
                    errorView(msg)
                }
            }
            .navigationTitle("PRO Local AI Lab")
            .navigationBarTitleDisplayMode(.large)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        showImportSheet = true
                    } label: {
                        Label("Import from Hugging Face", systemImage: "link.badge.plus")
                    }
                    .disabled(scannedDevice == nil)
                }
            }
            .sheet(isPresented: $showImportSheet) {
                if let device = scannedDevice {
                    ImportModelSheet(device: device)
                }
            }
        }
        .task { await vm.scan() }
    }

    /// The import sheet judges RAM/disk against the same scan the cards use,
    /// so it stays unavailable until that scan lands.
    private var scannedDevice: DeviceInfo? {
        if case .ready(let device, _) = vm.state { return device }
        return nil
    }

    enum ModalityFilter: String, CaseIterable, Identifiable {
        case all, text, vision, audio, tts
        var id: String { rawValue }
        var label: String {
            switch self {
            case .all: return "All"
            case .text: return "Text"
            case .vision: return "Vision"
            case .audio: return "Audio-in"
            case .tts: return "TTS"
            }
        }
        func matches(_ m: AIModel) -> Bool {
            switch self {
            case .all: return true
            case .text: return m.capabilities == .textOnly
            case .vision: return m.capabilities.contains(.imageIn)
            case .audio: return m.capabilities.contains(.audioIn)
            case .tts: return m.capabilities.contains(.audioOut)
            }
        }
    }

    private func applyFilters(_ models: [AIModel]) -> [AIModel] {
        models.filter { m in
            (backendFilter == nil || m.backend.family == backendFilter) && modalityFilter.matches(m)
        }
    }

    private var filterBar: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                Menu {
                    Button("All backends") { backendFilter = nil }
                    ForEach(BackendFamily.allCases, id: \.self) { fam in
                        Button(fam.displayName) { backendFilter = fam }
                    }
                } label: {
                    Label(backendFilter?.displayName ?? "Backend: all", systemImage: "cpu")
                        .font(.caption)
                        .padding(.horizontal, 10).padding(.vertical, 6)
                        .background(.regularMaterial, in: Capsule())
                }

                ForEach(ModalityFilter.allCases) { f in
                    Button {
                        modalityFilter = f
                    } label: {
                        Text(f.label)
                            .font(.caption)
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(modalityFilter == f ? Color.blue.opacity(0.2) : Color.clear,
                                        in: Capsule())
                            .overlay(Capsule().stroke(modalityFilter == f ? Color.blue : Color.secondary.opacity(0.4), lineWidth: 1))
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Scanning

    private var scanningView: some View {
        VStack(spacing: 16) {
            ProgressView()
                .scaleEffect(1.4)
            Text("Scanning device...")
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Ready

    private func readyView(device: DeviceInfo, result: RecommendationResult) -> some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {

                    DeviceInfoView(device: device)

                    // Said once for the whole list rather than per card.
                    HFSourceBanner()

                    // Recommendation
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recommendation")
                            .font(.caption)
                            .foregroundStyle(.secondary)

                        Text(result.reason)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 2)

                        if let recommended = result.recommended {
                            ModelCardView(
                                model: recommended,
                                isRecommended: true,
                                isHighlighted: highlightedModelID == recommended.id
                            )
                            .id(recommended.id)
                        }
                    }

                    importedSection

                    filterBar

                    if result.allCompatible.count > 1 {
                        let otherCompatible = applyFilters(Array(result.allCompatible.dropFirst()))
                        if !otherCompatible.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Other compatible models")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)

                                ForEach(otherCompatible) { model in
                                    ModelCardView(
                                        model: model,
                                        isRecommended: false,
                                        isHighlighted: highlightedModelID == model.id
                                    )
                                    .id(model.id)
                                }
                            }
                        }
                    }

                    if !result.incompatible.isEmpty {
                        let filteredIncompat = applyFilters(result.incompatible)
                        if !filteredIncompat.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Not enough resources")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)

                                ForEach(filteredIncompat) { model in
                                    ModelCardView(
                                        model: model,
                                        isRecommended: false,
                                        isHighlighted: highlightedModelID == model.id
                                    )
                                    .opacity(0.45)
                                    .id(model.id)
                                }
                            }
                        }
                    }
                }
                .padding(16)
            }
            .refreshable { await vm.scan() }
            .onChange(of: focusedModelID) { _, newValue in
                guard let id = newValue else { return }
                handleFocusRequest(id: id, proxy: proxy)
            }
            .onAppear {
                // If the deep link arrived before the model list finished
                // scanning, react now that the cards actually exist.
                if let id = focusedModelID {
                    handleFocusRequest(id: id, proxy: proxy)
                }
            }
        }
    }

    // MARK: - Imported models

    /// Sourced from the registry rather than the catalog — these exist only
    /// because the user pasted a link, and they must show up while they are
    /// still downloading, not just once installed.
    @ViewBuilder
    private var importedSection: some View {
        let imported = store.importedModels
        if !imported.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Imported models")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ForEach(imported) { model in
                    ModelCardView(
                        model: model,
                        isRecommended: false,
                        isHighlighted: highlightedModelID == model.id
                    )
                    .id(model.id)
                    .contextMenu {
                        Button(role: .destructive) {
                            ModelDownloader.shared.delete(model)
                            store.unregister(model)
                        } label: {
                            Label("Remove from list", systemImage: "trash")
                        }
                    }
                }
            }
        }
    }

    private func handleFocusRequest(id: String, proxy: ScrollViewProxy) {
        withAnimation(.easeInOut(duration: 0.4)) {
            proxy.scrollTo(id, anchor: .center)
        }
        highlightedModelID = id
        // Consume the deep link so it doesn't re-fire when the user
        // switches tabs. The local highlight keeps glowing a bit longer.
        focusedModelID = nil
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 2_200_000_000)
            if highlightedModelID == id {
                withAnimation(.easeOut(duration: 0.4)) {
                    highlightedModelID = nil
                }
            }
        }
    }

    // MARK: - Error

    private func errorView(_ message: String) -> some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle")
                .font(.largeTitle)
                .foregroundStyle(.orange)
            Text(message)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Retry") {
                Task { await vm.scan() }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

#Preview {
    SetupView(focusedModelID: .constant(nil))
}
