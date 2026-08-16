//
//  ContentView.swift
//  local_ai_test
//
//  Created by Daniil on 12.04.2026.
//

import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0
    /// When non-nil the Models tab scrolls to and highlights the matching
    /// model card. Consumed once, then cleared. Used for deep links like
    /// "install Kokoro" from the chat's Speak-response prompt.
    @State private var focusedModelID: String?

    var body: some View {
        TabView(selection: $selectedTab) {
            SetupView(focusedModelID: $focusedModelID)
                .tabItem { Label("Models", systemImage: "server.rack") }
                .tag(0)

            ChatView(selectedTab: $selectedTab, focusedModelID: $focusedModelID)
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                .tag(1)
        }
        // Best-effort catalog refresh; failures leave the bundled copy in place.
        .task { await RemoteCatalog.shared.refresh() }
    }
}

#Preview {
    ContentView()
}
