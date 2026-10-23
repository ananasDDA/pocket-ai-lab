//
//  ContentView.swift
//  local_ai_test
//
//  Created by Daniil on 12.04.2026.
//

import SwiftUI

struct ContentView: View {
    @Environment(\.scenePhase) private var scenePhase
    @State private var selectedTab = 0
    /// When non-nil the Models tab scrolls to and highlights the matching
    /// model card. Consumed once, then cleared. Used for deep links like
    /// "install Kokoro" from the chat's Speak-response prompt.
    @State private var focusedModelID: String?

    /// The build test's live screen. Bound to
    /// `BuildTestRunner.shared.isRunning` rather than to a button, and owned
    /// HERE rather than by the Diagnostics section that starts it, for two
    /// reasons:
    ///
    ///   * a sweep started from the Xcode test target has no view behind it,
    ///     and used to leave the tester staring at an idle app for hours;
    ///   * one owner means the cover can never be presented twice - the
    ///     in-app button just starts the runner and gets out of the way.
    @State private var buildTestVisible = false
    /// Set when a sweep ends, so its report comes up right after the live
    /// screen goes down.
    @State private var finishedBuildTest: BuildTestReport?
    /// The report `BuildTestLiveView` handed back, held until the cover's own
    /// `onDismiss:` fires - i.e. until UIKit says the dismissal actually
    /// finished - so the report sheet is never asked to present while the
    /// cover is still animating away underneath it.
    @State private var pendingBuildTestReport: BuildTestReport?

    /// The one-shot "what's new" sheet for the version just installed. Held
    /// locally rather than read straight off `UpdateChecker` so it is never
    /// asked to present while the build-test cover owns the screen — a sweep
    /// that survived an update would otherwise fight it for the presentation
    /// slot on the very first launch.
    @State private var whatsNew: ReleaseInfo?
    /// True only once the sheet's body has actually reached the screen. A
    /// request made while a DESCENDANT owns the presentation slot (ChatView's
    /// Diagnostics sheet) is silently dropped by SwiftUI, and without this
    /// flag the non-nil `whatsNew` would then block every later attempt - the
    /// sheet would be lost for the rest of the install.
    @State private var whatsNewOnScreen = false

    var body: some View {
        TabView(selection: $selectedTab) {
            SetupView(focusedModelID: $focusedModelID)
                .tabItem { Label("Models", systemImage: "server.rack") }
                .tag(0)

            ChatView(
                selectedTab: $selectedTab,
                focusedModelID: $focusedModelID,
                onDiagnosticsDismiss: handleDiagnosticsDismiss
            )
                .tabItem { Label("Chat", systemImage: "bubble.left.and.bubble.right") }
                .tag(1)
        }
        // Best-effort catalog refresh; failures leave the bundled copy in place.
        .task { await RemoteCatalog.shared.refresh() }
        // Update check: same best-effort contract, and throttled inside the
        // checker. Driven off `.active` alone - including the initial value -
        // so the What's New sheet is only ever raised at a moment when this
        // view really owns the screen, and so a request that got dropped is
        // asked for again on the next foreground.
        .onChange(of: scenePhase, initial: true) { _, phase in
            guard phase == .active else { return }
            Task {
                await UpdateChecker.shared.checkIfNeeded()
                presentWhatsNewIfNeeded()
            }
        }
        // A sweep already in flight when this view first appears (the app was
        // relaunched by the test host mid-run) never fires `onChange`.
        .onAppear {
            if BuildTestRunner.shared.isRunning { buildTestVisible = true }
        }
        // A journal left behind means the previous process died mid-sweep:
        // turn it into a report before anything else can start a new run,
        // and start listening for what MetricKit has to say about the death.
        .task {
            CrashDiagnosticsCollector.shared.start()
            BuildTestRunner.shared.recoverInterruptedRun()
        }
        .fullScreenCover(isPresented: $buildTestVisible, onDismiss: {
            // Fires once UIKit has actually finished taking the cover down,
            // so the report sheet's presentation never races that animation.
            guard let report = pendingBuildTestReport else { return }
            pendingBuildTestReport = nil
            finishedBuildTest = report
        }) {
            BuildTestLiveView { report in
                pendingBuildTestReport = report
                buildTestVisible = false
            }
        }
        .sheet(item: $whatsNew, onDismiss: {
            // Covers both a real dismissal and a presentation SwiftUI took
            // back down again: either way nothing is on screen, so the
            // request is cleared and `presentWhatsNewIfNeeded` may re-arm it.
            whatsNewOnScreen = false
            whatsNew = nil
        }) { release in
            WhatsNewSheet(release: release) {
                UpdateChecker.shared.markWhatsNewSeen()
                whatsNew = nil
            }
            .interactiveDismissDisabled()
            .onAppear { whatsNewOnScreen = true }
        }
        .sheet(item: $finishedBuildTest) { report in
            NavigationStack {
                BuildTestReportView(report: report)
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Done") { finishedBuildTest = nil }
                        }
                    }
            }
        }
    }

    /// Raises the What's New sheet once the checker says it is owed, unless
    /// something else already has the screen. It stays owed until the user
    /// taps Continue, so a skipped launch simply shows it on the next one.
    private func presentWhatsNewIfNeeded() {
        // The build test owns the screen outright; never race it.
        guard !buildTestVisible, finishedBuildTest == nil else { return }

        guard whatsNew == nil else {
            // A request is already outstanding. If it never actually reached
            // the screen, a descendant owned the presentation slot when it
            // was made - drop it here so the next call (the next `.active`,
            // or the Diagnostics sheet going away) asks again. Re-assigning
            // the same item in place would be a no-op to SwiftUI.
            if !whatsNewOnScreen { whatsNew = nil }
            return
        }
        whatsNew = UpdateChecker.shared.whatsNew
    }

    /// ChatView's Diagnostics sheet has finished going away, so the
    /// presentation slot is free again: start the build test's live screen if
    /// one is owed, and otherwise retry a What's New sheet that this very
    /// sheet may have swallowed.
    private func handleDiagnosticsDismiss() {
        presentBuildTestCoverIfNeeded()
        presentWhatsNewIfNeeded()
    }

    /// Presents the live screen for a sweep the in-app "Start build test"
    /// button kicked off. Called from `ChatView`'s Diagnostics sheet
    /// `onDismiss:` - i.e. once that sheet has actually finished going away,
    /// not merely once the runner reports `isRunning`, which can land while
    /// the dismissal animation is still playing and get silently dropped.
    ///
    /// The `isRunning` guard also covers a sweep that ends (e.g. an empty
    /// eligible list) before the Diagnostics sheet finishes dismissing: with
    /// nothing left running, the cover has nothing to show and must not
    /// flash on screen.
    private func presentBuildTestCoverIfNeeded() {
        guard BuildTestRunner.shared.isRunning, !buildTestVisible else { return }
        buildTestVisible = true
    }
}

#Preview {
    ContentView()
}
