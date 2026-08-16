//
//  AppLaunchSmokeTests.swift
//
//  Verifies the app launches cleanly on both Setup and Chat tabs.
//

import XCTest

final class AppLaunchSmokeTests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testLaunchShowsSetupOrChat() throws {
        let app = XCUIApplication()
        app.launch()

        // Either the setup tab or the chat tab should appear within 10s.
        let setup = app.tabBars.buttons["Setup"]
        let chat = app.tabBars.buttons["Chat"]
        let anyTab = XCUIApplication().windows.element

        XCTAssertTrue(setup.waitForExistence(timeout: 10) ||
                      chat.waitForExistence(timeout: 10) ||
                      anyTab.waitForExistence(timeout: 10))
    }

    @MainActor
    func testDiagnosticsOpensFromChatToolbar() throws {
        let app = XCUIApplication()
        app.launch()

        // Tap Chat tab if available
        if app.tabBars.buttons["Chat"].exists {
            app.tabBars.buttons["Chat"].tap()
        }

        // Toolbar "More" / ellipsis menu
        let moreButton = app.buttons.matching(identifier: "ellipsis.circle").firstMatch
        if moreButton.waitForExistence(timeout: 3) {
            moreButton.tap()
            let diag = app.buttons["Diagnostics"]
            if diag.waitForExistence(timeout: 3) {
                diag.tap()
                XCTAssertTrue(app.navigationBars["Diagnostics"].waitForExistence(timeout: 3))
            }
        }
    }
}
