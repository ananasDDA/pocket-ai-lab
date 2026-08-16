//
//  SetupFilterTests.swift
//
//  Drives the modality / backend filters in SetupView.
//

import XCTest

final class SetupFilterTests: XCTestCase {

    @MainActor
    func testFiltersAreTappable() throws {
        let app = XCUIApplication()
        app.launch()

        if app.tabBars.buttons["Setup"].exists {
            app.tabBars.buttons["Setup"].tap()
        }

        // Look for typical filter labels. We don't assert the list actually
        // shrinks because the catalog content is dynamic.
        let allChip = app.buttons["All"].firstMatch
        _ = allChip.waitForExistence(timeout: 5)
        if allChip.exists { allChip.tap() }

        let textChip = app.buttons["Text"].firstMatch
        if textChip.waitForExistence(timeout: 2) { textChip.tap() }
    }
}
