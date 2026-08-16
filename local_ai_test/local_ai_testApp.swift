//
//  local_ai_testApp.swift
//  local_ai_test
//
//  Created by Daniil on 12.04.2026.
//

import SwiftUI

class AppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        handleEventsForBackgroundURLSession identifier: String,
        completionHandler: @escaping () -> Void
    ) {
        ModelDownloader.shared.backgroundCompletionHandler = completionHandler
    }
}

@main
struct local_ai_testApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
