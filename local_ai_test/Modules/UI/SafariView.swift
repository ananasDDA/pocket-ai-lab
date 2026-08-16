//
//  SafariView.swift
//  local_ai_test
//
//  In-app browser for "Try online". SFSafariViewController rather than a
//  WKWebView so the user keeps the address bar and can see they left the app's
//  own on-device world.
//

import SafariServices
import SwiftUI

/// Wraps a URL so it can drive `.sheet(item:)`.
struct WebLink: Identifiable {
    let id = UUID()
    let url: URL
}

struct SafariView: UIViewControllerRepresentable {

    let url: URL

    func makeUIViewController(context: Context) -> SFSafariViewController {
        let configuration = SFSafariViewController.Configuration()
        configuration.entersReaderIfAvailable = false
        return SFSafariViewController(url: url, configuration: configuration)
    }

    func updateUIViewController(_ controller: SFSafariViewController, context: Context) {}
}
