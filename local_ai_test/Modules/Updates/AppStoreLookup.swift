//
//  AppStoreLookup.swift
//  local_ai_test
//
//  Asks Apple which version the App Store is currently serving, so the app
//  never tells a user to go update to something that is not there yet.
//
//  PRIVACY: this is a GET to itunes.apple.com carrying nothing but the app's
//  own bundle id. No user data, no device data, no identifiers, no cookies
//  are attached, and the answer is a single version string that is thrown
//  away on the next check.
//

import Foundation

nonisolated enum AppStoreLookup {

    /// The App Store record id (apps.apple.com/app/id6802253142). Looked up
    /// by numeric id, not bundle id: the bundleId form returned zero results
    /// for this app on 2026-09-03 while the id form answered correctly, and
    /// the id is what the store URL is built from anyway.
    static let appStoreID = "6802253142"

    /// `country` is required for a meaningful answer: the lookup is
    /// storefront-scoped and an unqualified query can miss an app that is
    /// live. "us" is the release storefront and always has the newest build.
    private static var endpoint: URL? {
        var components = URLComponents(string: "https://itunes.apple.com/lookup")
        components?.queryItems = [
            URLQueryItem(name: "id", value: appStoreID),
            URLQueryItem(name: "country", value: "us")
        ]
        return components?.url
    }

    private struct LookupResponse: Decodable {
        struct Entry: Decodable { let version: String }
        let results: [Entry]
    }

    /// The version the store page serves right now, or nil for "unknown" —
    /// offline, rate-limited, a malformed answer, or an app that has never
    /// been published. Callers must treat nil as "no information", never as
    /// "no update": see `UpdateChecker.evaluate()`.
    static func storeVersion() async -> SemanticVersion? {
        guard let endpoint else { return nil }

        var request = URLRequest(url: endpoint)
        request.timeoutInterval = 10
        request.cachePolicy = .reloadIgnoringLocalCacheData

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let decoded = try? JSONDecoder().decode(LookupResponse.self, from: data),
              let first = decoded.results.first,
              !first.version.isEmpty
        else { return nil }

        return SemanticVersion(first.version)
    }
}
