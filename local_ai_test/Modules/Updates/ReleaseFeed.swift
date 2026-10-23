//
//  ReleaseFeed.swift
//  local_ai_test
//
//  `releases.json` ships in the bundle AND lives in the public repo, but the
//  two copies feed two different things and must not be confused:
//
//    * the BUNDLED copy describes the build the user is running. It is the
//      only source for the What's New sheet.
//    * the REMOTE copy (fetched into Application Support) describes a release
//      that is already on the App Store. It is the only source for the update
//      banner.
//
//  Mixing them is what makes an app advertise an update to itself, so the
//  split is enforced here rather than left to callers. Every failure path
//  simply leaves the banner off — it is a nicety, never a blocker.
//

import Foundation
import Observation

// MARK: - On-disk locations

enum ReleaseFile {

    /// Path of `releases.json` inside the GitHub repo. The bundled resource
    /// and this path must stay in sync — the remote copy is literally the
    /// same file, served raw.
    static let remoteURL = URL(
        string: "https://raw.githubusercontent.com/ananasDDA/pocket-ai-lab/main/local_ai_test/Resources/releases.json"
    )

    /// Downloaded copy. Application Support (not Documents) because the user
    /// has no business seeing it in Files.app, and it is reproducible.
    static var cacheURL: URL? {
        guard let dir = FileManager.default.urls(
            for: .applicationSupportDirectory, in: .userDomainMask
        ).first else { return nil }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("releases-cache.json")
    }

    static func decode(_ data: Data) -> ReleaseInfo? {
        guard let release = try? JSONDecoder().decode(ReleaseInfo.self, from: data),
              release.schemaVersion == ReleaseInfo.supportedSchemaVersion,
              !release.version.isEmpty
        else { return nil }
        return release
    }

    static var bundledURL: URL? {
        // `Bundle.main` covers the app and app-hosted unit tests; the
        // fallbacks keep the lookup working if the resource ever moves into
        // its own bundle.
        if let url = Bundle.main.url(forResource: "releases", withExtension: "json") {
            return url
        }
        return Bundle.allBundles.compactMap {
            $0.url(forResource: "releases", withExtension: "json")
        }.first
    }

    /// The copy compiled into this build. Describes the release this binary
    /// IS (or the one it was cut for), which is what the What's New sheet
    /// needs — the remote copy may already describe a later one.
    static func loadBundled() -> ReleaseInfo? {
        guard let url = bundledURL, let data = try? Data(contentsOf: url) else { return nil }
        return decode(data)
    }

    /// The cached REMOTE copy, and nothing else. Deliberately does not fall
    /// back to the bundled resource: the bundled file is committed before the
    /// build is even submitted, so a `published: true` left in it would make
    /// the shipped app advertise an update to itself — offline, and
    /// undismissable whenever `minimumSupportedVersion` matches. No cache
    /// means no banner until the first successful refresh.
    static func loadCachedRemote() -> ReleaseInfo? {
        guard let cacheURL, let data = try? Data(contentsOf: cacheURL) else { return nil }
        return decode(data)
    }

    static func writeCache(_ data: Data) {
        guard let cacheURL else { return }
        try? data.write(to: cacheURL, options: .atomic)
    }
}

// MARK: - Remote refresh

@MainActor
@Observable
final class ReleaseFeed {

    static let shared = ReleaseFeed()

    /// The banner's only source: the cached copy of the REMOTE file, or nil
    /// until a refresh has succeeded at least once. Never the bundled
    /// resource — see `ReleaseFile.loadCachedRemote()`.
    private(set) var release: ReleaseInfo?
    private(set) var lastRefresh: Date?

    /// The copy compiled into this binary. Never replaced by a remote fetch
    /// and never used for the banner: it feeds the What's New sheet only,
    /// which is the one thing that must describe the running build.
    let bundled: ReleaseInfo?

    @ObservationIgnored
    private var isRefreshing = false

    private init() {
        bundled = ReleaseFile.loadBundled()
        release = ReleaseFile.loadCachedRemote()
    }

    /// Best-effort update. Never throws and never surfaces an error: a stale
    /// release note is strictly better than a spinner the user did not ask
    /// for.
    func refresh() async {
        // Stands down while a benchmark is being measured - see
        // `BenchmarkRunGate`.
        guard !BenchmarkRunGate.shared.isRunning else { return }
        guard !isRefreshing, let url = ReleaseFile.remoteURL else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.cachePolicy = .reloadIgnoringLocalCacheData

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let fetched = ReleaseFile.decode(data)
        else { return }

        ReleaseFile.writeCache(data)
        release = fetched
        lastRefresh = Date()
    }
}
