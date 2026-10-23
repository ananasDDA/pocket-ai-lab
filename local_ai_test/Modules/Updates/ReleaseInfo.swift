//
//  ReleaseInfo.swift
//  local_ai_test
//
//  Wire format of `releases.json`. Like `catalog.json` and `benchmarks.json`
//  this file ships in the bundle AND lives in the public repo, so a release
//  that has already shipped can be described to users running an older
//  build without another App Store round trip.
//

import Foundation

/// One release, as written in `releases.json`.
///
///     { "schemaVersion": 1,
///       "version": "1.0.1", "build": 2, "published": false,
///       "releasedAt": "2026-09-05",
///       "minimumSupportedVersion": "1.0.1",   // public repo only
///       "appStoreURL": "https://apps.apple.com/app/id6802253142",
///       "notes": ["Short user-facing bullet"] }
///
/// `published` is the release switch: the file may be committed as soon as
/// the build is submitted, but the banner stays hidden until the App Store
/// page actually serves the new version and the flag is flipped to true.
nonisolated struct ReleaseInfo: Codable, Sendable, Hashable, Identifiable {

    /// Bumping this in the app makes it ignore older remote documents, and
    /// bumping it in the remote file makes older app builds ignore it —
    /// either way the bundled copy keeps working.
    static let supportedSchemaVersion = 1

    let schemaVersion: Int
    let version: String
    let build: Int
    /// False while the release is still in review. See the type doc.
    let published: Bool
    /// `yyyy-MM-dd`, or nil for a release with no announced date.
    let releasedAt: String?
    /// Installs older than this are told the update is required rather than
    /// merely available. Exists for builds that shipped broken — 1.0 went out
    /// without the llama.cpp engine, and a user on it has no way to know.
    ///
    /// Set in the PUBLIC repo copy only, at publish time. The bundled copy
    /// omits it: only the remote file drives the banner (see `ReleaseFeed`),
    /// and a value committed alongside the build it ships in would be aimed
    /// at that build itself.
    let minimumSupportedVersion: String?
    let appStoreURL: String
    let notes: [String]

    var id: String { "\(version)+\(build)" }

    /// Hand-written so a hand-edited `releases.json` may omit the optional
    /// keys entirely instead of spelling out `null`.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        version = try c.decode(String.self, forKey: .version)
        build = try c.decodeIfPresent(Int.self, forKey: .build) ?? 0
        published = try c.decodeIfPresent(Bool.self, forKey: .published) ?? false
        releasedAt = try c.decodeIfPresent(String.self, forKey: .releasedAt)
        minimumSupportedVersion = try c.decodeIfPresent(
            String.self, forKey: .minimumSupportedVersion)
        appStoreURL = try c.decodeIfPresent(String.self, forKey: .appStoreURL)
            ?? ReleaseInfo.defaultAppStoreURL
        notes = try c.decodeIfPresent([String].self, forKey: .notes) ?? []
    }

    init(
        schemaVersion: Int = ReleaseInfo.supportedSchemaVersion,
        version: String,
        build: Int,
        published: Bool,
        releasedAt: String? = nil,
        minimumSupportedVersion: String? = nil,
        appStoreURL: String = ReleaseInfo.defaultAppStoreURL,
        notes: [String]
    ) {
        self.schemaVersion = schemaVersion
        self.version = version
        self.build = build
        self.published = published
        self.releasedAt = releasedAt
        self.minimumSupportedVersion = minimumSupportedVersion
        self.appStoreURL = appStoreURL
        self.notes = notes
    }

    /// Used when the JSON omits the key. App Store id 6802253142.
    static let defaultAppStoreURL = "https://apps.apple.com/app/id6802253142"

    // MARK: - Derived

    var appVersion: AppVersion { AppVersion(version: version, build: build) }

    var semanticVersion: SemanticVersion { SemanticVersion(version) }

    var minimumSupported: SemanticVersion? {
        minimumSupportedVersion.map(SemanticVersion.init)
    }

    /// Falls back to the canonical App Store link rather than to nil: a
    /// mistyped URL in the JSON must not leave the user with a dead button.
    var storeLink: URL {
        URL(string: appStoreURL) ?? URL(string: Self.defaultAppStoreURL)!
    }

    private static let dateParser: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.timeZone = TimeZone(secondsFromGMT: 0)
        formatter.dateFormat = "yyyy-MM-dd"
        return formatter
    }()

    var releaseDate: Date? {
        releasedAt.flatMap { Self.dateParser.date(from: $0) }
    }

    /// Localized medium date for the banner header, or nil when the JSON
    /// carried no date at all.
    var releaseDateLabel: String? {
        guard let releaseDate else { return nil }
        return releaseDate.formatted(date: .abbreviated, time: .omitted)
    }

    /// True when this release is strictly ahead of what is running, version
    /// first and build as the tie-break.
    ///
    /// NOT the banner rule - `UpdateChecker.decide` deliberately requires a
    /// newer MARKETING version and lets the build only suppress, because a
    /// build-number tie-break once nagged users who were already on the
    /// release. Use this for ordering, not for deciding.
    func isNewer(than installed: AppVersion) -> Bool {
        installed < appVersion
    }
}
