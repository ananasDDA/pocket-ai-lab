//
//  HFMetadataStore.swift
//  local_ai_test
//
//  Cosmetic Hugging Face metadata for model cards: download/like counts, the
//  pipeline tag, the last push date and the owner's avatar. Everything here is
//  best-effort — a card without network looks exactly like it did before, just
//  without the metrics row, so every failure path is silent.
//

import Foundation
import Observation

// MARK: - Wire types

/// The subset of `/api/models/{repo}` the cards actually render. Deliberately
/// separate from `HFRepoInfo`: that one carries the full sibling list, which is
/// megabytes for large repos and useless for a metrics row.
///
/// Decode-only on purpose — `lastModified` arrives as an ISO 8601 string, and
/// what gets persisted is `HFMetadataStore.RepoEntry`, not this.
nonisolated struct HFRepoStats: Decodable, Sendable, Equatable {
    var downloads: Int = 0
    var likes: Int = 0
    var pipelineTag: String?
    var lastModified: Date?

    init(downloads: Int = 0, likes: Int = 0, pipelineTag: String? = nil, lastModified: Date? = nil) {
        self.downloads = downloads
        self.likes = likes
        self.pipelineTag = pipelineTag
        self.lastModified = lastModified
    }

    private enum CodingKeys: String, CodingKey {
        case downloads, likes, lastModified
        case pipelineTag = "pipeline_tag"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Every field is optional: HF omits `pipeline_tag` for plenty of repos
        // and a missing `likes` must not cost us the whole row.
        self.init(
            downloads: try c.decodeIfPresent(Int.self, forKey: .downloads) ?? 0,
            likes: try c.decodeIfPresent(Int.self, forKey: .likes) ?? 0,
            pipelineTag: try c.decodeIfPresent(String.self, forKey: .pipelineTag),
            lastModified: (try c.decodeIfPresent(String.self, forKey: .lastModified))
                .flatMap(HFRepoStats.parseTimestamp)
        )
    }

    /// HF sends `2024-09-25T17:12:35.000Z`, but not every repo carries the
    /// fractional part, so both spellings have to parse.
    static func parseTimestamp(_ string: String) -> Date? {
        let fractional = ISO8601DateFormatter()
        fractional.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = fractional.date(from: string) { return date }
        let plain = ISO8601DateFormatter()
        plain.formatOptions = [.withInternetDateTime]
        return plain.date(from: string)
    }
}

/// What an avatar lookup told us. "No avatar" and "could not ask" are cached
/// differently: the first is a fact, the second is worth retrying.
nonisolated enum HFAvatarLookup: Sendable, Equatable {
    case found(URL)
    case absent
    case failed
}

/// Everything a card needs to draw its Hugging Face header.
nonisolated struct HFModelMetadata: Sendable, Equatable {
    var downloads: Int
    var likes: Int
    var pipelineTag: String?
    var lastModified: Date?
    var avatarURL: URL?
}

// MARK: - Network

/// Stateless half of the store, kept off the main actor so decoding never runs
/// on it. Nothing here throws — callers only care whether a value arrived.
nonisolated enum HFMetadataFetcher {

    static func stats(repo: String) async -> HFRepoStats? {
        guard let url = URL(string: "\(HuggingFaceAPI.webHost)/api/models/\(repo)") else {
            return nil
        }
        guard let (data, response) = try? await URLSession.shared.data(from: url),
              (response as? HTTPURLResponse)?.statusCode == 200
        else { return nil }
        return try? JSONDecoder().decode(HFRepoStats.self, from: data)
    }

    /// Owners can be organizations or plain users and the endpoints are
    /// separate, so an org miss falls through to the user namespace.
    static func avatarURL(owner: String) async -> HFAvatarLookup {
        var sawFailure = false
        for namespace in ["organizations", "users"] {
            switch await lookup(path: "/api/\(namespace)/\(owner)/avatar") {
            case .found(let url): return .found(url)
            case .absent:         continue
            case .failed:         sawFailure = true
            }
        }
        return sawFailure ? .failed : .absent
    }

    private struct AvatarResponse: Decodable {
        var avatarUrl: String?
    }

    private static func lookup(path: String) async -> HFAvatarLookup {
        guard let url = URL(string: HuggingFaceAPI.webHost + path) else { return .absent }
        guard let (data, response) = try? await URLSession.shared.data(from: url) else {
            return .failed
        }
        switch (response as? HTTPURLResponse)?.statusCode {
        case 200:  break
        case 404:  return .absent
        default:   return .failed
        }
        guard let decoded = try? JSONDecoder().decode(AvatarResponse.self, from: data),
              let raw = decoded.avatarUrl, !raw.isEmpty
        else { return .absent }
        // Some accounts get a site-relative path instead of a CDN URL.
        let absolute = raw.hasPrefix("/") ? HuggingFaceAPI.webHost + raw : raw
        return URL(string: absolute).map { .found($0) } ?? .absent
    }
}

// MARK: - Store

/// Process-wide cache in front of `HFMetadataFetcher`, persisted so a relaunch
/// draws complete cards immediately instead of popping metrics in one by one.
@MainActor
@Observable
final class HFMetadataStore {

    static let shared = HFMetadataStore()

    /// Stats move slowly; a day-old download count is indistinguishable from a
    /// live one at this size on screen.
    nonisolated static let ttl: TimeInterval = 24 * 60 * 60

    private var repos: [String: RepoEntry] = [:]
    /// Keyed by owner, not by repo — one organization backs a dozen cards.
    private var avatars: [String: AvatarEntry] = [:]

    @ObservationIgnored private var inFlightRepos: Set<String> = []
    @ObservationIgnored private var inFlightOwners: Set<String> = []

    private init() {
        load()
    }

    // MARK: Reading

    /// Cached metadata, or nil while nothing has arrived yet. Stale entries are
    /// still returned: showing yesterday's count beats blanking the row while
    /// `ensureLoaded` refreshes it.
    func metadata(for repo: String) -> HFModelMetadata? {
        let entry = repos[repo]
        let avatar = Self.owner(of: repo).flatMap { avatars[$0]?.url }
        guard entry != nil || avatar != nil else { return nil }
        return HFModelMetadata(
            downloads: entry?.downloads ?? 0,
            likes: entry?.likes ?? 0,
            pipelineTag: entry?.pipelineTag,
            lastModified: entry?.lastModified,
            avatarURL: avatar
        )
    }

    // MARK: Loading

    /// Idempotent and cheap to call from every card's `.task`: it returns
    /// immediately unless the entry is missing, expired, or already in flight.
    func ensureLoaded(repo: String) {
        guard let owner = Self.owner(of: repo) else { return }
        loadStats(repo: repo)
        loadAvatar(owner: owner)
    }

    private func loadStats(repo: String) {
        if let entry = repos[repo], !Self.isExpired(fetchedAt: entry.fetchedAt) { return }
        guard !inFlightRepos.contains(repo) else { return }
        inFlightRepos.insert(repo)

        Task {
            let stats = await HFMetadataFetcher.stats(repo: repo)
            inFlightRepos.remove(repo)
            guard let stats else { return }
            repos[repo] = RepoEntry(stats: stats, fetchedAt: Date())
            save()
        }
    }

    private func loadAvatar(owner: String) {
        if let entry = avatars[owner], !Self.isExpired(fetchedAt: entry.fetchedAt) { return }
        guard !inFlightOwners.contains(owner) else { return }
        inFlightOwners.insert(owner)

        Task {
            let lookup = await HFMetadataFetcher.avatarURL(owner: owner)
            inFlightOwners.remove(owner)
            switch lookup {
            case .found(let url):
                avatars[owner] = AvatarEntry(url: url, fetchedAt: Date())
            case .absent:
                // Cache the negative too, otherwise every scroll re-asks.
                avatars[owner] = AvatarEntry(url: nil, fetchedAt: Date())
            case .failed:
                return
            }
            save()
        }
    }

    // MARK: Pure helpers

    /// Owner half of `owner/repo`, or nil if the string is not a repo id.
    nonisolated static func owner(of repo: String) -> String? {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: true)
        guard parts.count >= 2 else { return nil }
        return String(parts[0])
    }

    /// A timestamp from the future means the clock moved; treat it as expired
    /// rather than trusting an entry that can never age out.
    nonisolated static func isExpired(
        fetchedAt: Date,
        now: Date = Date(),
        ttl: TimeInterval = HFMetadataStore.ttl
    ) -> Bool {
        let age = now.timeIntervalSince(fetchedAt)
        return age < 0 || age >= ttl
    }

    /// `1234` → `1.2K`, `3_400_000` → `3.4M`. Matches how huggingface.co
    /// abbreviates the same counters.
    nonisolated static func compactCount(_ value: Int) -> String {
        switch value {
        case 1_000_000...: return String(format: "%.1fM", Double(value) / 1_000_000)
        case 1_000...:     return String(format: "%.1fK", Double(value) / 1_000)
        default:           return "\(value)"
        }
    }

    // MARK: Persistence

    private struct RepoEntry: Codable {
        var downloads: Int = 0
        var likes: Int = 0
        var pipelineTag: String?
        var lastModified: Date?
        var fetchedAt: Date

        init(stats: HFRepoStats, fetchedAt: Date) {
            self.downloads = stats.downloads
            self.likes = stats.likes
            self.pipelineTag = stats.pipelineTag
            self.lastModified = stats.lastModified
            self.fetchedAt = fetchedAt
        }
    }

    private struct AvatarEntry: Codable {
        var url: URL?
        var fetchedAt: Date
    }

    private struct CacheFile: Codable {
        var schemaVersion: Int
        var repos: [String: RepoEntry]
        var avatars: [String: AvatarEntry]
    }

    private static let cacheSchemaVersion = 1

    @ObservationIgnored
    private lazy var cacheURL: URL? = {
        guard let dir = FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first
        else { return nil }
        // Application Support is not created for us on iOS.
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("hf-metadata-cache.json")
    }()

    private func load() {
        guard let cacheURL,
              let data = try? Data(contentsOf: cacheURL),
              let file = try? JSONDecoder().decode(CacheFile.self, from: data),
              file.schemaVersion == Self.cacheSchemaVersion
        else { return }
        repos = file.repos
        avatars = file.avatars
    }

    private func save() {
        guard let cacheURL else { return }
        let file = CacheFile(
            schemaVersion: Self.cacheSchemaVersion,
            repos: repos,
            avatars: avatars
        )
        guard let data = try? JSONEncoder().encode(file) else { return }
        try? data.write(to: cacheURL, options: .atomic)
    }
}
