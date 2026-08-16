//
//  HuggingFaceAPI.swift
//  local_ai_test
//
//  Read-only client for the public Hugging Face model API. Used by the
//  "import from link" flow to work out what a repo actually contains before
//  the downloader commits to gigabytes of traffic.
//

import Foundation

// MARK: - Errors

nonisolated enum HuggingFaceError: Error, LocalizedError, Equatable {
    case invalidLink(String)
    case notFound(String)
    case gated(String)
    case restricted(String)
    case badResponse(Int)
    case network(String)
    case decoding(String)
    case unsupportedRepo(String)

    var errorDescription: String? {
        switch self {
        case .invalidLink(let text):
            return "\"\(text)\" is not a Hugging Face model link. "
                + "Use huggingface.co/owner/repo or just owner/repo."
        case .notFound(let repo):
            return "No model repo named \(repo) on Hugging Face. Check the spelling."
        case .gated(let repo):
            return "\(repo) is gated. Open huggingface.co/\(repo), accept the license, "
                + "then use a mirror repo — this app downloads anonymously and cannot "
                + "pass your account token."
        case .restricted(let repo):
            return "\(repo) is private. Only its owner can download it."
        case .badResponse(let code):
            return "Hugging Face answered HTTP \(code). Try again in a moment."
        case .network(let message):
            return "Network error: \(message)"
        case .decoding(let message):
            return "Unexpected response from Hugging Face: \(message)"
        case .unsupportedRepo(let reason):
            return reason
        }
    }
}

// MARK: - Wire types

nonisolated struct HFSibling: Decodable, Sendable, Hashable {
    var rfilename: String
    var size: Int64?

    init(rfilename: String, size: Int64? = nil) {
        self.rfilename = rfilename
        self.size = size
    }
}

/// HF reports `gated` as `false`, `"auto"` or `"manual"` — three JSON types in
/// one field, so it needs a hand-written decoder.
nonisolated enum HFGatedStatus: Sendable, Equatable {
    case notGated
    /// License must be accepted, but access is granted instantly.
    case auto
    /// A human reviews each request.
    case manual

    var requiresLicenseAcceptance: Bool { self != .notGated }
}

nonisolated extension HFGatedStatus: Decodable {
    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let flag = try? container.decode(Bool.self) {
            self = flag ? .manual : .notGated
            return
        }
        switch (try? container.decode(String.self))?.lowercased() {
        case "auto":            self = .auto
        case "manual":          self = .manual
        case "false", "none":   self = .notGated
        case .none:             self = .notGated
        // Unknown non-empty value: assume the strictest reading.
        case .some:             self = .manual
        }
    }
}

nonisolated struct HFRepoInfo: Sendable {
    var id: String = ""
    /// Commit the response describes. Downloads pin to it.
    var sha: String?
    var gated: HFGatedStatus = .notGated
    var isPrivate: Bool = false
    var downloads: Int = 0
    var likes: Int = 0
    var tags: [String] = []
    var siblings: [HFSibling] = []
    var spaces: [String] = []
}

nonisolated extension HFRepoInfo: Decodable {

    private enum CodingKeys: String, CodingKey {
        case id, sha, gated, downloads, likes, tags, siblings, spaces
        case isPrivate = "private"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        // Everything except `id` is optional: the API omits fields for some
        // repos, and a missing `likes` must not fail the whole import.
        self.init(
            id: try c.decodeIfPresent(String.self, forKey: .id) ?? "",
            sha: try c.decodeIfPresent(String.self, forKey: .sha),
            gated: try c.decodeIfPresent(HFGatedStatus.self, forKey: .gated) ?? .notGated,
            isPrivate: try c.decodeIfPresent(Bool.self, forKey: .isPrivate) ?? false,
            downloads: try c.decodeIfPresent(Int.self, forKey: .downloads) ?? 0,
            likes: try c.decodeIfPresent(Int.self, forKey: .likes) ?? 0,
            tags: try c.decodeIfPresent([String].self, forKey: .tags) ?? [],
            siblings: try c.decodeIfPresent([HFSibling].self, forKey: .siblings) ?? [],
            spaces: try c.decodeIfPresent([String].self, forKey: .spaces) ?? []
        )
    }
}

// MARK: - API

enum HuggingFaceAPI {

    /// `nonisolated` so off-main-actor fetchers (the card metadata store) can
    /// build URLs without hopping back.
    nonisolated static let webHost = "https://huggingface.co"

    // MARK: Link parsing

    /// Accepts a full repo URL (`https://huggingface.co/owner/repo/tree/main`,
    /// `hf.co/owner/repo?x=1`) or a bare `owner/repo`, and returns the
    /// canonical `owner/repo`. Returns nil for anything else, including links
    /// to spaces/datasets and to other sites.
    static func parseRepoID(from string: String) -> String? {
        var text = string.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return nil }

        // Query and fragment can contain slashes; cut them before splitting.
        if let cut = text.firstIndex(where: { $0 == "?" || $0 == "#" }) {
            text = String(text[..<cut])
        }

        let hadScheme = text.contains("://")
        if let scheme = text.range(of: "://") {
            text = String(text[scheme.upperBound...])
        }

        var matchedHost = false
        for host in ["www.huggingface.co", "huggingface.co", "www.hf.co", "hf.co"]
        where text.lowercased().hasPrefix(host) {
            text = String(text.dropFirst(host.count))
            matchedHost = true
            break
        }

        // A scheme that did not resolve to a HF host points somewhere else.
        if hadScheme && !matchedHost { return nil }

        var parts = text.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
        // huggingface.co/models/owner/repo is a valid canonical form too.
        if parts.first?.lowercased() == "models" { parts.removeFirst() }
        // Spaces and datasets are different resources, not model repos.
        if let first = parts.first?.lowercased(), first == "spaces" || first == "datasets" {
            return nil
        }
        guard parts.count >= 2 else { return nil }

        let owner = parts[0]
        let repo = parts[1]
        guard isValidComponent(owner), isValidComponent(repo) else { return nil }
        return owner + "/" + repo
    }

    private static func isValidComponent(_ component: String) -> Bool {
        guard !component.isEmpty, component != ".", component != ".." else { return false }
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            .union(CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyz"))
            .union(CharacterSet(charactersIn: "0123456789-_."))
        return component.unicodeScalars.allSatisfy { allowed.contains($0) }
    }

    // MARK: Repo info

    static func fetchRepoInfo(repo: String) async throws -> HFRepoInfo {
        guard let url = URL(string: "\(webHost)/api/models/\(repo)?blobs=true") else {
            throw HuggingFaceError.invalidLink(repo)
        }

        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(from: url)
        } catch {
            throw HuggingFaceError.network(error.localizedDescription)
        }

        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        switch status {
        case 200:  break
        case 404:  throw HuggingFaceError.notFound(repo)
        // HF answers 401/403 both for gated repos and for private ones; the
        // body does not reliably distinguish them, so report the common case.
        case 401, 403: throw HuggingFaceError.gated(repo)
        default:   throw HuggingFaceError.badResponse(status)
        }

        do {
            var info = try JSONDecoder().decode(HFRepoInfo.self, from: data)
            if info.id.isEmpty { info.id = repo }
            return info
        } catch {
            throw HuggingFaceError.decoding(error.localizedDescription)
        }
    }

    // MARK: Spaces

    /// Spaces that demo this model, most relevant first. Never throws — a
    /// missing "Try online" button is not worth failing an import over.
    static func fetchSpaces(repo: String) async -> [String] {
        var components = URLComponents(string: "\(webHost)/api/models/\(repo)")
        components?.queryItems = [URLQueryItem(name: "expand[]", value: "spaces")]
        guard let url = components?.url else { return [] }

        guard let (data, response) = try? await URLSession.shared.data(from: url),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let info = try? JSONDecoder().decode(HFRepoInfo.self, from: data)
        else { return [] }
        return info.spaces
    }

    /// Where "Try online" should send the user: a demo Space if one exists,
    /// otherwise the model page itself.
    static func onlineDemoURL(repo: String, spaces: [String]) -> URL? {
        if let space = spaces.first {
            return URL(string: "\(webHost)/spaces/\(space)")
        }
        return URL(string: "\(webHost)/\(repo)")
    }

    // MARK: config.json

    /// `max_position_embeddings` from the repo's config.json, if it is cheap
    /// to get. Returns nil for repos without one (most GGUF repos).
    static func fetchContextLength(repo: String, revision: String?) async -> Int? {
        let rev = revision ?? "main"
        guard let url = URL(string: "\(webHost)/\(repo)/resolve/\(rev)/config.json") else {
            return nil
        }
        var request = URLRequest(url: url)
        request.timeoutInterval = 10

        guard let (data, response) = try? await URLSession.shared.data(for: request),
              (response as? HTTPURLResponse)?.statusCode == 200,
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }

        if let value = json["max_position_embeddings"] as? Int { return value }
        // Multimodal configs nest the language model's settings.
        if let nested = json["text_config"] as? [String: Any],
           let value = nested["max_position_embeddings"] as? Int {
            return value
        }
        return nil
    }
}
