//
//  SemanticVersion.swift
//  local_ai_test
//
//  Numeric comparison for the dotted version strings that come out of
//  `Bundle.main`, `releases.json` and the iTunes lookup API. String
//  comparison is wrong for exactly the case that matters ("1.0.10" < "1.0.9"
//  lexicographically), so every comparison in the update path goes through
//  this type.
//

import Foundation

/// A dotted numeric version. Non-numeric components and anything after them
/// are ignored, so a hand-typed "1.2.0-beta" still compares as 1.2.0 rather
/// than failing to parse and silencing the whole update check.
nonisolated struct SemanticVersion: Comparable, Hashable, Sendable, CustomStringConvertible {

    /// Always at least one element. Trailing zeroes are kept as written —
    /// comparison pads instead, so "1.0" == "1.0.0".
    let components: [Int]

    init(_ string: String) {
        let parsed = string
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ".")
            .map { part -> Int? in
                // Leading digits only: "3beta" -> 3, "beta" -> nil.
                let digits = part.prefix { $0.isNumber }
                return digits.isEmpty ? nil : Int(digits)
            }

        var values: [Int] = []
        for value in parsed {
            guard let value else { break }
            values.append(value)
        }
        components = values.isEmpty ? [0] : values
    }

    /// Missing components count as zero, so 1.0 < 1.0.1 and 1.0 == 1.0.0.
    static func < (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        let count = max(lhs.components.count, rhs.components.count)
        for index in 0..<count {
            let left = index < lhs.components.count ? lhs.components[index] : 0
            let right = index < rhs.components.count ? rhs.components[index] : 0
            if left != right { return left < right }
        }
        return false
    }

    static func == (lhs: SemanticVersion, rhs: SemanticVersion) -> Bool {
        let count = max(lhs.components.count, rhs.components.count)
        for index in 0..<count {
            let left = index < lhs.components.count ? lhs.components[index] : 0
            let right = index < rhs.components.count ? rhs.components[index] : 0
            if left != right { return false }
        }
        return true
    }

    func hash(into hasher: inout Hasher) {
        // Must agree with `==`: trailing zeroes are not part of the identity.
        var trimmed = components
        while trimmed.count > 1, trimmed.last == 0 { trimmed.removeLast() }
        hasher.combine(trimmed)
    }

    var description: String { components.map(String.init).joined(separator: ".") }
}

// MARK: - Version + build

/// A marketing version paired with its build number. Two releases can carry
/// the same `CFBundleShortVersionString` while the build differs (TestFlight
/// does this constantly), so the build breaks the tie whenever the marketing
/// versions compare equal.
///
/// The build is a `SemanticVersion` rather than an `Int` on purpose:
/// `CFBundleVersion` is only required to be a dotted numeric string, and
/// "1.0.1" used to parse as `Int` → nil → 0, which made an installed build
/// look older than every release and nagged users who were already on it.
/// See `UpdateChecker.decide` — the build now only ever SUPPRESSES the
/// banner, it can never raise one.
nonisolated struct AppVersion: Comparable, Hashable, Sendable, CustomStringConvertible {

    let version: SemanticVersion
    let build: SemanticVersion

    init(version: SemanticVersion, build: SemanticVersion) {
        self.version = version
        self.build = build
    }

    init(version: SemanticVersion, build: Int) {
        self.init(version: version, build: SemanticVersion(String(build)))
    }

    init(version: String, build: Int) {
        self.init(version: SemanticVersion(version), build: build)
    }

    init(version: String, build: String) {
        self.init(version: SemanticVersion(version), build: SemanticVersion(build))
    }

    static func < (lhs: AppVersion, rhs: AppVersion) -> Bool {
        if lhs.version != rhs.version { return lhs.version < rhs.version }
        return lhs.build < rhs.build
    }

    var description: String { "\(version) (\(build))" }

    /// What is installed right now. Both fields are parsed leniently, so a
    /// `CFBundleVersion` of "1.0.1", "2" or even "2-internal" all compare
    /// numerically instead of collapsing to zero.
    static var installed: AppVersion {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "0"
        let build = info?["CFBundleVersion"] as? String ?? "0"
        return AppVersion(version: short, build: build)
    }
}
