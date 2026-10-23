//
//  UpdateChecker.swift
//  local_ai_test
//
//  Decides whether the Models tab shows an "update available" banner, and
//  whether the What's New sheet is owed after an update just landed.
//
//  Two independent sources have to agree before the banner appears, because
//  each one alone gets it wrong in a way the user would feel:
//
//    * the REMOTE `releases.json` knows what the release CONTAINS and when we
//      consider it announceable, but it is committed before review finishes.
//      (The bundled copy is never consulted here - see `ReleaseFeed`.)
//    * the iTunes lookup knows what the App Store actually SERVES, but it
//      knows nothing about release notes and lags behind by up to a few hours
//      after approval.
//
//  So: the feed proposes, the store confirms, and an unreachable store falls
//  back to trusting the feed's `published` flag. The whole rule is the pure
//  `decide(...)` below, which is where the unit tests aim; `evaluate()` only
//  feeds it the persisted state.
//

import Foundation
import Observation

@MainActor
@Observable
final class UpdateChecker {

    static let shared = UpdateChecker()

    /// Non-nil when the banner should be on screen. Never set for a release
    /// that is not strictly newer than what is running.
    private(set) var available: ReleaseInfo?

    /// True when the installed version is older than the release's
    /// `minimumSupportedVersion` — i.e. this build is known-broken. The
    /// banner then runs a hotter gradient, loses its "Later" button, and can
    /// never be collapsed.
    private(set) var isMandatory = false

    /// True when the user tapped "Later" on exactly this release. The banner
    /// stays on screen either way — this only picks the compact single-line
    /// strip over the full card. A mandatory update is never collapsed.
    private(set) var isCollapsed = false

    /// Presented once per installed version. Driven off the BUNDLED
    /// `releases.json`, never the remote one: it describes this binary.
    private(set) var whatsNew: ReleaseInfo?

    @ObservationIgnored
    private var isChecking = false

    private init() {}

    // MARK: - Persistence

    private enum Keys {
        /// Marketing version the user tapped "Later" on. "Later" collapses
        /// the banner to a one-line strip rather than hiding it, so this is
        /// a display preference and not a suppression.
        static let collapsedVersion = "updates.collapsedVersion"
        /// The key the same preference used before "Later" meant "collapse"
        /// instead of "hide". Read once, so an existing install carries its
        /// choice over as a collapsed strip instead of a re-opened card.
        static let legacyDismissedVersion = "updates.dismissedVersion"
        /// Date of the last network check (feed + store lookup).
        static let lastCheck = "updates.lastCheck"
        /// Whether that last check actually reached anything. A check where
        /// both calls failed is retried much sooner — see `dueInterval`.
        static let lastCheckSucceeded = "updates.lastCheckSucceeded"
        /// Last store version seen, kept so a throttled launch still has an
        /// answer to compare against.
        static let storeVersion = "updates.storeVersion"
        /// When that store version was fetched. Without it a stale value
        /// could hide a legitimate update forever.
        static let storeVersionCheckedAt = "updates.storeVersionCheckedAt"
        /// Installed version whose What's New sheet has been shown.
        static let lastSeenVersion = "updates.lastSeenVersion"

        #if DEBUG
        /// DEBUG only. Set it to true to force the banner on a dev build:
        ///   UserDefaults.standard.set(true, forKey: "debug.forceUpdateBanner")
        static let forceUpdateBanner = "debug.forceUpdateBanner"
        /// DEBUG only. Makes the forced banner the non-dismissable variant.
        static let forceUpdateMandatory = "debug.forceUpdateMandatory"

        /// DEBUG only. Three input substitutions applied inside `evaluate()`
        /// before `decide()` is called, so `decide()` itself runs unchanged
        /// and the real decision path can be demoed end to end without
        /// publishing anything. Readable as launch arguments — UserDefaults
        /// picks up `-key value` automatically — e.g.:
        ///
        ///   xcrun devicectl device process launch --device <id> self.pro-ai-lab -- \
        ///     -debug.installedVersion 1.0.0 -debug.storeVersion 1.0.1 \
        ///     -debug.releaseJSON '{"schemaVersion":1,"version":"1.0.1","build":2,"published":true,"releasedAt":"2026-09-05","appStoreURL":"https://apps.apple.com/app/id6802253142","notes":["..."]}'
        ///
        /// Overrides the installed MARKETING version compared against the
        /// release. The build stays whatever is actually installed, unless
        /// the value carries a trailing "+build", e.g. "1.0.0+1".
        static let debugInstalledVersion = "debug.installedVersion"
        /// DEBUG only. Overrides the App Store lookup result; the
        /// substituted answer is always treated as fresh (storeAge = 0).
        static let debugStoreVersion = "debug.storeVersion"
        /// DEBUG only. An inline `releases.json` document, decoded with the
        /// same `ReleaseFile.decode` validation as the bundled/remote files
        /// and used instead of the cached remote feed for this evaluation
        /// only — it is never written to the cache. While this key is set,
        /// `checkIfNeeded()` also skips the real feed refresh so a
        /// concurrent network answer cannot clobber the demo mid-flight.
        static let debugReleaseJSON = "debug.releaseJSON"
        /// `-debug.persistOverrides YES` copies the three overrides above
        /// from the volatile launch-argument domain into the persistent one,
        /// so a demo survives a manual relaunch exactly like a cached remote
        /// release would. `-debug.clearOverrides YES` removes them again.
        static let debugPersistOverrides = "debug.persistOverrides"
        static let debugClearOverrides = "debug.clearOverrides"
        #endif
    }

    /// Network checks are cheap but pointless more than a few times a day —
    /// releases do not land that often, and every foreground would otherwise
    /// hit Apple.
    nonisolated static let checkInterval: TimeInterval = 6 * 60 * 60

    /// Used instead of `checkInterval` after a check where nothing answered.
    /// A launch in airplane mode must not burn the whole six-hour window.
    nonisolated static let retryInterval: TimeInterval = 15 * 60

    /// How long a store answer stays usable. Past this it is treated as
    /// "never asked" rather than as evidence, so a version cached back when
    /// the store still served the old build cannot suppress the banner
    /// indefinitely.
    nonisolated static let storeVersionTTL: TimeInterval = 48 * 60 * 60

    private var defaults: UserDefaults { .standard }

    /// Parsed, not raw: "1.0" and "1.0.0" are the same collapse. Falls back
    /// to the pre-rename key so an upgrade does not re-open a card the user
    /// already put away.
    private var collapsedVersion: SemanticVersion? {
        let raw = defaults.string(forKey: Keys.collapsedVersion)
            ?? defaults.string(forKey: Keys.legacyDismissedVersion)
        return raw.map(SemanticVersion.init)
    }

    /// nil means "we have never had a usable answer from Apple".
    private var cachedStoreVersion: SemanticVersion? {
        get { defaults.string(forKey: Keys.storeVersion).map(SemanticVersion.init) }
        set {
            defaults.set(newValue?.description, forKey: Keys.storeVersion)
            defaults.set(Date(), forKey: Keys.storeVersionCheckedAt)
        }
    }

    /// Age of `cachedStoreVersion`, or nil when there is no timestamp — which
    /// includes values written by a build that predates the timestamp, and is
    /// treated as "unknown" by `decide`.
    private var cachedStoreVersionAge: TimeInterval? {
        guard let stamped = defaults.object(forKey: Keys.storeVersionCheckedAt) as? Date else {
            return nil
        }
        return Date().timeIntervalSince(stamped)
    }

    // MARK: - Installed build

    private var installed: AppVersion { AppVersion.installed }

    // MARK: - Entry points

    /// Called on launch and on every `scenePhase == .active`. Re-evaluates
    /// from whatever is already cached every time; only the network part is
    /// throttled, so a dismissal or a fresh install is reflected immediately.
    func checkIfNeeded() async {
        guard !isChecking else { return }
        isChecking = true
        defer { isChecking = false }

        #if DEBUG
        let overrideKeys = [Keys.debugInstalledVersion, Keys.debugStoreVersion, Keys.debugReleaseJSON]
        if defaults.bool(forKey: Keys.debugClearOverrides) {
            overrideKeys.forEach { defaults.removeObject(forKey: $0) }
        } else if defaults.bool(forKey: Keys.debugPersistOverrides) {
            // `string(forKey:)` reads the argument domain first; `set` writes
            // the persistent domain, which is what later launches read.
            for key in overrideKeys {
                if let value = defaults.string(forKey: key) { defaults.set(value, forKey: key) }
            }
        }
        #endif

        refreshWhatsNew()

        if isDue {
            // A `debug.releaseJSON` override already stands in for the feed
            // this evaluation; fetching for real would race it and make the
            // demo non-deterministic, so skip rather than let it clobber.
            #if DEBUG
            let skipFeedRefresh = defaults.string(forKey: Keys.debugReleaseJSON) != nil
            #else
            let skipFeedRefresh = false
            #endif

            // `lastRefresh` only moves on a fetch that decoded, so this is a
            // real "the feed answered", not "the call returned".
            let refreshMark = ReleaseFeed.shared.lastRefresh
            if !skipFeedRefresh {
                await ReleaseFeed.shared.refresh()
            }
            let feedAnswered = !skipFeedRefresh && ReleaseFeed.shared.lastRefresh != refreshMark

            // A failed lookup must not erase a previously confirmed store
            // version — "we could not ask today" is not "it disappeared".
            var lookupAnswered = false
            if let store = await AppStoreLookup.storeVersion() {
                cachedStoreVersion = store
                lookupAnswered = true
            }

            // Stamping the window on a check where nothing answered would
            // hand an offline launch a six-hour silence it did not earn.
            defaults.set(Date(), forKey: Keys.lastCheck)
            defaults.set(feedAnswered || lookupAnswered, forKey: Keys.lastCheckSucceeded)
        }

        evaluate()
    }

    private var isDue: Bool {
        guard let last = defaults.object(forKey: Keys.lastCheck) as? Date else { return true }
        let interval = defaults.bool(forKey: Keys.lastCheckSucceeded)
            ? Self.checkInterval
            : Self.retryInterval
        return Date().timeIntervalSince(last) >= interval
    }

    /// "Later". Collapses the banner to the one-line strip for this
    /// marketing version; it never hides it. The next release brings the full
    /// card back, and a mandatory update refuses to collapse at all.
    func collapse() {
        guard let available, !isMandatory else { return }
        // Normalised, so "1.0.1" collapsed and "1.0.1.0" served still match.
        defaults.set(available.semanticVersion.description, forKey: Keys.collapsedVersion)
        defaults.removeObject(forKey: Keys.legacyDismissedVersion)
        isCollapsed = true
    }

    /// Tapping the compact strip. Re-expanding is remembered too, so the card
    /// stays open across launches until "Later" is tapped again.
    func expand() {
        defaults.removeObject(forKey: Keys.collapsedVersion)
        defaults.removeObject(forKey: Keys.legacyDismissedVersion)
        isCollapsed = false
    }

    /// The user has seen the What's New sheet for the running version.
    func markWhatsNewSeen() {
        defaults.set(installed.version.description, forKey: Keys.lastSeenVersion)
        whatsNew = nil
    }

    // MARK: - Show rule

    /// What the banner should do. Kept as a value so `decide` can stay pure
    /// and be tested without `UserDefaults`, a bundle or the network.
    nonisolated struct Decision: Equatable, Sendable {
        var showsBanner: Bool
        var isMandatory: Bool

        static let hidden = Decision(showsBanner: false, isMandatory: false)
    }

    /// The banner appears when ALL of these hold:
    ///
    ///   1. the release's MARKETING version is strictly newer than the
    ///      installed one. The build number never raises the banner: a
    ///      `CFBundleVersion` the app cannot read as a number used to fall
    ///      back to 0 and nag users who were already on the release.
    ///   2. the installed build is behind the release's build. Build numbers
    ///      are monotonic across releases here (the checklist bumps
    ///      `CURRENT_PROJECT_VERSION` every time), so an installed build at
    ///      or beyond the release's is a TestFlight copy of something newer
    ///      and must not be told to downgrade. This test can only SUPPRESS.
    ///   3. the feed says `published: true`;
    ///   4. the App Store lookup reports a version >= the feed's version,
    ///      OR the lookup is unknown — offline, rate-limited, never answered,
    ///      or older than `storeVersionTTL` — in which case rule 3 is trusted
    ///      on its own. The fallback is deliberate: a user with no
    ///      connectivity to itunes.apple.com is exactly the user most likely
    ///      to be stuck on a broken build, and the worst case is a banner
    ///      whose App Store page briefly still shows the old version.
    ///
    /// A release the user dismissed is then hidden again — unless the
    /// installed version is below `minimumSupportedVersion`, which no "Later"
    /// can silence.
    ///
    /// - Parameters:
    ///   - storeAge: age of `storeVersion`; nil, or older than
    ///     `storeVersionTTL`, makes the store answer count as unknown.
    ///   - dismissed: the marketing version to suppress, parsed. `evaluate()`
    ///     now passes nil: "Later" collapses the banner instead of hiding it,
    ///     so nothing in the app suppresses a release any more. The rule stays
    ///     here — it is the general one, and it stays under test.
    nonisolated static func decide(
        installed: AppVersion,
        release: ReleaseInfo?,
        storeVersion: SemanticVersion?,
        storeAge: TimeInterval?,
        dismissed: SemanticVersion?
    ) -> Decision {
        guard let release else { return .hidden }

        let mandatory = release.minimumSupported.map { installed.version < $0 } ?? false

        // 1. strictly newer marketing version. The build number is not
        //    consulted: build strings are not guaranteed to grow across
        //    marketing versions, and a newer version is newer regardless of
        //    what CFBundleVersion the two sides happen to carry.
        guard installed.version < release.semanticVersion else { return .hidden }

        // 2. announceable
        guard release.published else { return .hidden }

        // 3. the store agrees, or has not answered recently enough to count
        let isFresh = storeAge.map { $0 >= 0 && $0 <= storeVersionTTL } ?? false
        if isFresh, let store = storeVersion, store < release.semanticVersion {
            return .hidden
        }

        // 4. dismissal, which a mandatory update overrides
        if !mandatory, let dismissed, dismissed == release.semanticVersion {
            return .hidden
        }

        return Decision(showsBanner: true, isMandatory: mandatory)
    }

    #if DEBUG
    /// `debug.installedVersion` override — see `Keys.debugInstalledVersion`.
    private var debugInstalledOverride: AppVersion? {
        guard let raw = defaults.string(forKey: Keys.debugInstalledVersion) else { return nil }
        let parts = raw.split(separator: "+", maxSplits: 1)
        let build = parts.count > 1 ? SemanticVersion(String(parts[1])) : installed.build
        return AppVersion(version: SemanticVersion(String(parts[0])), build: build)
    }

    /// `debug.storeVersion` override — see `Keys.debugStoreVersion`.
    private var debugStoreVersionOverride: SemanticVersion? {
        defaults.string(forKey: Keys.debugStoreVersion).map(SemanticVersion.init)
    }

    /// `debug.releaseJSON` override — see `Keys.debugReleaseJSON`. Decoded
    /// with the same validation as the bundled/remote files; never written
    /// to the cache.
    /// Accepts the document either as raw JSON or base64-encoded JSON. The
    /// base64 form exists because a launch argument that starts with `{` is
    /// parsed by UserDefaults as an old-style property list dictionary and
    /// never reaches `string(forKey:)`.
    private var debugReleaseOverride: ReleaseInfo? {
        guard let raw = defaults.string(forKey: Keys.debugReleaseJSON) else { return nil }
        let data: Data?
        if raw.trimmingCharacters(in: .whitespaces).hasPrefix("{") {
            data = raw.data(using: .utf8)
        } else {
            data = Data(base64Encoded: raw, options: .ignoreUnknownCharacters)
        }
        guard let data else { return nil }
        return ReleaseFile.decode(data)
    }
    #endif

    private func evaluate() {
        #if DEBUG
        if defaults.bool(forKey: Keys.forceUpdateBanner) {
            let forced = ReleaseFeed.shared.release ?? Self.debugSampleRelease
            available = forced
            isMandatory = defaults.bool(forKey: Keys.forceUpdateMandatory)
            isCollapsed = !isMandatory && collapsedVersion == forced.semanticVersion
            return
        }
        #endif

        // The three `debug.*` overrides are input substitutions applied here,
        // before `decide()` runs — `decide()` itself is never touched.
        #if DEBUG
        let effectiveInstalled = debugInstalledOverride ?? installed
        let effectiveRelease = debugReleaseOverride ?? ReleaseFeed.shared.release
        let effectiveStoreVersion = debugStoreVersionOverride ?? cachedStoreVersion
        let effectiveStoreAge = debugStoreVersionOverride != nil ? 0 : cachedStoreVersionAge
        #else
        let effectiveInstalled = installed
        let effectiveRelease = ReleaseFeed.shared.release
        let effectiveStoreVersion = cachedStoreVersion
        let effectiveStoreAge = cachedStoreVersionAge
        #endif

        // `dismissed: nil` on purpose. "Later" no longer suppresses the
        // banner — it collapses it — so the visibility rule must not consult
        // it. `decide` keeps the parameter (and its tests keep exercising it)
        // as the general rule; this caller simply has nothing to suppress.
        let decision = Self.decide(
            installed: effectiveInstalled,
            release: effectiveRelease,
            storeVersion: effectiveStoreVersion,
            storeAge: effectiveStoreAge,
            dismissed: nil
        )

        available = decision.showsBanner ? effectiveRelease : nil
        isMandatory = decision.showsBanner && decision.isMandatory
        isCollapsed = !isMandatory
            && decision.showsBanner
            && collapsedVersion == effectiveRelease?.semanticVersion
    }

    /// The sheet is owed when the bundled notes describe the version that is
    /// actually running (so a stale bundled file cannot advertise a release
    /// the user does not have) and that version has not been acknowledged
    /// yet. An update from 1.0 to 1.0.1 sees it once; every later launch does
    /// not, and a first-ever install does not see it at all.
    private func refreshWhatsNew() {
        guard let bundled = ReleaseFeed.shared.bundled, !bundled.notes.isEmpty else {
            whatsNew = nil
            return
        }
        let installedVersion = installed.version
        guard bundled.semanticVersion == installedVersion else {
            whatsNew = nil
            return
        }

        let lastSeen = defaults.string(forKey: Keys.lastSeenVersion).map(SemanticVersion.init)
        if let lastSeen, lastSeen == installedVersion {
            whatsNew = nil
            return
        }

        // "What's new" only means something against a previous version. On a
        // first-ever install there is nothing to contrast it with, so the
        // acknowledgement is written silently and the sheet is skipped.
        if lastSeen == nil, isFirstEverLaunch {
            defaults.set(installedVersion.description, forKey: Keys.lastSeenVersion)
            whatsNew = nil
            return
        }

        whatsNew = bundled
    }

    /// Cheap "this app has never been used" probe: nothing has ever been
    /// downloaded or imported, and no benchmark has ever been run. Both
    /// stores are already loaded by the time the Models tab draws, and the
    /// model registry is checked first so the common case never touches the
    /// reports directory.
    private var isFirstEverLaunch: Bool {
        guard InstalledModelsStore.shared.knownModels.isEmpty else { return false }
        return BenchmarkReportStore.shared.reports.isEmpty
    }

    #if DEBUG
    /// Only reached when `releases.json` is missing entirely, which the debug
    /// override should still be able to exercise.
    private static let debugSampleRelease = ReleaseInfo(
        version: "9.9.9",
        build: 999,
        published: true,
        releasedAt: "2026-09-05",
        minimumSupportedVersion: nil,
        notes: [
            "Debug banner - releases.json was not found",
            "Set debug.forceUpdateBanner to false to hide"
        ]
    )
    #endif
}
