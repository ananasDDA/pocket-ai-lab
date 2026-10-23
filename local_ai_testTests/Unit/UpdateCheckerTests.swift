//
//  UpdateCheckerTests.swift
//  local_ai_testTests
//
//  The update banner is the one piece of UI the app puts in front of a user
//  unprompted, and the two ways it can be wrong are both expensive: nagging
//  someone who is already on the newest build, or staying silent for someone
//  stuck on a build that shipped without an engine.
//
//  Everything here is pure. `UpdateChecker.decide` takes the whole world as
//  arguments — installed version, feed document, store answer and its age,
//  previous dismissal — so the decision table can be exercised without
//  `UserDefaults`, a bundle, or the network.
//

import Testing
import Foundation
@testable import local_ai_test

// MARK: - Version parsing

struct SemanticVersionTests {

    @Test func padsMissingComponents() {
        #expect(SemanticVersion("1.0") == SemanticVersion("1.0.0"))
        #expect(SemanticVersion("1") == SemanticVersion("1.0.0.0"))
        #expect(SemanticVersion("1.0").hashValue == SemanticVersion("1.0.0").hashValue)
    }

    @Test func comparesNumericallyNotLexicographically() {
        #expect(SemanticVersion("1.0.9") < SemanticVersion("1.0.10"))
        #expect(SemanticVersion("1.10") > SemanticVersion("1.9"))
        #expect(SemanticVersion("2.0") > SemanticVersion("1.999.999"))
    }

    @Test func strictOrderingIsIrreflexive() {
        #expect(!(SemanticVersion("1.0.1") < SemanticVersion("1.0.1")))
        #expect(!(SemanticVersion("1.0.1") < SemanticVersion("1.0.1.0")))
    }

    @Test func junkFailsClosedToZero() {
        // Nothing parses, so it sorts below every real version — which can
        // only ever make the banner MORE eager, never hide a real update.
        #expect(SemanticVersion("beta") == SemanticVersion("0"))
        #expect(SemanticVersion("") == SemanticVersion("0"))
        #expect(SemanticVersion("beta") < SemanticVersion("1.0"))
    }

    @Test func takesLeadingDigitsAndStopsAtTheFirstJunkComponent() {
        #expect(SemanticVersion("1.2.0-beta") == SemanticVersion("1.2.0"))
        #expect(SemanticVersion("3beta.1") == SemanticVersion("3.1"))
        // "rc" kills the component and everything after it.
        #expect(SemanticVersion("1.rc.5") == SemanticVersion("1"))
    }
}

// MARK: - Installed version + build

struct AppVersionTests {

    @Test func buildBreaksTiesOnlyWhenVersionsAreEqual() {
        #expect(AppVersion(version: "1.0.1", build: 1) < AppVersion(version: "1.0.1", build: 2))
        // The lower build does not make the older marketing version win.
        #expect(AppVersion(version: "1.0.1", build: 9) > AppVersion(version: "1.0.0", build: 99))
    }

    @Test func parsesDottedBuildNumbersLeniently() {
        // The bug this replaced: `Int("1.0.1")` is nil, so the installed
        // build collapsed to 0 and looked older than every release.
        #expect(AppVersion(version: "1.0.1", build: "1.0.1").build == SemanticVersion("1.0.1"))
        #expect(AppVersion(version: "1.0.1", build: "2").build == SemanticVersion("2"))
        #expect(AppVersion(version: "1.0.1", build: "beta").build == SemanticVersion("0"))
    }
}

// MARK: - The decision table

struct UpdateDecisionTests {

    private static let installedVersion = "1.0.0"
    private static let installedBuild = "1"

    private func installed(
        version: String = installedVersion,
        build: String = installedBuild
    ) -> AppVersion {
        AppVersion(version: version, build: build)
    }

    private func release(
        version: String = "1.0.1",
        build: Int = 2,
        published: Bool = true,
        minimumSupportedVersion: String? = nil
    ) -> ReleaseInfo {
        ReleaseInfo(
            version: version,
            build: build,
            published: published,
            releasedAt: "2026-09-05",
            minimumSupportedVersion: minimumSupportedVersion,
            notes: ["llama.cpp models work again - the engine is back"]
        )
    }

    private func decide(
        installed: AppVersion? = nil,
        release: ReleaseInfo? = nil,
        storeVersion: String? = "1.0.1",
        storeAge: TimeInterval? = 60,
        dismissed: String? = nil
    ) -> UpdateChecker.Decision {
        UpdateChecker.decide(
            installed: installed ?? self.installed(),
            release: release ?? self.release(),
            storeVersion: storeVersion.map(SemanticVersion.init),
            storeAge: storeAge,
            dismissed: dismissed.map(SemanticVersion.init)
        )
    }

    // MARK: Happy path

    @Test func showsWhenNewerPublishedAndTheStoreConfirms() {
        let decision = decide()
        #expect(decision.showsBanner)
        #expect(!decision.isMandatory)
    }

    @Test func hidesWithoutAFeed() {
        // No cached remote document - the bundled copy is never a banner
        // source, so a fresh install stays quiet until the first refresh.
        let decision = UpdateChecker.decide(
            installed: installed(),
            release: nil,
            storeVersion: SemanticVersion("9.9.9"),
            storeAge: 60,
            dismissed: nil
        )
        #expect(decision == .hidden)
    }

    // MARK: Version and build

    @Test func hidesWhenTheReleaseIsNotNewer() {
        #expect(decide(release: release(version: "1.0.0")) == .hidden)
        #expect(decide(release: release(version: "0.9.9")) == .hidden)
    }

    @Test func hidesWhenOnlyTheBuildIsNewer() {
        // Same marketing version, higher build: the store serves what the
        // user already has. The build must never raise the banner.
        #expect(decide(release: release(version: "1.0.0", build: 99)) == .hidden)
    }

    @Test func aDottedInstalledBuildNoLongerNags() {
        // Regression: `CFBundleVersion` of "1.0.1" used to parse as 0, so the
        // build tie-break fired against a user already on 1.0.1.
        let onTheRelease = installed(version: "1.0.1", build: "1.0.1")
        #expect(decide(installed: onTheRelease) == .hidden)
    }

    @Test func showsANewerVersionRegardlessOfBuildNumbers() {
        // Build strings are not comparable across marketing versions, so an
        // installed build equal to or above the release's must not hide a
        // strictly newer version (1.0.0 (2) and 1.0.0 (7) vs 1.0.1 (2)).
        #expect(decide(installed: installed(build: "2")).showsBanner)
        #expect(decide(installed: installed(build: "7")).showsBanner)
    }

    // MARK: Published flag

    @Test func hidesWhileUnpublished() {
        #expect(decide(release: release(published: false)) == .hidden)
    }

    // MARK: Store confirmation

    @Test func hidesWhenTheStoreStillServesSomethingOlder() {
        #expect(decide(storeVersion: "1.0.0") == .hidden)
    }

    @Test func showsWhenTheStoreIsUnknown() {
        // Offline, rate-limited, or never asked: the feed is trusted alone.
        #expect(decide(storeVersion: nil, storeAge: nil).showsBanner)
    }

    @Test func showsWhenTheStoreIsAhead() {
        #expect(decide(storeVersion: "1.1.0").showsBanner)
    }

    @Test func treatsAStaleStoreAnswerAsUnknown() {
        let stale = UpdateChecker.storeVersionTTL + 60
        // The same "1.0.0" that suppresses the banner while fresh...
        #expect(decide(storeVersion: "1.0.0", storeAge: 60) == .hidden)
        // ...stops counting once it is older than the TTL, so a value cached
        // before the release propagated cannot hide it forever.
        #expect(decide(storeVersion: "1.0.0", storeAge: stale).showsBanner)
    }

    @Test func treatsAnUntimestampedStoreAnswerAsUnknown() {
        // Written by a build that predated the timestamp.
        #expect(decide(storeVersion: "1.0.0", storeAge: nil).showsBanner)
    }

    @Test func acceptsAStoreAnswerRightUpToTheTTL() {
        let fresh = UpdateChecker.storeVersionTTL - 1
        #expect(decide(storeVersion: "1.0.0", storeAge: fresh) == .hidden)
    }

    // MARK: Dismissal

    @Test func hidesADismissedRelease() {
        #expect(decide(dismissed: "1.0.1") == .hidden)
    }

    @Test func matchesTheDismissalOnParsedVersionsNotStrings() {
        // "1.0.1" and "1.0.1.0" are the same release to a user.
        #expect(decide(dismissed: "1.0.1.0") == .hidden)
        #expect(decide(release: release(version: "1.1"),
                       storeVersion: "1.1",
                       dismissed: "1.1.0") == .hidden)
    }

    @Test func aDismissalOfAnotherVersionDoesNotCarryOver() {
        #expect(decide(dismissed: "1.0.0").showsBanner)
        #expect(decide(dismissed: "1.0.2").showsBanner)
    }

    // MARK: Mandatory

    @Test func mandatoryWhenBelowTheMinimumSupportedVersion() {
        let decision = decide(release: release(minimumSupportedVersion: "1.0.1"))
        #expect(decision.showsBanner)
        #expect(decision.isMandatory)
    }

    @Test func mandatoryOverridesADismissal() {
        let decision = decide(
            release: release(minimumSupportedVersion: "1.0.1"),
            dismissed: "1.0.1"
        )
        #expect(decision.showsBanner)
        #expect(decision.isMandatory)
    }

    @Test func notMandatoryWhenTheInstalledVersionMeetsTheMinimum() {
        let decision = decide(release: release(minimumSupportedVersion: "1.0.0"))
        #expect(decision.showsBanner)
        #expect(!decision.isMandatory)
    }

    @Test func mandatoryStillObeysTheOtherGates() {
        // "Required" is a reason to ignore a dismissal, not a reason to
        // advertise a release the store does not serve yet.
        let unpublished = release(published: false, minimumSupportedVersion: "1.0.1")
        #expect(decide(release: unpublished) == .hidden)

        let notYetServed = release(minimumSupportedVersion: "1.0.1")
        #expect(decide(release: notYetServed, storeVersion: "1.0.0") == .hidden)
    }
}
