//
//  BackendLinkageTests.swift
//  local_ai_testTests
//
//  R2 regression guard: an App Store build once shipped without the
//  llama.cpp XCFramework linked, while catalog.json still recommended a
//  llama.cpp model -- the user got a raw developer error instead of a
//  usable app. `BackendAvailability` is the single source of truth for
//  "is this backend actually compiled into this binary"; this test cross
//  checks it against the bundled catalog so a mismatch like that fails a
//  build here instead of on a real device.
//
//  Runs on device only: the vendored llama XCFramework ships without a
//  simulator slice, so this test target only builds for device. That's
//  fine -- it just needs to compile cleanly and run there.
//

import Testing
import Foundation
@testable import local_ai_test

struct BackendLinkageTests {

    /// Every backend any catalog entry declares must be linked in, except
    /// Apple Intelligence, whose availability is a device/OS runtime
    /// question (model support, region, the toggle being on) rather than a
    /// compile-time one -- `BackendAvailability.isLinked` always reports it
    /// as linked, so there is nothing meaningful to assert there.
    /// Known gap: `isLinked(.llamaCppVision)` only proves the `llama` module
    /// itself is linked, not that the XCFramework it links was built with
    /// libmtmd (mtmd.h / mtmd-helper.h). A build that links a plain llama.cpp
    /// XCFramework without mtmd would pass this check while still failing at
    /// runtime for vision models. That half is covered separately by the
    /// `nm` check in RELEASE_CHECKLIST.md step 2, not by this test.
    @Test func everyCatalogBackendIsLinkedInThisBuild() throws {
        let url = try #require(CatalogFile.bundledURL, "catalog.json is not in the app bundle")
        let data = try Data(contentsOf: url)
        let models = try #require(CatalogFile.decode(data), "bundled catalog.json failed to decode")

        let backendsInUse = Set(models.map(\.backend))
        #expect(!backendsInUse.isEmpty, "catalog.json declared no models")

        for backend in backendsInUse where backend != .appleIntelligence {
            #expect(
                BackendAvailability.isLinked(backend),
                "catalog.json offers \(backend.rawValue) models but this build does not link that backend. This is exactly the App Store bug where a recommended model showed a raw developer error to a user -- either link the backend or drop it from the catalog."
            )
        }
    }

    /// `ModelCatalog.builtIn` is a separate, compiled-in source from the
    /// bundled `catalog.json` above -- it is what the app falls back to when
    /// neither the cache nor the bundled resource can be read, so it can
    /// drift out of sync with catalog.json and needs the same guard.
    ///
    /// (The cached/remote catalog `RemoteCatalog` can fetch at runtime is not
    /// checked here -- it is covered live by the recommender's `isLinked`
    /// filter rather than by a build-time test.)
    @Test func everyBuiltInCatalogBackendIsLinkedInThisBuild() throws {
        let backendsInUse = Set(ModelCatalog.builtIn.map(\.backend))
        #expect(!backendsInUse.isEmpty, "ModelCatalog.builtIn declared no models")

        for backend in backendsInUse where backend != .appleIntelligence {
            #expect(
                BackendAvailability.isLinked(backend),
                "ModelCatalog.builtIn offers \(backend.rawValue) models but this build does not link that backend."
            )
        }
    }
}
