//
//  CoreMLCompilerTests.swift
//
//  Exercises the progress + cache behaviour of `CoreMLCompiler`. Real
//  compilation is covered by the device-level Core ML tests; here we
//  focus on the error paths.
//

import Testing
import Foundation
@testable import local_ai_test

struct CoreMLCompilerTests {

    @Test func missingPackageThrows() async {
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("nonexistent.mlpackage")
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cache) }

        do {
            _ = try await CoreMLCompiler.compileIfNeeded(
                mlpackageURL: tmp,
                cacheDirectory: cache,
                estimatedDurationSeconds: 1,
                progress: { _ in }
            )
            Issue.record("Should have thrown for missing package")
        } catch {
            // expected
        }
    }

    @Test func cachedModelIsReturnedImmediately() async throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cache) }

        // Simulate a pre-existing .mlmodelc
        let cached = cache.appendingPathComponent("compiled.mlmodelc")
        try FileManager.default.createDirectory(at: cached, withIntermediateDirectories: true)

        let fakePackage = cache.appendingPathComponent("dummy.mlpackage")

        var progressCalled = false
        let url = try await CoreMLCompiler.compileIfNeeded(
            mlpackageURL: fakePackage,
            cacheDirectory: cache,
            estimatedDurationSeconds: 5
        ) { _ in progressCalled = true }

        #expect(url.path == cached.path)
        #expect(progressCalled, "Progress should have been called with 1.0 for cache hit")
    }

    @Test func clearCacheRemovesCompiled() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: cache) }

        let compiled = cache.appendingPathComponent("compiled.mlmodelc")
        try FileManager.default.createDirectory(at: compiled, withIntermediateDirectories: true)
        #expect(FileManager.default.fileExists(atPath: compiled.path))

        CoreMLCompiler.clearCache(in: cache)
        #expect(!FileManager.default.fileExists(atPath: compiled.path))
    }
}
