//
//  DownloaderFilterTests.swift
//
//  Catalog-driven sanity checks for the download planning logic. The real
//  filter is private inside `ModelDownloader`; here we assert the catalog
//  carries the metadata the filter relies on (preferredGGUFFilename, mmproj
//  filename, and file layout).
//

import Testing
import Foundation
@testable import local_ai_test

struct DownloaderFilterTests {

    @Test func catalogGGUFModelsHavePreferredFilename() {
        for model in ModelCatalog.all where model.fileLayout == .singleGGUF {
            #expect(model.preferredGGUFFilename != nil, "\(model.id) needs preferredGGUFFilename")
            #expect(model.preferredGGUFFilename?.hasSuffix(".gguf") == true)
        }
    }

    @Test func catalogMmprojModelsHaveBothFilenames() {
        for model in ModelCatalog.all where model.fileLayout == .ggufWithMmproj {
            #expect(model.preferredGGUFFilename != nil)
            #expect(model.mmprojFilename != nil)
            #expect(model.mmprojFilename?.hasPrefix("mmproj") == true)
        }
    }

    @Test func catalogCoreMLModelsMarkCompilation() {
        for model in ModelCatalog.all where model.fileLayout == .coreMLPackage {
            #expect(model.requiresCompilation)
        }
    }
}
