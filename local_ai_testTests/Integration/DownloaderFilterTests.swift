//
//  DownloaderFilterTests.swift
//
//  Catalog-driven sanity checks for the download planning logic: the catalog
//  must carry the metadata the filter relies on (preferredGGUFFilename, mmproj
//  filename, preferredPackage and file layout), plus direct coverage of
//  `ModelDownloader.filterFiles(_:for:)` for multi-package Core ML repos.
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

    /// A `coreMLPackage` entry must pin the container it wants, because the
    /// layout now covers both an `.mlpackage` (needs compiling on device) and
    /// an already-compiled `.mlmodelc` (does not), and repos ship both. The
    /// pin and `requiresCompilation` have to agree, or the downloader either
    /// compiles nothing it should or compiles something already compiled.
    ///
    /// Asserted over `builtIn`, not `all`: `all` serves whatever catalog the
    /// device last fetched from the public repo, so on a real phone this would
    /// be grading someone else's file instead of this build's.
    @Test func catalogCoreMLEntriesAgreeOnCompilation() {
        for model in ModelCatalog.builtIn where model.fileLayout == .coreMLPackage {
            let container = try? #require(model.preferredPackage)
            guard let container else { continue }
            if container.hasSuffix(".mlmodelc") {
                #expect(!model.requiresCompilation, "\(model.id) ships compiled; it must not be recompiled")
            } else {
                #expect(model.requiresCompilation, "\(model.id) ships an .mlpackage; it must be compiled")
            }
        }
    }

    /// The Core ML text entry must pin the ALREADY-COMPILED container. Its
    /// repo ships the `.mlpackage` next to the `.mlmodelc` built from it, and
    /// the unpinned fallback picks the smaller one — which is the `.mlpackage`,
    /// putting an on-device compile back into a path that does not need one.
    @Test func coreMLTextEntryPinsItsCompiledBundle() {
        let model = ModelCatalog.builtIn.first {
            $0.id == "anziank/grio-qwen2.5-1.5b-instruct-coreml-stateful-int8"
        }
        #expect(model?.preferredPackage == "Qwen2.5-1.5B-Instruct-stateful-int8.mlmodelc")
        #expect(model?.requiresCompilation == false)
    }

    // MARK: - Core ML package selection

    /// Shape of apple/mistral-coreml: two packages side by side plus the
    /// tokenizer/config files at the repo root.
    private static let twoPackageRepo: [HFSibling] = [
        HFSibling(rfilename: ".gitattributes", size: 1_000),
        HFSibling(rfilename: "README.md", size: 2_000),
        HFSibling(rfilename: "config.json", size: 700),
        HFSibling(rfilename: "tokenizer.json", size: 1_800_000),
        HFSibling(rfilename: "tokenizer_config.json", size: 900),
        HFSibling(rfilename: "special_tokens_map.json", size: 400),
        HFSibling(rfilename: "StatefulMistral7BInstructFP16.mlpackage/Manifest.json", size: 500),
        HFSibling(rfilename: "StatefulMistral7BInstructFP16.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 20_000),
        HFSibling(rfilename: "StatefulMistral7BInstructFP16.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 14_500_000_000),
        HFSibling(rfilename: "StatefulMistral7BInstructInt4.mlpackage/Manifest.json", size: 500),
        HFSibling(rfilename: "StatefulMistral7BInstructInt4.mlpackage/Data/com.apple.CoreML/model.mlmodel", size: 20_000),
        HFSibling(rfilename: "StatefulMistral7BInstructInt4.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 4_080_000_000),
        // A tokenizer copy inside the package we are not taking must not
        // sneak past the suffix rules.
        HFSibling(rfilename: "StatefulMistral7BInstructFP16.mlpackage/tokenizer.json", size: 1_800_000),
    ]

    private static func coreMLModel(preferredPackage: String?) -> AIModel {
        AIModel(
            id: "test/coreml",
            name: "Test Core ML",
            family: "Test",
            parameterSize: "7B",
            quantization: "int4",
            backend: .coreML,
            fileLayout: .coreMLPackage,
            ramRequiredGB: 4.2,
            diskSizeGB: 4.3,
            huggingFaceRepo: "test/coreml",
            contextLength: 4096,
            quality: .great,
            requiresCompilation: true,
            preferredPackage: preferredPackage
        )
    }

    @Test func preferredPackageKeepsOnlyThatPackageAndRootConfigs() throws {
        let kept = try ModelDownloader.filterFiles(
            Self.twoPackageRepo,
            for: Self.coreMLModel(preferredPackage: "StatefulMistral7BInstructInt4.mlpackage")
        ).map(\.rfilename)

        #expect(!kept.contains { $0.hasPrefix("StatefulMistral7BInstructFP16.mlpackage/") })
        #expect(kept.contains("StatefulMistral7BInstructInt4.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        #expect(kept.contains("StatefulMistral7BInstructInt4.mlpackage/Manifest.json"))
        for root in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"] {
            #expect(kept.contains(root), "root \(root) must be downloaded")
        }
        #expect(!kept.contains("README.md"))

        let bytes = try ModelDownloader.filterFiles(
            Self.twoPackageRepo,
            for: Self.coreMLModel(preferredPackage: "StatefulMistral7BInstructInt4.mlpackage")
        ).compactMap(\.size).reduce(0, +)
        #expect(bytes < 5_000_000_000, "must not pull the 14.5 GB FP16 package")
    }

    @Test func withoutPreferenceTheSmallestPackageWins() throws {
        let kept = try ModelDownloader.filterFiles(
            Self.twoPackageRepo, for: Self.coreMLModel(preferredPackage: nil)
        ).map(\.rfilename)

        #expect(kept.contains("StatefulMistral7BInstructInt4.mlpackage/Data/com.apple.CoreML/weights/weight.bin"))
        #expect(!kept.contains { $0.hasPrefix("StatefulMistral7BInstructFP16.mlpackage/") })
        #expect(kept.contains("tokenizer.json"))
    }

    @Test func singlePackageRepoIsUnchanged() throws {
        let files: [HFSibling] = [
            HFSibling(rfilename: "README.md", size: 2_000),
            HFSibling(rfilename: "config.json", size: 700),
            HFSibling(rfilename: "tokenizer.json", size: 1_800_000),
            HFSibling(rfilename: "Model.mlpackage/Manifest.json", size: 500),
            HFSibling(rfilename: "Model.mlpackage/Data/com.apple.CoreML/weights/weight.bin", size: 4_000_000_000),
        ]
        let kept = Set(try ModelDownloader.filterFiles(files, for: Self.coreMLModel(preferredPackage: nil)).map(\.rfilename))

        #expect(kept == [
            "config.json",
            "tokenizer.json",
            "Model.mlpackage/Manifest.json",
            "Model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        ])
    }

    /// A pin that no longer matches must FAIL the download, not quietly take
    /// the smallest container. The pin is what makes the catalog entry's disk
    /// size, RAM figure and `requiresCompilation` flag true: an `.mlmodelc`
    /// pin exists exactly because the smallest container is the uncompiled
    /// `.mlpackage` twin, so the old fallback installed a model that needed an
    /// on-device compile and twice the reserved disk.
    @Test func unknownPreferredPackageFailsTheDownload() {
        #expect(throws: ModelDownloadError.preferredPackageMissing(
            package: "Missing.mlpackage", repo: "test/coreml"
        )) {
            try ModelDownloader.filterFiles(
                Self.twoPackageRepo, for: Self.coreMLModel(preferredPackage: "Missing.mlpackage")
            )
        }
    }

    /// The pin only gates the layout that has containers to choose between.
    @Test func matchingPreferredPackageStillResolves() throws {
        let kept = try ModelDownloader.filterFiles(
            Self.twoPackageRepo,
            for: Self.coreMLModel(preferredPackage: "StatefulMistral7BInstructInt4.mlpackage")
        ).map(\.rfilename)
        #expect(kept.contains("StatefulMistral7BInstructInt4.mlpackage/Manifest.json"))
    }
}
