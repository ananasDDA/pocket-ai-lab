//
//  HuggingFaceImportTests.swift
//  local_ai_testTests
//
//  Covers the pure parts of the "import from a Hugging Face link" pipeline:
//  link parsing, format detection, the gated-field decoder and the RAM/disk
//  estimator. No network.
//

import Testing
import Foundation
@testable import local_ai_test

/// Device with `processAllowance` 0 so estimates exercise the
/// `totalRAM - 3` fallback the expectations in these suites were written
/// against.
private func estimateDevice(ramGB: Double) -> DeviceInfo {
    DeviceInfo(
        identifier: "iPhone15,2",
        marketingName: "iPhone Test",
        chip: "A17 Pro",
        totalRAM: UInt64(ramGB * 1_073_741_824),
        processAllowance: 0,
        metalWorkingSet: 0,
        iOSVersion: "26.1",
        freeDiskSpace: Int64(100 * 1_073_741_824),
        appleIntelligence: .notSupported
    )
}

// MARK: - Link parsing

struct RepoIDParsingTests {

    @Test func acceptsBareRepoID() {
        #expect(HuggingFaceAPI.parseRepoID(from: "mlx-community/gemma-2-2b-it-4bit")
                == "mlx-community/gemma-2-2b-it-4bit")
    }

    @Test func acceptsFullURLs() {
        let expected = "bartowski/Llama-3.2-3B-Instruct-GGUF"
        let inputs = [
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
            "http://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
            "https://www.huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
            "huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
            "https://hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
            "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/",
            "  https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF  "
        ]
        for input in inputs {
            #expect(HuggingFaceAPI.parseRepoID(from: input) == expected, "failed for \(input)")
        }
    }

    @Test func stripsTreeBlobAndQuery() {
        let expected = "unsloth/gemma-3-4b-it-GGUF"
        #expect(HuggingFaceAPI.parseRepoID(
            from: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/tree/main") == expected)
        #expect(HuggingFaceAPI.parseRepoID(
            from: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF/blob/main/a/b.gguf") == expected)
        #expect(HuggingFaceAPI.parseRepoID(
            from: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF?library=true") == expected)
        #expect(HuggingFaceAPI.parseRepoID(
            from: "https://huggingface.co/unsloth/gemma-3-4b-it-GGUF#files") == expected)
    }

    @Test func acceptsCanonicalModelsPrefix() {
        #expect(HuggingFaceAPI.parseRepoID(from: "https://huggingface.co/models/apple/mistral-coreml")
                == "apple/mistral-coreml")
    }

    @Test func rejectsGarbage() {
        let rejected = [
            "", "   ", "not-a-repo", "/", "//", "owner/",
            "/repo-only", "owner//repo",
            "https://example.com/owner/repo",
            "https://github.com/owner/repo",
            "owner/repo with spaces",
            "owner/re$po",
            "https://huggingface.co/spaces/owner/demo",
            "https://huggingface.co/datasets/owner/data"
        ]
        for input in rejected {
            #expect(HuggingFaceAPI.parseRepoID(from: input) == nil, "should reject \(input)")
        }
    }
}

// MARK: - gated decoder

struct HFGatedDecodingTests {

    private func decodeRepo(gatedLiteral: String) throws -> HFRepoInfo {
        let json = #"{"id":"a/b","gated":\#(gatedLiteral),"siblings":[]}"#
        return try JSONDecoder().decode(HFRepoInfo.self, from: Data(json.utf8))
    }

    @Test func decodesBooleanFalse() throws {
        #expect(try decodeRepo(gatedLiteral: "false").gated == .notGated)
    }

    @Test func decodesAutoAndManual() throws {
        #expect(try decodeRepo(gatedLiteral: "\"auto\"").gated == .auto)
        #expect(try decodeRepo(gatedLiteral: "\"manual\"").gated == .manual)
    }

    @Test func treatsBooleanTrueAsGated() throws {
        #expect(try decodeRepo(gatedLiteral: "true").gated == .manual)
        #expect(try decodeRepo(gatedLiteral: "true").gated.requiresLicenseAcceptance)
    }

    @Test func missingFieldMeansNotGated() throws {
        let json = #"{"id":"a/b","siblings":[]}"#
        let info = try JSONDecoder().decode(HFRepoInfo.self, from: Data(json.utf8))
        #expect(info.gated == .notGated)
        #expect(!info.gated.requiresLicenseAcceptance)
    }

    @Test func decodesPrivateAndCounters() throws {
        let json = #"""
        {"id":"a/b","private":true,"downloads":1234,"likes":7,
         "tags":["text-generation"],"sha":"abc123","siblings":[{"rfilename":"x.gguf","size":10}]}
        """#
        let info = try JSONDecoder().decode(HFRepoInfo.self, from: Data(json.utf8))
        #expect(info.isPrivate)
        #expect(info.downloads == 1234)
        #expect(info.likes == 7)
        #expect(info.sha == "abc123")
        #expect(info.siblings.first?.size == 10)
    }
}

// MARK: - Format detection

struct ImportAnalyzerTests {

    private func files(_ names: [String]) -> [HFSibling] {
        names.map { HFSibling(rfilename: $0, size: 1_000) }
    }

    @Test func detectsPlainGGUF() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["README.md", "Model-Q4_K_M.gguf", "Model-Q8_0.gguf"]),
            tags: []
        )
        #expect(formats[0].backend == .llamaCpp)
        #expect(formats[0].layout == .singleGGUF)
    }

    @Test func detectsVisionGGUFViaMmproj() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["gemma-3-4b-it-Q4_K_M.gguf", "mmproj-F16.gguf"]),
            tags: []
        )
        #expect(formats[0].backend == .llamaCppVision)
        #expect(formats[0].layout == .ggufWithMmproj)
    }

    @Test func detectsMLX() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["config.json", "model.safetensors", "tokenizer.json"]),
            tags: ["text-generation"]
        )
        #expect(formats[0].backend == .mlx)
        #expect(formats[0].layout == .huggingFaceMLX)
    }

    @Test func detectsMLXVisionFromTags() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["config.json", "model-00001-of-00002.safetensors"]),
            tags: ["image-text-to-text"]
        )
        #expect(formats[0].backend == .mlxVision)
    }

    @Test func safetensorsWithoutConfigIsNotMLX() {
        #expect(throws: HuggingFaceError.self) {
            _ = try ImportAnalyzer.detectFormats(files: self.files(["model.safetensors"]), tags: [])
        }
    }

    @Test func detectsCoreML() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["StatefulMistral.mlpackage/Manifest.json", "tokenizer.json"]),
            tags: []
        )
        #expect(formats[0].backend == .coreML)
        #expect(formats[0].layout == .coreMLPackage)
    }

    @Test func detectsWhisperGGML() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["ggml-tiny.en.bin", "ggml-base.bin"]),
            tags: []
        )
        #expect(formats[0].backend == .whisperCpp)
        #expect(formats[0].layout == .whisperGGML)
    }

    @Test func offersMLXAsAlternativeWhenBothFormatsExist() throws {
        let formats = try ImportAnalyzer.detectFormats(
            files: files(["Model-Q4_K_M.gguf", "config.json", "model.safetensors"]),
            tags: []
        )
        #expect(formats.count == 2)
        #expect(formats[0].layout == .singleGGUF)
        #expect(formats.contains { $0.layout == .huggingFaceMLX })
    }

    @Test func unsupportedRepoThrows() {
        #expect(throws: HuggingFaceError.self) {
            _ = try ImportAnalyzer.detectFormats(
                files: self.files(["README.md", "notes.txt"]), tags: []
            )
        }
    }

    // MARK: Quantizations

    @Test func quantsAreSortedAndExcludeProjector() {
        let list = [
            HFSibling(rfilename: "Model-Q8_0.gguf", size: 8_000),
            HFSibling(rfilename: "mmproj-F16.gguf", size: 500),
            HFSibling(rfilename: "Model-Q4_K_M.gguf", size: 4_000)
        ]
        let quants = ImportAnalyzer.ggufQuants(in: list)
        #expect(quants.map(\.filename) == ["Model-Q4_K_M.gguf", "Model-Q8_0.gguf"])
        #expect(quants.map(\.label) == ["Q4_K_M", "Q8_0"])
    }

    @Test func defaultQuantPrefersQ4KM() {
        let quants = ImportAnalyzer.ggufQuants(in: [
            HFSibling(rfilename: "M-Q2_K.gguf", size: 2),
            HFSibling(rfilename: "M-Q4_K_M.gguf", size: 4),
            HFSibling(rfilename: "M-Q8_0.gguf", size: 8)
        ])
        #expect(ImportAnalyzer.defaultQuant(from: quants)?.label == "Q4_K_M")
    }

    @Test func defaultQuantFallsBackToMiddle() {
        let quants = ImportAnalyzer.ggufQuants(in: [
            HFSibling(rfilename: "M-Q2_K.gguf", size: 2),
            HFSibling(rfilename: "M-Q5_K_S.gguf", size: 5),
            HFSibling(rfilename: "M-Q8_0.gguf", size: 8)
        ])
        #expect(ImportAnalyzer.defaultQuant(from: quants)?.label == "Q5_K_S")
        #expect(ImportAnalyzer.defaultQuant(from: []) == nil)
    }

    @Test func quantLabelHandlesOddNames() {
        #expect(ImportAnalyzer.quantLabel(for: "Llama-3.2-3B-Instruct-IQ3_XS.gguf") == "IQ3_XS")
        #expect(ImportAnalyzer.quantLabel(for: "some/dir/Model-f16.gguf") == "F16")
        #expect(ImportAnalyzer.quantLabel(for: "weights.gguf") == "weights")
    }

    @Test func mmprojPrefersF16() {
        let list = [
            HFSibling(rfilename: "mmproj-BF16.gguf", size: 1),
            HFSibling(rfilename: "mmproj-F16.gguf", size: 2)
        ]
        #expect(ImportAnalyzer.mmprojFilename(in: list) == "mmproj-F16.gguf")
    }

    @Test func whisperPicksSmallestTopLevelBin() {
        let list = [
            HFSibling(rfilename: "ggml-large-v3.bin", size: 3_000),
            HFSibling(rfilename: "ggml-tiny.en.bin", size: 70),
            HFSibling(rfilename: "nested/ggml-base.bin", size: 1)
        ]
        #expect(ImportAnalyzer.whisperFilename(in: list) == "ggml-tiny.en.bin")
    }
}

// MARK: - Resource estimation

struct ResourceEstimateTests {

    private let gb = 1_073_741_824.0

    @Test func ggufRAMFormula() {
        let bytes = Int64(2.0 * gb)
        let ram = ImportAnalyzer.ramEstimateGB(layout: .singleGGUF, weightBytes: bytes)
        #expect(abs(ram - (2.0 * 1.15 + 0.8)) < 0.001)
    }

    @Test func mlxRAMFormula() {
        let bytes = Int64(4.0 * gb)
        let ram = ImportAnalyzer.ramEstimateGB(layout: .huggingFaceMLX, weightBytes: bytes)
        #expect(abs(ram - (4.0 * 1.2 + 0.8)) < 0.001)
    }

    @Test func usableRAMReservesThreeGB() {
        #expect(abs(ImportAnalyzer.usableRAMGB(totalRAMBytes: UInt64(8 * gb)) - 5.0) < 0.001)
    }

    @Test func verdictFitsWithHeadroom() {
        // 8 GB device → 5 GB usable; a 1 GB GGUF needs ~1.95 GB.
        let estimate = ImportAnalyzer.estimate(
            layout: .singleGGUF,
            weightBytes: Int64(1.0 * gb),
            downloadBytes: Int64(1.0 * gb),
            device: estimateDevice(ramGB: 8),
            freeDiskBytes: Int64(50 * gb)
        )
        #expect(estimate.ramVerdict == .fits)
        #expect(estimate.diskFits)
    }

    @Test func verdictTightWithinOneGBOfLimit() {
        // 8 GB device → 5 GB usable; a 3.6 GB GGUF needs ~4.94 GB.
        let estimate = ImportAnalyzer.estimate(
            layout: .singleGGUF,
            weightBytes: Int64(3.6 * gb),
            downloadBytes: Int64(3.6 * gb),
            device: estimateDevice(ramGB: 8),
            freeDiskBytes: Int64(50 * gb)
        )
        #expect(estimate.ramVerdict == .tight)
    }

    @Test func verdictWontFit() {
        let estimate = ImportAnalyzer.estimate(
            layout: .huggingFaceMLX,
            weightBytes: Int64(8.0 * gb),
            downloadBytes: Int64(8.0 * gb),
            device: estimateDevice(ramGB: 8),
            freeDiskBytes: Int64(50 * gb)
        )
        #expect(estimate.ramVerdict == .wontFit)
    }

    @Test func diskShortageIsReportedSeparatelyFromRAM() {
        let estimate = ImportAnalyzer.estimate(
            layout: .singleGGUF,
            weightBytes: Int64(1.0 * gb),
            downloadBytes: Int64(1.0 * gb),
            device: estimateDevice(ramGB: 16),
            freeDiskBytes: Int64(0.5 * gb)
        )
        #expect(estimate.ramVerdict == .fits)
        #expect(!estimate.diskFits)
    }
}

// MARK: - Imported descriptor

struct ImportedModelBuilderTests {

    private func analysis(repo: String, files: [HFSibling], tags: [String] = []) throws -> ImportAnalysis {
        var info = HFRepoInfo()
        info.id = repo
        info.sha = "deadbeef"
        info.siblings = files
        info.tags = tags
        return try ImportAnalyzer.analyze(info)
    }

    @Test func ggufImportPinsQuantAndRevision() throws {
        let result = try analysis(repo: "bartowski/Qwen2.5-7B-Instruct-GGUF", files: [
            HFSibling(rfilename: "Qwen2.5-7B-Instruct-Q4_K_M.gguf", size: 4_000_000_000),
            HFSibling(rfilename: "Qwen2.5-7B-Instruct-Q8_0.gguf", size: 8_000_000_000)
        ])
        let sizing = result.sizing(layout: .singleGGUF, quant: result.defaultQuant)
        let estimate = ImportAnalyzer.estimate(
            layout: .singleGGUF,
            weightBytes: sizing.weights,
            downloadBytes: sizing.download,
            device: estimateDevice(ramGB: 16),
            freeDiskBytes: Int64(64 * 1_073_741_824)
        )
        let model = ImportedModelBuilder.makeModel(
            analysis: result,
            format: result.detected,
            quant: result.defaultQuant,
            estimate: estimate,
            contextLength: 32768
        )

        #expect(model.source == .imported)
        #expect(model.revision == "deadbeef")
        // The id carries the quant so two quants of one repo can coexist.
        #expect(model.id == "bartowski/Qwen2.5-7B-Instruct-GGUF#Q4_K_M")
        #expect(model.preferredGGUFFilename == "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
        #expect(model.backend == .llamaCpp)
        #expect(model.fileLayout == .singleGGUF)
        #expect(model.family == "Qwen")
        #expect(model.parameterSize == "7B")
        #expect(model.contextLength == 32768)
        #expect(model.huggingFaceRepo == "bartowski/Qwen2.5-7B-Instruct-GGUF")
    }

    @Test func mlxImportKeepsRepoIDAsModelID() throws {
        let result = try analysis(repo: "mlx-community/SmolLM-1.7B-Instruct-4bit", files: [
            HFSibling(rfilename: "config.json", size: 1_000),
            HFSibling(rfilename: "model.safetensors", size: 1_000_000_000)
        ])
        let model = ImportedModelBuilder.makeModel(
            analysis: result,
            format: result.detected,
            quant: nil,
            estimate: ImportAnalyzer.estimate(
                layout: .huggingFaceMLX, weightBytes: 1_000_000_000,
                downloadBytes: 1_000_001_000,
                device: estimateDevice(ramGB: 8),
                freeDiskBytes: Int64(64 * 1_073_741_824)
            ),
            contextLength: nil
        )
        #expect(model.id == "mlx-community/SmolLM-1.7B-Instruct-4bit")
        #expect(model.preferredGGUFFilename == nil)
        #expect(model.family == "SmolLM")
        // No config.json read → documented default.
        #expect(model.contextLength == 4096)
    }

    @Test func familyFallsBackToOwner() {
        #expect(ImportedModelBuilder.family(from: "acme/mystery-model") == "acme")
        #expect(ImportedModelBuilder.parameterSize(from: "acme/mystery-model") == "?")
    }

    @Test func displayNameDropsFormatSuffix() {
        let name = ImportedModelBuilder.displayName(
            repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", quant: nil
        )
        #expect(name == "Llama 3.2 3B Instruct")
    }
}
