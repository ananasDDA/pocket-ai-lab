//
//  GGUFNamingTests.swift
//
//  Which `.gguf` in a repo is the vision projector.
//
//  The rule used to be "basename starts with mmproj", which imported
//  PaddleOCR-VL - whose official release names the projector
//  `PaddleOCR-VL-1.6-GGUF-mmproj.gguf` - as a text-only model, so the app
//  refused image input for a vision model and offered the 840 MB projector
//  as a weight file.
//

import Testing
import Foundation
@testable import local_ai_test

struct GGUFNamingTests {

    @Test func projectorsAreRecognisedWhereverTheWordSits() {
        // Prefix style: bartowski, unsloth, ggml-org.
        #expect(GGUFNaming.isProjector("mmproj-F16.gguf"))
        #expect(GGUFNaming.isProjector("mmproj-model-f16.gguf"))
        // Suffix style: PaddlePaddle's own release and community requants.
        #expect(GGUFNaming.isProjector("PaddleOCR-VL-1.6-GGUF-mmproj.gguf"))
        #expect(GGUFNaming.isProjector("PaddleOCR-VL-1.6-mmproj.gguf"))
        #expect(GGUFNaming.isProjector("Qwen2-VL-7B-Instruct-mmproj-f16.gguf"))
        // Case and folders do not matter.
        #expect(GGUFNaming.isProjector("MMPROJ-F16.GGUF"))
        #expect(GGUFNaming.isProjector("vision/mmproj-f16.gguf"))
    }

    @Test func weightsAreNotProjectors() {
        #expect(!GGUFNaming.isProjector("PaddleOCR-VL-1.6-GGUF.gguf"))
        #expect(!GGUFNaming.isProjector("gemma-3-4b-it-Q4_K_M.gguf"))
        #expect(!GGUFNaming.isProjector("Llama-3.2-3B-Instruct-Q4_K_M.gguf"))
    }

    @Test func onlyGGUFsCountAsWeights() {
        #expect(GGUFNaming.isWeights("PaddleOCR-VL-1.6-GGUF.gguf"))
        #expect(!GGUFNaming.isWeights("PaddleOCR-VL-1.6-GGUF-mmproj.gguf"))
        #expect(!GGUFNaming.isWeights("README.md"))
        #expect(!GGUFNaming.isWeights("tokenizer.json"))
    }

    // The whole repo, decided the way the importer decides it.
    @Test func theOfficialPaddleOCRRepoImportsAsAVisionModel() throws {
        let files = [
            HFSibling(rfilename: "PaddleOCR-VL-1.6-GGUF.gguf", size: 935_000_000),
            HFSibling(rfilename: "PaddleOCR-VL-1.6-GGUF-mmproj.gguf", size: 881_000_000),
            HFSibling(rfilename: "chat_template.jinja", size: 1_400),
            HFSibling(rfilename: "README.md", size: 12_000),
        ]
        let formats = try ImportAnalyzer.detectFormats(files: files, tags: [])
        #expect(formats.first?.backend == .llamaCppVision)
        #expect(formats.first?.layout == .ggufWithMmproj)
        #expect(ImportAnalyzer.mmprojFilename(in: files) == "PaddleOCR-VL-1.6-GGUF-mmproj.gguf")
        // The projector is not a quantization choice, and not a weight file.
        let quants = ImportAnalyzer.ggufQuants(in: files)
        #expect(quants.map(\.filename) == ["PaddleOCR-VL-1.6-GGUF.gguf"])
    }

    @Test func aTextOnlyGGUFRepoStillImportsAsText() throws {
        let files = [
            HFSibling(rfilename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf", size: 2_000_000_000),
            HFSibling(rfilename: "README.md", size: 1_000),
        ]
        let formats = try ImportAnalyzer.detectFormats(files: files, tags: [])
        #expect(formats.first?.backend == .llamaCpp)
        #expect(formats.first?.layout == .singleGGUF)
        #expect(ImportAnalyzer.mmprojFilename(in: files) == nil)
    }
}

// MARK: - PaddleOCR / ERNIE prompt

struct ErniePromptTests {

    @Test func paddleOCRAndErnieIdsMapToTheErnieFamily() {
        #expect(PromptFormatter.detectFamily(modelId: "PaddlePaddle/PaddleOCR-VL-1.6-GGUF") == .ernie)
        #expect(PromptFormatter.detectFamily(modelId: "baidu/ERNIE-4.5-0.3B-PT") == .ernie)
        // Unrelated ids keep their families.
        #expect(PromptFormatter.detectFamily(modelId: "unsloth/gemma-3-4b-it-GGUF") == .gemma)
        #expect(PromptFormatter.detectFamily(modelId: "mlx-community/Qwen2.5-3B-Instruct-4bit") == .qwen)
    }

    // Follows chat_template.jinja in PaddlePaddle's GGUF release.
    @Test func theTranscriptMatchesTheOfficialTemplate() {
        let prompt = PromptFormatter.format(
            turns: [.user("Read this page."), .assistant("Done."), .user("And this one?")],
            family: .ernie
        )
        #expect(prompt == "<|begin_of_sentence|>User: Read this page.\nAssistant:\nDone.</s>"
                + "User: And this one?\nAssistant:\n")
    }

    @Test func theSystemTurnIsBareTextInFront() {
        let prompt = PromptFormatter.format(
            turns: [.system("You are an OCR engine."), .user("Go.")],
            family: .ernie
        )
        #expect(prompt.hasPrefix("<|begin_of_sentence|>You are an OCR engine.\nUser: Go."))
    }

    // An OCR model has to see the page before the instruction about it.
    @Test func theImageMarkerLeadsTheUserTurn() {
        let prepared = VisionPromptBuilder.build(
            turns: [.user("Convert to markdown.", attachments: [.image(Data([0x01]))])],
            family: .ernie,
            budgetTokens: 8192
        )
        let marker = prepared.prompt.range(of: "<__media__>")!.lowerBound
        let text = prepared.prompt.range(of: "Convert to markdown.")!.lowerBound
        #expect(marker < text)
    }

    @Test func otherFamiliesKeepTheMarkerAfterTheText() {
        let prepared = VisionPromptBuilder.build(
            turns: [.user("What is this?", attachments: [.image(Data([0x01]))])],
            family: .gemma,
            budgetTokens: 8192
        )
        let marker = prepared.prompt.range(of: "<__media__>")!.lowerBound
        let text = prepared.prompt.range(of: "What is this?")!.lowerBound
        #expect(text < marker)
    }
}
