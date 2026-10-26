//
//  FirstFieldDataTests.swift
//  local_ai_testTests
//
//  The fixes that came out of the first day of 1.1 measurement data: the
//  import fit decision (split GGUF sets, projector-only and incomplete
//  selections, memory margins), which backends may be the chat model, the
//  empty-reply notice, and the cause read out of a failed llama.cpp load.
//  Pure logic only - no network, no model, no device state.
//

import Testing
import Foundation
@testable import local_ai_test

private let gib: Double = 1_073_741_824

private func file(_ name: String, gb: Double) -> HFSibling {
    HFSibling(rfilename: name, size: Int64(gb * gib))
}

private func analysis(_ files: [HFSibling], architecture: String? = nil) throws -> ImportAnalysis {
    var result = try ImportAnalyzer.analyze(HFRepoInfo(id: "someone/Model-GGUF", siblings: files))
    result.ggufArchitecture = architecture
    return result
}

private func quant(_ result: ImportAnalysis, _ label: String) -> GGUFQuantOption? {
    result.ggufQuants.first { $0.label == label }
}

// MARK: - Split GGUF naming

@MainActor
struct SplitGGUFNamingTests {

    @Test func parsesLlamaCppSplitNames() {
        let part = GGUFNaming.splitPart("Q8_0/gemma-27b-Q8_0-00002-of-00008.gguf")
        #expect(part == GGUFNaming.SplitPart(prefix: "Q8_0/gemma-27b-Q8_0", index: 2, count: 8))
        #expect(GGUFNaming.splitPart("gemma-27b-Q8_0.gguf") == nil)
        #expect(GGUFNaming.splitPart("model-0001-of-0003.gguf") == nil, "llama.cpp needs five digits")
        #expect(GGUFNaming.splitPart("model-00004-of-00003.gguf") == nil)
        #expect(GGUFNaming.splitFilename(prefix: "a/m", index: 3, count: 4) == "a/m-00003-of-00004.gguf")
    }

    @Test func siblingsComeInPartOrder() {
        let names = ["m-00002-of-00003.gguf", "m-00001-of-00003.gguf", "other-00001-of-00002.gguf",
                     "m-00003-of-00003.gguf", "mmproj-F16.gguf"]
        #expect(GGUFNaming.splitSiblings(of: "m-00001-of-00003.gguf", in: names)
                == ["m-00001-of-00003.gguf", "m-00002-of-00003.gguf", "m-00003-of-00003.gguf"])
        #expect(GGUFNaming.splitSiblings(of: "single.gguf", in: names).isEmpty)
    }

    @Test func importanceMatricesAreNotWeights() {
        #expect(!GGUFNaming.isWeights("imatrix_unsloth.gguf"))
        #expect(GGUFNaming.isWeights("Model-Q4_K_M.gguf"))
    }
}

// MARK: - Quantization grouping

@MainActor
struct SplitQuantGroupingTests {

    /// The field case: a 27B repo whose big quants are split into parts of a
    /// few GB each. 1.1 offered every part as its own "quantization".
    private func gemma27B() throws -> ImportAnalysis {
        var files = (1...8).map { file("Q8_0/gemma-27b-Q8_0-0000\($0)-of-00008.gguf", gb: 3.6) }
        files += (1...4).map { file("Q5_K_M/gemma-27b-Q5_K_M-0000\($0)-of-00004.gguf", gb: 4.8) }
        files += [file("gemma-27b-Q2_K.gguf", gb: 10.5), file("mmproj-F16.gguf", gb: 0.8)]
        return try analysis(files)
    }

    @Test func partsOfOneQuantizationAreOneChoiceSizedAsTheirSum() throws {
        let result = try gemma27B()
        #expect(result.ggufQuants.map(\.label) == ["Q2_K", "Q5_K_M", "Q8_0"])
        let q8 = try #require(quant(result, "Q8_0"))
        #expect(q8.parts.count == 8)
        #expect(q8.filename == "Q8_0/gemma-27b-Q8_0-00001-of-00008.gguf")
        #expect(abs(Double(q8.size) / gib - 28.8) < 0.01)
        #expect(q8.isSplit && q8.isComplete)
        #expect(result.sizing(layout: .ggufWithMmproj, quant: q8).weights == q8.size)
    }

    @Test func anIncompleteSetIsMarkedAndNeverTheDefault() throws {
        let result = try analysis([
            file("m-Q4_K_M-00001-of-00003.gguf", gb: 2), file("m-Q4_K_M-00003-of-00003.gguf", gb: 1),
            file("m-Q2_K.gguf", gb: 1.5),
        ])
        let q4 = try #require(quant(result, "Q4_K_M"))
        #expect(!q4.isComplete)
        #expect(q4.missingParts == [2])
        #expect(result.defaultQuant?.label == "Q2_K")
    }

    @Test func importanceMatrixIsNotOffered() throws {
        let result = try analysis([file("m-Q4_K_M.gguf", gb: 2), file("imatrix.gguf", gb: 0.01)])
        #expect(result.ggufQuants.map(\.label) == ["Q4_K_M"])
    }
}

// MARK: - Fit decision

@MainActor
struct ImportFitDecisionTests {

    @Test func verdictUsesTheRecommenderMargin() {
        #expect(ImportFit.verdict(ramGB: 8.5, budgetGB: 10) == .fits)
        #expect(ImportFit.verdict(ramGB: 8.6, budgetGB: 10) == .tight)
        #expect(ImportFit.verdict(ramGB: 10, budgetGB: 10) == .tight)
        #expect(ImportFit.verdict(ramGB: 10.1, budgetGB: 10) == .wontFit)
    }

    #if !MAC_STANDIN
    @Test func marginMatchesTheRecommender() {
        #expect(ImportFit.comfortMargin == ModelRecommender.recommendationMargin)
    }
    #endif

    /// iPhone 17 Pro Max: llama.cpp budget 8186 MB.
    private let proMaxBudget = 8186.0 / 1024
    private let gguf = ImportFormat(backend: .llamaCpp, layout: .singleGGUF)

    @Test func theFieldCaseIsRefusedAndSaysNothingInTheRepoFits() throws {
        var files = (1...4).map { file("Q5_K_M/gemma-27b-Q5_K_M-0000\($0)-of-00004.gguf", gb: 4.8) }
        files.append(file("gemma-27b-Q2_K.gguf", gb: 10.5))
        let result = try analysis(files)
        let decision = ImportFit.decide(
            analysis: result, format: gguf, quant: quant(result, "Q5_K_M"),
            budgetGB: proMaxBudget, deviceName: "iPhone 17 Pro Max"
        )
        #expect(decision.kind == .refuse)
        #expect(decision.blocksDownload)
        guard case .tooLarge(let needed, _)? = decision.problem else {
            Issue.record("expected tooLarge, got \(String(describing: decision.problem))")
            return
        }
        #expect(needed > 20)
        #expect(decision.alternative == nil)
        let message = try #require(decision.message)
        #expect(message.contains("iPhone 17 Pro Max"))
        #expect(message.contains("Nothing in this repo fits"))
        #expect(message.contains("about 9B parameters or fewer at Q4_K_M"))
    }

    private func qwen4B() throws -> ImportAnalysis {
        try analysis([
            file("Qwen3-4B-Q3_K_M.gguf", gb: 2.0), file("Qwen3-4B-Q4_K_M.gguf", gb: 2.5),
            file("Qwen3-4B-Q8_0.gguf", gb: 4.3), file("Qwen3-4B-F16.gguf", gb: 8.0),
        ])
    }

    @Test func tooLargeOffersTheLargestQuantThatFits() throws {
        let result = try qwen4B()
        let decision = ImportFit.decide(analysis: result, format: gguf, quant: quant(result, "Q8_0"),
                                        budgetGB: 5.1, deviceName: "iPhone 15")
        #expect(decision.kind == .refuse)
        #expect(decision.alternative?.label == "Q4_K_M")
        #expect(decision.message?.contains("Q4_K_M from this repo needs about 3.7 GB and fits.") == true)
    }

    @Test func aboveTheMarginWarnsButAllowsAndSuggestsRoomierQuant() throws {
        let result = try qwen4B()
        // Q4_K_M needs ~3.68 GB; 85% of 4.0 is 3.4.
        let decision = ImportFit.decide(analysis: result, format: gguf, quant: quant(result, "Q4_K_M"),
                                        budgetGB: 4.0, deviceName: "iPhone 13")
        #expect(decision.kind == .tight)
        #expect(!decision.blocksDownload)
        #expect(decision.alternative?.label == "Q3_K_M")
        #expect(decision.message?.contains("more than 85%") == true)
    }

    @Test func comfortableFitHasNoMessage() throws {
        let result = try qwen4B()
        let decision = ImportFit.decide(analysis: result, format: gguf, quant: quant(result, "Q4_K_M"),
                                        budgetGB: proMaxBudget, deviceName: "iPhone 17 Pro Max")
        #expect(decision == .fits)
    }

    @Test func projectorOnlyRepoIsRefused() throws {
        let result = try analysis([file("mmproj-model-f16.gguf", gb: 0.3)])
        let decision = ImportFit.decide(analysis: result, format: result.detected, quant: nil,
                                        budgetGB: proMaxBudget, deviceName: "iPhone")
        #expect(decision.kind == .refuse)
        #expect(decision.problem == .projectorOnly)
        #expect(decision.message?.contains("vision projector") == true)
    }

    @Test func supportFilesOnlyRepoIsRefused() throws {
        let result = try analysis([file("imatrix.gguf", gb: 0.01)])
        let decision = ImportFit.decide(analysis: result, format: result.detected, quant: nil,
                                        budgetGB: proMaxBudget, deviceName: "iPhone")
        #expect(decision.problem == .noWeights)
    }

    @Test func incompleteSplitIsRefusedWithACompleteAlternative() throws {
        let result = try analysis([
            file("m-Q4_K_M-00001-of-00003.gguf", gb: 1), file("m-Q4_K_M-00003-of-00003.gguf", gb: 1),
            file("m-Q2_K.gguf", gb: 1.5),
        ])
        let decision = ImportFit.decide(analysis: result, format: gguf, quant: quant(result, "Q4_K_M"),
                                        budgetGB: proMaxBudget, deviceName: "iPhone")
        #expect(decision.kind == .refuse)
        #expect(decision.problem == .incompleteSplit(missing: [2], expected: 3))
        #expect(decision.alternative?.label == "Q2_K")
        #expect(decision.message?.contains("missing part 2") == true)
    }

    @Test func completeSplitThatFitsIsAllowed() throws {
        let result = try analysis([
            file("m-Q4_K_M-00001-of-00002.gguf", gb: 1.2), file("m-Q4_K_M-00002-of-00002.gguf", gb: 1.2),
        ])
        let decision = ImportFit.decide(analysis: result, format: gguf, quant: result.defaultQuant,
                                        budgetGB: proMaxBudget, deviceName: "iPhone")
        #expect(decision.kind == .fits)
    }

    @Test func unsupportedArchitectureIsRefusedButAProjectorAnswerIsNot() throws {
        let files = [file("m-Q4_K_M.gguf", gb: 2), file("mmproj-F16.gguf", gb: 0.5)]
        let vision = ImportFormat(backend: .llamaCppVision, layout: .ggufWithMmproj)

        let unknown = try analysis(files, architecture: "fancynet")
        let refused = ImportFit.decide(analysis: unknown, format: vision, quant: nil,
                                       budgetGB: proMaxBudget, deviceName: "iPhone")
        #expect(refused.problem == .unsupportedArchitecture("fancynet"))

        let clip = try analysis(files, architecture: "clip")
        #expect(ImportFit.decide(analysis: clip, format: vision, quant: nil,
                                 budgetGB: proMaxBudget, deviceName: "iPhone").kind == .fits)
        let gemma = try analysis(files, architecture: "gemma3")
        #expect(ImportFit.decide(analysis: gemma, format: vision, quant: nil,
                                 budgetGB: proMaxBudget, deviceName: "iPhone").kind == .fits)
    }

    @Test func localFileUsesTheSameMemoryRules() {
        let fits = ImportFit.decideLocal(fileSizeBytes: Int64(2 * gib), budgetGB: proMaxBudget,
                                         deviceName: "iPhone")
        #expect(fits.kind == .fits)
        let tooBig = ImportFit.decideLocal(fileSizeBytes: Int64(16 * gib), budgetGB: proMaxBudget,
                                           deviceName: "iPhone")
        #expect(tooBig.kind == .refuse)
        #expect(tooBig.message?.contains("Look for a smaller model") == true)
    }

    @Test func parameterHintScalesWithTheBudget() {
        #expect(ImportFit.maxParametersBillions(budgetGB: 8.0, layout: .singleGGUF) == 9)
        #expect(ImportFit.maxParametersBillions(budgetGB: 5.1, layout: .singleGGUF) == 5)
        #expect(ImportFit.maxParametersBillions(budgetGB: 1.0, layout: .singleGGUF) == 0)
    }
}

// MARK: - Local file validation

@MainActor
struct LocalGGUFValidationTests {

    private func header(arch: String? = "llama", kind: String? = nil,
                        splitIndex: Int? = nil, splitCount: Int? = nil) -> GGUFHeader {
        GGUFHeader(version: 3, architecture: arch, name: nil, fileTypeLabel: nil,
                   fileKind: kind, splitIndex: splitIndex, splitCount: splitCount)
    }

    @Test func anOrdinaryModelPasses() throws {
        try LocalGGUFFile.validate(filename: "m.gguf", header: header())
        try LocalGGUFFile.validate(filename: "m.gguf", header: header(arch: nil))
    }

    @Test func aSplitPartIsRefused() {
        #expect(throws: LocalImportError.splitPart("m.gguf", index: 1, count: 3)) {
            try LocalGGUFFile.validate(filename: "m.gguf", header: header(splitIndex: 0, splitCount: 3))
        }
    }

    @Test func projectorsAdaptersAndMatricesAreRefused() {
        #expect(throws: LocalImportError.projectorOnly("v.gguf")) {
            try LocalGGUFFile.validate(filename: "v.gguf", header: header(arch: "clip"))
        }
        #expect(throws: LocalImportError.notAModel("l.gguf", kind: "adapter")) {
            try LocalGGUFFile.validate(filename: "l.gguf", header: header(kind: "adapter"))
        }
        #expect(throws: LocalImportError.notAModel("imatrix.gguf", kind: "imatrix")) {
            try LocalGGUFFile.validate(filename: "imatrix.gguf", header: header(arch: nil))
        }
    }

    @Test func anUnsupportedArchitectureIsRefused() {
        #expect(throws: LocalImportError.unsupportedArchitecture("m.gguf", architecture: "fancynet")) {
            try LocalGGUFFile.validate(filename: "m.gguf", header: header(arch: "fancynet"))
        }
        #expect(LlamaArchitectureSupport.verdict(for: "Gemma3") == .supported)
        #expect(LlamaArchitectureSupport.verdict(for: "qwen35") == .supported)
    }
}

// MARK: - Chat model eligibility

@MainActor
struct ChatModelEligibilityTests {

    @Test func speechBackendsCannotBeTheChatModel() {
        let speech: Set<ModelBackend> = [.whisperCpp, .coreMLWhisper, .coreMLKokoro]
        for backend in ModelBackend.allCases {
            #expect(backend.canRunChat == !speech.contains(backend), "\(backend)")
        }
    }

    private func model(_ backend: ModelBackend, _ capabilities: ModelCapabilities) -> AIModel {
        AIModel(id: backend.rawValue, name: "M", family: "f", parameterSize: "1B", quantization: "q",
                backend: backend, capabilities: capabilities, ramRequiredGB: 1, diskSizeGB: 1,
                huggingFaceRepo: "a/b", contextLength: 4096, quality: .good)
    }

    @Test func eligibilityNeedsAChatBackendAndTextInAndOut() {
        #expect(model(.llamaCpp, .textOnly).canBeChatModel)
        #expect(model(.llamaCppVision, .vision).canBeChatModel)
        #expect(model(.mlxAudio, .audioLLM).canBeChatModel)
        #expect(model(.appleIntelligence, .textOnly).canBeChatModel)
        #expect(!model(.whisperCpp, .speechToText).canBeChatModel)
        #expect(!model(.coreMLWhisper, .speechToText).canBeChatModel)
        #expect(!model(.coreMLKokoro, .textToSpeech).canBeChatModel)
        // A mislabelled import still cannot slip through on capabilities.
        #expect(!model(.whisperCpp, .textOnly).canBeChatModel)
    }
}

// MARK: - Empty replies

@MainActor
struct EmptyReplyTests {

    @Test func zeroTokensMeansItEndedImmediately() {
        #expect(EmptyReply.detect(text: "", tokensGenerated: 0) == .endedImmediately)
        #expect(EmptyReply.detect(text: " \n", tokensGenerated: nil) == .endedImmediately)
        #expect(EmptyReply.detect(text: "", tokensGenerated: 3) == .blank)
        #expect(EmptyReply.detect(text: "Hi", tokensGenerated: 0) == nil)
    }

    @Test func theNoticePointsAtTheTemplateMenu() {
        let message = EmptyReply.message(for: .endedImmediately)
        #expect(message.contains("chat template"))
        #expect(message.contains("Chat template"))
        #expect(!message.contains("(Empty response)"))
    }
}

// MARK: - Load failure causes

@MainActor
struct LlamaLoadFailureTests {

    @Test func namesTheCauseFromTheLog() {
        func cause(_ line: String, _ stage: LlamaLoadFailure.Stage = .model) -> LlamaLoadCause {
            LlamaLoadFailure.classify(log: ["llama_model_loader: loaded meta data", line], stage: stage)
        }
        #expect(cause("llama_model_load: error loading model: error loading model architecture: "
                      + "unknown model architecture: 'fancynet'") == .unsupportedArchitecture("fancynet"))
        #expect(cause("llama_model_load: error loading model: illegal split file idx: 1 (file: "
                      + "/m-00002-of-00003.gguf), model must be loaded with the first split")
                == .notFirstSplitPart)
        #expect(cause("llama_model_loader: failed to load GGUF split from /m-00002-of-00003.gguf")
                == .missingSplitParts)
        #expect(cause("llama_model_load: error loading model: failed to open /d/m-00002-of-00003.gguf: "
                      + "No such file or directory") == .missingSplitParts)
        #expect(cause("ggml_metal_buffer_init: error: failed to allocate buffer, size = 5120.00 MiB")
                == .outOfMemory)
        #expect(cause("llama_model_load: error loading model: done_getting_tensors: wrong number of "
                      + "tensors; expected 300, got 290") == .incompatibleFile)
        #expect(cause("llama_model_load: error loading model: tensor 'blk.3.attn_q.weight' data is not "
                      + "within the file bounds, model is corrupted or incomplete") == .damagedFile)
        #expect(cause("gguf_init_from_file_impl: invalid magic characters: 'abcd', expected 'GGUF'")
                == .damagedFile)
        #expect(cause("nothing useful") == .unknown)
        #expect(cause("nothing useful", .context) == .outOfMemory)
    }

    @Test func errorCarriesExplanationAndExcerpt() {
        let log = ["llama_model_loader: - kv 0: general.architecture str = fancynet",
                   "llama_model_load: error loading model: unknown model architecture: 'fancynet'"]
        let error = LlamaLoadError(cause: LlamaLoadFailure.classify(log: log, stage: .model),
                                   logExcerpt: LlamaLoadFailure.excerpt(from: log))
        #expect(error.cause.summary == "This model's architecture is not supported")
        #expect(error.localizedDescription.contains("fancynet"))
        #expect(error.logExcerpt.count == 1)
        #expect(error.detail.contains("llama.cpp log:"))
    }

    @Test func captureJoinsFragmentsAndSkipsDebug() {
        let capture = LlamaLogCapture()
        capture.append(level: 2, text: "before\n")
        let mark = capture.mark()
        capture.append(level: 4, text: "error: failed ")
        capture.append(level: 5, text: "to allocate\nnext line\n")
        capture.append(level: 1, text: "debug noise\n")
        #expect(capture.lines(since: mark) == ["error: failed to allocate", "next line"])
    }
}

// MARK: - Downloader

#if !MAC_STANDIN
@MainActor
struct SplitDownloadTests {

    @Test func everyPartOfTheChosenSetIsDownloaded() throws {
        let files = [
            HFSibling(rfilename: "Q4/m-Q4_K_M-00001-of-00002.gguf", size: 1),
            HFSibling(rfilename: "Q4/m-Q4_K_M-00002-of-00002.gguf", size: 1),
            HFSibling(rfilename: "Q8/m-Q8_0-00001-of-00002.gguf", size: 1),
            HFSibling(rfilename: "Q8/m-Q8_0-00002-of-00002.gguf", size: 1),
        ]
        let model = AIModel(id: "a/m#Q4_K_M", name: "M", family: "f", parameterSize: "1B",
                            quantization: "Q4_K_M", backend: .llamaCpp, fileLayout: .singleGGUF,
                            ramRequiredGB: 1, diskSizeGB: 1, huggingFaceRepo: "a/m", contextLength: 4096,
                            quality: .good, preferredGGUFFilename: "Q4/m-Q4_K_M-00001-of-00002.gguf",
                            source: .imported)
        let chosen = try ModelDownloader.filterFiles(files, for: model).map(\.rfilename)
        #expect(chosen == ["Q4/m-Q4_K_M-00001-of-00002.gguf", "Q4/m-Q4_K_M-00002-of-00002.gguf"])
    }
}
#endif
