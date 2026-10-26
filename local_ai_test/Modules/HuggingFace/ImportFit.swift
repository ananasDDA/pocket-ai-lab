//
//  ImportFit.swift
//  local_ai_test
//
//  The import sheet's go / warn / refuse decision, made before a single byte
//  is downloaded or copied. Pure over its inputs so the rules are testable.
//
//  Why it exists: the first field data from 1.1 had an iPhone 17 Pro Max
//  (llama.cpp budget ~8 GB) import three files of 27B GGUF models - a 3.9 GB
//  part of a split Q8_0, a 5 GB part of a split Q5_K_M and a 307 MB file that
//  was not a whole model - and fail to load them seven times. Each file was
//  judged on its own size, so every one "fit". A split model is judged as
//  the sum of its parts now, a selection that cannot load is refused, and the
//  refusal says what in the same repo would fit instead.
//

import Foundation

// MARK: - Decision

struct ImportDecision: Sendable, Equatable {

    enum Kind: Sendable, Equatable {
        /// Loads with room to spare.
        case fits
        /// Loads, but above the recommender's comfort margin: allowed, with a
        /// warning.
        case tight
        /// Cannot load. Nothing is downloaded.
        case refuse
    }

    enum Problem: Sendable, Equatable {
        /// Needs more memory than the device budget.
        case tooLarge(neededGB: Double, budgetGB: Double)
        /// Fits, but above `ImportFit.comfortMargin` of the budget.
        case tight(neededGB: Double, budgetGB: Double)
        /// The GGUF files are vision projectors only.
        case projectorOnly
        /// GGUF files exist but none of them is model weights (an importance
        /// matrix, for example).
        case noWeights
        /// A split model whose listing lacks some parts.
        case incompleteSplit(missing: [Int], expected: Int)
        /// The bundled llama.cpp has no implementation for this architecture.
        case unsupportedArchitecture(String)
    }

    let kind: Kind
    let problem: Problem?
    /// A quantization from the same listing that fits (comfortably when one
    /// does), offered when the current choice is refused or tight.
    let alternative: GGUFQuantOption?
    /// Why, in plain words, plus what would fit instead. nil when it fits.
    let message: String?

    static let fits = ImportDecision(kind: .fits, problem: nil, alternative: nil, message: nil)

    var blocksDownload: Bool { kind == .refuse }
}

// MARK: - Rules

enum ImportFit {

    /// Share of the device budget above which a model counts as tight. Same
    /// figure as `ModelRecommender.recommendationMargin` (asserted by a unit
    /// test), so the import sheet and the recommender agree on "comfortable".
    static let comfortMargin = 0.85

    /// Bits per weight of Q4_K_M, the quantization the "smaller model" hint is
    /// phrased in.
    private static let q4KMBitsPerWeight = 4.85

    static func verdict(ramGB: Double, budgetGB: Double) -> ResourceVerdict {
        if ramGB > budgetGB { return .wontFit }
        if ramGB > budgetGB * comfortMargin { return .tight }
        return .fits
    }

    // MARK: Hugging Face listing

    /// Decision for downloading `quant` (nil for non-GGUF layouts) of
    /// `analysis` through `format`, on a device that gives that backend
    /// `budgetGB`.
    static func decide(
        analysis: ImportAnalysis,
        format: ImportFormat,
        quant: GGUFQuantOption?,
        budgetGB: Double,
        deviceName: String
    ) -> ImportDecision {
        let isGGUF = format.layout == .singleGGUF || format.layout == .ggufWithMmproj

        guard isGGUF else {
            let weights = analysis.sizing(layout: format.layout, quant: nil).weights
            let ram = ImportAnalyzer.ramEstimateGB(layout: format.layout, weightBytes: weights)
            return memoryDecision(
                neededGB: ram, budgetGB: budgetGB, deviceName: deviceName,
                layout: format.layout, current: nil, candidates: []
            )
        }

        // Nothing loadable at all: projector-only or support-file-only repos.
        if analysis.ggufQuants.isEmpty {
            let hasProjector = analysis.files.contains {
                $0.rfilename.lowercased().hasSuffix(".gguf") && GGUFNaming.isProjector($0.rfilename)
            }
            return refusal(hasProjector ? .projectorOnly : .noWeights, alternative: nil)
        }

        // HF reports one architecture per repo; for a projector-only answer
        // in a repo that does have weights it described the wrong file, so
        // only an unsupported language architecture refuses.
        if case .unsupported(let arch)? = LlamaArchitectureSupport.verdict(for: analysis.ggufArchitecture) {
            return refusal(.unsupportedArchitecture(arch), alternative: nil)
        }

        guard let chosen = quant ?? analysis.defaultQuant else {
            return refusal(.noWeights, alternative: nil)
        }
        let candidates = analysis.ggufQuants.map {
            ($0, ImportAnalyzer.ramEstimateGB(layout: format.layout, weightBytes: $0.size))
        }

        if !chosen.isComplete {
            let alternative = bestAlternative(
                to: chosen, among: candidates, budgetGB: budgetGB, requireSmaller: false
            )
            var decision = refusal(
                .incompleteSplit(missing: chosen.missingParts, expected: chosen.expectedParts),
                alternative: alternative.map(\.0)
            )
            if let alternative {
                decision = decision.appending(alternativeSentence(alternative, budgetGB: budgetGB))
            }
            return decision
        }

        let ram = ImportAnalyzer.ramEstimateGB(layout: format.layout, weightBytes: chosen.size)
        return memoryDecision(
            neededGB: ram, budgetGB: budgetGB, deviceName: deviceName,
            layout: format.layout, current: chosen, candidates: candidates
        )
    }

    // MARK: Single local file

    /// Decision for a GGUF the user picked in Files. The file's own problems
    /// (projector, split part, unsupported architecture) are refused by
    /// `LocalGGUFFile.validate` before this is asked; this is memory only.
    static func decideLocal(
        fileSizeBytes: Int64, budgetGB: Double, deviceName: String
    ) -> ImportDecision {
        let ram = ImportAnalyzer.ramEstimateGB(layout: .singleGGUF, weightBytes: fileSizeBytes)
        return memoryDecision(
            neededGB: ram, budgetGB: budgetGB, deviceName: deviceName,
            layout: .singleGGUF, current: nil, candidates: []
        )
    }

    // MARK: Memory

    private static func memoryDecision(
        neededGB: Double,
        budgetGB: Double,
        deviceName: String,
        layout: ModelFileLayout,
        current: GGUFQuantOption?,
        candidates: [(GGUFQuantOption, Double)]
    ) -> ImportDecision {
        switch verdict(ramGB: neededGB, budgetGB: budgetGB) {
        case .fits:
            return .fits

        case .wontFit:
            var message = "This model needs about \(gb(neededGB)) of memory, but this "
                + "\(deviceName) can spare about \(gb(budgetGB)) for it. It would download "
                + "and then fail to load."
            let alternative = current.flatMap {
                bestAlternative(to: $0, among: candidates, budgetGB: budgetGB, requireSmaller: true)
            }
            if let alternative {
                message += " " + alternativeSentence(alternative, budgetGB: budgetGB)
            } else {
                message += " " + (candidates.count > 1 ? "Nothing in this repo fits. " : "")
                    + smallerModelHint(budgetGB: budgetGB, layout: layout)
            }
            return ImportDecision(
                kind: .refuse,
                problem: .tooLarge(neededGB: neededGB, budgetGB: budgetGB),
                alternative: alternative?.0,
                message: message
            )

        case .tight:
            let percent = Int((comfortMargin * 100).rounded())
            var message = "This model needs about \(gb(neededGB)) of the \(gb(budgetGB)) this "
                + "\(deviceName) can spare, more than \(percent)%. It should load, but a long "
                + "chat or apps in the background can push it over the limit."
            // Only a choice that is comfortable is worth suggesting here.
            let alternative = current.flatMap {
                bestAlternative(to: $0, among: candidates, budgetGB: budgetGB, requireSmaller: true)
            }.flatMap { verdict(ramGB: $0.1, budgetGB: budgetGB) == .fits ? $0 : nil }
            if let alternative {
                message += " \(alternative.0.label) (about \(gb(alternative.1))) leaves more room."
            }
            return ImportDecision(
                kind: .tight,
                problem: .tight(neededGB: neededGB, budgetGB: budgetGB),
                alternative: alternative?.0,
                message: message
            )
        }
    }

    /// The largest complete quantization that fits comfortably, else the
    /// largest that fits at all. With `requireSmaller`, only ones smaller
    /// than `current` qualify.
    private static func bestAlternative(
        to current: GGUFQuantOption,
        among candidates: [(GGUFQuantOption, Double)],
        budgetGB: Double,
        requireSmaller: Bool
    ) -> (GGUFQuantOption, Double)? {
        let pool = candidates.filter { option, _ in
            option.id != current.id && option.isComplete
                && (!requireSmaller || option.size < current.size)
        }
        let comfortable = pool.filter { verdict(ramGB: $0.1, budgetGB: budgetGB) == .fits }
        let loadable = pool.filter { verdict(ramGB: $0.1, budgetGB: budgetGB) != .wontFit }
        return (comfortable.isEmpty ? loadable : comfortable).max { $0.0.size < $1.0.size }
    }

    private static func alternativeSentence(
        _ alternative: (GGUFQuantOption, Double), budgetGB: Double
    ) -> String {
        let (option, ram) = alternative
        let fit = verdict(ramGB: ram, budgetGB: budgetGB) == .fits ? "fits" : "fits, but only just"
        return "\(option.label) from this repo needs about \(gb(ram)) and \(fit)."
    }

    /// "Look for a smaller model: about 9B parameters or fewer at Q4_K_M
    /// fits." Inverts the RAM estimate at the comfort margin.
    static func smallerModelHint(budgetGB: Double, layout: ModelFileLayout) -> String {
        let billions = maxParametersBillions(budgetGB: budgetGB, layout: layout)
        guard billions >= 1 else {
            return "Look for a much smaller model, under 1B parameters."
        }
        let quant = (layout == .singleGGUF || layout == .ggufWithMmproj) ? "Q4_K_M" : "4-bit"
        return "Look for a smaller model: about \(billions)B parameters or fewer at \(quant) fits."
    }

    /// Whole billions of parameters a 4-bit model may have to stay inside the
    /// comfort margin of `budgetGB`.
    static func maxParametersBillions(budgetGB: Double, layout: ModelFileLayout) -> Int {
        let target = budgetGB * comfortMargin
        // Weight bytes whose RAM estimate equals `target` (the estimate is
        // linear: weights * slope + constant).
        let constant = ImportAnalyzer.ramEstimateGB(layout: layout, weightBytes: 0)
        let slope = ImportAnalyzer.ramEstimateGB(
            layout: layout, weightBytes: Int64(ImportAnalyzer.bytesPerGB)
        ) - constant
        guard slope > 0 else { return 0 }
        let weightsGB = (target - constant) / slope
        let gbPerBillion = 1e9 * (q4KMBitsPerWeight / 8) / ImportAnalyzer.bytesPerGB
        return max(Int((weightsGB / gbPerBillion).rounded(.down)), 0)
    }

    // MARK: Messages

    private static func refusal(
        _ problem: ImportDecision.Problem, alternative: GGUFQuantOption?
    ) -> ImportDecision {
        ImportDecision(kind: .refuse, problem: problem, alternative: alternative,
                       message: message(for: problem))
    }

    static func message(for problem: ImportDecision.Problem) -> String {
        switch problem {
        case .tooLarge(let needed, let budget):
            return "This model needs about \(gb(needed)) of memory, but this device can "
                + "spare about \(gb(budget)) for it."
        case .tight(let needed, let budget):
            return "This model needs about \(gb(needed)) of the \(gb(budget)) this device can spare."
        case .projectorOnly:
            return "This repo only has a vision projector (mmproj). A projector adds image "
                + "input to a model but is not a model itself, so there is nothing here to "
                + "chat with. Import the repo that has the model's weights instead."
        case .noWeights:
            return "This repo has no GGUF model weights, only support files such as an "
                + "importance matrix. There is nothing here llama.cpp can load."
        case .incompleteSplit(let missing, let expected):
            let list = missing.map(String.init).joined(separator: ", ")
            let noun = missing.count == 1 ? "part" : "parts"
            return "This quantization is split into \(expected) files, but the repo is "
                + "missing \(noun) \(list). llama.cpp needs every part to load it."
        case .unsupportedArchitecture(let arch):
            return "This model's architecture (\(arch)) is not supported by the llama.cpp "
                + "version in this app (\(LlamaArchitectureSupport.llamaRevision)). It would "
                + "download and then fail to load."
        }
    }

    static func gb(_ value: Double) -> String {
        value < 1 ? String(format: "%.2f GB", value) : String(format: "%.1f GB", value)
    }
}

private extension ImportDecision {
    func appending(_ sentence: String) -> ImportDecision {
        ImportDecision(kind: kind, problem: problem, alternative: alternative,
                       message: [message, sentence].compactMap { $0 }.joined(separator: " "))
    }
}
