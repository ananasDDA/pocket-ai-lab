//
//  BenchmarkTypes.swift
//  local_ai_test
//
//  Result types produced by `BenchmarkEngine`. Everything here is a plain
//  value type — Codable, Sendable and `nonisolated` — so a report can be
//  written to disk from a background task, handed to the release test
//  harness, or diffed between builds without touching the main actor.
//
//  The scoring rule is deliberately narrow: the overall score is the quality
//  pass-rate and nothing else. Throughput and memory are reported as raw
//  numbers next to the device that produced them, because folding a
//  device-dependent tok/s into an absolute 0-100 score would mean the same
//  model "scores" differently on an iPhone 13 and an iPhone 17 Pro — which
//  says something about the phone, not about the model.
//
//  THE SCORE, EXACTLY. Over the tasks that were graded:
//
//      score = max(0, (Σ pass_t − Σ chance_t) / Σ (1 − chance_t)) × 100
//
//  pass_t is 1 or 0, and chance_t is the task's guessing floor
//  (`BenchmarkTask.chance`, 1/n for an n-choice task, 0 otherwise). This is
//  the Open LLM Leaderboard normalisation, (accuracy − chance) /
//  (1 − chance) clipped at 0, pooled over tasks so it also works for a suite
//  that mixes free-answer and multiple-choice items:
//  https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/normalization
//
//  Every check in the bundled suite is a free answer (chance 0), so today
//  the formula IS the flat pass rate, passed / graded × 100, and every
//  existing report keeps its number. It is deliberately a pass rate over ALL
//  tasks, not an average of category scores: with categories of 57 and 4
//  tasks, averaging would make one retrieval task worth five points. The
//  same formula over one category's tasks gives `CategoryScore.normalizedScore`.
//
//  A run that hit its time budget (`QualityResults.truncated`) is scored over
//  the tasks it completed. That number is real but it is a different exam
//  from the full suite, so it is shown as partial ("61 / 100, 73 of 118
//  tasks") and never compared with another run.
//

import Foundation

// MARK: - Report

/// One complete benchmark run of one model on one device.
nonisolated struct BenchmarkReport: Codable, Sendable, Hashable, Identifiable {

    /// UUID string. Stored as a `String` rather than a `UUID` so hand-written
    /// fixtures and the harness's JSON stay trivially readable.
    let id: String
    let date: Date

    /// Version of the app that produced the report — a run is only comparable
    /// against runs of the same build family.
    let appVersion: String
    let appBuild: String

    let modelId: String
    let modelName: String
    /// `ModelBackend.displayName` ("MLX", "llama.cpp", …), not the raw value,
    /// because reports are read by humans more often than by code.
    let backend: String
    let quantization: String
    let parameterSize: String

    let device: DeviceSnapshot
    let performance: PerformanceMetrics
    /// `nil` when the run was `.performanceOnly` or the quality stage was
    /// skipped. A report without quality has no overall score.
    let quality: QualityResults?

    /// Suite version behind `overallScore`, or `nil` for a performance-only
    /// report.
    ///
    /// IMPORTANT FOR CONSUMERS: scores are ONLY comparable within one suite
    /// version. Adding, removing or rewording a single task changes what
    /// "72 / 100" means, so a UI that ranks or diffs two reports must either
    /// compare same-version reports only, or show the version next to the
    /// score (e.g. "72 / 100 - suite v2"). Comparing a v1 score to a v2 score
    /// is not a regression signal; it is two different exams.
    var suiteVersion: Int? { quality?.suiteVersion }

    /// Which suite graded this report — `BenchmarkSuiteDocument.builtInID`
    /// for the bundled one and for every report written before suites could
    /// be imported.
    var suiteIdentity: String? {
        guard quality != nil else { return nil }
        return quality?.suiteID ?? BenchmarkSuiteDocument.builtInID
    }

    /// Name to print next to a score. The stored name for an imported suite,
    /// "Built-in" for ours.
    var suiteDisplayName: String? {
        guard let quality else { return nil }
        return quality.suiteName ?? "Built-in"
    }

    /// True when both reports were graded by the same suite, i.e. their
    /// `overallScore`s may be compared directly.
    ///
    /// BOTH halves matter. The version catches "we added three hard tasks";
    /// the identity catches "this one was graded by a suite somebody pasted
    /// in", which shares nothing with ours but a version number that happens
    /// to be the same integer.
    ///
    /// A PARTIAL run (time budget reached, `QualityResults.truncated`) is
    /// never comparable, not even with another partial run of the same
    /// suite: the two stopped at different tasks, so they sat different
    /// exams.
    func isScoreComparable(with other: BenchmarkReport) -> Bool {
        guard !isPartial, !other.isPartial else { return false }
        guard let mine = suiteVersion, let theirs = other.suiteVersion else { return false }
        guard let mineID = suiteIdentity, let theirsID = other.suiteIdentity else { return false }
        return mine == theirs && mineID == theirsID
    }

    init(
        id: String = UUID().uuidString,
        date: Date = Date(),
        appVersion: String,
        appBuild: String,
        modelId: String,
        modelName: String,
        backend: String,
        quantization: String,
        parameterSize: String,
        device: DeviceSnapshot,
        performance: PerformanceMetrics,
        quality: QualityResults?
    ) {
        self.id = id
        self.date = date
        self.appVersion = appVersion
        self.appBuild = appBuild
        self.modelId = modelId
        self.modelName = modelName
        self.backend = backend
        self.quantization = quantization
        self.parameterSize = parameterSize
        self.device = device
        self.performance = performance
        self.quality = quality
    }

    /// 0…100 — the chance-normalised pass rate over the graded tasks (see
    /// the file header for the formula; with no multiple-choice tasks it is
    /// exactly passed / graded × 100). `nil` when the quality stage did not
    /// run: a performance-only report is a measurement, not a grade, and
    /// must not be displayed as "0 / 100".
    var overallScore: Double? {
        guard let quality, quality.completed > 0 else { return nil }
        return quality.normalizedScore * 100
    }

    /// Rounded form for badges and list rows.
    var overallScoreRounded: Int? {
        overallScore.map { Int($0.rounded()) }
    }

    /// True when the quality stage stopped at its time budget before the
    /// last task. The score is then over the completed tasks only.
    var isPartial: Bool { quality?.truncated ?? false }

    /// "73 of 118 tasks" for a partial run, `nil` otherwise.
    var partialTasksLabel: String? {
        guard let quality, quality.truncated else { return nil }
        return "\(quality.completed) of \(quality.total) tasks"
    }

    /// The score as every list row, history entry and log line prints it:
    /// "72/100", or "61/100, 73 of 118 tasks" for a partial run, so a
    /// cut-short score can never pass for a full one. `nil` without a score.
    var scoreLabel: String? {
        guard let score = overallScoreRounded else { return nil }
        guard let partial = partialTasksLabel else { return "\(score)/100" }
        return "\(score)/100, \(partial)"
    }

    /// One-line digest for logs and the harness's console output.
    ///
    /// v2 appends the decode spread when it was measured and a "throttled"
    /// marker (with the observed drop, when it is known) when the numbers
    /// must not be compared.
    var summaryLine: String {
        var score = overallScoreRounded.map { "\($0)/100" } ?? "perf-only"
        if let quality, quality.truncated {
            score += " partial (\(quality.completed)/\(quality.total) tasks)"
        }
        var line = String(
            format: "%@ [%@] %@ - %.1f tok/s gen, %.1f s load, %.0f MB peak",
            modelName, backend, score,
            performance.generationTokensPerSecond,
            performance.loadSeconds,
            performance.peakMemoryMB
        )
        if let sd = performance.generationTokensPerSecondStdDev {
            line += String(format: " (SD %.2f, n=%d)", sd, performance.generationRepetitions)
        }
        if performance.thermallyConstrained {
            if let drop = performance.throttleDropPercent {
                line += String(format: " [throttled -%.0f%% - not comparable]", drop)
            } else {
                line += " [throttled - not comparable]"
            }
        }
        return line
    }
}

// MARK: - Device

/// The parts of `DeviceInfo` worth freezing into a report. Deliberately a
/// separate type: `DeviceInfo` carries live budget math that changes meaning
/// between releases, while a report has to stay readable years later.
nonisolated struct DeviceSnapshot: Codable, Sendable, Hashable {
    /// "iPhone17,1"
    let identifier: String
    /// "iPhone 16 Pro"
    let marketingName: String
    /// "A18 Pro"
    let chip: String
    let ramGB: Double
    let iOSVersion: String

    init(
        identifier: String,
        marketingName: String,
        chip: String,
        ramGB: Double,
        iOSVersion: String
    ) {
        self.identifier = identifier
        self.marketingName = marketingName
        self.chip = chip
        self.ramGB = ramGB
        self.iOSVersion = iOSVersion
    }
}

extension DeviceSnapshot {

    /// Snapshot of the device the app is running on. Main-actor because
    /// `DeviceScanner` is — call it once per run, not once per task.
    static func current() -> DeviceSnapshot {
        DeviceSnapshot(DeviceScanner.shared.scan())
    }

    init(_ info: DeviceInfo) {
        self.init(
            identifier: info.identifier,
            marketingName: info.marketingName,
            chip: info.chip,
            ramGB: (Double(info.totalRAM) / 1_073_741_824 * 10).rounded() / 10,
            iOSVersion: info.iOSVersion
        )
    }
}

// MARK: - Statistics

/// The two statistics the benchmark publishes. Split out of the metrics type
/// so a test can pin them and so every caller uses the SAME definition of
/// "standard deviation" — the sample one.
nonisolated enum BenchmarkStatistics {

    static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    /// SAMPLE standard deviation (Bessel-corrected, n-1 in the denominator),
    /// because three repetitions are a sample of the device's behaviour, not
    /// the entire population of possible runs. `nil` for fewer than two
    /// values: one measurement has no spread, and reporting 0 would claim a
    /// precision that was never measured.
    static func sampleStandardDeviation(_ values: [Double]) -> Double? {
        guard values.count > 1 else { return nil }
        let m = mean(values)
        let sumOfSquares = values.reduce(0) { $0 + ($1 - m) * ($1 - m) }
        return (sumOfSquares / Double(values.count - 1)).squareRoot()
    }

    /// Standard deviation as a percentage of the mean (coefficient of
    /// variation). The number to show next to a tok/s figure: "12.4 tok/s
    /// ±3%" tells a reader whether the run was stable. `nil` when there is no
    /// spread to report or the mean is zero.
    static func relativeSpreadPercent(_ values: [Double]) -> Double? {
        guard let sd = sampleStandardDeviation(values) else { return nil }
        let m = mean(values)
        guard m > 0 else { return nil }
        return sd / m * 100
    }
}

// MARK: - Standard protocol (pp512 / tg128)

/// Our approximation of the community pp512 / tg128 convention that
/// llama-bench prints and that PocketPal, DeviceMark and the apple-silicon
/// LLM benchmarks quote. A reader who knows those numbers can put this phone
/// next to their laptop without reading our methodology first.
///
/// It is an APPROXIMATION, and the report says so in three ways:
///   • the prompt length is ESTIMATED from characters (engines do not hand
///     back a prefill token count), so `promptTokensEstimated` is ~512, not
///     exactly 512;
///   • prefill throughput is derived from time to first token, which is
///     prompt processing PLUS one decode step;
///   • the decode figure is the average of `repetitions` runs, measured
///     first-token-to-last-token exactly like `generationTokensPerSecond`.
///
/// Comparability rule: these numbers are only comparable across runs that
/// used the SAME prompt. `promptId` freezes that — bump it (and expect the
/// numbers to move) if `BenchmarkEngine.standardProtocolPrompt` is ever
/// edited. Never edit the prompt and keep the id.
nonisolated struct StandardProtocolMetrics: Codable, Sendable, Hashable {

    /// Identifier of the frozen prompt this measurement used, e.g. "pp512-v1".
    /// Two runs with different `promptId`s are NOT comparable.
    let promptId: String

    /// Character-based estimate of the frozen prompt's token count. Targets
    /// 512; the real count depends on the model's tokenizer.
    let promptTokensEstimated: Int

    /// Token cap the decode half ran under. 128 by convention (tg128).
    let generationMaxTokens: Int

    /// Prompt-processing throughput on the frozen prompt — the "pp512" half.
    let prefillTokensPerSecond: Double

    /// Average decode throughput over `repetitions` runs — the "tg128" half.
    let generationTokensPerSecond: Double

    /// Sample standard deviation of `generationRunsTokensPerSecond`.
    let generationTokensPerSecondStdDev: Double?

    /// Raw per-repetition decode rates, in run order.
    let generationRunsTokensPerSecond: [Double]

    /// How many measured decode repetitions were actually completed. May be
    /// smaller than the number requested when a repetition produced too few
    /// tokens to time.
    let repetitions: Int

    init(
        promptId: String,
        promptTokensEstimated: Int,
        generationMaxTokens: Int,
        prefillTokensPerSecond: Double,
        generationTokensPerSecond: Double,
        generationTokensPerSecondStdDev: Double? = nil,
        generationRunsTokensPerSecond: [Double] = [],
        repetitions: Int
    ) {
        self.promptId = promptId
        self.promptTokensEstimated = promptTokensEstimated
        self.generationMaxTokens = generationMaxTokens
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.generationTokensPerSecond = generationTokensPerSecond
        self.generationTokensPerSecondStdDev = generationTokensPerSecondStdDev
        self.generationRunsTokensPerSecond = generationRunsTokensPerSecond
        self.repetitions = repetitions
    }

    private enum CodingKeys: String, CodingKey {
        case promptId, promptTokensEstimated, generationMaxTokens
        case prefillTokensPerSecond, generationTokensPerSecond
        case generationTokensPerSecondStdDev, generationRunsTokensPerSecond
        case repetitions
    }

    /// Tolerant of hand-written and older files: only the two headline rates
    /// are required.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            promptId: try c.decodeIfPresent(String.self, forKey: .promptId) ?? "unknown",
            promptTokensEstimated: try c.decodeIfPresent(Int.self, forKey: .promptTokensEstimated) ?? 0,
            generationMaxTokens: try c.decodeIfPresent(Int.self, forKey: .generationMaxTokens) ?? 0,
            prefillTokensPerSecond: try c.decode(Double.self, forKey: .prefillTokensPerSecond),
            generationTokensPerSecond: try c.decode(Double.self, forKey: .generationTokensPerSecond),
            generationTokensPerSecondStdDev: try c.decodeIfPresent(
                Double.self, forKey: .generationTokensPerSecondStdDev
            ),
            generationRunsTokensPerSecond: try c.decodeIfPresent(
                [Double].self, forKey: .generationRunsTokensPerSecond
            ) ?? [],
            repetitions: try c.decodeIfPresent(Int.self, forKey: .repetitions) ?? 1
        )
    }

    /// "pp512 ~1180 t/s - tg128 24.6 t/s" — the shape llama-bench readers
    /// expect, with the tilde admitting the estimate.
    var summaryLine: String {
        String(
            format: "pp512 ~%.0f t/s - tg128 %.1f t/s",
            prefillTokensPerSecond, generationTokensPerSecond
        )
    }
}

// MARK: - Performance

/// Raw throughput / footprint numbers. Never scored, always reported.
///
/// v2 adds repetitions: the decode probe runs several times and this type
/// carries the average (under the ORIGINAL v1 field name), the sample
/// standard deviation, and the raw per-repetition rates. Every v2 field is
/// optional or defaulted on decode, so a v1 report written by an older build
/// still decodes without migration.
nonisolated struct PerformanceMetrics: Codable, Sendable, Hashable {

    /// Bumped when the MEASUREMENT METHODOLOGY changes, not when a field is
    /// added. v1: single-shot probes, no warmup. v2: warmup generation,
    /// repeated decode probe, thermal gating, optional standard protocol.
    /// A v1 report decodes as `metricsVersion == 1`; consumers use it to
    /// decide whether two reports may be compared at all.
    static let currentMetricsVersion = 2

    /// Wall clock of `InferenceManager.prepare(model:)`, measured from a
    /// fully unloaded engine (a warm re-load would be meaninglessly fast).
    /// This IS the cold metric and is deliberately never warmed up or
    /// repeated: there is only one first load.
    let loadSeconds: Double

    /// Prompt-processing throughput, estimated as prompt tokens ÷ time to
    /// first token on a fixed ~300-token prompt. Engines do not report a
    /// prefill token count, so the token count is estimated from the prompt
    /// text (see `BenchmarkEngine.estimatedTokenCount`). Comparable between
    /// models on the same device, indicative across devices.
    ///
    /// v2: the AVERAGE over `prefillRunsTokensPerSecond`. The "estimated"
    /// caveat above still applies in full.
    let prefillTokensPerSecond: Double

    /// Decode throughput, measured from the FIRST token to the last one, so
    /// prefill time is excluded. This is the number users experience as
    /// "how fast does it type".
    ///
    /// v2: the AVERAGE over `generationRunsTokensPerSecond`. Field name kept
    /// from v1 on purpose so every existing consumer keeps working.
    let generationTokensPerSecond: Double

    /// Highest `phys_footprint` sampled during the run, in MB — the figure
    /// jetsam charges and Xcode's memory gauge shows.
    let peakMemoryMB: Double

    /// `InferenceManager.memoryBudgetMB(for:)` at the time of the run, i.e.
    /// how much this backend was allowed to use on this device.
    let memoryBudgetMB: Int64

    /// "nominal" / "fair" / "serious" / "critical".
    ///
    /// v2: `thermalStateBefore` is sampled AFTER the cool-down gate, i.e. it
    /// is the state the measurement actually started in rather than the state
    /// the user tapped Run in.
    ///
    /// These readings are INFORMATION ONLY. They no longer decide
    /// `thermallyConstrained`: on a modern phone every sustained LLM run
    /// reaches `.serious` within minutes, so a rule based on the state
    /// flagged every row of a 12-model sweep and said nothing. What the flag
    /// is decided by now is the measured slowdown - see `throttleVerdict`.
    let thermalStateBefore: String
    let thermalStateAfter: String

    /// Tokens produced by the generation probe (not by the quality suite).
    ///
    /// v2 CHANGED MEANING: the SUM across every measured decode repetition,
    /// so a 3-repetition run of a 200-token probe reports ~600. Per-repetition
    /// counts are in `generationRunsTokenCounts`.
    let tokensGenerated: Int

    /// Wall clock of the whole benchmark, load phase included.
    let wallClockSeconds: Double

    // MARK: v2 additions - all optional or defaulted on decode

    /// Sample standard deviation of `generationRunsTokensPerSecond`, in tok/s.
    /// `nil` on a v1 report and on any run with fewer than two measured
    /// repetitions. Show it as "±" next to the average.
    let generationTokensPerSecondStdDev: Double?

    /// Raw per-repetition decode rates in run order — the honest record.
    /// The first entry is the first MEASURED repetition (a separate, shorter
    /// warmup generation ran before it when `warmupPerformed` is true), so a
    /// visibly slower first entry is a real cold-cache effect and not an
    /// artefact of the harness. `nil` on a v1 report.
    let generationRunsTokensPerSecond: [Double]?

    /// Token count of each decode repetition, same order as
    /// `generationRunsTokensPerSecond`. Cheap to record and it makes the
    /// rates auditable.
    let generationRunsTokenCounts: [Int]?

    /// Raw per-repetition prefill rates. Two repetitions by default — enough
    /// to catch a wild outlier, not enough for a meaningful deviation, so no
    /// standard deviation is published for prefill.
    let prefillRunsTokensPerSecond: [Double]?

    /// True when an unmeasured warmup generation ran between the load and the
    /// first measured probe. Always true for a normal v2 run; false on v1
    /// reports and when warmup was explicitly disabled.
    let warmupPerformed: Bool

    /// True when THROTTLING WAS DETECTED IN THE MEASUREMENTS themselves —
    /// see `throttleVerdict(runs:protocolRate:probeMean:)` for the exact
    /// rule. The JSON key keeps its v2 name for compatibility; it has always
    /// meant "this run's throughput is not comparable", and it still does.
    ///
    /// It is NOT derived from `ProcessInfo.thermalState` any more. Evidence
    /// from a 12-model sweep on an iPhone Air: the device reaches `.serious`
    /// within minutes for EVERY model, small and large alike, so a
    /// state-based rule flagged all of them and carried no information. The
    /// per-repetition decode rates separate the two cases cleanly: a small
    /// model holds its rate (30.5, 30.4, 28.1 tok/s) while a large one falls
    /// apart (11.5, 9.1, 4.0 tok/s).
    ///
    /// A FLAGGED RUN'S THROUGHPUT NUMBERS ARE NOT COMPARABLE — not with other
    /// devices, not with other models, not with an earlier run of the same
    /// model: the figures describe how far the phone had ramped down by the
    /// time each repetition ran. Consumers must flag such a report and should
    /// exclude it from any ranking.
    let thermallyConstrained: Bool

    /// The observed max-to-min drop across `generationRunsTokensPerSecond`,
    /// in percent — "speed fell 46% across runs". `nil` when it could not be
    /// measured (a v1 report, or fewer than two measured repetitions), which
    /// is not the same as zero.
    ///
    /// Recorded whenever it is measurable, including on runs that stayed
    /// under the threshold, so a reader can see the spread that produced (or
    /// did not produce) the verdict.
    let throttleDropPercent: Double?

    /// Hottest thermal state observed at any probe or task boundary during the
    /// run, using the same lowercase spellings as `thermalStateBefore`.
    /// `nil` on a v1 report.
    let peakThermalState: String?

    /// The pp512 / tg128 block, when it ran. `nil` on a v1 report and when
    /// `BenchmarkOptions.runStandardProtocol` was turned off.
    let standardProtocol: StandardProtocolMetrics?

    /// See `currentMetricsVersion`. Decodes as 1 for a v1 report.
    let metricsVersion: Int

    init(
        loadSeconds: Double,
        prefillTokensPerSecond: Double,
        generationTokensPerSecond: Double,
        peakMemoryMB: Double,
        memoryBudgetMB: Int64,
        thermalStateBefore: String,
        thermalStateAfter: String,
        tokensGenerated: Int,
        wallClockSeconds: Double,
        generationTokensPerSecondStdDev: Double? = nil,
        generationRunsTokensPerSecond: [Double]? = nil,
        generationRunsTokenCounts: [Int]? = nil,
        prefillRunsTokensPerSecond: [Double]? = nil,
        warmupPerformed: Bool = false,
        thermallyConstrained: Bool = false,
        throttleDropPercent: Double? = nil,
        peakThermalState: String? = nil,
        standardProtocol: StandardProtocolMetrics? = nil,
        metricsVersion: Int = PerformanceMetrics.currentMetricsVersion
    ) {
        self.loadSeconds = loadSeconds
        self.prefillTokensPerSecond = prefillTokensPerSecond
        self.generationTokensPerSecond = generationTokensPerSecond
        self.peakMemoryMB = peakMemoryMB
        self.memoryBudgetMB = memoryBudgetMB
        self.thermalStateBefore = thermalStateBefore
        self.thermalStateAfter = thermalStateAfter
        self.tokensGenerated = tokensGenerated
        self.wallClockSeconds = wallClockSeconds
        self.generationTokensPerSecondStdDev = generationTokensPerSecondStdDev
        self.generationRunsTokensPerSecond = generationRunsTokensPerSecond
        self.generationRunsTokenCounts = generationRunsTokenCounts
        self.prefillRunsTokensPerSecond = prefillRunsTokensPerSecond
        self.warmupPerformed = warmupPerformed
        self.thermallyConstrained = thermallyConstrained
        self.throttleDropPercent = throttleDropPercent
        self.peakThermalState = peakThermalState
        self.standardProtocol = standardProtocol
        self.metricsVersion = metricsVersion
    }

    private enum CodingKeys: String, CodingKey {
        case loadSeconds, prefillTokensPerSecond, generationTokensPerSecond
        case peakMemoryMB, memoryBudgetMB
        case thermalStateBefore, thermalStateAfter
        case tokensGenerated, wallClockSeconds
        case generationTokensPerSecondStdDev, generationRunsTokensPerSecond
        case generationRunsTokenCounts, prefillRunsTokensPerSecond
        case warmupPerformed, thermallyConstrained, throttleDropPercent
        case peakThermalState
        case standardProtocol, metricsVersion
    }

    /// Hand-written so a v1 report — which has none of the v2 keys — still
    /// decodes. Every v2 key goes through `decodeIfPresent`; the v1 keys stay
    /// required, because a file missing `loadSeconds` is corrupt, not old.
    /// `encode` stays synthesized and omits the nil optionals.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            loadSeconds: try c.decode(Double.self, forKey: .loadSeconds),
            prefillTokensPerSecond: try c.decode(Double.self, forKey: .prefillTokensPerSecond),
            generationTokensPerSecond: try c.decode(Double.self, forKey: .generationTokensPerSecond),
            peakMemoryMB: try c.decode(Double.self, forKey: .peakMemoryMB),
            memoryBudgetMB: try c.decode(Int64.self, forKey: .memoryBudgetMB),
            thermalStateBefore: try c.decode(String.self, forKey: .thermalStateBefore),
            thermalStateAfter: try c.decode(String.self, forKey: .thermalStateAfter),
            tokensGenerated: try c.decode(Int.self, forKey: .tokensGenerated),
            wallClockSeconds: try c.decode(Double.self, forKey: .wallClockSeconds),
            generationTokensPerSecondStdDev: try c.decodeIfPresent(
                Double.self, forKey: .generationTokensPerSecondStdDev
            ),
            generationRunsTokensPerSecond: try c.decodeIfPresent(
                [Double].self, forKey: .generationRunsTokensPerSecond
            ),
            generationRunsTokenCounts: try c.decodeIfPresent(
                [Int].self, forKey: .generationRunsTokenCounts
            ),
            prefillRunsTokensPerSecond: try c.decodeIfPresent(
                [Double].self, forKey: .prefillRunsTokensPerSecond
            ),
            warmupPerformed: try c.decodeIfPresent(Bool.self, forKey: .warmupPerformed) ?? false,
            thermallyConstrained: try c.decodeIfPresent(Bool.self, forKey: .thermallyConstrained) ?? false,
            // Absent on every report written before this rule existed: such a
            // file keeps the flag exactly as it was stored, without a drop.
            throttleDropPercent: try c.decodeIfPresent(Double.self, forKey: .throttleDropPercent),
            peakThermalState: try c.decodeIfPresent(String.self, forKey: .peakThermalState),
            standardProtocol: try c.decodeIfPresent(
                StandardProtocolMetrics.self, forKey: .standardProtocol
            ),
            // A file without the key predates v2 by definition.
            metricsVersion: try c.decodeIfPresent(Int.self, forKey: .metricsVersion) ?? 1
        )
    }

    /// Share of the backend's memory budget the run actually used, 0…1+.
    /// `nil` when the budget was not measurable (simulator, unit tests).
    var memoryHeadroomFraction: Double? {
        guard memoryBudgetMB > 0 else { return nil }
        return peakMemoryMB / Double(memoryBudgetMB)
    }

    /// True when the device heated up during the run. Informational only:
    /// almost every sustained run heats the phone, so this is context next to
    /// the thermal chips, not a reason to distrust the numbers. That call is
    /// `thermallyConstrained`'s.
    var thermalDegraded: Bool {
        Self.thermalRank(thermalStateAfter) > Self.thermalRank(thermalStateBefore)
    }

    /// Number of measured decode repetitions behind
    /// `generationTokensPerSecond`. 1 for a v1 report.
    var generationRepetitions: Int {
        generationRunsTokensPerSecond?.count ?? 1
    }

    /// Standard deviation as a percentage of the average decode rate — the
    /// number to render as "24.6 tok/s ±4%". `nil` when there was no spread
    /// to measure.
    var generationSpreadPercent: Double? {
        guard let runs = generationRunsTokensPerSecond else { return nil }
        return BenchmarkStatistics.relativeSpreadPercent(runs)
    }

    /// A blunt gate for consumers: false means "do not put this number in a
    /// league table". Currently only detected throttling disqualifies a run.
    var throughputIsComparable: Bool { !thermallyConstrained }

    // MARK: - Throttle detection

    /// A run is thrown out when the slowdown between its own repetitions
    /// exceeds this many percent. 20% is comfortably above the run-to-run
    /// noise seen on stable models (a few percent) and far below the collapse
    /// seen on models that do throttle (24% to 65% in the reference sweep).
    static let throttleDropThresholdPercent: Double = 20

    /// THE rule behind `thermallyConstrained`, as a pure function so it can
    /// be tested without a device, an engine or a report.
    ///
    /// A run is flagged when EITHER of these holds:
    ///
    ///   1. the decode rate fell by more than
    ///      `throttleDropThresholdPercent` between the best and the worst
    ///      measured repetition — `(max - min) / max`. With fewer than two
    ///      usable repetitions the rule cannot fire: one measurement has no
    ///      trend, and guessing one would be worse than saying nothing.
    ///   2. the standard-protocol tg128 rate came in more than
    ///      `throttleDropThresholdPercent` below the decode-probe mean, when
    ///      both exist. tg128 runs LAST, so a phone that ramped down over the
    ///      course of the run shows up here even if the three decode
    ///      repetitions themselves were consistent.
    ///
    /// - Parameters:
    ///   - runs: per-repetition decode rates in run order, tok/s.
    ///   - protocolRate: `StandardProtocolMetrics.generationTokensPerSecond`,
    ///     or `nil` when the pp512/tg128 probe did not run.
    ///   - probeMean: mean of `runs`, or `nil` when nothing was measured.
    /// - Returns: the verdict, plus the max-to-min drop in percent whenever
    ///   it was measurable — reported even when it stayed under the
    ///   threshold, and `nil` when there were not two rates to compare.
    static func throttleVerdict(
        runs: [Double]?,
        protocolRate: Double?,
        probeMean: Double?
    ) -> (constrained: Bool, dropPercent: Double?) {
        // Zero and non-finite rates are not measurements: a repetition that
        // produced too few tokens to time must not read as a 100% collapse.
        let usable = (runs ?? []).filter { $0.isFinite && $0 > 0 }

        var dropPercent: Double?
        if usable.count > 1, let best = usable.max(), let worst = usable.min(), best > 0 {
            dropPercent = (best - worst) / best * 100
        }

        var constrained = (dropPercent ?? 0) > throttleDropThresholdPercent

        if !constrained,
           let protocolRate, protocolRate.isFinite, protocolRate > 0,
           let probeMean, probeMean.isFinite, probeMean > 0 {
            let shortfall = (probeMean - protocolRate) / probeMean * 100
            constrained = shortfall > throttleDropThresholdPercent
        }

        return (constrained, dropPercent)
    }

    private static func thermalRank(_ state: String) -> Int {
        switch state {
        case "nominal":  return 0
        case "fair":     return 1
        case "serious":  return 2
        case "critical": return 3
        default:         return -1
        }
    }
}

// MARK: - Quality

/// Outcome of the verifiable task suite. Nothing here involves an LLM judge:
/// every task is checked by `BenchmarkCheck`, so two runs of the same suite
/// on the same answers always agree.
nonisolated struct QualityResults: Codable, Sendable, Hashable {

    /// `BenchmarkSuiteDocument.suiteVersion` of the suite that ran.
    ///
    /// SCORES ARE ONLY COMPARABLE WITHIN ONE SUITE VERSION. A pass-rate is a
    /// mark on one specific exam: adding a hard task, dropping an easy one or
    /// rewording a prompt all move every model's score without any model
    /// changing. Any UI that ranks models, shows a delta against a previous
    /// run, or aggregates across reports must group by this value and show it
    /// next to the score. See `BenchmarkReport.isScoreComparable(with:)`.
    let suiteVersion: Int
    let categories: [CategoryScore]

    /// `BenchmarkSuiteDocument.identity` of the suite that ran. `nil` in
    /// reports written before suites could be imported, which all came from
    /// the bundled suite — so a missing value reads as
    /// `BenchmarkSuiteDocument.builtInID`.
    ///
    /// A version number alone stopped being a sufficient key once anyone
    /// could import a suite: two different suites both at version 1 are two
    /// different exams, and a delta between them is noise. Comparisons key on
    /// this id AND the version.
    let suiteID: String?

    /// Name of the suite, frozen at run time so a report still says what it
    /// was graded on after that suite has been deleted from the device.
    let suiteName: String?

    /// Tasks the run SET OUT to grade: the suite after the category filter
    /// and `maxTasks`. Equal to `completed` unless the run was truncated.
    /// JSON key `total`; a report written before time budgets existed has
    /// none and decodes it as `completed`, which is what it was.
    let total: Int

    /// True when the quality stage stopped at `BenchmarkOptions.qualityBudget`
    /// before the last task. `false` in every older report.
    let truncated: Bool

    init(
        suiteVersion: Int,
        categories: [CategoryScore],
        suiteID: String? = nil,
        suiteName: String? = nil,
        total: Int? = nil,
        truncated: Bool = false
    ) {
        self.suiteVersion = suiteVersion
        self.categories = categories
        self.suiteID = suiteID
        self.suiteName = suiteName
        let completed = categories.reduce(0) { $0 + $1.total }
        self.total = max(total ?? completed, completed)
        self.truncated = truncated
    }

    private enum CodingKeys: String, CodingKey {
        case suiteVersion, categories, suiteID, suiteName
        case completed, total, truncated
    }

    /// Tolerant of every report written before `total` / `truncated`.
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            suiteVersion: try c.decode(Int.self, forKey: .suiteVersion),
            categories: try c.decode([CategoryScore].self, forKey: .categories),
            suiteID: try c.decodeIfPresent(String.self, forKey: .suiteID),
            suiteName: try c.decodeIfPresent(String.self, forKey: .suiteName),
            total: try c.decodeIfPresent(Int.self, forKey: .total),
            truncated: try c.decodeIfPresent(Bool.self, forKey: .truncated) ?? false
        )
    }

    /// `completed` is written for readers of the raw JSON (the release
    /// report script among them) and ignored on decode: it is always the sum
    /// of the category counters.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(suiteVersion, forKey: .suiteVersion)
        try c.encode(categories, forKey: .categories)
        try c.encodeIfPresent(suiteID, forKey: .suiteID)
        try c.encodeIfPresent(suiteName, forKey: .suiteName)
        try c.encode(completed, forKey: .completed)
        try c.encode(total, forKey: .total)
        try c.encode(truncated, forKey: .truncated)
    }

    var passed: Int { categories.reduce(0) { $0 + $1.passed } }

    /// Tasks actually graded. The score is computed over these.
    var completed: Int { categories.reduce(0) { $0 + $1.total } }

    /// 0…1, raw passed / graded with no chance correction. Zero tasks means
    /// zero rather than a division by zero; callers that care about "did
    /// anything run" check `completed`.
    var passRate: Double {
        completed > 0 ? Double(passed) / Double(completed) : 0
    }

    /// 0…1, the chance-normalised score over every graded task (see the file
    /// header). Equals `passRate` when no task has a guessing floor.
    var normalizedScore: Double {
        BenchmarkScoring.normalized(
            passed: passed,
            graded: completed,
            chanceSum: categories.reduce(0) { $0 + $1.chanceSum }
        )
    }

    /// All task results across every category, in suite order.
    var allTasks: [TaskResult] { categories.flatMap(\.tasks) }
}

// MARK: - Scoring

/// The one implementation of the score formula in the file header, so the
/// overall score and every category score are computed the same way.
nonisolated enum BenchmarkScoring {

    /// (passed − Σchance) / (graded − Σchance), clipped to 0…1.
    ///
    /// `graded − Σchance` is Σ(1 − chance_t); `passed − Σchance` is
    /// Σpass_t − Σchance_t. Zero graded tasks scores 0, and so does the
    /// degenerate case where every task is a certain guess (chance 1): such
    /// a suite measures nothing.
    static func normalized(passed: Int, graded: Int, chanceSum: Double) -> Double {
        guard graded > 0 else { return 0 }
        let floor = min(max(chanceSum, 0), Double(graded))
        let denominator = Double(graded) - floor
        guard denominator > 0 else { return 0 }
        return min(max((Double(passed) - floor) / denominator, 0), 1)
    }

    /// Convenience over raw outcomes, mostly for tests and the docs:
    /// `[(true, 0), (false, 0.25)]` and so on.
    static func normalized(_ outcomes: [(passed: Bool, chance: Double)]) -> Double {
        normalized(
            passed: outcomes.filter(\.passed).count,
            graded: outcomes.count,
            chanceSum: outcomes.reduce(0) { $0 + min(max($1.chance, 0), 1) }
        )
    }
}

nonisolated struct CategoryScore: Codable, Sendable, Hashable, Identifiable {

    /// Matches `BenchmarkTask.category` ("math", "instructions", …).
    let category: String
    let passed: Int
    let total: Int
    let tasks: [TaskResult]

    var id: String { category }

    init(category: String, tasks: [TaskResult]) {
        self.category = category
        self.tasks = tasks
        self.passed = tasks.filter(\.passed).count
        self.total = tasks.count
    }

    /// Decode-only path: older files carry the counters, so trust them
    /// instead of recomputing from a possibly truncated task list.
    init(category: String, passed: Int, total: Int, tasks: [TaskResult]) {
        self.category = category
        self.passed = passed
        self.total = total
        self.tasks = tasks
    }

    var passRate: Double { total > 0 ? Double(passed) / Double(total) : 0 }

    /// Σ chance over this category's graded tasks. A task missing from a
    /// truncated older file counts as chance 0, which is what every task was
    /// before the field existed.
    var chanceSum: Double { tasks.reduce(0) { $0 + $1.guessingFloor } }

    /// 0…1, this category's score by the same formula as the overall one.
    /// What the report's category bars show.
    var normalizedScore: Double {
        BenchmarkScoring.normalized(passed: passed, graded: total, chanceSum: chanceSum)
    }
}

nonisolated struct TaskResult: Codable, Sendable, Hashable, Identifiable {

    /// `BenchmarkTask.id`, e.g. "math_word_02".
    let taskId: String
    let passed: Bool
    /// Wall clock of this single generation, including prompt processing.
    let latencySeconds: Double

    /// The model's raw answer, truncated to `answerLimit` characters. Reports
    /// are persisted and shared; a runaway model that emitted 8k tokens of
    /// repetition must not turn a report into a megabyte of noise.
    let answer: String

    /// Why the task did not produce a usable answer — "timed out", an engine
    /// error, … `nil` for a normal (passing or failing) generation.
    let error: String?

    /// `BenchmarkTask.chance` at run time, frozen into the report so the
    /// score can be recomputed from the file alone. `nil` (and absent from
    /// the JSON) for a free-answer task, i.e. for every task so far.
    let chance: Double?

    var id: String { taskId }

    /// `chance` with nil read as 0.
    var guessingFloor: Double { min(max(chance ?? 0, 0), 1) }

    /// Characters kept from the model's answer. Long enough to see why a
    /// check failed (the answer plus a bit of preamble), short enough that a
    /// 30-task report stays well under 100 KB.
    static let answerLimit = 500

    init(
        taskId: String,
        passed: Bool,
        latencySeconds: Double,
        answer: String,
        error: String? = nil,
        chance: Double = 0
    ) {
        self.taskId = taskId
        self.passed = passed
        self.latencySeconds = latencySeconds
        self.answer = Self.truncate(answer)
        self.error = error
        self.chance = chance > 0 ? min(chance, 1) : nil
    }

    /// Truncation marks itself, so a report reader never mistakes a clipped
    /// answer for the model stopping early.
    static func truncate(_ text: String, limit: Int = TaskResult.answerLimit) -> String {
        guard text.count > limit else { return text }
        return String(text.prefix(limit)) + "… [truncated]"
    }
}

// MARK: - Errors

/// Failures that abort a whole run. Per-task failures are recorded in
/// `TaskResult.error` instead — one model refusing one task is a result, not
/// an error.
nonisolated enum BenchmarkError: Error, LocalizedError, Sendable, Equatable {

    /// `InferenceManager.prepare(model:)` did not reach `.ready`.
    case loadFailed(String)
    /// The engine vanished mid-run — almost always the memory-warning
    /// unload in `InferenceManager`. The run is abandoned: everything
    /// measured after an unload would be measuring a different process state.
    case engineUnloadedMidRun
    /// The performance probe produced no tokens at all.
    case generationFailed(String)
    /// Neither the cached nor the bundled `benchmarks.json` could be read.
    case suiteUnavailable
    /// The run named an imported suite that is no longer on the device.
    case customSuiteMissing(String)
    /// The model cannot take text in and produce text out, so the text-only
    /// v1 suite does not apply to it (whisper, Kokoro TTS, …).
    case unsupportedModel(String)

    var errorDescription: String? {
        switch self {
        case .loadFailed(let detail):
            return "Could not load the model for benchmarking: \(detail)"
        case .engineUnloadedMidRun:
            return "The model was unloaded during the benchmark (most likely "
                + "memory pressure). Close other apps and run it again."
        case .generationFailed(let detail):
            return "The model produced no output: \(detail)"
        case .suiteUnavailable:
            return "The benchmark task suite could not be loaded."
        case .customSuiteMissing(let id):
            return "The imported suite this run asked for (\(id)) is no longer on this device."
        case .unsupportedModel(let name):
            return "\(name) is not a text model, so the benchmark suite does not apply to it."
        }
    }
}

// MARK: - App version

/// Version/build of the running app, as stamped into every report.
nonisolated enum BenchmarkAppInfo {

    static var version: String {
        Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "0.0"
    }

    static var build: String {
        Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "0"
    }
}

// MARK: - Thermal naming

nonisolated extension ProcessInfo.ThermalState {
    /// Lowercase, stable spelling used in reports. Distinct from
    /// `displayName` (title-cased, for the UI) on purpose: report strings are
    /// compared across builds and must never be localized or restyled.
    var reportName: String {
        switch self {
        case .nominal:  return "nominal"
        case .fair:     return "fair"
        case .serious:  return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    /// 0…3, hottest highest. `ProcessInfo.ThermalState` is an `Int` enum
    /// whose raw values already order this way, but the raw values are not
    /// documented as stable, so the mapping is spelled out. An unknown future
    /// case ranks as the hottest known one: a state Apple added above
    /// `.critical` is not a state to benchmark in.
    var severityRank: Int {
        switch self {
        case .nominal:  return 0
        case .fair:     return 1
        case .serious:  return 2
        case .critical: return 3
        @unknown default: return 3
        }
    }

    /// The line the benchmark refuses to measure across: at `.serious` iOS
    /// has already cut the CPU and GPU clocks, so throughput measured here
    /// describes the temperature, not the model.
    var isThermallyConstrained: Bool { severityRank >= 2 }
}
