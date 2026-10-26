//
//  BenchmarkEngine.swift
//  local_ai_test
//
//  The shared benchmark runner. Two frontends sit on top of it: the in-app
//  "test this model" flow, and the release harness that sweeps the whole
//  catalog from the device test target. Both call exactly one method:
//
//      let report = try await BenchmarkEngine.shared.run(
//          model: model,
//          options: .full
//      ) { progress in ... }
//
//  Structure of a v2 run:
//    0. Gate    — if the device is at .serious or hotter, wait up to 60 s for
//                 it to cool. Measuring a throttled phone measures the phone.
//    1. Load    — unload whatever is resident, then time `prepare(model:)`
//                 from cold. A warm re-load measures nothing useful. This is
//                 the ONLY cold measurement and it is never repeated.
//    2. Warmup  — one short unmeasured generation, so the first measured
//                 repetition is not paying for kernel compilation.
//    3. Perf    — the ~300-token prefill probe, twice; the decode probe,
//                 three times. Reported as average + sample standard
//                 deviation + every raw rate.
//                 `phys_footprint` is sampled from before the load until the
//                 run ends, so one high-water mark covers every phase.
//    4. pp512/tg128 — one extra probe on a frozen 512-token prompt with a
//                 128-token cap, so the report carries a number that people
//                 who read llama-bench output already understand.
//    5. Quality — every suite task, one at a time, fresh conversation state
//                 between them, each under its own timeout. Tasks run in a
//                 fixed, category-interleaved order
//                 (`BenchmarkTaskOrder`), and an optional wall-clock budget
//                 (`BenchmarkOptions.qualityBudget`) can stop the stage at a
//                 task boundary; the report is then marked partial.
//
//  Everything the model generates — warmup, probes and quality tasks alike —
//  runs GREEDILY (`GenerationParameters.deterministic`: temperature 0, topK 1,
//  fixed seed), so a re-run produces the same answers wherever the backend
//  honours the seed. Sampling noise is not something a benchmark should be
//  averaging over.
//
//  The thermal state is sampled at every probe and task boundary, not just at
//  the ends, and every reading is reported (`thermalStateBefore` / `After` /
//  `peakThermalState`). It is INFORMATION ONLY: a sustained run drives any
//  current iPhone to `.serious` within minutes whatever the model, so a
//  state-based verdict flagged every row of a 12-model sweep and told a
//  reader nothing.
//
//  `thermallyConstrained` (the JSON name is kept) is decided instead by the
//  MEASUREMENTS: `PerformanceMetrics.throttleVerdict(runs:protocolRate:
//  probeMean:)` flags a run whose decode rate fell more than 20% between its
//  best and worst repetition, or whose tg128 rate landed more than 20% below
//  the decode-probe mean. The verdict is frozen right after the last
//  throughput probe, so the quality stage that follows may heat the device
//  further without discrediting numbers taken before it.
//
//  It delegates all engine work to `InferenceManager`: no backend is touched
//  directly here, so a new backend is benchmarkable the day it is routable.
//
//  Robustness rules that shaped the code:
//    • Cancellation is checked between every phase and every task; a
//      cancelled run unloads the engine and throws `CancellationError`.
//    • The app's memory-warning handler can unload the model mid-run. That is
//      detected (the manager's `currentModelId` changes) and aborts the run
//      with `BenchmarkError.engineUnloadedMidRun` instead of silently
//      reporting a page of failed tasks.
//    • A task that times out or errors is a FAILED task, not a failed run.
//

import Foundation
import Darwin
import UIKit

// MARK: - Protocol constants

/// The v2 measurement protocol, in one place.
///
/// A `nonisolated enum` rather than statics on `BenchmarkEngine` because
/// `BenchmarkOptions` is a nonisolated value type and reads these as default
/// arguments — a default expression cannot touch a main-actor member.
///
/// Changing any of these changes what the published numbers mean. Treat them
/// the way the fixed prompts are treated: a change is a methodology change,
/// and `PerformanceMetrics.currentMetricsVersion` goes up with it.
nonisolated enum BenchmarkDefaults {

    /// Measured repetitions of the decode probe. Three is the smallest count
    /// that yields a standard deviation worth printing (llama-bench's default
    /// is five; on a phone that is another minute of heat for very little
    /// extra confidence, and heat is exactly what we are trying not to add).
    static let decodeRepetitions = 3

    /// Measured repetitions of the prefill probe. Two: enough to notice a
    /// wild outlier, not enough for a deviation, so none is published.
    static let prefillRepetitions = 2

    /// Repetitions of the standard-protocol (pp512 / tg128) decode half.
    static let standardProtocolRepetitions = 2

    /// Tokens the unmeasured warmup generation asks for. Long enough to walk
    /// the whole path once — prompt formatting, prefill, several decode steps,
    /// the first Metal kernel compilations, the first KV-cache allocation —
    /// short enough to cost a second or two.
    static let warmupTokens = 16

    /// Token cap for the tg128 half of the standard protocol. 128 by
    /// community convention; do not "improve" it.
    static let standardProtocolGenerationTokens = 128

    /// How long the run waits for a hot device to cool before giving up and
    /// measuring anyway (with `thermallyConstrained` set).
    static let thermalWaitTimeout: TimeInterval = 60

    /// Poll interval while waiting.
    static let thermalPollInterval: TimeInterval = 5

    /// The exact label the cool-down wait reports through the progress
    /// callback, at stage `.preparing`.
    static let thermalWaitLabel = "Waiting for the device to cool down"

    /// Quality-stage budget of an in-app Standard run: 20 minutes. Enough
    /// for the whole built-in suite on a 3B GGUF (about 6 minutes) or an 8B
    /// one at 10 tok/s (about 15); a 1.5B Core ML model, which needs about an
    /// hour, stops part-way with a partial score instead of holding the
    /// phone hostage. The Full choice, the build test and the release
    /// harness run without a budget.
    static let standardQualityBudget: TimeInterval = 20 * 60
}

// MARK: - Options

/// What to run and how patient to be.
nonisolated struct BenchmarkOptions: Sendable, Hashable {

    enum Stages: String, Sendable, Codable, CaseIterable {
        /// Load + throughput + memory only. Fast (~20 s), no score.
        case performanceOnly
        /// Everything. Produces an overall score.
        case full
    }

    var stages: Stages

    /// Per-task ceiling for the quality stage. A 1B model on an old phone
    /// needs ~10 s for a 96-token answer; 60 s is generous without letting a
    /// looping model stall the whole suite.
    var taskTimeout: TimeInterval

    /// Wall-clock budget for the whole quality stage, in seconds. `nil` is
    /// unlimited: every selected task runs.
    ///
    /// The budget is checked BEFORE each task, so the stage stops cleanly at
    /// a task boundary: the task in flight when it runs out is finished and
    /// graded (still under `taskTimeout`, which bounds the overrun). The
    /// report then records `completed` < `total`, `truncated = true`, and a
    /// score over the completed tasks, which is shown as partial and never
    /// compared (`BenchmarkReport.isScoreComparable(with:)`). A zero,
    /// negative or non-finite value is read as unlimited.
    var qualityBudget: TimeInterval?

    /// Restricts the quality stage to these categories. `nil` runs all of
    /// them. The harness uses it to split a long sweep across runs.
    var categories: Set<String>?

    /// Which suite grades this run. `nil` is the bundled/remote
    /// `benchmarks.json`; anything else is the id of an imported suite (see
    /// `CustomBenchmarkSuiteFile`). Stamped into the report so a score from
    /// somebody else's suite is never compared against ours.
    var suiteID: String?

    /// Caps the number of quality tasks. `nil` runs the whole suite; the
    /// harness's smoke mode sets it to a handful.
    var maxTasks: Int?

    /// Token cap for the throughput probe. 200 is long enough that startup
    /// jitter stops mattering and short enough to keep a run under a minute.
    var generationMaxTokens: Int

    /// Unload the model when the run finishes successfully. The in-app flow
    /// leaves it loaded (the user just picked it); the harness unloads to
    /// keep peak memory honest for the next model in the sweep.
    var unloadWhenFinished: Bool

    // MARK: v2 knobs

    /// Measured repetitions of the decode probe. The report carries the
    /// average, the sample standard deviation and every raw rate. 1 disables
    /// the deviation (there is nothing to deviate from) but is otherwise
    /// valid, and is what the smoke profile uses.
    var decodeRepetitions: Int

    /// Measured repetitions of the prefill probe. No deviation is published
    /// for prefill, so this exists mostly so a fast profile can drop it to 1.
    var prefillRepetitions: Int

    /// Run one unmeasured short generation between the load and the first
    /// measured probe. On by default and there is no good reason to turn it
    /// off outside tests: without it the first probe pays for kernel
    /// compilation and the first KV-cache allocation, which is a property of
    /// the harness rather than of the model.
    var performWarmup: Bool

    /// Run the pp512 / tg128 standard-protocol probe. ON for BOTH `.full`
    /// and `.performanceOnly`: it is a performance measurement, and a
    /// performance-only run is exactly when someone wants a number they can
    /// compare against llama-bench output. Costs one ~512-token prefill plus
    /// two 128-token generations.
    var runStandardProtocol: Bool

    /// Wait for a hot device to cool before measuring (see
    /// `BenchmarkDefaults.thermalWaitTimeout`). Turning it off does not turn
    /// off the `thermallyConstrained` flag — that is measured from the decode
    /// rates either way, so a run that starts hot and ramps down is still
    /// caught; the run is just not delayed.
    var thermalGating: Bool

    /// Issues the caller found before the run started - a cool-down or power
    /// wait that ran out (`RunConditionsGate`). They go into
    /// `PerformanceMetrics.measurementIssues` ahead of whatever the run sees
    /// itself, so the run is flagged like one measured in Low Power Mode.
    var preRunIssues: [String] = []

    init(
        stages: Stages = .full,
        taskTimeout: TimeInterval = 60,
        qualityBudget: TimeInterval? = nil,
        categories: Set<String>? = nil,
        suiteID: String? = nil,
        maxTasks: Int? = nil,
        generationMaxTokens: Int = 200,
        unloadWhenFinished: Bool = false,
        decodeRepetitions: Int = BenchmarkDefaults.decodeRepetitions,
        prefillRepetitions: Int = BenchmarkDefaults.prefillRepetitions,
        performWarmup: Bool = true,
        runStandardProtocol: Bool = true,
        thermalGating: Bool = true
    ) {
        self.stages = stages
        self.taskTimeout = taskTimeout
        self.qualityBudget = Self.sanitizedBudget(qualityBudget)
        self.categories = categories
        self.suiteID = suiteID
        self.maxTasks = maxTasks
        self.generationMaxTokens = generationMaxTokens
        self.unloadWhenFinished = unloadWhenFinished
        // A repetition count below 1 would produce a report with no
        // measurement at all, so it is clamped rather than trusted.
        self.decodeRepetitions = max(1, decodeRepetitions)
        self.prefillRepetitions = max(1, prefillRepetitions)
        self.performWarmup = performWarmup
        self.runStandardProtocol = runStandardProtocol
        self.thermalGating = thermalGating
    }

    /// THE truncation rule, as a pure function so it can be tested without a
    /// model: true when the quality stage must stop instead of starting the
    /// task at `index`, `elapsed` seconds after the stage began. Never true
    /// for the first task, so a budgeted run always grades something.
    func qualityBudgetExhausted(beforeTaskAt index: Int, elapsed: TimeInterval) -> Bool {
        guard let qualityBudget, index > 0 else { return false }
        return elapsed >= qualityBudget
    }

    /// `nil` for anything that is not a usable positive duration.
    static func sanitizedBudget(_ budget: TimeInterval?) -> TimeInterval? {
        guard let budget, budget.isFinite, budget > 0 else { return nil }
        return budget
    }

    /// Load + performance + the whole quality suite, however long it takes.
    /// The in-app "Full" choice, and what the build test and the release
    /// harness use by default.
    static let full = BenchmarkOptions()

    /// The in-app default: like `.full`, but the quality stage stops after
    /// `BenchmarkDefaults.standardQualityBudget` and a cut-short run is
    /// reported as partial.
    static let standard = BenchmarkOptions(qualityBudget: BenchmarkDefaults.standardQualityBudget)

    /// Load + performance, no score. Still repeats the decode probe and still
    /// runs the standard protocol — that is the point of a speed-only run.
    static let performanceOnly = BenchmarkOptions(stages: .performanceOnly)

    /// Everything, but only the first few tasks — for CI smoke runs.
    /// Single-shot probes and no standard protocol: a smoke run checks that
    /// the pipeline works, and its numbers are explicitly not publishable.
    static let smoke = BenchmarkOptions(
        taskTimeout: 30,
        maxTasks: 5,
        decodeRepetitions: 1,
        prefillRepetitions: 1,
        runStandardProtocol: false,
        thermalGating: false
    )
}

// MARK: - Progress

/// Pushed to the caller's closure on the main actor, several times per stage.
nonisolated struct BenchmarkProgress: Sendable, Hashable {

    enum Stage: String, Sendable, Codable, CaseIterable {
        case preparing
        case loading
        case performance
        case quality
        case finishing

        var displayName: String {
            switch self {
            case .preparing:   return "Preparing"
            case .loading:     return "Loading model"
            case .performance: return "Measuring speed"
            case .quality:     return "Running tasks"
            case .finishing:   return "Finishing"
            }
        }
    }

    let stage: Stage
    /// 0…1 across the WHOLE run, not within the stage — a progress bar can
    /// bind to it directly.
    let fraction: Double
    /// Human-readable current step, e.g. "Math word - task 7 of 28".
    let label: String
    /// Quality-stage counters; both 0 until the quality stage starts.
    let completedTasks: Int
    let totalTasks: Int
    /// True only while the run is parked in the thermal cool-down wait, which
    /// is overhead rather than progress. A frontend that wants to present the
    /// wait differently should read this rather than compare `label` against
    /// `BenchmarkDefaults.thermalWaitLabel`.
    let isThermalWait: Bool

    init(
        stage: Stage,
        fraction: Double,
        label: String,
        completedTasks: Int = 0,
        totalTasks: Int = 0,
        isThermalWait: Bool = false
    ) {
        self.stage = stage
        self.fraction = min(max(fraction, 0), 1)
        self.label = label
        self.completedTasks = completedTasks
        self.totalTasks = totalTasks
        self.isThermalWait = isThermalWait
    }
}

// MARK: - Engine

@MainActor
final class BenchmarkEngine {

    static let shared = BenchmarkEngine()

    private let inference: InferenceManager
    private let storage: BenchmarkSuiteStorage

    /// Injection point for tests. No default arguments on purpose: a default
    /// expression is evaluated in a nonisolated context, and both singletons
    /// are main-actor bound.
    init(inference: InferenceManager, storage: BenchmarkSuiteStorage) {
        self.inference = inference
        self.storage = storage
    }

    init() {
        self.inference = .shared
        self.storage = .shared
    }

    // MARK: Tuning constants

    /// Fewest tokens the decode probe must produce for its tok/s to count as a
    /// measurement. The rate needs at least two tokens to exist at all, and a
    /// two-token window is pure jitter; below this floor the run fails loudly
    /// instead of publishing a number nobody should compare.
    static let minimumDecodeProbeTokens = 8

    // MARK: Fixed probe prompts

    /// ~220 words (~300 BPE tokens) of ordinary prose. Prefill throughput is
    /// measured on it: long enough that prompt processing dominates the time
    /// to first token, dull enough that no model can shortcut it.
    /// Never edit it without bumping the app version in the report — prefill
    /// numbers are only comparable across identical prompts.
    static let prefillProbePrompt = """
    Read the following passage and then reply with the single word "ok".

    The harbour town woke slowly on weekday mornings. Fishing boats returned \
    with the tide a little after five, and by six the market under the stone \
    arches was loud with crates being stacked, ice being shovelled, and the \
    steady argument of gulls overhead. The bakery on the corner opened at \
    half past six, and its owner, who had run the place for thirty-one years, \
    still ground her own flour twice a week in a mill her grandfather had \
    installed. Tourists rarely arrived before nine, which suited everyone: \
    the first three hours belonged to the people who actually lived there. \
    In winter the same square was almost empty, the boats fewer, the market \
    reduced to four stalls under a canvas roof that leaked in exactly one \
    place. The town council had voted eleven times over two decades to repair \
    that roof, and eleven times the money had gone somewhere more urgent — a \
    seawall, a school boiler, a stretch of road that flooded every autumn. \
    The leak had become a local joke, then a local landmark, and finally a \
    small point of pride, the sort of imperfection a place keeps on purpose \
    because fixing it would make it like everywhere else.

    Reply with the single word "ok".
    """

    /// Fixed decode probe. Open-ended enough that every model fills the token
    /// budget, so tok/s is measured over a full 200-token window rather than
    /// over a model that stopped after eight words.
    static let generationProbePrompt = """
    Write a detailed, continuous description of a train journey through a \
    mountain valley in autumn. Describe the landscape, the weather, the \
    sounds inside the carriage, and the passengers. Do not use bullet points \
    or headings; write flowing prose and keep going until you are stopped.
    """

    /// Short prompt for the unmeasured warmup generation. Trivial on purpose:
    /// the point is to walk the engine's whole path once (prompt formatting,
    /// prefill, several decode steps, the first Metal kernel compilations,
    /// the first KV-cache allocation) at the lowest possible cost. Nothing it
    /// produces is recorded.
    static let warmupPrompt = "Count from one to eight, separated by commas."

    // MARK: Standard protocol (pp512 / tg128)

    /// Identifier of the frozen standard-protocol prompt, copied into
    /// `StandardProtocolMetrics.promptId`. Bump it (to "pp512-v2", …) in the
    /// same commit that edits `standardProtocolPrompt`, never separately.
    static let standardProtocolPromptId = "pp512-v1"

    /// FROZEN 2048-character filler passage — 512 tokens under the app's
    /// 4-characters-per-token estimate — used for the pp512 half of the
    /// standard protocol.
    ///
    /// DO NOT EDIT THIS STRING. Not a word, not a space. Prompt-processing
    /// throughput is a function of prompt length, so changing it silently
    /// makes every previously published pp512 number incomparable with every
    /// future one, across all devices and all models, with nothing in the
    /// report to show that anything moved. If it genuinely must change, add a
    /// new constant, bump `standardProtocolPromptId`, and expect the numbers
    /// to shift.
    ///
    /// It is deliberately dull, self-contained, English, and free of
    /// instructions a model might follow at length: the only thing being
    /// measured here is how fast the engine can chew through 512 tokens.
    static let standardProtocolPrompt = """
    Read the following passage and then reply with the single word "ok".

    The regional depot receives goods on every weekday between six in the \
    morning and two in the afternoon. Each pallet arriving at the loading \
    bay carries a paper docket and a printed label. The docket lists the \
    supplier, the order number, the number of cartons, and the date the \
    pallet left the supplier's warehouse. The label repeats the order number \
    in a machine-readable form so that a handheld scanner can read it \
    without a keyboard entry. A clerk compares the two, and when they \
    disagree the pallet is moved to the discrepancy lane at the north end of \
    the bay, where it waits until somebody telephones the supplier.

    Goods that pass the check are moved by forklift to one of four holding \
    zones. Zone one holds items that must stay below eight degrees. Zone two \
    holds items that must stay dry. Zone three holds heavy items that cannot \
    be stacked more than two pallets high. Zone four holds everything else. \
    A supervisor walks the zones twice a day, once shortly after the morning \
    intake and once before the depot closes, and records the temperature of \
    zone one in a bound notebook that has been kept in the same drawer since \
    the depot opened.

    Picking begins at three in the afternoon. A picker receives a list on a \
    handheld terminal, walks the aisles in the order the list gives, and \
    scans each carton as it is placed on the trolley. The terminal refuses \
    to advance to the next line until the scan matches the line it expects, \
    which prevents a picker from taking the right quantity of the wrong \
    item. Completed trolleys are wrapped, weighed, and staged against the \
    outbound doors in delivery-route order, so that the first van to leave \
    in the morning is loaded last and stands nearest the door.

    Nothing about this arrangement is clever. It was assembled over eleven \
    years out of small corrections, each one made after something went \
    wrong, and the people who work there can explain the reason behind every \
    rule. That is why the audits pass.

    Reply with the single word "ok".
    """

    // MARK: - Entry point

    /// Runs a benchmark and returns its report.
    ///
    /// - Parameters:
    ///   - model: catalog or imported descriptor. Its files must already be
    ///     downloaded — the engine does not download anything.
    ///   - options: which stages to run, timeouts, filters.
    ///   - progress: called on the main actor as the run advances.
    /// - Throws: `BenchmarkError` for run-level failures, `CancellationError`
    ///   when the calling task is cancelled. Per-task failures never throw.
    @discardableResult
    func run(
        model: AIModel,
        options: BenchmarkOptions = .full,
        progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> BenchmarkReport {

        // v1 is text-only: a Whisper or Kokoro entry would score 0 on every
        // task for reasons that say nothing about the model.
        guard model.capabilities.contains(.textIn),
              model.capabilities.contains(.textOut) else {
            throw BenchmarkError.unsupportedModel(model.name)
        }

        progress(BenchmarkProgress(stage: .preparing, fraction: 0, label: model.name))

        // Everything measured from here on is a number somebody may publish,
        // so the app's background refreshes stand down until the run is over
        // (see `BenchmarkRunGate`). The `defer` covers failure and
        // cancellation as well as success.
        BenchmarkRunGate.shared.begin()
        defer { BenchmarkRunGate.shared.end() }

        // The suite is resolved BEFORE the model is loaded: discovering that
        // benchmarks.json is missing after a 40-second load would be rude.
        // An imported suite is read from its own file rather than from the
        // shared snapshot, which only ever holds the built-in one.
        let suite: BenchmarkSuiteDocument
        if let suiteID = options.suiteID, suiteID != BenchmarkSuiteDocument.builtInID {
            guard let stored = CustomBenchmarkSuiteFile.load(id: suiteID) else {
                throw BenchmarkError.customSuiteMissing(suiteID)
            }
            suite = stored.document
        } else {
            suite = storage.current
        }
        var tasks: [BenchmarkTask] = []
        if options.stages == .full {
            guard !suite.isEmpty else { throw BenchmarkError.suiteUnavailable }
            tasks = suite.qualityRunOrder(categories: options.categories, limit: options.maxTasks)
            guard !tasks.isEmpty else { throw BenchmarkError.suiteUnavailable }
        }

        // Everything from here on lives inside one do/catch, the cool-down
        // wait included: Cancel pressed while the run is parked waiting for
        // the device to cool must leave the engine unloaded exactly like any
        // other abort, which is the contract `run` documents.
        do {
            // Thermal gate, before anything expensive happens. A phone that is
            // already at .serious has its clocks cut, so measuring now would
            // produce numbers about the temperature rather than about the
            // model.
            if options.thermalGating {
                try await waitForCoolDevice(progress: progress)
            }

            // Created AFTER the wait. `ThermalTracker` seeds its peak from the
            // state it is constructed in, so a tracker built before the gate
            // would report a run that started hot, cooled down and then
            // measured clean as having peaked at `.serious` - contradicting
            // `thermalBefore` right next to it in the same report.
            let thermal = ThermalTracker()

            // Sampled AFTER the cool-down wait: this is the state the
            // measurement actually starts in, which is what a reader of the
            // report needs.
            let thermalBefore = ProcessInfo.processInfo.thermalState.reportName

            // Started AFTER the wait too. The gate can idle for a full minute,
            // and billing that to `wallClockSeconds` would make a gated run
            // look slower than the same work on a cool phone.
            let runStart = Date()

            // One sampler for the WHOLE run rather than one per probe. The cold
            // load and the quality generations (KV cache growth) are the
            // likeliest peaks, and sampling them after the fact misses a spike
            // that has already been released. Started before the load and torn
            // down in a defer, so every exit path is covered: success, failure
            // and cancellation alike.
            let sampler = MemorySampler()
            sampler.start()
            defer { sampler.stop() }

            try Task.checkCancellation()

            // MARK: 1. Load

            progress(BenchmarkProgress(stage: .loading, fraction: 0.02, label: "Loading \(model.name)"))
            let loadSeconds = try await loadFromCold(model: model)
            try Task.checkCancellation()
            thermal.sample()

            // MARK: 2. Warmup (unmeasured)

            // Everything the first generation pays for once — Metal kernel
            // compilation, the first KV-cache allocation, tokenizer and
            // prompt-template setup — belongs to the harness, not to the
            // model. Doing it here means repetition 1 of the decode probe is
            // a real measurement rather than a systematic outlier. The cold
            // number that DOES matter, `loadSeconds`, was already taken above
            // and is untouched by this.
            var warmupPerformed = false
            if options.performWarmup {
                progress(BenchmarkProgress(stage: .performance, fraction: 0.10, label: "Warming up"))
                warmupPerformed = await runWarmup(options: options, modelId: model.id)
                try Task.checkCancellation()
                thermal.sample()
            }

            // MARK: 3. Performance

            progress(BenchmarkProgress(stage: .performance, fraction: 0.12, label: "Prompt processing"))

            // Prefill: engines do not report a prompt token count, so the
            // prompt is estimated from its text and divided by the time to
            // first token. TTFT on a ~300-token prompt is prompt processing
            // plus one decode step — close enough for a comparison between
            // models on one device, which is what this number is for. The
            // ESTIMATED caveat survives v2 in full; repeating the probe makes
            // the number steadier, not more exact.
            let prefillRuns = try await measurePrefill(
                prompt: Self.prefillProbePrompt,
                repetitions: options.prefillRepetitions,
                options: options,
                modelId: model.id,
                thermal: thermal,
                label: "prompt processing",
                progress: progress,
                fractionFrom: 0.12,
                fractionTo: 0.17
            )
            let prefillTokensPerSecond = BenchmarkStatistics.mean(prefillRuns)

            progress(BenchmarkProgress(stage: .performance, fraction: 0.17, label: "Token generation"))
            let decode = try await measureDecode(
                prompt: Self.generationProbePrompt,
                maxTokens: options.generationMaxTokens,
                repetitions: options.decodeRepetitions,
                options: options,
                modelId: model.id,
                thermal: thermal,
                label: "token generation",
                progress: progress,
                fractionFrom: 0.17,
                fractionTo: 0.26
            )

            // MARK: 4. Standard protocol (pp512 / tg128)

            var standardProtocol: StandardProtocolMetrics?
            if options.runStandardProtocol {
                standardProtocol = try await runStandardProtocol(
                    options: options,
                    modelId: model.id,
                    thermal: thermal,
                    progress: progress,
                    fractionFrom: 0.26,
                    fractionTo: 0.30
                )
            }

            // The standard-protocol helper swallows its own failures (a
            // missing courtesy number must not throw away a good run), which
            // would also swallow a cancellation. Re-check here so Cancel
            // pressed during pp512 stops the run instead of falling through
            // into the quality stage.
            try Task.checkCancellation()

            // Every throughput number is now measured, so the throttle
            // verdict can be settled from the rates themselves: did the
            // device hold its speed across the repetitions, and did tg128
            // (which ran last, on the hottest phone of the run) land near the
            // decode-probe mean? Quality scores are graded by deterministic
            // checks and do not depend on clock speed, so the stage that
            // follows needs no such caveat and cannot change this verdict.
            thermal.sample()
            let throttle = PerformanceMetrics.throttleVerdict(
                runs: decode.rates,
                protocolRate: standardProtocol?.generationTokensPerSecond,
                probeMean: decode.rates.isEmpty
                    ? nil
                    : BenchmarkStatistics.mean(decode.rates)
            )
            // Taken here, with the throughput figures, not at the end: the
            // quality stage is graded by deterministic checks, so a low power
            // or background stint during it does not touch these numbers.
            var measurementIssues = options.preRunIssues
            for issue in thermal.issues where !measurementIssues.contains(issue) {
                measurementIssues.append(issue)
            }

            // MARK: 5. Quality

            var quality: QualityResults?
            if options.stages == .full {
                let outcome = try await runQuality(
                    tasks: tasks,
                    suite: suite,
                    options: options,
                    modelId: model.id,
                    thermal: thermal,
                    progress: progress
                )
                quality = QualityResults(
                    suiteVersion: suite.suiteVersion,
                    categories: outcome.categories,
                    suiteID: suite.identity,
                    suiteName: suite.name,
                    total: tasks.count,
                    truncated: outcome.truncated
                )
            }
            let completedTasks = quality?.completed ?? tasks.count

            progress(BenchmarkProgress(
                stage: .finishing,
                fraction: 0.99,
                label: "Writing report",
                completedTasks: completedTasks,
                totalTasks: tasks.count
            ))

            // Assembled at the very end: the sampler's high-water mark and the
            // thermal state are only final once the quality stage — the
            // longest and usually the hottest part of the run — is over.
            thermal.sample()
            let performance = PerformanceMetrics(
                loadSeconds: loadSeconds,
                prefillTokensPerSecond: prefillTokensPerSecond,
                generationTokensPerSecond: BenchmarkStatistics.mean(decode.rates),
                peakMemoryMB: sampler.peakMB,
                memoryBudgetMB: InferenceManager.memoryBudgetMB(for: model.backend),
                thermalStateBefore: thermalBefore,
                thermalStateAfter: ProcessInfo.processInfo.thermalState.reportName,
                tokensGenerated: decode.tokenCounts.reduce(0, +),
                wallClockSeconds: Date().timeIntervalSince(runStart),
                generationTokensPerSecondStdDev:
                    BenchmarkStatistics.sampleStandardDeviation(decode.rates),
                generationRunsTokensPerSecond: decode.rates,
                generationRunsTokenCounts: decode.tokenCounts,
                prefillRunsTokensPerSecond: prefillRuns,
                warmupPerformed: warmupPerformed,
                thermallyConstrained: throttle.constrained || !measurementIssues.isEmpty,
                throttleDropPercent: throttle.dropPercent,
                peakThermalState: thermal.peak.reportName,
                measurementIssues: measurementIssues.isEmpty ? nil : measurementIssues,
                standardProtocol: standardProtocol
            )

            if options.unloadWhenFinished {
                await inference.unloadCurrent()
            }

            let report = BenchmarkReport(
                appVersion: BenchmarkAppInfo.version,
                appBuild: BenchmarkAppInfo.build,
                modelId: model.id,
                modelName: model.name,
                backend: model.backend.displayName,
                quantization: model.quantization,
                parameterSize: model.parameterSize,
                device: DeviceSnapshot.current(),
                performance: performance,
                quality: quality
            )

            progress(BenchmarkProgress(
                stage: .finishing,
                fraction: 1,
                label: "Done",
                completedTasks: completedTasks,
                totalTasks: tasks.count
            ))
            return report

        } catch {
            // Any abort leaves the engine unloaded rather than half-warm:
            // a caller that retries must start from the same cold state the
            // load measurement assumes.
            await inference.unloadCurrent()
            throw error
        }
    }

    // MARK: - Thermal gate

    /// Waits for the device to leave `.serious`/`.critical` before measuring.
    ///
    /// Polls every `BenchmarkDefaults.thermalPollInterval` seconds for at most
    /// `BenchmarkDefaults.thermalWaitTimeout` seconds. If the device is still
    /// hot when the wait runs out the run PROCEEDS — refusing to benchmark
    /// would be worse than benchmarking with a warning — and it carries its
    /// true starting temperature into the report: the `ThermalTracker` the
    /// caller builds immediately after this returns seeds its peak from the
    /// still-hot state. Whether the numbers are usable is then decided by the
    /// rates the run actually produces, not by that reading.
    ///
    /// Never throws except on cancellation: pressing Cancel during a
    /// cool-down wait must stop the run immediately.
    private func waitForCoolDevice(
        progress: @escaping (BenchmarkProgress) -> Void
    ) async throws {
        guard ProcessInfo.processInfo.thermalState.isThermallyConstrained else { return }

        let deadline = Date().addingTimeInterval(BenchmarkDefaults.thermalWaitTimeout)
        while Date() < deadline {
            try Task.checkCancellation()

            let remaining = max(0, deadline.timeIntervalSinceNow)
            let elapsed = BenchmarkDefaults.thermalWaitTimeout - remaining
            progress(BenchmarkProgress(
                stage: .preparing,
                // Kept inside the first 2% of the bar: the wait is overhead,
                // not progress through the benchmark.
                fraction: 0.02 * (elapsed / BenchmarkDefaults.thermalWaitTimeout),
                label: BenchmarkDefaults.thermalWaitLabel,
                isThermalWait: true
            ))

            try await Task.sleep(
                nanoseconds: UInt64(
                    min(BenchmarkDefaults.thermalPollInterval, max(remaining, 0.1))
                    * 1_000_000_000
                )
            )
            if !ProcessInfo.processInfo.thermalState.isThermallyConstrained { return }
        }
    }

    // MARK: - Phase 1: load

    /// Unloads whatever is resident, then times a cold `prepare`.
    private func loadFromCold(model: AIModel) async throws -> TimeInterval {
        await inference.unloadCurrent()
        // Give iOS a moment to reclaim the freed pages — the same 200 ms
        // `InferenceManager.prepare` uses between engines. Without it, the
        // first load's peak memory includes the previous model's corpse.
        try? await Task.sleep(nanoseconds: 200_000_000)
        try Task.checkCancellation()

        let start = Date()
        await inference.prepare(model: model)
        let elapsed = Date().timeIntervalSince(start)

        // `prepare` is not cancellable, so a Cancel pressed during a 40-second
        // cold load is inert until it returns. This is the earliest point at
        // which cancellation can take effect; making the load itself
        // interruptible is an InferenceManager change, not a benchmark one.
        try Task.checkCancellation()

        guard case .ready = inference.state, inference.currentModelId == model.id else {
            let detail = inference.lastError?.detail
                ?? "the engine did not reach a ready state"
            throw BenchmarkError.loadFailed(detail)
        }
        return elapsed
    }

    // MARK: - Phase 2: warmup

    /// One short, unmeasured generation. Returns true when it actually
    /// produced tokens.
    ///
    /// Deliberately forgiving: a warmup that times out or errors is not a
    /// failed run. Its whole job is to move one-off costs out of the first
    /// measured repetition, and if it could not, the measured probes will say
    /// so loudly enough on their own. The conversation is reset afterwards so
    /// the warmup's KV cache never reaches a probe.
    private func runWarmup(options: BenchmarkOptions, modelId: String) async -> Bool {
        let sample = try? await generate(
            turns: [.user(Self.warmupPrompt)],
            maxTokens: BenchmarkDefaults.warmupTokens,
            timeout: max(options.taskTimeout, 60),
            modelId: modelId
        )
        await inference.resetConversation()
        return (sample?.tokenCount ?? 0) > 0
    }

    // MARK: - Phase 3: repeated probes

    /// Runs the prefill probe `repetitions` times and returns one rate per
    /// successful repetition.
    ///
    /// Throws only when NOT ONE repetition produced a first token: a single
    /// flaky repetition costs a data point, not the run.
    private func measurePrefill(
        prompt: String,
        repetitions: Int,
        options: BenchmarkOptions,
        modelId: String,
        thermal: ThermalTracker,
        label: String,
        progress: @escaping (BenchmarkProgress) -> Void,
        fractionFrom: Double,
        fractionTo: Double
    ) async throws -> [Double] {

        let promptTokens = Self.estimatedTokenCount(prompt)
        var rates: [Double] = []
        var lastFailure: String?

        for index in 0..<repetitions {
            try Task.checkCancellation()
            progress(BenchmarkProgress(
                stage: .performance,
                fraction: Self.lerp(fractionFrom, fractionTo, Double(index) / Double(repetitions)),
                label: repetitions > 1
                    ? "Prompt processing - run \(index + 1) of \(repetitions)"
                    : "Prompt processing"
            ))

            // Fresh state per repetition, or repetition 2 would prefill
            // against a cache that already holds repetition 1's prompt and
            // report an absurd rate.
            await inference.resetConversation()

            let sample = try await probe(
                prompt: prompt,
                maxTokens: 4,
                timeout: max(options.taskTimeout, 90),
                modelId: modelId,
                label: label
            )
            thermal.sample()

            // No first token means nothing was measured. Falling back to the
            // wall clock here used to publish the probe's timeout duration as
            // if it were a real prompt-processing rate.
            guard let seconds = sample.firstTokenSeconds, seconds > 0 else {
                lastFailure = sample.errorDescription
                    ?? "the prompt processing probe never produced a token"
                continue
            }
            rates.append(Double(promptTokens) / seconds)
        }

        guard !rates.isEmpty else {
            throw BenchmarkError.generationFailed(
                lastFailure ?? "the prompt processing probe never produced a token"
            )
        }
        return rates
    }

    /// Runs the decode probe `repetitions` times.
    ///
    /// The per-repetition token floor is the v1 rule (`minimumDecodeProbeTokens`)
    /// applied per run: a repetition below it contributes nothing, and the
    /// whole measurement fails only when EVERY repetition fell short. That is
    /// the one place v2 is more forgiving than v1, and it has to be: three
    /// chances to trip a hard failure would make the benchmark three times
    /// flakier for no gain in honesty, since every surviving rate is recorded
    /// raw anyway.
    private func measureDecode(
        prompt: String,
        maxTokens: Int,
        repetitions: Int,
        options: BenchmarkOptions,
        modelId: String,
        thermal: ThermalTracker,
        label: String,
        progress: @escaping (BenchmarkProgress) -> Void,
        fractionFrom: Double,
        fractionTo: Double
    ) async throws -> DecodeMeasurement {

        var rates: [Double] = []
        var tokenCounts: [Int] = []
        var lastFailure: String?

        for index in 0..<repetitions {
            try Task.checkCancellation()
            progress(BenchmarkProgress(
                stage: .performance,
                fraction: Self.lerp(fractionFrom, fractionTo, Double(index) / Double(repetitions)),
                label: repetitions > 1
                    ? "Token generation - run \(index + 1) of \(repetitions)"
                    : "Token generation"
            ))

            await inference.resetConversation()

            let sample = try await probe(
                prompt: prompt,
                maxTokens: maxTokens,
                timeout: max(options.taskTimeout, 180),
                modelId: modelId,
                label: label
            )

            // A cancellation during the decode probe unwinds the stream and
            // hands back an empty sample. Checking here keeps that case a
            // CancellationError instead of the token-count guard below turning
            // a deliberate Cancel into a "benchmark failed" alert.
            try Task.checkCancellation()
            thermal.sample()

            // A probe that emitted one or two tokens cannot produce a decode
            // rate (the rate needs at least two tokens and a stable window),
            // so a handful of tokens is the floor for calling the measurement
            // real. Quality tasks are held to no such minimum: a one-word
            // answer there is a legitimate result.
            guard sample.tokenCount >= Self.minimumDecodeProbeTokens,
                  sample.decodeTokensPerSecond > 0 else {
                lastFailure = sample.errorDescription
                    ?? "the generation probe produced \(sample.tokenCount) token(s), "
                    + "too few to measure a generation speed"
                continue
            }
            rates.append(sample.decodeTokensPerSecond)
            tokenCounts.append(sample.tokenCount)
        }

        guard !rates.isEmpty else {
            throw BenchmarkError.generationFailed(
                lastFailure ?? "the generation probe produced no measurable tokens"
            )
        }
        return DecodeMeasurement(rates: rates, tokenCounts: tokenCounts)
    }

    // MARK: - Phase 4: standard protocol

    /// The pp512 / tg128 block: one prefill measurement on the frozen
    /// 512-token prompt, then `BenchmarkDefaults.standardProtocolRepetitions`
    /// decode runs capped at 128 tokens.
    ///
    /// Same methodology as the main probes — TTFT for prefill, first-token to
    /// last-token for decode, greedy sampling — so the two blocks in a report
    /// are internally consistent. What it adds is the community's prompt
    /// length and token cap, which is what makes the numbers legible to
    /// somebody who reads llama-bench output.
    ///
    /// The tg128 half deliberately uses the ordinary decode prompt rather
    /// than the 512-token one, exactly as llama-bench measures pp and tg as
    /// two separate runs: generating 128 tokens on top of a 512-token context
    /// would fold prefill cost back into the decode figure.
    ///
    /// Never fatal: if the standard protocol cannot be measured the report
    /// simply has no `standardProtocol` block. Losing an extra courtesy
    /// number is not a reason to throw away a completed load, prefill, decode
    /// and quality run.
    private func runStandardProtocol(
        options: BenchmarkOptions,
        modelId: String,
        thermal: ThermalTracker,
        progress: @escaping (BenchmarkProgress) -> Void,
        fractionFrom: Double,
        fractionTo: Double
    ) async throws -> StandardProtocolMetrics? {

        try Task.checkCancellation()
        progress(BenchmarkProgress(
            stage: .performance,
            fraction: fractionFrom,
            label: "Standard protocol - pp512"
        ))

        let promptTokens = Self.estimatedTokenCount(Self.standardProtocolPrompt)

        let prefillRates = try? await measurePrefill(
            prompt: Self.standardProtocolPrompt,
            repetitions: 1,
            options: options,
            modelId: modelId,
            thermal: thermal,
            label: "standard protocol prompt processing",
            progress: { _ in },
            fractionFrom: fractionFrom,
            fractionTo: fractionFrom
        )
        guard let prefillRates, !prefillRates.isEmpty else { return nil }

        try Task.checkCancellation()
        progress(BenchmarkProgress(
            stage: .performance,
            fraction: Self.lerp(fractionFrom, fractionTo, 0.4),
            label: "Standard protocol - tg128"
        ))

        let decode = try? await measureDecode(
            prompt: Self.generationProbePrompt,
            maxTokens: BenchmarkDefaults.standardProtocolGenerationTokens,
            repetitions: BenchmarkDefaults.standardProtocolRepetitions,
            options: options,
            modelId: modelId,
            thermal: thermal,
            label: "standard protocol token generation",
            progress: { _ in },
            fractionFrom: fractionFrom,
            fractionTo: fractionTo
        )
        guard let decode, !decode.rates.isEmpty else { return nil }

        return StandardProtocolMetrics(
            promptId: Self.standardProtocolPromptId,
            promptTokensEstimated: promptTokens,
            generationMaxTokens: BenchmarkDefaults.standardProtocolGenerationTokens,
            prefillTokensPerSecond: BenchmarkStatistics.mean(prefillRates),
            generationTokensPerSecond: BenchmarkStatistics.mean(decode.rates),
            generationTokensPerSecondStdDev:
                BenchmarkStatistics.sampleStandardDeviation(decode.rates),
            generationRunsTokensPerSecond: decode.rates,
            repetitions: decode.rates.count
        )
    }

    /// Linear interpolation for progress fractions.
    private static func lerp(_ from: Double, _ to: Double, _ t: Double) -> Double {
        from + (to - from) * min(max(t, 0), 1)
    }

    // MARK: - Phase 5: quality

    /// What the quality stage hands back to `run`.
    private struct QualityOutcome {
        /// Graded tasks, grouped by category in SUITE order (not run order),
        /// so a report's breakdown reads the same whatever order ran.
        let categories: [CategoryScore]
        /// The time budget ran out before the last task.
        let truncated: Bool
    }

    private func runQuality(
        tasks: [BenchmarkTask],
        suite: BenchmarkSuiteDocument,
        options: BenchmarkOptions,
        modelId: String,
        thermal: ThermalTracker,
        progress: @escaping (BenchmarkProgress) -> Void
    ) async throws -> QualityOutcome {

        // Quality occupies 0.30…0.98 of the progress bar; the load, warmup,
        // performance and standard-protocol phases own everything before it.
        let base = 0.30
        let span = 0.68

        let stageStart = Date()
        let budget = options.qualityBudget
        var results: [String: TaskResult] = [:]
        var truncated = false

        for (index, task) in tasks.enumerated() {
            try Task.checkCancellation()

            // Budget check at the task boundary, never mid-task: a half-run
            // task has no answer to grade.
            if options.qualityBudgetExhausted(
                beforeTaskAt: index,
                elapsed: Date().timeIntervalSince(stageStart)
            ) {
                truncated = true
                break
            }

            // Task boundary: the quality stage is the longest and hottest
            // part of a run, so this is where a device most often crosses
            // into `.serious` and the report's throughput numbers stop being
            // comparable.
            thermal.sample()

            // With a budget the bar tracks whichever is further along, the
            // tasks or the clock, so a slow model's bar does not sit at 40%
            // and then jump to the end when the budget runs out.
            var stageFraction = Double(index) / Double(tasks.count)
            if let budget {
                stageFraction = max(stageFraction, Date().timeIntervalSince(stageStart) / budget)
            }
            progress(BenchmarkProgress(
                stage: .quality,
                fraction: base + span * min(stageFraction, 1),
                // The category, not the raw task id: "math_word_02" means
                // nothing to the person watching the progress bar.
                label: "\(Self.categoryDisplayName(task.category)) - "
                    + "task \(index + 1) of \(tasks.count)",
                completedTasks: index,
                totalTasks: tasks.count
            ))

            // Fresh state per task: no shared history, no KV cache carried
            // over. A benchmark where task 12 can see task 11's answer is
            // measuring the harness, not the model.
            await inference.resetConversation()

            let turns = Self.qualityTurns(for: task)

            let started = Date()
            let sample: GenerationSample
            do {
                sample = try await generate(
                    turns: turns,
                    maxTokens: task.maxTokens,
                    timeout: options.taskTimeout,
                    modelId: modelId
                )
            } catch is TaskTimeout {
                // A stalled task fails and the suite carries on — one slow
                // model must not cost the other 27 results.
                results[task.id] = TaskResult(
                    taskId: task.id,
                    passed: false,
                    latencySeconds: Date().timeIntervalSince(started),
                    answer: "",
                    error: "timed out after \(Int(options.taskTimeout)) s",
                    chance: task.chance
                )
                continue
            }

            results[task.id] = Self.grade(task: task, sample: sample)
        }

        // Rebuilt in suite order from whatever completed. Categories with no
        // completed task are left out, exactly as a category filter does.
        let categories = suite.categories.compactMap { category -> CategoryScore? in
            let graded = suite.tasks(in: category).compactMap { results[$0.id] }
            return graded.isEmpty ? nil : CategoryScore(category: category, tasks: graded)
        }
        return QualityOutcome(categories: categories, truncated: truncated)
    }

    /// Exactly what one quality task sends: its own system prompt when it
    /// has one, then its prompt. Never the chat's system prompt
    /// (`SystemPromptStore`): scores have to stay comparable with every
    /// report made before the chat had one, and with each other no matter
    /// what a user typed into their chat settings.
    static func qualityTurns(for task: BenchmarkTask) -> [ChatTurn] {
        var turns: [ChatTurn] = []
        if let system = task.systemPrompt, !system.isEmpty {
            turns.append(.system(system))
        }
        turns.append(.user(task.prompt))
        return turns
    }

    /// Grades one generation. A leading `<think>...</think>` block is
    /// removed first (`BenchmarkChecker.stripLeadingReasoning`), so a
    /// reasoning model's correct answer is not failed for its preamble; an
    /// answer cut off INSIDE the block fails, with the reason recorded. The
    /// report keeps the graded text, which is what a reader needs to see.
    private static func grade(task: BenchmarkTask, sample: GenerationSample) -> TaskResult {
        let reasoning = BenchmarkChecker.stripLeadingReasoning(sample.text)

        if reasoning.unterminated {
            return TaskResult(
                taskId: task.id,
                passed: false,
                latencySeconds: sample.wallClockSeconds,
                answer: sample.text,
                error: sample.errorDescription
                    ?? "stopped inside a <think> block before answering",
                chance: task.chance
            )
        }

        let passed = sample.errorDescription == nil
            && task.check.evaluate(answer: reasoning.answer)
        return TaskResult(
            taskId: task.id,
            passed: passed,
            latencySeconds: sample.wallClockSeconds,
            answer: reasoning.answer,
            error: sample.errorDescription,
            chance: task.chance
        )
    }

    // MARK: - Generation plumbing

    /// Performance-stage generation. A timeout here is fatal to the run —
    /// unlike a quality task, a probe that never finishes leaves nothing
    /// meaningful to report.
    private func probe(
        prompt: String,
        maxTokens: Int,
        timeout: TimeInterval,
        modelId: String,
        label: String
    ) async throws -> GenerationSample {
        do {
            return try await generate(
                turns: [.user(prompt)],
                maxTokens: maxTokens,
                timeout: timeout,
                modelId: modelId
            )
        } catch is TaskTimeout {
            throw BenchmarkError.generationFailed(
                "\(label) did not finish within \(Int(timeout)) s"
            )
        }
    }

    /// What every benchmark generation sends: `deterministic` with the
    /// step's token cap. No model sampling, ever (see `generate`).
    nonisolated static func generationParameters(maxTokens: Int) -> GenerationParameters {
        var parameters = GenerationParameters.deterministic
        parameters.maxTokens = maxTokens
        return parameters
    }

    /// One generation, fully consumed, with a timeout.
    ///
    /// GREEDY DECODING. Every generation the benchmark performs — warmup,
    /// prefill probe, decode probe, standard protocol and every quality task
    /// — goes through this one method, and it always passes
    /// `GenerationParameters.deterministic`: temperature 0, topP 1, topK 1,
    /// repetition penalty 1, seed 42. `InferenceManager.generate(turns:
    /// parameters:)` takes those parameters directly, so no other file needs
    /// to change for the benchmark to be deterministic.
    ///
    /// What each backend can do with those values differs, and the differences
    /// are worth knowing before comparing two backends token for token:
    ///   • llama.cpp — temperature, topP, topK, repetition penalty and seed
    ///     are all mapped onto the sampler chain.
    ///   • MLX — temperature, topP and repetition penalty are mapped; `topK`
    ///     and `seed` are dropped (MLX's samplers hold a private random state
    ///     seeded from the clock). At temperature 0 MLX samples with argmax,
    ///     so neither omission changes what the benchmark measures.
    ///   • Apple Intelligence — `GenerationOptions` carries only sampling
    ///     mode, temperature and a token cap. Temperature 0 becomes
    ///     `.greedy`; `topP` and the repetition penalty have no counterpart
    ///     and are dropped.
    ///   • Core ML — temperature (0 switches sampling off entirely), topP and
    ///     repetition penalty are mapped; `topK` and `seed` are dropped.
    ///
    /// A model's catalog `sampling` never reaches the benchmark: it is
    /// merged only on the chat path (`ChatViewModel.parameters(for:)`), and
    /// `deterministic` leaves min-p and the presence penalty off.
    ///
    /// The honest caveat that survives all of it: greedy sampling removes the
    /// deliberate randomness, not the non-determinism that lives inside an
    /// engine. A backend that resolves ties between equally-likely tokens
    /// differently between runs, or that reduces in a different order on the
    /// GPU, can still produce different text from the same prompt.
    private func generate(
        turns: [ChatTurn],
        maxTokens: Int,
        timeout: TimeInterval,
        modelId: String
    ) async throws -> GenerationSample {

        guard inference.currentModelId == modelId else {
            throw BenchmarkError.engineUnloadedMidRun
        }

        let parameters = Self.generationParameters(maxTokens: maxTokens)

        let stream = inference.generate(turns: turns, parameters: parameters)
        let start = Date()

        // The consumption runs in its own task so the timeout can cancel it;
        // cancelling the task terminates the stream, which cancels the engine
        // (`continuation.onTermination`).
        let work = Task { @MainActor () -> GenerationSample in
            var text = ""
            var tokenCount = 0
            var firstTokenAt: Date?
            var lastTokenAt: Date?
            var reportedTokens: Int?
            var failure: String?

            do {
                for try await output in stream {
                    switch output {
                    case .textDelta(let chunk):
                        if firstTokenAt == nil { firstTokenAt = Date() }
                        lastTokenAt = Date()
                        tokenCount += 1
                        text += chunk
                    case .diagnostic(.finished(let tokens, _)):
                        // Engines count real tokens; the delta count is only
                        // a proxy (a chunk can carry more than one token).
                        reportedTokens = tokens
                    case .diagnostic, .audioChunk:
                        break
                    }
                }
            } catch is CancellationError {
                throw CancellationError()
            } catch {
                // A mid-stream engine error still leaves whatever text
                // arrived, which is worth recording in the report.
                failure = error.localizedDescription
            }

            // Cancelling a task that is iterating an AsyncThrowingStream makes
            // `next()` return nil, so the loop above exits NORMALLY and throws
            // nothing. Without this check a watchdog cancel would look like a
            // finished generation and the partial text would be graded as a
            // real answer. Re-raising here is what makes the timeout path (and
            // the caller's cancellation) reachable at all.
            if Task.isCancelled { throw CancellationError() }

            let end = Date()
            let decodeWindow = (firstTokenAt.map { (lastTokenAt ?? end).timeIntervalSince($0) }) ?? 0
            let counted = reportedTokens ?? tokenCount

            return GenerationSample(
                text: text.trimmingCharacters(in: .whitespacesAndNewlines),
                tokenCount: counted,
                wallClockSeconds: end.timeIntervalSince(start),
                firstTokenSeconds: firstTokenAt.map { $0.timeIntervalSince(start) },
                // Decode rate excludes prefill: measured from the first token
                // to the last, over the tokens that came after the first one.
                decodeTokensPerSecond: decodeWindow > 0 && counted > 1
                    ? Double(counted - 1) / decodeWindow
                    : 0,
                errorDescription: failure
            )
        }

        // Timeout watchdog. `work.cancel()` unwinds the stream and, thanks to
        // the `Task.isCancelled` re-raise at the end of `work`, surfaces as a
        // CancellationError, which is translated into `TaskTimeout` below.
        let watchdog = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(max(timeout, 1) * 1_000_000_000))
            guard !Task.isCancelled else { return }
            work.cancel()
        }
        defer { watchdog.cancel() }

        do {
            let sample = try await withTaskCancellationHandler {
                try await work.value
            } onCancel: {
                work.cancel()
            }

            // The memory-warning handler can drop the engine between the
            // start and the end of a generation. Everything measured after
            // that point belongs to a different process state.
            guard inference.currentModelId == modelId else {
                throw BenchmarkError.engineUnloadedMidRun
            }
            return sample
        } catch is CancellationError {
            // Distinguish "the watchdog fired" from "the caller cancelled":
            // only the latter must abort the whole run.
            if Task.isCancelled { throw CancellationError() }
            throw TaskTimeout()
        }
    }

    /// Rough token count for a prompt: ~4 characters per token, the usual
    /// English approximation for BPE tokenizers. Only used for the prefill
    /// rate, where a consistent estimate matters more than an exact one —
    /// the same prompt is used for every model, so the bias is constant.
    static func estimatedTokenCount(_ text: String) -> Int {
        max(1, Int((Double(text.count) / 4.0).rounded()))
    }

    /// Turns a suite category id into something showable: "math_word" becomes
    /// "Math word", "instructions" becomes "Instructions". Deliberately dumb —
    /// categories are lowercase ASCII identifiers, so splitting on the
    /// separators and capitalising is all this needs to do.
    static func categoryDisplayName(_ category: String) -> String {
        let words = category
            .split(whereSeparator: { $0 == "_" || $0 == "-" || $0 == " " })
            .map { $0.prefix(1).uppercased() + $0.dropFirst() }
        return words.isEmpty ? "Task" : words.joined(separator: " ")
    }
}

// MARK: - Convenience

extension BenchmarkEngine {

    /// Runs a benchmark and persists the report. The common case for both
    /// frontends — the in-app flow wants the run in the history, and the
    /// harness wants an artefact on disk even if the sweep is interrupted
    /// later. A failed *write* does not lose the report: it is returned
    /// anyway and the throw is left to `store.save` callers who care.
    @discardableResult
    func runAndStore(
        model: AIModel,
        options: BenchmarkOptions = .full,
        store: BenchmarkReportStore? = nil,
        progress: @escaping (BenchmarkProgress) -> Void = { _ in }
    ) async throws -> BenchmarkReport {
        let report = try await run(model: model, options: options, progress: progress)
        _ = try? (store ?? .shared).save(report)
        return report
    }
}

// MARK: - Support types

/// Thrown internally when a per-task watchdog fires. Never escapes `run`.
private struct TaskTimeout: Error {}

/// Result of a repeated decode probe: one rate and one token count per
/// successful repetition, in run order.
private struct DecodeMeasurement {
    let rates: [Double]
    let tokenCounts: [Int]
}

/// Remembers the hottest thermal state seen at any probe or task boundary.
///
/// `ProcessInfo.thermalState` is a point sample, and a run can heat the phone
/// into `.serious` in the middle of the quality stage and drift back to
/// `.fair` before the final reading — which would produce a report whose two
/// endpoint chips both read `.fair` for a run that spent two minutes hot.
/// Sampling at every boundary and keeping the maximum is what makes
/// `PerformanceMetrics.peakThermalState` honest.
///
/// The peak is REPORTED, not judged on: whether the numbers are comparable is
/// decided by `PerformanceMetrics.throttleVerdict(runs:protocolRate:
/// probeMean:)` from the measured rates.
///
/// A class rather than a struct so the sampling calls scattered through an
/// async run all update the same value without `inout` gymnastics.
@MainActor
final class ThermalTracker {

    private(set) var peak: ProcessInfo.ThermalState

    /// Every `MeasurementConditions` issue seen at any boundary, first-seen
    /// order, no repeats. Becomes `PerformanceMetrics.measurementIssues`.
    private(set) var issues: [String] = []

    init() {
        peak = ProcessInfo.processInfo.thermalState
        noteIssues()
    }

    /// Takes a reading and keeps it if it is the hottest so far.
    func sample() {
        let now = ProcessInfo.processInfo.thermalState
        if now.severityRank > peak.severityRank { peak = now }
        noteIssues()
    }

    private func noteIssues() {
        for issue in MeasurementConditions.currentIssues() where !issues.contains(issue) {
            issues.append(issue)
        }
    }

    /// True once the device has been seen at `.serious` or worse.
    ///
    /// Informational: this used to decide `thermallyConstrained`, and it was
    /// true for every model in a 12-model sweep — small and large, fast and
    /// slow — which is why the flag is now measured from the decode rates
    /// instead. Kept because "did this run ever get hot" is still worth
    /// asking of a `ThermalTracker`.
    var constrained: Bool { peak.isThermallyConstrained }
}

/// Device states under which a throughput measurement describes the OS's
/// power policy rather than the model.
///
/// Both produce a slow but perfectly steady run (repetitions agree to a
/// hundredth of a token per second), which the throttle rule reads as "not
/// throttled" and the report would publish as a slow model. A night sweep on
/// an iPhone Air did exactly that: from one model on, every engine (MLX and
/// llama.cpp alike) ran 7-11x slower with the phone staying cool, and
/// downloads stalled at 0%.
///
/// `.inactive` is deliberately not an issue: a notification banner or the
/// Control Center makes the app inactive without restricting the GPU.
@MainActor
enum MeasurementConditions {

    static func currentIssues() -> [String] {
        var issues: [String] = []
        if ProcessInfo.processInfo.isLowPowerModeEnabled {
            issues.append("low power mode")
        }
        if UIApplication.shared.applicationState == .background {
            issues.append("app in background")
        }
        return issues
    }

    /// One line for the harness console: the issues plus the battery, which
    /// is context (a sweep drains a phone whose charging pauses when hot)
    /// rather than a reason to discard a run.
    static func logLine() -> String {
        let device = UIDevice.current
        if !device.isBatteryMonitoringEnabled { device.isBatteryMonitoringEnabled = true }
        let battery: String
        switch device.batteryState {
        case .charging: battery = "charging"
        case .full: battery = "full"
        case .unplugged: battery = "unplugged"
        case .unknown: battery = "unknown"
        @unknown default: battery = "unknown"
        }
        let level = device.batteryLevel >= 0
            ? String(format: "%.0f%%", device.batteryLevel * 100)
            : "?"
        let issues = currentIssues()
        return "low power \(ProcessInfo.processInfo.isLowPowerModeEnabled ? "on" : "off"), "
            + "app \(UIApplication.shared.applicationState == .background ? "background" : "foreground"), "
            + "battery \(level) \(battery), thermal \(ProcessInfo.processInfo.thermalState.reportName)"
            + (issues.isEmpty ? "" : " - numbers taken now are invalid")
    }
}

/// Everything one generation tells us.
private struct GenerationSample {
    let text: String
    let tokenCount: Int
    let wallClockSeconds: TimeInterval
    /// Time to first token — prompt processing plus one decode step.
    let firstTokenSeconds: TimeInterval?
    /// Decode rate with prefill excluded.
    let decodeTokensPerSecond: Double
    /// Engine error that ended the stream early, if any.
    let errorDescription: String?
}

// MARK: - Memory probing

/// `phys_footprint` reader — the same figure jetsam charges and Xcode's
/// memory gauge shows. Same `TASK_VM_INFO` pattern as `DeviceScanner` and
/// `LiveDeviceMetrics`; kept here as a `nonisolated` helper so the sampler
/// can call it from any context.
nonisolated enum MemoryProbe {

    /// Current footprint in MB, 0 when the mach call fails (simulator quirks,
    /// sandboxed test runners).
    static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size
        )
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }
}

/// Polls `phys_footprint` while a generation runs. A generation yields to the
/// main actor between tokens, so a main-actor polling loop does get scheduled;
/// 150 ms is frequent enough to catch a KV-cache spike and cheap enough to be
/// invisible in the throughput numbers.
@MainActor
final class MemorySampler {

    private(set) var peakMB: Double = 0
    private var task: Task<Void, Never>?

    func start() {
        stop()
        peakMB = MemoryProbe.footprintMB()
        task = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.peakMB = max(self.peakMB, MemoryProbe.footprintMB())
                try? await Task.sleep(nanoseconds: 150_000_000)
            }
        }
    }

    func stop() {
        task?.cancel()
        task = nil
        peakMB = max(peakMB, MemoryProbe.footprintMB())
    }
}
