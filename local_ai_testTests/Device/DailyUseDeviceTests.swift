//
//  DailyUseDeviceTests.swift
//
//  How one catalog model behaves in realistic everyday use: a ten-turn chat,
//  a long article to summarise, an email to rewrite, data to extract, two
//  translations, two word problems, a small coding task, a six-turn trip
//  plan whose constraints change, questions about a pasted lease (one of
//  them not answered by it), questions about things that do not exist,
//  short chats in other languages, four formatting instructions, a 40-turn
//  casual chat with five pasted texts that outgrows both llama.cpp context
//  windows, photo questions for vision models (two synthetic images, two
//  real photos), and eight back-to-back answers to watch throttling.
//
//  Changed 2026-10-24: `long_session` grew from 20 short turns (about
//  1,500 prompt tokens at the end, so no engine ever trimmed) to 40 longer
//  ones with five pasted texts (about 8,200-9,300), so history trimming is
//  actually exercised in both the 4096-token text window and the
//  8192-token vision window. Its `trim_events` check now fails when an
//  engine that reports trimming never trimmed. Results before and after
//  the change are not comparable for this scenario.
//
//  Apple Intelligence (`apple-intelligence`) runs too: there is nothing to
//  download, the session keeps the conversation itself, and the framework
//  reports no token counts, prefill or cache figures - the step's
//  `token_source` says `estimated_chars` and the rates are estimates from
//  ~4 characters per token.
//
//  It is the opposite of the benchmark's exam. The benchmark asks hundreds of
//  short questions greedily (temperature 0, ~96 tokens) and grades them
//  automatically. This suite sends what the chat screen sends - the app's
//  default system prompt (`SystemPrompt.defaultText`, through the same
//  `SystemPrompt.requestTurns`), the user's turns, the chat's sampling
//  (`ChatViewModel.parameters(for:)`: `GenerationParameters.chat` with the
//  model's catalog `sampling` merged over it, unless DAILY_SAMPLING says
//  otherwise) with its 512-token reply cap, the whole conversation re-sent
//  every turn, the model's own chat template -
//  and records every prompt and every full answer verbatim, so a person (or
//  a grading model) can read the transcripts afterwards. The automatic
//  checks in `DailyUseChecks` only catch the obvious failures.
//
//  One deviation from the chat screen: the seed is fixed (42), so a re-run
//  on the same build samples the same answers on the backends that honour a
//  seed (llama.cpp, Core ML, Apple Intelligence). MLX ignores it.
//
//  ---------------------------------------------------------------------
//  Running it (one model per invocation)
//  ---------------------------------------------------------------------
//
//      TEST_RUNNER_DAILY_ENABLED=1 \
//      TEST_RUNNER_DAILY_MODEL_ID=unsloth/Qwen3.5-2B-GGUF \
//      xcodebuild test \
//        -project local_ai_test.xcodeproj -scheme local_ai_test \
//        -destination 'platform=iOS,id=<hardware UDID>' \
//        -only-testing:local_ai_testTests/DailyUseDeviceTests \
//        2>&1 | tee daily-<model>.log
//
//  `Scripts/daily_use_night.sh` runs a list of models this way, one
//  process each, and tabulates the results.
//
//  The variables MUST be environment variables placed BEFORE `xcodebuild`.
//  Xcode copies every `TEST_RUNNER_<NAME>` from its environment into the
//  test runner's environment as `<NAME>`. Written after `xcodebuild` they
//  become build settings and are silently ignored: the test then reports
//  "skipped" (or passes having run nothing).
//
//  ---------------------------------------------------------------------
//  Environment variables
//  ---------------------------------------------------------------------
//
//    DAILY_ENABLED=1         Required. Without it the test is skipped.
//    DAILY_MODEL_ID=<id>     Required when enabled. One catalog model id.
//    DAILY_SCENARIOS=a,b     Optional subset, by scenario name (see
//                            `DailyScenario`). Default: all of them; the two
//                            vision scenarios run only for models whose
//                            catalog capabilities include image input.
//    DAILY_KEEP_MODEL=1      Do not delete a model this run downloaded.
//                            A model that was already installed is never
//                            deleted either way.
//    DAILY_TIMEOUT_SEC=N     Per-generation timeout. Default 180.
//    DAILY_NO_SYSTEM_PROMPT=1 Send no system prompt, as the chat did before
//                            1.2 (for before/after comparisons).
//    DAILY_NO_THERMAL_GATE=1 Do not wait for the phone to cool before the
//                            load and each scenario. The power gate stays.
//
//  Conditions gate (`RunConditionsGate`, shared with the catalog sweep),
//  before the load and before every scenario:
//
//    DAILY_COOLDOWN_TARGET=nominal|fair|off  Default nominal.
//    DAILY_COOLDOWN_SETTLE_SEC=N  Extra idle time once the target is reached
//                            after a wait. Default 60.
//    DAILY_COOLDOWN_MAX_SEC=N  Cap of one wait. Default 1200. A scenario
//                            measured after the cap ran out carries the
//                            reason in `measurement_issues`.
//    Power: Low Power Mode off and charging/full or 40% battery or more,
//    checked before the download too.
//    DAILY_STOP_ON_BAD_POWER=0|1  Power still bad at the cap: 1 (default)
//                            prints `DAILY_NIGHT_STOP <reason>`, skips the
//                            rest, and `daily_use_night.sh` ends the night;
//                            0 measures and flags.
//    DAILY_DOWNLOAD_STALL_SEC=N  Fail a download that moves no bytes for N
//                            seconds (default 300; 0 = only the 30-min
//                            budget). A lost network path is waited for
//                            first.
//    DAILY_SAMPLING=mode     Which sampling the run sends (`DailySampling`):
//                            `catalog` (default) as the chat screen does,
//                            the model's catalog `sampling` over the global
//                            chat parameters; `global` the global chat
//                            parameters only; `authors` the model authors'
//                            values from `DailyUseAuthorsSampling.json` in
//                            the test bundle, even though the bundled catalog
//                            carries none - the global-vs-authors A/B without
//                            changing the app. Any other value fails the run.
//                            The values used are in each result's
//                            `parameters` (with `sampling`: the mode).
//
//  ---------------------------------------------------------------------
//  Output
//  ---------------------------------------------------------------------
//
//  One line per scenario, `DAILY_RESULT_JSON {...}`, and a final
//  `DAILY_SUMMARY_JSON {...}`, each a single line of JSON in the test log.
//  The same data is rewritten after every scenario to
//  `Documents/DailyUse/<model>-<timestamp>.json` in the app container, so a
//  crash keeps everything finished before it. `Scripts/daily_use_extract.py`
//  splits a log into one file per scenario and prints a table of numbers.
//
//  Per generation ("step"): the prompt, the full answer, time to first
//  token, tokens, decode tok/s (prefill excluded), the engine's own figures
//  (including, for llama.cpp, how many prompt tokens came from the KV cache
//  and how many were decoded),
//  peak `phys_footprint` during the step, the thermal state after it, and
//  the context situation (see below).
//
//  ---------------------------------------------------------------------
//  Context window
//  ---------------------------------------------------------------------
//
//  Every turn is sent, as the chat does, even when the conversation no
//  longer fits. The engines handle that themselves: both llama.cpp engines
//  drop older exchanges in batches, keeping the system turn and a short
//  first exchange (the text engine also cuts inside a single turn that is
//  too long on its own), Core ML
//  trims or fails. Each step therefore records the window the engine opens,
//  the room it reserves for the reply, an estimate of the prompt, whether an
//  overflow was expected, and whether the first user turn still reached the
//  model (in the rendered prompt and not trimmed by the engine afterwards,
//  per the llama.cpp engines' cache report). A forgotten fact in
//  `chat_recall` can then be told apart from a trimmed one. An engine error
//  is recorded as the step's outcome (`context_full`, `out_of_memory`,
//  `timeout`, ...) and the scenario stops; the next scenario still runs.
//
//  ---------------------------------------------------------------------
//  Verdict
//  ---------------------------------------------------------------------
//
//  The test fails only when the run could not happen at all (no model id,
//  unknown model, backend not linked, download or load failure) or when
//  every scenario failed. Bad answers do not fail it: grading is the
//  reader's job.
//

import Foundation
import Testing
import UIKit
@testable import local_ai_test

extension Tag {
    /// Realistic-use transcripts for one model. Long-running, device only.
    @Tag static var dailyUse: Self
}

// MARK: - Configuration

/// Read from the process environment; `TEST_RUNNER_DAILY_X` arrives as
/// `DAILY_X`.
struct DailyUseConfig {
    let enabled: Bool
    let modelID: String?
    /// Names as given; `nil` means every scenario.
    let scenarioNames: [String]?
    let keepModel: Bool
    let timeout: TimeInterval
    let thermalGate: Bool
    /// The chat's default system prompt, or nil with DAILY_NO_SYSTEM_PROMPT.
    let systemPrompt: String?
    /// Cool-down, power and download-stall rules (`DAILY_COOLDOWN_*`,
    /// `DAILY_STOP_ON_BAD_POWER`, `DAILY_DOWNLOAD_STALL_SEC`).
    let conditions: RunConditionsPolicy
    /// DAILY_SAMPLING; nil when the value is not a known mode.
    let sampling: DailySamplingMode?
    /// DAILY_SAMPLING as given, for the error message.
    let samplingRaw: String?

    static let defaultTimeout: TimeInterval = 180

    static func fromEnvironment(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> DailyUseConfig {
        func value(_ key: String) -> String? {
            guard let raw = environment[key]?.trimmingCharacters(in: .whitespacesAndNewlines),
                  !raw.isEmpty else { return nil }
            return raw
        }
        func flag(_ key: String) -> Bool {
            ["1", "true", "yes"].contains(value(key)?.lowercased() ?? "")
        }
        let names = value("DAILY_SCENARIOS")?
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespaces).lowercased() }
            .filter { !$0.isEmpty }
        return DailyUseConfig(
            enabled: flag("DAILY_ENABLED"),
            modelID: value("DAILY_MODEL_ID"),
            scenarioNames: (names?.isEmpty ?? true) ? nil : names,
            keepModel: flag("DAILY_KEEP_MODEL"),
            timeout: value("DAILY_TIMEOUT_SEC").flatMap(TimeInterval.init).flatMap { $0 > 0 ? $0 : nil }
                ?? defaultTimeout,
            thermalGate: !flag("DAILY_NO_THERMAL_GATE"),
            systemPrompt: flag("DAILY_NO_SYSTEM_PROMPT") ? nil : SystemPrompt.defaultText,
            conditions: RunConditionsPolicy.fromEnvironment(environment, prefix: "DAILY_"),
            sampling: DailySamplingMode.parse(value("DAILY_SAMPLING")),
            samplingRaw: value("DAILY_SAMPLING")
        )
    }
}

// MARK: - Scenarios

enum DailyScenario: String, CaseIterable, Codable {
    case chatRecall = "chat_recall"
    case summarizeLong = "summarize_long"
    case rewriteEmail = "rewrite_email"
    case extractJSON = "extract_json"
    case translate = "translate"
    case reasoningShort = "reasoning_short"
    case codeSmall = "code_small"
    case adviceFollowup = "advice_followup"
    case docQA = "doc_qa"
    case unknownEntity = "unknown_entity"
    case nonEnglishChat = "non_english_chat"
    case formatFollow = "format_follow"
    case longSession = "long_session"
    case visionDescribe = "vision_describe"
    case visionReadText = "vision_read_text"
    case visionPhotoReal = "vision_photo_real"
    case sustained = "sustained"

    var needsVision: Bool {
        self == .visionDescribe || self == .visionReadText || self == .visionPhotoReal
    }
}

// MARK: - Result records (the JSON the grader reads)

struct DailyContextInfo: Codable {
    /// Window the engine opens for this model, when known.
    var contextTokens: Int?
    /// Tokens the engine keeps free for the reply.
    var replyReserve: Int?
    /// ~4 characters per token over every turn sent, plus the vision
    /// builder's per-image and per-turn estimates.
    var promptTokensEstimate: Int
    var turnsSent: Int
    /// Estimate > window - reserve: the engine will have trimmed or failed.
    var overflowExpected: Bool?
    /// The first user turn reached the model: it is in the prompt the
    /// engine rendered AND the engine did not cut it afterwards (tokens
    /// trimmed from the front, or turns dropped, per its cache report).
    /// `nil` when the engine reports neither or there is no history.
    var firstTurnKept: Bool?
    var renderedPromptChars: Int?
    var templateSource: String?
}

struct DailyStep: Codable {
    var index: Int
    var label: String
    var prompt: String
    var hasImage: Bool
    /// Full answer, whitespace-trimmed, reasoning blocks included.
    var answer: String
    /// ok, empty, timeout, context_full, out_of_memory, engine_error,
    /// model_unloaded, cancelled, model_not_loaded, guardrail,
    /// unsupported_language.
    var outcome: String
    var error: String?
    var ttftMs: Double?
    var tokens: Int
    /// Where `tokens` comes from: `engine` (the engine's own count),
    /// `stream_deltas` (one per streamed piece) or `estimated_chars`
    /// (~4 characters per token; Apple Intelligence streams snapshots, not
    /// tokens, and reports no count).
    var tokenSource: String
    /// Decode rate, prefill excluded: (tokens - 1) / (last token - first token).
    var tps: Double?
    var engineTtftMs: Double?
    var engineTps: Double?
    /// Engine-reported prompt tokens and the time spent decoding them. With
    /// llama.cpp's prefix cache only the new tail is decoded on later turns,
    /// so `prefillTps` is filled for the first step of a conversation only.
    var prefillTokens: Int?
    var prefillMs: Double?
    var prefillTps: Double?
    /// KV prefix reuse, from the engine's cache report (llama.cpp engines
    /// only): prompt tokens taken from the previous turn's cache, prompt
    /// tokens decoded this turn, and how the cache was rewound (`empty`,
    /// `prefix`, `checkpoint`, `cleared`).
    var reusedTokens: Int?
    var evaluatedTokens: Int?
    var cacheRewind: String?
    /// History trimming this turn, from the same cache report: whole turns
    /// left out, tokens cut from the front as a last resort, tokens cut
    /// inside one over-long turn. Nil when the engine sends no report.
    var droppedTurns: Int?
    var droppedFrontTokens: Int?
    var truncatedTokens: Int?
    var maxTokens: Int
    var hitTokenCap: Bool
    var startsWithReasoning: Bool
    var wallMs: Double
    var peakMb: Double
    var thermalAfter: String
    var context: DailyContextInfo
}

struct DailyCheck: Codable {
    var name: String
    var passed: Bool
    var detail: String
}

struct DailyParameters: Codable {
    /// DAILY_SAMPLING mode the values came from.
    var sampling: String
    var temperature: Float
    var topP: Float
    var topK: Int
    var minP: Float
    var repetitionPenalty: Float
    var presencePenalty: Float
    var maxTokens: Int
    var seed: UInt64?
    var seedHonouredByBackend: Bool
    var systemPrompt: String?

    init(_ p: GenerationParameters, sampling: DailySamplingMode, backend: ModelBackend, systemPrompt: String?) {
        self.sampling = sampling.rawValue
        temperature = p.temperature
        topP = p.topP
        topK = p.topK
        minP = p.minP
        repetitionPenalty = p.repetitionPenalty
        presencePenalty = p.presencePenalty
        maxTokens = p.maxTokens
        seed = p.seed
        switch backend {
        case .mlx, .mlxVision, .mlxAudio: seedHonouredByBackend = false
        default: seedHonouredByBackend = true
        }
        self.systemPrompt = systemPrompt
    }
}

struct DailyScenarioResult: Codable {
    var schema = 2
    var modelId: String
    var modelName: String
    var backend: String
    var scenario: String
    var device: DeviceSnapshot
    var appVersion: String
    var parameters: DailyParameters
    var startedAt: String
    /// ok, partial (some steps ran), failed (none did), skipped, error.
    var outcome: String
    var error: String?
    var thermalBefore: String
    var thermalAfter: String
    var peakThermal: String
    var cooldownWaitS: Double
    var modelReloaded: Bool
    var wallClockS: Double
    /// What the grader should know that is not in the prompts (the content
    /// of a generated image, the expected answers).
    var reference: String?
    var score: Int?
    var scoreMax: Int?
    var steps: [DailyStep]
    var checks: [DailyCheck]
    /// Why the numbers may not describe the model: a gate that ran out
    /// before the scenario, Low Power Mode or the background during it.
    /// `nil` when there is nothing to say.
    var measurementIssues: [String]? = nil
}

struct DailyScenarioSummary: Codable {
    var scenario: String
    var outcome: String
    var stepsOk: Int
    var stepsTotal: Int
    var checksPassed: Int
    var checksTotal: Int
    var score: Int?
    var scoreMax: Int?
    var medianTtftMs: Double?
    var medianTps: Double?
    var maxPeakMb: Double?
    var wallClockS: Double
}

struct DailyLoadInfo: Codable {
    var outcome: String
    var error: String?
    var loadMs: Double?
    var peakMb: Double?
    var footprintAfterMb: Double?
}

struct DailyRunSummary: Codable {
    var schema = 1
    var modelId: String
    var modelName: String
    var backend: String
    var device: DeviceSnapshot
    var appVersion: String
    var appBuild: String
    var startedAt: String
    var complete: Bool
    var wallClockS: Double
    var requestedScenarios: [String]
    var ignoredScenarioNames: [String]
    var downloadedHere: Bool
    var downloadS: Double?
    var deletedAfter: Bool
    var load: DailyLoadInfo?
    var thermalStart: String
    var thermalEnd: String
    var scenarios: [DailyScenarioSummary]
    var jsonFile: String?
    var error: String?
}

private struct DailyFile: Codable {
    var summary: DailyRunSummary
    var scenarios: [DailyScenarioResult]
}

enum DailyUseError: Error, CustomStringConvertible {
    case download(String)
    case image(String)
    case fixture(String)
    case load(String)

    var description: String {
        switch self {
        case .download(let m): return "download failed: \(m)"
        case .image(let m): return "test image: \(m)"
        case .fixture(let m): return "test fixture: \(m)"
        case .load(let m): return "load failed: \(m)"
        }
    }
}

// MARK: - Test

@Suite(.tags(.device, .dailyUse, .performance), .serialized)
struct DailyUseDeviceTests {

    @Test(
        "Daily use: one catalog model through realistic chat scenarios",
        .enabled(
            if: DailyUseConfig.fromEnvironment().enabled,
            "set TEST_RUNNER_DAILY_ENABLED=1 and TEST_RUNNER_DAILY_MODEL_ID=<id> in front of xcodebuild"
        )
    )
    @MainActor
    func dailyUse() async throws {
        let config = DailyUseConfig.fromEnvironment()

        guard let modelID = config.modelID else {
            Issue.record("DAILY_MODEL_ID is required (TEST_RUNNER_DAILY_MODEL_ID=<catalog id>)")
            return
        }
        guard let model = ModelCatalog.model(id: modelID) else {
            Issue.record("model not in the catalog: \(modelID)")
            return
        }
        guard model.backend.isLinked else {
            Issue.record("backend \(model.backend.rawValue) is not linked into this build")
            return
        }
        guard model.capabilities.contains(.textIn), model.capabilities.contains(.textOut) else {
            Issue.record("\(modelID) is not a text-in/text-out model")
            return
        }

        guard let sampling = config.sampling else {
            Issue.record("DAILY_SAMPLING must be catalog, global or authors, not \"\(config.samplingRaw ?? "")\"")
            return
        }
        var authorsSampling: ModelSampling?
        if sampling == .authors {
            guard let data = DailyFixtures.bundledResource(DailyAuthorsSampling.resourceName) else {
                Issue.record("\(DailyAuthorsSampling.resourceName) is missing from the test bundle")
                return
            }
            do {
                authorsSampling = try DailyAuthorsSampling.decode(data).models[model.id]
            } catch {
                Issue.record("\(DailyAuthorsSampling.resourceName) does not decode: \(error)")
                return
            }
        }

        let runner = DailyUseRunner(
            model: model, config: config, sampling: sampling, authorsSampling: authorsSampling
        )
        let summary = await runner.run()

        if let path = summary.jsonFile,
           let data = FileManager.default.contents(atPath: path),
           let json = String(data: data, encoding: .utf8) {
            Attachment.record(json, named: (path as NSString).lastPathComponent)
        }

        #expect(summary.error == nil, "\(summary.error ?? "")")
        let ran = summary.scenarios.filter { $0.outcome != "skipped" }
        #expect(
            ran.isEmpty || ran.contains { $0.outcome == "ok" || $0.outcome == "partial" },
            "every scenario failed - see the DAILY_RESULT_JSON lines"
        )
    }
}

// MARK: - Runner

@MainActor
final class DailyUseRunner {

    private let model: AIModel
    private let config: DailyUseConfig
    private let inference = InferenceManager.shared
    private let device: DeviceSnapshot
    private let startedAt = Date()
    private var results: [DailyScenarioResult] = []
    private var summary: DailyRunSummary
    private let fileURL: URL?
    /// Set when the conditions gate gave up on the power; the rest of the
    /// run is skipped and the night script stops on `DAILY_NIGHT_STOP`.
    private var stopReason: String?
    private let sampling: DailySamplingMode
    /// What every scenario sends before its own tweaks (seed, cap).
    private let baseParameters: GenerationParameters
    private let authorsEntryMissing: Bool

    /// Pause after a timed-out generation, so the engine's own cancellation
    /// can land before the next scenario sends a prompt.
    private static let settleAfterTimeout: UInt64 = 2_000_000_000

    init(model: AIModel, config: DailyUseConfig, sampling: DailySamplingMode, authorsSampling: ModelSampling?) {
        self.model = model
        self.config = config
        self.sampling = sampling
        self.baseParameters = DailySampling.parameters(
            mode: sampling,
            global: ChatViewModel.parameters,
            catalogSampling: ModelSampling.effective(
                for: model, catalogEntry: ModelCatalog.model(id: model.id)
            ),
            authorsSampling: authorsSampling
        )
        self.authorsEntryMissing = sampling == .authors && authorsSampling == nil
        let device = DeviceSnapshot.current()
        self.device = device

        let stamp: String = {
            let f = DateFormatter()
            f.locale = Locale(identifier: "en_US_POSIX")
            f.dateFormat = "yyyyMMdd-HHmmss"
            return f.string(from: Date())
        }()
        let safeID = model.id.replacingOccurrences(of: "/", with: "_")
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("DailyUse", isDirectory: true)
        if let dir {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        fileURL = dir?.appendingPathComponent("\(safeID)-\(stamp).json")

        summary = DailyRunSummary(
            modelId: model.id,
            modelName: model.name,
            backend: model.backend.rawValue,
            device: device,
            appVersion: BenchmarkAppInfo.version,
            appBuild: BenchmarkAppInfo.build,
            startedAt: ISO8601DateFormatter().string(from: startedAt),
            complete: false,
            wallClockS: 0,
            requestedScenarios: [],
            ignoredScenarioNames: [],
            downloadedHere: false,
            downloadS: nil,
            deletedAfter: false,
            load: nil,
            thermalStart: ProcessInfo.processInfo.thermalState.reportName,
            thermalEnd: ProcessInfo.processInfo.thermalState.reportName,
            scenarios: [],
            jsonFile: fileURL?.path,
            error: nil
        )
    }

    // MARK: Top level

    func run() async -> DailyRunSummary {
        let scenarios = selectedScenarios()
        summary.requestedScenarios = scenarios.map(\.rawValue)
        log("model \(model.id) (\(model.backend.rawValue)) on \(device.marketingName), "
            + "iOS \(device.iOSVersion), app \(BenchmarkAppInfo.version) (\(BenchmarkAppInfo.build))")
        log("scenarios: \(summary.requestedScenarios.joined(separator: ", "))")
        let p = baseParameters
        log("sampling \(sampling.rawValue): temp \(p.temperature), top_p \(p.topP), top_k \(p.topK), "
            + "min_p \(p.minP), repeat \(p.repetitionPenalty), presence \(p.presencePenalty)")
        if authorsEntryMissing {
            log("WARNING: no authors' values for \(model.id) in \(DailyAuthorsSampling.resourceName) - "
                + "running with the global parameters")
        }
        if !summary.ignoredScenarioNames.isEmpty {
            log("ignored unknown scenario names: \(summary.ignoredScenarioNames.joined(separator: ", "))")
        }
        guard !scenarios.isEmpty else {
            return finish(error: "DAILY_SCENARIOS named no known scenario")
        }
        log("conditions: \(config.conditions.summaryLine)"
            + (config.thermalGate ? "" : " (thermal gate off)"))

        // Power first: a phone on a draining battery or in Low Power Mode is
        // not worth a download.
        if await gateConditions(checkThermal: false).stop {
            return finish(error: "stopped: \(stopReason ?? "power")")
        }

        // Install.
        var owned = false
        if BuildTestRunner.requiresDownload(model) {
            if BuildTestRunner.isInstalled(model) {
                // A marker left by a crashed harness run: those bytes are a
                // harness's to remove, not the user's.
                owned = BuildTestRunner.isOwned(model)
                log(owned ? "installed by an earlier harness run - will be removed afterwards"
                          : "already installed - will be left in place")
            } else {
                if let reason = BuildTestRunner.insufficientSpaceReason(for: model) {
                    return finish(error: reason)
                }
                if config.conditions.downloadStallTimeout != nil {
                    _ = try? await RunConditionsWaiter.waitForNetwork(
                        maxWait: config.conditions.maxWait, log: { self.log($0) }
                    )
                }
                let started = Date()
                do {
                    try await download(ownedByRun: &owned)
                    summary.downloadS = Date().timeIntervalSince(started)
                } catch {
                    summary.downloadedHere = owned
                    await cleanUp(owned: owned)
                    return finish(error: "\(error)")
                }
            }
        }
        summary.downloadedHere = owned

        // Load from cold, once, timed.
        if await gateConditions(checkThermal: true).stop {
            await cleanUp(owned: owned)
            return finish(error: "stopped: \(stopReason ?? "power")")
        }
        await inference.unloadCurrent()
        let loadSampler = MemorySampler()
        loadSampler.start()
        let loadStart = Date()
        await inference.prepare(model: model)
        loadSampler.stop()
        let loadMs = Date().timeIntervalSince(loadStart) * 1000
        guard inference.state == .ready, inference.currentModelId == model.id else {
            let message = inference.lastError?.detail ?? "\(inference.state)"
            summary.load = DailyLoadInfo(outcome: "failed", error: message, loadMs: loadMs,
                                         peakMb: loadSampler.peakMB, footprintAfterMb: MemoryProbe.footprintMB())
            await cleanUp(owned: owned)
            return finish(error: "load failed: \(message)")
        }
        summary.load = DailyLoadInfo(outcome: "ok", error: nil, loadMs: loadMs,
                                     peakMb: loadSampler.peakMB, footprintAfterMb: MemoryProbe.footprintMB())
        log(String(format: "loaded in %.0f ms, peak %.0f MB", loadMs, loadSampler.peakMB))

        for scenario in scenarios {
            let result = await runScenario(scenario)
            results.append(result)
            summary.scenarios.append(Self.summarize(result))
            print("DAILY_RESULT_JSON " + Self.encode(result))
            persist()
            if stopReason != nil { break }
        }

        await inference.unloadCurrent()
        await cleanUp(owned: owned)
        return finish(error: stopReason.map { "stopped: \($0)" })
    }

    private func selectedScenarios() -> [DailyScenario] {
        let all = DailyScenario.allCases
        guard let names = config.scenarioNames else { return all }
        summary.ignoredScenarioNames = names.filter { DailyScenario(rawValue: $0) == nil }
        let wanted = Set(names.compactMap(DailyScenario.init(rawValue:)))
        return all.filter { wanted.contains($0) }
    }

    private func finish(error: String?) -> DailyRunSummary {
        summary.error = error
        summary.complete = error == nil
        summary.wallClockS = Date().timeIntervalSince(startedAt)
        summary.thermalEnd = ProcessInfo.processInfo.thermalState.reportName
        if let error { log("ERROR: \(error)") }
        persist()
        print("DAILY_SUMMARY_JSON " + Self.encode(summary))
        return summary
    }

    private func cleanUp(owned: Bool) async {
        guard owned else { return }
        if config.keepModel {
            log("DAILY_KEEP_MODEL=1: keeping \(model.id)")
            return
        }
        log("deleting \(model.id) (downloaded by this run)")
        await BuildTestRunner.deleteAndWait(model)
        summary.deletedAfter = true
    }

    // MARK: Scenario plumbing

    /// Everything a scenario needs to send turns and collect steps.
    private final class Conversation {
        var turns: [ChatTurn] = []
        var steps: [DailyStep] = []
        /// Set once a step fails in a way that ends the conversation.
        var stopped = false
        /// A separate chat inside the scenario stopped early; the others
        /// still ran.
        var partial = false
    }

    /// Runs `body` in a new chat, as "Clear chat" starts one, and appends
    /// its steps to the scenario's. Returns the new chat's steps, indexed
    /// as they appear in the scenario.
    @discardableResult
    private func inFreshChat(
        _ scenarioChat: Conversation,
        _ body: (Conversation) async -> Void
    ) async -> [DailyStep] {
        let fresh = Conversation()
        await inference.resetConversation()
        await body(fresh)
        let offset = scenarioChat.steps.count
        let steps = fresh.steps.map { step -> DailyStep in
            var s = step
            s.index += offset
            return s
        }
        scenarioChat.steps.append(contentsOf: steps)
        if fresh.stopped { scenarioChat.partial = true }
        return steps
    }

    private func runScenario(_ scenario: DailyScenario) async -> DailyScenarioResult {
        var result = DailyScenarioResult(
            modelId: model.id,
            modelName: model.name,
            backend: model.backend.rawValue,
            scenario: scenario.rawValue,
            device: device,
            appVersion: BenchmarkAppInfo.version,
            parameters: DailyParameters(
                parameters(for: scenario),
                sampling: sampling,
                backend: model.backend,
                systemPrompt: config.systemPrompt
            ),
            startedAt: ISO8601DateFormatter().string(from: Date()),
            outcome: "ok",
            error: nil,
            thermalBefore: "",
            thermalAfter: "",
            peakThermal: "",
            cooldownWaitS: 0,
            modelReloaded: false,
            wallClockS: 0,
            reference: nil,
            score: nil,
            scoreMax: nil,
            steps: [],
            checks: []
        )

        if scenario.needsVision, !model.capabilities.contains(.imageIn) {
            result.outcome = "skipped"
            result.error = "model has no image input"
            result.thermalBefore = ProcessInfo.processInfo.thermalState.reportName
            result.thermalAfter = result.thermalBefore
            result.peakThermal = result.thermalBefore
            log("[\(scenario.rawValue)] skipped: no image input")
            return result
        }

        let gate = await gateConditions(checkThermal: true)
        result.cooldownWaitS = gate.waited
        result.thermalBefore = ProcessInfo.processInfo.thermalState.reportName
        if gate.stop {
            result.outcome = "skipped"
            result.error = "stopped: " + gate.issues.joined(separator: "; ")
            result.thermalAfter = result.thermalBefore
            result.peakThermal = result.thermalBefore
            result.measurementIssues = gate.issues
            log("[\(scenario.rawValue)] skipped: run stopped by the conditions gate")
            return result
        }
        let thermal = ThermalTracker()
        let started = Date()
        log("[\(scenario.rawValue)] start (thermal \(result.thermalBefore))")

        do {
            result.modelReloaded = try await ensureLoaded()
            // A new chat, as "Clear chat" starts one.
            await inference.resetConversation()

            let conversation = Conversation()
            try await play(scenario, into: conversation, result: &result, thermal: thermal)
            result.steps = conversation.steps

            let ran = conversation.steps.filter { $0.outcome == "ok" || $0.outcome == "empty" }
            if conversation.steps.isEmpty || ran.isEmpty {
                result.outcome = "failed"
            } else if ran.count < conversation.steps.count || conversation.stopped || conversation.partial {
                result.outcome = "partial"
            }
        } catch {
            result.outcome = result.steps.isEmpty ? "error" : "partial"
            result.error = "\(error)"
        }

        thermal.sample()
        result.thermalAfter = ProcessInfo.processInfo.thermalState.reportName
        result.peakThermal = thermal.peak.reportName
        var issues = gate.issues
        for issue in thermal.issues where !issues.contains(issue) { issues.append(issue) }
        result.measurementIssues = issues.isEmpty ? nil : issues
        result.wallClockS = Date().timeIntervalSince(started)
        let passed = result.checks.filter(\.passed).count
        log(String(format: "[%@] %@ in %.0f s, %d steps, checks %d/%d",
                   scenario.rawValue, result.outcome, result.wallClockS,
                   result.steps.count, passed, result.checks.count))
        return result
    }

    /// Returns true when the model had to be loaded again (the memory
    /// warning handler unloaded it during an earlier scenario).
    private func ensureLoaded() async throws -> Bool {
        if inference.currentModelId == model.id, inference.state == .ready { return false }
        let wasLoaded = inference.currentModelId == model.id
        await inference.prepare(model: model)
        guard inference.state == .ready, inference.currentModelId == model.id else {
            throw DailyUseError.load(inference.lastError?.detail ?? "\(inference.state)")
        }
        return !wasLoaded
    }

    private func parameters(for scenario: DailyScenario) -> GenerationParameters {
        var p = baseParameters
        p.seed = 42
        if scenario == .sustained { p.maxTokens = DailyFixtures.sustainedMaxTokens }
        return p
    }

    /// Sends one user turn on top of the conversation, the way the chat
    /// does, and records the step. Returns the step (also appended).
    @discardableResult
    private func send(
        _ prompt: String,
        label: String,
        image: Data? = nil,
        in conversation: Conversation,
        scenario: DailyScenario,
        thermal: ThermalTracker
    ) async -> DailyStep? {
        guard !conversation.stopped else { return nil }

        let attachments: [ChatAttachment] = image.map {
            [.image(ImagePreprocessor.normalize($0) ?? $0)]
        } ?? []
        conversation.turns.append(.user(prompt, attachments: attachments))

        let parameters = parameters(for: scenario)
        let requestTurns = SystemPrompt.requestTurns(
            history: conversation.turns,
            systemPrompt: config.systemPrompt,
            capabilities: model.capabilities
        )
        let measured = await generate(turns: requestTurns, parameters: parameters)
        thermal.sample()

        let step = makeStep(
            index: conversation.steps.count,
            label: label,
            prompt: prompt,
            hasImage: image != nil,
            measured: measured,
            parameters: parameters,
            turns: requestTurns
        )
        conversation.steps.append(step)
        log(String(format: "[%@] %@: %@, %d tok, ttft %@ ms, %@ tok/s, peak %.0f MB, %@",
                   scenario.rawValue, label, step.outcome, step.tokens,
                   step.ttftMs.map { String(format: "%.0f", $0) } ?? "-",
                   step.tps.map { String(format: "%.1f", $0) } ?? "-",
                   step.peakMb, step.thermalAfter))

        switch step.outcome {
        case "ok", "empty":
            // The chat stores the trimmed reply; an empty one is shown as an
            // explanation but re-sent as an empty assistant turn.
            conversation.turns.append(.assistant(step.answer))
        default:
            conversation.stopped = true
            if step.outcome == "timeout" {
                try? await Task.sleep(nanoseconds: Self.settleAfterTimeout)
            }
        }
        return step
    }

    // MARK: Scenario bodies

    private func play(
        _ scenario: DailyScenario,
        into c: Conversation,
        result: inout DailyScenarioResult,
        thermal: ThermalTracker
    ) async throws {
        typealias F = DailyFixtures
        typealias K = DailyUseChecks

        switch scenario {

        case .chatRecall:
            for (i, prompt) in F.recallTurns.enumerated() {
                await send(prompt, label: "turn \(i + 1)", in: c, scenario: scenario, thermal: thermal)
            }
            result.reference = "Facts stated in turn 1: " + F.recallFacts.joined(separator: ", ")
            result.scoreMax = F.recallFacts.count
            if let last = c.steps.last, c.steps.count == F.recallTurns.count {
                let recall = K.recalledFacts(in: last.answer, facts: F.recallFacts)
                result.score = recall.score
                for fact in F.recallFacts {
                    result.checks.append(DailyCheck(
                        name: "recalls_\(fact.lowercased())",
                        passed: recall.found[fact] ?? false,
                        detail: (recall.found[fact] ?? false) ? "found in turn 10" : "absent from turn 10"
                    ))
                }
            } else {
                result.checks.append(DailyCheck(name: "recall", passed: false, detail: "turn 10 not reached"))
            }

        case .summarizeLong:
            let step = await send(F.article + "\n\n" + F.summarizeInstruction,
                                  label: "summary", in: c, scenario: scenario, thermal: thermal)
            let bullets = step.map { K.bulletCount(in: $0.answer) } ?? 0
            result.checks.append(DailyCheck(
                name: "four_to_six_bullets",
                passed: step?.outcome == "ok" && (4...6).contains(bullets),
                detail: "\(bullets) bullet lines (asked for 5)"
            ))
            result.reference = "Article: \(K.wordCount(in: F.article)) words, about the history of public libraries."

        case .rewriteEmail:
            let step = await send(F.emailInstruction + "\n\n" + F.sloppyEmail,
                                  label: "rewrite", in: c, scenario: scenario, thermal: thermal)
            let words = step.map { K.wordCount(in: $0.answer) } ?? 0
            result.checks.append(DailyCheck(
                name: "under_120_words",
                passed: step?.outcome == "ok" && words > 0 && words < 120,
                detail: "\(words) words"
            ))

        case .extractJSON:
            let step = await send(F.extractInstruction + "\n\n" + F.appointmentText,
                                  label: "extract", in: c, scenario: scenario, thermal: thermal)
            let json = K.jsonKeys(in: step?.answer ?? "", required: F.extractKeys)
            result.checks.append(DailyCheck(
                name: "json_with_required_keys",
                passed: step?.outcome == "ok" && json.passed,
                detail: json.parsed
                    ? (json.missing.isEmpty ? "all keys present" : "missing: \(json.missing.joined(separator: ", "))")
                    : "no JSON object found"
            ))
            let time = json.values["time"] ?? ""
            result.checks.append(DailyCheck(
                name: "time_is_3_30_pm",
                passed: time.contains("3:30") || time.contains("15:30"),
                detail: "time = \(time.isEmpty ? "(none)" : time)"
            ))
            let people = json.values["people"] ?? ""
            result.checks.append(DailyCheck(
                name: "people_include_ruiz_and_ana",
                passed: people.contains("ruiz") && people.contains("ana"),
                detail: "people = \(people.isEmpty ? "(none)" : people)"
            ))
            result.reference = "Expected: date 2026-03-12 (Thursday, March 12), time 3:30 PM, "
                + "place Riverside Health Clinic, 14 Harbor Road; people Dr. Elena Ruiz and Ana (and the writer)."

        case .translate:
            await send(F.translateToSpanish, label: "spanish", in: c, scenario: scenario, thermal: thermal)
            await send(F.translateToGerman, label: "german", in: c, scenario: scenario, thermal: thermal)
            // Non-empty alone passed an English echo of the paragraph.
            let expected = ["spanish": "es", "german": "de"]
            for step in c.steps {
                let guess = K.detectLanguage(step.answer)
                let language = expected[step.label] ?? "?"
                result.checks.append(DailyCheck(
                    name: "\(step.label)_in_\(language)",
                    passed: step.outcome == "ok" && guess.language == language,
                    detail: "\(K.wordCount(in: step.answer)) words, detected "
                        + "\(guess.language ?? "unclear") (\(guess.detail))"
                ))
            }
            if c.steps.count < 2 {
                result.checks.append(DailyCheck(name: "german_in_de", passed: false, detail: "not reached"))
            }

        case .reasoningShort:
            // Two separate questions, each in a fresh chat.
            let bill = await send(F.billProblem, label: "bill_split", in: c, scenario: scenario, thermal: thermal)
            let billFinal = bill.flatMap { K.finalNumber(in: $0.answer) }
            result.checks.append(DailyCheck(
                name: "bill_split_24_78",
                passed: bill?.outcome == "ok" && billFinal.map { abs($0 - F.billAnswer) < 0.005 } == true,
                detail: "final number \(billFinal.map { String($0) } ?? "none"), expected \(F.billAnswer)"
            ))

            let second = Conversation()
            await inference.resetConversation()
            let zone = await send(F.timeZoneProblem, label: "time_zone", in: second, scenario: scenario, thermal: thermal)
            let offset = c.steps.count
            c.steps.append(contentsOf: second.steps.map { var s = $0; s.index += offset; return s })
            let times = zone.map { K.clockTimes(in: $0.answer) } ?? []
            // "11:00 AM (previous day)" names the right clock time on the
            // wrong day; the call is the same afternoon/morning.
            let dayShift = K.mentionsDayShift(zone?.answer ?? "")
            result.checks.append(DailyCheck(
                name: "time_zone_11_00_am",
                passed: zone?.outcome == "ok"
                    && K.clockAnswerMatches(zone?.answer ?? "", expected: F.timeZoneAnswer,
                                            allowedOthers: F.timeZoneAllowedOthers)
                    && !dayShift,
                detail: "times mentioned: " + (times.isEmpty ? "none" : times.map(Self.clock).joined(separator: ", "))
                    + (dayShift ? "; moves the call to another day" : "; same day")
            ))
            c.stopped = c.stopped || second.stopped
            result.reference = "Expected: 84 x 1.18 = 99.12, / 4 = 24.78 each; 4:00 PM London = 11:00 AM New York."

        case .codeSmall:
            let step = await send(F.codePrompt, label: "dedupe", in: c, scenario: scenario, thermal: thermal)
            let code = K.gradedText(step?.answer ?? "")
            result.checks.append(DailyCheck(name: "has_def", passed: code.contains("def "), detail: ""))
            result.checks.append(DailyCheck(name: "has_return", passed: code.contains("return"), detail: ""))
            result.checks.append(DailyCheck(
                name: "has_docstring",
                passed: code.contains("\"\"\"") || code.contains("'''"),
                detail: "informational"
            ))
            // The code is not run; this reads it for the two failures the
            // grader saw: list(set(x)) and str(item) keys.
            let order = K.dedupeKeepsOrder(step?.answer ?? "")
            result.checks.append(DailyCheck(name: "keeps_order", passed: order.passed, detail: order.reason))

        case .adviceFollowup:
            for (i, prompt) in F.tripTurns.enumerated() {
                await send(prompt, label: "turn \(i + 1)", in: c, scenario: scenario, thermal: thermal)
            }
            result.reference = "Constraints set along the way: 3 days in Porto, 300 euros in total without "
                + "flights (turn 1), a cheaper day 2 (turn 2), vegetarian (turn 3). The turn-6 checklist "
                + "should keep all of them and end with a total at or under 300 euros."
            if c.steps.count >= 3 {
                let adjusted = c.steps[2]
                let meat = K.meatSuggestions(in: adjusted.answer)
                result.checks.append(DailyCheck(
                    name: "turn3_no_meat_or_fish",
                    passed: adjusted.outcome == "ok" && meat.isEmpty,
                    detail: meat.isEmpty ? "none suggested" : "suggests: \(meat.joined(separator: ", "))"
                ))
            }
            guard c.steps.count == F.tripTurns.count, let last = c.steps.last, last.outcome == "ok" else {
                result.checks.append(DailyCheck(
                    name: "checklist", passed: false, detail: "turn \(F.tripTurns.count) not reached"
                ))
                break
            }
            let lines = K.bulletCount(in: last.answer)
            result.checks.append(DailyCheck(
                name: "checklist_is_a_list", passed: lines >= 3, detail: "\(lines) list lines"
            ))
            let veg = K.mentioned(K.vegetarianWords, in: last.answer)
            let meat = K.meatSuggestions(in: last.answer)
            result.checks.append(DailyCheck(
                name: "checklist_vegetarian_food",
                passed: !veg.isEmpty && meat.isEmpty,
                detail: "vegetarian words: " + (veg.isEmpty ? "none" : veg.joined(separator: ", "))
                    + "; meat/fish: " + (meat.isEmpty ? "none" : meat.joined(separator: ", "))
            ))
            let euros = K.amounts(in: last.answer, currency: .euro)
            let top = euros.max()
            result.checks.append(DailyCheck(
                name: "checklist_within_300_euros",
                passed: top.map { $0 <= F.tripBudget } ?? false,
                detail: euros.isEmpty
                    ? "no euro amount in the checklist"
                    : "euro amounts: " + euros.map { String(format: "%g", $0) }.joined(separator: ", ")
            ))

        case .docQA:
            for (i, question) in F.leaseQuestions.enumerated() {
                let prompt = i == 0 ? F.leaseIntro + "\n\n" + F.lease + "\n\n" + question.text : question.text
                await send(prompt, label: question.label, in: c, scenario: scenario, thermal: thermal)
            }
            func answer(_ label: String) -> DailyStep? { c.steps.first { $0.label == label && $0.outcome == "ok" } }
            let dates = answer("dates")?.answer ?? ""
            result.checks.append(DailyCheck(
                name: "dates_apr_1_and_mar_31",
                passed: K.mentionsDate(dates, month: 4, day: 1) && K.mentionsDate(dates, month: 3, day: 31),
                detail: "start \(K.mentionsDate(dates, month: 4, day: 1) ? "found" : "missing"), "
                    + "end \(K.mentionsDate(dates, month: 3, day: 31) ? "found" : "missing")"
            ))
            let rent = answer("rent")?.answer ?? ""
            let dueDay = !K.mentioned(["3rd", "third"], in: rent).isEmpty
            result.checks.append(DailyCheck(
                name: "rent_1450_due_3rd",
                passed: K.containsNumber(1450, in: rent) && dueDay,
                detail: "1450 \(K.containsNumber(1450, in: rent) ? "found" : "missing"), "
                    + "3rd \(dueDay ? "found" : "missing")"
            ))
            let deposit = answer("deposit")?.answer ?? ""
            result.checks.append(DailyCheck(
                name: "deposit_2175_within_21_days",
                passed: K.containsNumber(2175, in: deposit) && K.containsNumber(21, in: deposit),
                detail: "numbers: " + K.numbers(in: deposit).map { String(format: "%g", $0) }.joined(separator: ", ")
            ))
            let pet = answer("pet")?.answer ?? ""
            result.checks.append(DailyCheck(
                name: "pet_one_cat_35",
                passed: K.containsNumber(35, in: pet) && K.mentions("cat", in: pet),
                detail: "numbers: " + K.numbers(in: pet).map { String(format: "%g", $0) }.joined(separator: ", ")
            ))
            let parking = answer("parking")?.answer ?? ""
            let invented = K.amounts(in: parking, currency: .dollar).filter { !F.leaseAmounts.contains($0) }
            let notStated = K.saysNotStated(parking)
            result.checks.append(DailyCheck(
                name: "parking_not_stated",
                passed: !parking.isEmpty && notStated && invented.isEmpty,
                detail: (notStated ? "says the lease does not cover it" : "no 'not stated' wording")
                    + (invented.isEmpty ? "" : "; invented amounts: "
                        + invented.map { String(format: "%g", $0) }.joined(separator: ", "))
            ))
            result.reference = F.leaseReference

        case .unknownEntity:
            for question in F.unknownQuestions {
                let steps = await inFreshChat(c) { chat in
                    await send(question.text, label: question.label, in: chat, scenario: scenario, thermal: thermal)
                }
                let reply = steps.first
                let admits = K.admitsNotKnowing(reply?.answer ?? "")
                result.checks.append(DailyCheck(
                    name: "\(question.label)_admits_unknown",
                    passed: reply?.outcome == "ok" && admits,
                    detail: "\(K.wordCount(in: reply?.answer ?? "")) words; "
                        + (admits ? "says it does not know" : "no admission found - read it for invented details")
                ))
            }
            result.reference = "None of these exist: " + F.unknownQuestions.map(\.label).joined(separator: ", ")
                + ". A good answer says so (or that it cannot verify it) and invents no details."

        case .nonEnglishChat:
            let chats = try F.nonEnglishChats()
            for chat in chats {
                await inFreshChat(c) { fresh in
                    for (i, prompt) in chat.turns.enumerated() {
                        await send(prompt, label: "\(chat.name) \(i + 1)", in: fresh,
                                   scenario: scenario, thermal: thermal)
                    }
                }
            }
            for chat in chats {
                for i in chat.turns.indices {
                    let label = "\(chat.name) \(i + 1)"
                    let step = c.steps.first { $0.label == label }
                    let guess = K.detectLanguage(step?.answer ?? "")
                    result.checks.append(DailyCheck(
                        name: "\(chat.name)_\(i + 1)_in_\(chat.language)",
                        passed: step?.outcome == "ok" && guess.language == chat.language,
                        detail: step == nil ? "not reached"
                            : "detected \(guess.language ?? "unclear") (\(guess.detail))"
                    ))
                }
            }
            result.reference = "Each chat is written in one language; every answer should be in that "
                + "language. Languages: " + chats.map(\.language).joined(separator: ", ") + "."

        case .formatFollow:
            let bullets = await inFreshChat(c) { chat in
                await send(F.formatBullets, label: "three_bullets", in: chat, scenario: scenario, thermal: thermal)
            }.first
            let bulletLines = K.bulletCount(in: bullets?.answer ?? "")
            result.checks.append(DailyCheck(
                name: "exactly_3_bullets",
                passed: bullets?.outcome == "ok" && bulletLines == 3,
                detail: "\(bulletLines) list lines"
            ))

            let table = await inFreshChat(c) { chat in
                await send(F.formatTable, label: "table", in: chat, scenario: scenario, thermal: thermal)
            }.first
            let parsed = K.markdownTable(in: table?.answer ?? "")
            let missing = parsed.map { K.missingColumns(F.formatTableColumns, in: $0) } ?? F.formatTableColumns
            result.checks.append(DailyCheck(
                name: "table_with_columns",
                passed: table?.outcome == "ok" && parsed != nil && missing.isEmpty && (parsed?.rows.count ?? 0) >= 3,
                detail: parsed.map {
                    "headers: \($0.headers.joined(separator: " | ")); \($0.rows.count) rows"
                        + (missing.isEmpty ? "" : "; missing: \(missing.joined(separator: ", "))")
                } ?? "no markdown table"
            ))

            let sentence = await inFreshChat(c) { chat in
                await send(F.formatSentence, label: "one_sentence", in: chat, scenario: scenario, thermal: thermal)
            }.first
            let sentences = K.sentenceCount(in: sentence?.answer ?? "")
            let words = K.wordCount(in: sentence?.answer ?? "")
            result.checks.append(DailyCheck(
                name: "one_sentence_under_20_words",
                passed: sentence?.outcome == "ok" && sentences == 1 && words < 20,
                detail: "\(sentences) sentence(s), \(words) words"
            ))

            let steps = await inFreshChat(c) { chat in
                await send(F.formatSteps, label: "numbered_steps", in: chat, scenario: scenario, thermal: thermal)
            }.first
            let numbers = K.numberedSteps(in: steps?.answer ?? "")
            result.checks.append(DailyCheck(
                name: "numbered_steps",
                passed: steps?.outcome == "ok" && K.stepsAreSequential(numbers),
                detail: numbers.isEmpty ? "no numbered lines" : "numbers: " + numbers.map(String.init).joined(separator: ", ")
            ))

        case .longSession:
            for (i, prompt) in F.longSessionTurns.enumerated() {
                await send(prompt, label: "turn \(i + 1)", in: c, scenario: scenario, thermal: thermal)
            }
            result.reference = "Facts stated in turn 1: " + F.longSessionFacts.joined(separator: ", ")
                + ". Five texts are pasted along the way (an article, an email draft, a lease, a recipe, "
                + "a school letter); the chat outgrows both "
                + "llama.cpp windows (4096 text, 8192 vision), so the engines trim history; "
                + "see each step's dropped_turns / first_turn_kept."
            result.scoreMax = F.longSessionFacts.count
            let trims = c.steps.filter {
                ($0.droppedTurns ?? 0) > 0 || ($0.droppedFrontTokens ?? 0) > 0 || ($0.truncatedTokens ?? 0) > 0
            }
            // The scenario exists to exercise trimming: an engine that
            // reports its trim figures and never trimmed means the chat did
            // not outgrow the window, which is a fault of the scenario.
            // Engines without the figures (MLX, Core ML, Apple) stay
            // informational.
            let reportsTrims = c.steps.contains { $0.droppedTurns != nil }
            result.checks.append(DailyCheck(
                name: "trim_events",
                passed: !reportsTrims || !trims.isEmpty,
                detail: (reportsTrims ? "" : "informational: ")
                    + "\(trims.count) of \(c.steps.count) turns trimmed history"
                    + (trims.first.map { " (first at \($0.label))" } ?? "")
            ))
            let early = Self.median(c.steps.prefix(5).compactMap(\.ttftMs))
            let late = Self.median(c.steps.suffix(5).compactMap(\.ttftMs))
            result.checks.append(DailyCheck(
                name: "ttft_late_vs_early",
                passed: true,
                detail: "informational: median ttft turns 1-5 "
                    + (early.map { String(format: "%.0f ms", $0) } ?? "-") + ", last 5 "
                    + (late.map { String(format: "%.0f ms", $0) } ?? "-")
            ))
            if let last = c.steps.last, c.steps.count == F.longSessionTurns.count {
                let recall = K.recalledFacts(in: last.answer, facts: F.longSessionFacts)
                result.score = recall.score
                let kept = last.context.firstTurnKept
                for fact in F.longSessionFacts {
                    let found = recall.found[fact] ?? false
                    result.checks.append(DailyCheck(
                        name: "recalls_\(fact.lowercased())",
                        passed: found,
                        detail: (found ? "found" : "absent") + " in turn \(F.longSessionTurns.count); first turn "
                            + (kept.map { $0 ? "still in the prompt" : "trimmed away" } ?? "unknown")
                    ))
                }
            } else {
                result.checks.append(DailyCheck(
                    name: "recall", passed: false, detail: "turn \(F.longSessionTurns.count) not reached"
                ))
            }

        case .visionPhotoReal:
            for photo in F.realPhotos {
                guard let data = F.bundledResource(photo.resource) else {
                    throw DailyUseError.image("\(photo.resource) is missing from the test bundle")
                }
                let steps = await inFreshChat(c) { chat in
                    await send(F.describePrompt, label: "\(photo.label) describe", image: data,
                               in: chat, scenario: scenario, thermal: thermal)
                    await send(photo.question, label: "\(photo.label) question",
                               in: chat, scenario: scenario, thermal: thermal)
                }
                let describe = steps.first { $0.label.hasSuffix("describe") }
                let question = steps.first { $0.label.hasSuffix("question") }
                let seen = K.mentioned(photo.describeKeywords, in: describe?.answer ?? "")
                result.checks.append(DailyCheck(
                    name: "\(photo.label)_describe_on_topic",
                    passed: describe?.outcome == "ok" && !seen.isEmpty,
                    detail: "mentions: " + (seen.isEmpty ? "none of " + photo.describeKeywords.joined(separator: "/")
                                                         : seen.joined(separator: ", "))
                ))
                let named = K.mentioned(photo.answerKeywords, in: question?.answer ?? "")
                result.checks.append(DailyCheck(
                    name: "\(photo.label)_question",
                    passed: question?.outcome == "ok" && !named.isEmpty,
                    detail: "expects " + photo.answerKeywords.joined(separator: "/") + "; found: "
                        + (named.isEmpty ? "none" : named.joined(separator: ", "))
                ))
            }
            result.reference = F.realPhotos.map { "\($0.label): \($0.reference)" }.joined(separator: "\n")

        case .visionDescribe:
            guard let photo = DailyFixtures.syntheticPhoto() else { throw DailyUseError.image("photo encode failed") }
            let step = await send(F.describePrompt, label: "describe", image: photo,
                                  in: c, scenario: scenario, thermal: thermal)
            let answer = step?.answer ?? ""
            // Both the colour and the shape: "a checkerboard" alone passed
            // the old shape-or-grid check.
            let shapes = K.mentioned(["rectangle", "rectangles", "rectangular", "square", "squares",
                                      "block", "blocks"], in: answer)
            let orange = K.mentions("orange", in: answer)
            result.checks.append(DailyCheck(
                name: "orange_rectangles",
                passed: step?.outcome == "ok" && orange && !shapes.isEmpty,
                detail: "orange \(orange ? "yes" : "no"), shapes: "
                    + (shapes.isEmpty ? "none" : shapes.joined(separator: ", "))
            ))
            result.reference = DailyFixtures.syntheticPhotoDescription

        case .visionReadText:
            guard let receipt = DailyFixtures.receiptImage() else { throw DailyUseError.image("receipt encode failed") }
            let step = await send(F.receiptPrompt, label: "receipt", image: receipt,
                                  in: c, scenario: scenario, thermal: thermal)
            let answer = step?.answer ?? ""
            result.checks.append(DailyCheck(
                name: "total_10_44",
                passed: step?.outcome == "ok" && K.containsNumber(F.receiptTotal, in: answer),
                detail: "numbers: " + K.numbers(in: answer).map { String($0) }.joined(separator: ", ")
            ))
            result.checks.append(DailyCheck(
                name: "shop_name",
                passed: step?.outcome == "ok" && K.mentions(F.receiptShop, in: answer),
                detail: "expects \"\(F.receiptShop)\""
            ))
            result.reference = DailyFixtures.receiptDescription

        case .sustained:
            for (i, prompt) in F.sustainedPrompts.enumerated() {
                await send(prompt, label: "gen \(i + 1)", in: c, scenario: scenario, thermal: thermal)
            }
            let rates = c.steps.compactMap(\.tps)
            if let first = rates.first, let last = rates.last, first > 0 {
                let drop = (first - last) / first * 100
                result.checks.append(DailyCheck(
                    name: "decode_drop_under_20_percent",
                    passed: drop < 20,
                    detail: String(format: "first %.1f tok/s, last %.1f tok/s, drop %.0f%%; thermal %@",
                                   first, last, drop,
                                   c.steps.map(\.thermalAfter).joined(separator: " > "))
                ))
            }
            result.checks.append(DailyCheck(
                name: "all_eight_completed",
                passed: c.steps.count == F.sustainedPrompts.count && c.steps.allSatisfy { $0.outcome == "ok" },
                detail: "\(c.steps.filter { $0.outcome == "ok" }.count) of \(F.sustainedPrompts.count)"
            ))
        }
    }

    // MARK: Generation

    @MainActor
    private final class WatchdogFlag {
        var value = false
    }

    private struct Measured {
        var text = ""
        var deltas = 0
        var start = Date()
        var end = Date()
        var firstTokenAt: Date?
        var lastTokenAt: Date?
        var reportedTokens: Int?
        var engineTps: Double?
        var engineTtftMs: Double?
        var prefillTokens: Int?
        var prefillMs: Double?
        var cache: PromptCacheReport?
        var rendered: RenderedPrompt?
        var error: String?
        var timedOut = false
        var peakMB: Double = 0
        var modelStillLoaded = true
    }

    /// One generation, fully consumed, under a watchdog. Mirrors
    /// `BenchmarkEngine.generate` (which is private), except that a timeout
    /// keeps the partial text instead of discarding it.
    private func generate(turns: [ChatTurn], parameters: GenerationParameters) async -> Measured {
        let sampler = MemorySampler()
        sampler.start()
        let start = Date()
        let stream = inference.generate(turns: turns, parameters: parameters)

        let work = Task { @MainActor () -> Measured in
            var m = Measured()
            m.start = start
            do {
                for try await output in stream {
                    switch output {
                    case .textDelta(let piece):
                        let now = Date()
                        if m.firstTokenAt == nil { m.firstTokenAt = now }
                        m.lastTokenAt = now
                        m.deltas += 1
                        m.text += piece
                    case .diagnostic(.firstTokenLatency(let ms)):
                        m.engineTtftMs = ms
                    case .diagnostic(.finished(let tokens, let tps)):
                        m.reportedTokens = tokens
                        m.engineTps = tps
                    case .diagnostic(.promptProcessed(let tokens, let ms)):
                        m.prefillTokens = tokens
                        m.prefillMs = ms
                    case .diagnostic(.promptRendered(let rendered)):
                        m.rendered = rendered
                    case .diagnostic(.promptCache(let report)):
                        m.cache = report
                    case .audioChunk:
                        break
                    }
                }
            } catch {
                m.error = error.localizedDescription
            }
            m.end = Date()
            return m
        }

        let fired = WatchdogFlag()
        let timeout = config.timeout
        let watchdog = Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(max(timeout, 1) * 1_000_000_000))
            guard !Task.isCancelled else { return }
            fired.value = true
            work.cancel()
        }

        var measured = await work.value
        watchdog.cancel()
        sampler.stop()
        measured.timedOut = fired.value
        measured.peakMB = sampler.peakMB
        measured.modelStillLoaded = inference.currentModelId == model.id
        return measured
    }

    private func makeStep(
        index: Int,
        label: String,
        prompt: String,
        hasImage: Bool,
        measured m: Measured,
        parameters: GenerationParameters,
        turns: [ChatTurn]
    ) -> DailyStep {
        let answer = m.text.trimmingCharacters(in: .whitespacesAndNewlines)
        let tokens: Int
        let tokenSource: String
        if let reported = m.reportedTokens {
            tokens = reported
            tokenSource = "engine"
        } else if model.backend == .appleIntelligence {
            tokens = BenchmarkEngine.estimatedTokenCount(m.text)
            tokenSource = "estimated_chars"
        } else {
            tokens = m.deltas
            tokenSource = "stream_deltas"
        }

        let outcome: String
        if m.timedOut {
            outcome = "timeout"
        } else if !m.modelStillLoaded {
            outcome = "model_unloaded"
        } else if let error = m.error {
            outcome = DailyUseChecks.errorKind(error)
        } else if answer.isEmpty {
            outcome = "empty"
        } else {
            outcome = "ok"
        }

        var tps: Double?
        if let first = m.firstTokenAt, let last = m.lastTokenAt, tokens > 1 {
            let window = last.timeIntervalSince(first)
            if window > 0 { tps = Double(tokens - 1) / window }
        }

        // Fresh conversation: the engine decoded the whole prompt, so the
        // prefill rate means something. Later turns reuse the prefix cache.
        var prefillTps: Double?
        if turns.count == 1, let pt = m.prefillTokens, let ms = m.prefillMs, ms > 0 {
            prefillTps = Double(pt) / (ms / 1000)
        }

        return DailyStep(
            index: index,
            label: label,
            prompt: prompt,
            hasImage: hasImage,
            answer: answer,
            outcome: outcome,
            error: m.timedOut ? "no complete answer within \(Int(config.timeout)) s" : m.error,
            ttftMs: m.firstTokenAt.map { $0.timeIntervalSince(m.start) * 1000 },
            tokens: tokens,
            tokenSource: tokenSource,
            tps: tps,
            engineTtftMs: m.engineTtftMs,
            engineTps: m.engineTps,
            prefillTokens: m.prefillTokens,
            prefillMs: m.prefillMs,
            prefillTps: prefillTps,
            reusedTokens: m.cache?.reusedTokens,
            evaluatedTokens: m.cache?.evaluatedTokens,
            cacheRewind: m.cache?.rewind.rawValue,
            droppedTurns: m.cache?.droppedTurns,
            droppedFrontTokens: m.cache?.droppedFrontTokens,
            truncatedTokens: m.cache?.truncatedTokens,
            maxTokens: parameters.maxTokens,
            hitTokenCap: tokens >= parameters.maxTokens,
            startsWithReasoning: DailyUseChecks.startsWithReasoning(answer),
            wallMs: m.end.timeIntervalSince(m.start) * 1000,
            peakMb: m.peakMB,
            thermalAfter: ProcessInfo.processInfo.thermalState.reportName,
            context: contextInfo(turns: turns, parameters: parameters, rendered: m.rendered, cache: m.cache)
        )
    }

    // MARK: Context

    /// The window the engine actually opens for this model, and the room it
    /// keeps for the reply - read from the engines' own constants where they
    /// are visible, so a change there shows up here.
    private func contextWindow(maxTokens: Int) -> (tokens: Int?, reserve: Int?) {
        switch model.backend {
        case .llamaCpp:
            #if canImport(llama)
            let ctx = Int(LlamaContext.contextSize)
            #else
            let ctx = model.contextLength
            #endif
            return (ctx, min(maxTokens, ctx / 2))
        case .llamaCppVision:
            #if canImport(llama)
            let ctx = Int(LlamaVisionContext.contextTokens)
            #else
            let ctx = model.contextLength
            #endif
            return (ctx, maxTokens + 64)
        case .coreML:
            let ctx = model.contextLength
            return (ctx > 0 ? ctx : nil, ctx > 0 ? min(maxTokens, ctx / 2) : nil)
        default:
            // MLX and Apple Intelligence open the model's own window and do
            // not trim; the catalog figure is the best available.
            return (model.contextLength > 0 ? model.contextLength : nil, nil)
        }
    }

    private func contextInfo(
        turns: [ChatTurn],
        parameters: GenerationParameters,
        rendered: RenderedPrompt?,
        cache: PromptCacheReport?
    ) -> DailyContextInfo {
        let text = turns.map(\.content).joined(separator: "\n")
        let images = turns.reduce(0) { $0 + $1.attachments.count }
        let estimate = BenchmarkEngine.estimatedTokenCount(text)
            + images * VisionPromptBuilder.imageTokenEstimate
            + turns.count * VisionPromptBuilder.perTurnOverhead
        let window = contextWindow(maxTokens: parameters.maxTokens)
        var overflow: Bool?
        if let ctx = window.tokens {
            overflow = estimate > ctx - (window.reserve ?? 0)
        }

        // What the engine actually kept, not just what it rendered: the
        // text engine trims tokens after rendering (see
        // `DailyUseChecks.firstTurnKept`).
        // Apple Intelligence's rendered prompt holds only the new message:
        // the session keeps the earlier turns itself and says nothing about
        // what it kept, so there is no evidence either way.
        let firstKept = rendered?.source == .session ? nil : DailyUseChecks.firstTurnKept(
            turns: turns,
            renderedText: rendered?.text,
            cache: cache
        )

        return DailyContextInfo(
            contextTokens: window.tokens,
            replyReserve: window.reserve,
            promptTokensEstimate: estimate,
            turnsSent: turns.count,
            overflowExpected: overflow,
            firstTurnKept: firstKept,
            renderedPromptChars: rendered?.text.count,
            templateSource: rendered?.source.label
        )
    }

    // MARK: Conditions

    /// Waits for the conditions the policy asks for (`RunConditionsWaiter`
    /// logs every wait). On a stop, remembers the reason and prints the
    /// `DAILY_NIGHT_STOP` line the night script looks for, once.
    private func gateConditions(checkThermal: Bool) async -> RunConditionsOutcome {
        let thermal = checkThermal && config.thermalGate
        guard thermal || config.conditions.powerGate else { return .clear }
        let outcome = (try? await RunConditionsWaiter.wait(
            policy: config.conditions,
            checkThermal: thermal,
            log: { self.log($0) }
        )) ?? .clear
        if outcome.stop, stopReason == nil {
            let reason = outcome.issues.joined(separator: "; ")
            stopReason = reason
            print("DAILY_NIGHT_STOP " + reason)
        }
        return outcome
    }

    // MARK: Download

    /// Same ownership rules as `BuildTestRunner.downloadAndWait` (private
    /// there): the run owns the files - and deletes them afterwards - only
    /// once it has seen bytes move on its behalf, or when the directory was
    /// empty before it started. The marker goes on disk the moment
    /// ownership is claimed, so a crash leaves proof behind.
    private func download(ownedByRun owned: inout Bool) async throws {
        let downloader = ModelDownloader.shared
        let startedEmpty = !BuildTestRunner.hasFilesOnDisk(model)
        log(String(format: "downloading %@ (~%.1f GB)", model.id, model.diskSizeGB))
        downloader.download(model)
        await Task.yield()
        try? await Task.sleep(nanoseconds: 250_000_000)

        var deadline = Date().addingTimeInterval(BuildTestRunner.downloadTimeout)
        var compileGranted = false
        var lastLog = Date.distantPast
        var firstRead = true
        var stallWatch = config.conditions.downloadStallTimeout.map { DownloadStallWatch(timeout: $0) }
        let transferStart = Date()

        while Date() < deadline {
            var observedBytes: Int64?
            var compiling = false
            switch downloader.state(for: model) {
            case .installed:
                if !owned, startedEmpty {
                    owned = true
                    BuildTestRunner.markOwned(model)
                }
                log("installed")
                return
            case .error(let message):
                if firstRead {
                    firstRead = false
                    try await Task.sleep(nanoseconds: 2_000_000_000)
                    continue
                }
                throw DailyUseError.download(message)
            case .downloading(let fraction, let done, let total):
                observedBytes = done
                if !owned {
                    owned = true
                    BuildTestRunner.markOwned(model)
                }
                if Date().timeIntervalSince(lastLog) >= 30 {
                    lastLog = Date()
                    log(String(format: "download %.0f%% (%.2f / %.2f GB)", fraction * 100,
                               Double(done) / 1_073_741_824, Double(total) / 1_073_741_824))
                }
            case .compiling(let fraction):
                compiling = true
                if !owned {
                    owned = true
                    BuildTestRunner.markOwned(model)
                }
                if !compileGranted {
                    compileGranted = true
                    deadline = Date().addingTimeInterval(BuildTestRunner.compileTimeout)
                }
                if Date().timeIntervalSince(lastLog) >= 30 {
                    lastLog = Date()
                    log(String(format: "compiling %.0f%%", fraction * 100))
                }
            case .fetchingManifest, .idle:
                break
            }
            firstRead = false
            if stallWatch != nil {
                let elapsed = Date().timeIntervalSince(transferStart)
                if compiling {
                    stallWatch?.reset(at: elapsed)
                } else if stallWatch?.observe(bytes: observedBytes, elapsed: elapsed) == true {
                    let network = NetworkPathWatcher.shared.stateDescription
                    log(String(format: "download: no bytes for %.0f s, network %@ - giving up on this model",
                               stallWatch?.idle(at: elapsed) ?? 0, network))
                    log("[conditions] " + MeasurementConditions.logLine())
                    throw DailyUseError.download("stalled: no bytes for "
                        + "\(Int(stallWatch?.timeout ?? 0)) s (network \(network))")
                }
            }
            try await Task.sleep(nanoseconds: 2_000_000_000)
        }
        throw DailyUseError.download("timed out")
    }

    // MARK: Output

    private func persist() {
        guard let fileURL else { return }
        summary.wallClockS = Date().timeIntervalSince(startedAt)
        let file = DailyFile(summary: summary, scenarios: results)
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        if let data = try? encoder.encode(file) {
            try? data.write(to: fileURL, options: .atomic)
        }
    }

    static func encode<T: Encodable>(_ value: T) -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        guard let data = try? encoder.encode(value), let s = String(data: data, encoding: .utf8) else {
            return "{\"error\":\"encoding failed\"}"
        }
        return s
    }

    static func summarize(_ r: DailyScenarioResult) -> DailyScenarioSummary {
        let ok = r.steps.filter { $0.outcome == "ok" }
        return DailyScenarioSummary(
            scenario: r.scenario,
            outcome: r.outcome,
            stepsOk: ok.count,
            stepsTotal: r.steps.count,
            checksPassed: r.checks.filter(\.passed).count,
            checksTotal: r.checks.count,
            score: r.score,
            scoreMax: r.scoreMax,
            medianTtftMs: median(ok.compactMap(\.ttftMs)),
            medianTps: median(ok.compactMap(\.tps)),
            maxPeakMb: r.steps.map(\.peakMb).max(),
            wallClockS: r.wallClockS
        )
    }

    static func median(_ values: [Double]) -> Double? {
        guard !values.isEmpty else { return nil }
        let s = values.sorted()
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    static func clock(_ minutes: Int) -> String {
        String(format: "%02d:%02d", minutes / 60, minutes % 60)
    }

    private func log(_ line: String) {
        print("[daily-use] " + line)
    }
}

// MARK: - Fixtures

/// Every input the suite sends. Fixed, English, written for this file.
enum DailyFixtures {

    // MARK: chat_recall

    static let recallFacts = ["Priya", "Lisbon", "Ziggy"]

    static let recallTurns = [
        "Hi! Quick intro before we chat: my name is Priya, I live in Lisbon, and I have a cat named Ziggy. How is your day going?",
        "I want to cook something quick for dinner tonight with chickpeas, spinach and rice. Any idea? Keep it short.",
        "Sounds good. What could I do this weekend if it rains the whole time? Give me three ideas.",
        "Can you explain what inflation is in simple words, as if I were twelve?",
        "Tell me a short, clean joke about computers.",
        "I keep forgetting to drink water during the day. Any practical tips?",
        "What is the difference between a virus and a bacterium, briefly?",
        "I am thinking of learning a musical instrument as an adult. Which one is easiest to start with, and why?",
        "Give me a two-sentence pep talk for a Monday morning.",
        "Thanks! One last thing: can you remind me what my name is, which city I live in, and what my cat is called?"
    ]

    // MARK: summarize_long

    static let summarizeInstruction = "Summarize in 5 bullet points."

    /// Original text written for this suite (about 1,750 words).
    static let article = """
    A Short History of the Public Library

    Most towns today have a building where anyone can walk in, sit down, and read for free. It is easy to treat that as a natural feature of civic life, like a post office or a park. In fact the public library is a fairly recent invention, and the idea that books should be available to everyone, paid for by everyone, took a long time to win acceptance. Its history is a story about technology, money, education and changing ideas of who deserves access to knowledge.

    Early collections

    Collections of written records are almost as old as writing itself. In the ancient Near East, temples and palaces kept clay tablets that recorded laws, contracts, prayers and astronomical observations. These archives were working tools for priests and officials rather than places for the public to read. The same was true of the famous library at Alexandria in Egypt, founded in the third century BCE. It gathered an enormous number of scrolls and attracted scholars from across the Mediterranean, but it served a small learned community under royal patronage. Ordinary residents of the city had no reason to expect a seat in its reading rooms.

    Rome came somewhat closer to the modern idea. Wealthy Romans kept private libraries in their villas, and several emperors and politicians paid for libraries attached to public buildings, including the large bath complexes. Educated citizens could consult works in Greek and Latin there. Literacy, however, was limited, and the books themselves were handwritten copies that took weeks to produce. A library was a sign of prestige as much as a service.

    Monasteries and universities

    After the decline of the western Roman Empire, much of the work of preserving texts in Europe moved to monasteries. Monks copied manuscripts by hand in rooms set aside for the purpose, and a monastery's book collection might number only a few hundred volumes. Books were so valuable that some were chained to desks to prevent theft. Similar traditions of copying and collecting existed elsewhere: great libraries grew in cities such as Baghdad, Cordoba and Cairo, and scholars in those centers translated, studied and extended works of philosophy, mathematics and medicine.

    From the twelfth century onward, the new universities of Europe built their own collections. Students and teachers needed texts, and colleges began to lend books under strict rules. Access was still restricted to members of the institution, and the general population, most of whom could not read, played no part in this world.

    The printing press changes the economics

    The arrival of movable type printing in Europe in the middle of the fifteenth century changed the cost of books dramatically. A printed edition could produce hundreds of identical copies in the time it once took to copy one by hand. Over the following centuries, books became cheaper, more varied and more common. Pamphlets, almanacs, religious works and later newspapers reached people who had never owned a manuscript.

    Cheaper books did not immediately produce public libraries. Many collections remained private, owned by aristocrats, churches, scholars or professional bodies. Some city governments and churches opened small reference collections, and a few generous donors left their books to a town on condition that local people could use them. These early experiments show that the idea of shared reading was in the air, even if it was not yet organized on a large scale.

    Subscription and lending libraries

    In the eighteenth century a new model appeared: the subscription library. Members paid a joining fee and an annual charge, and in return they could borrow from a shared collection that no single member could have afforded alone. One well known example was founded in Philadelphia in 1731 by a group of tradesmen and artisans who wanted access to books on science, history and practical subjects. Similar societies appeared in many towns in Britain and North America.

    At around the same time, commercial circulating libraries became popular. Shops rented out novels and other light reading for a small fee per volume or per season. Critics worried that these businesses encouraged frivolous reading, but they proved that there was a large demand for books among people who could not buy them. Mechanics' institutes, created in the nineteenth century to educate working men in science and technical subjects, also ran libraries for their members.

    All of these institutions had one thing in common: they depended on fees. That kept out the poorest readers, the very people that reformers increasingly believed would benefit most from books.

    Tax-supported libraries

    The decisive step was the decision to fund libraries from local taxes and to open them free of charge to all residents. In the middle of the nineteenth century, legislation in Britain allowed towns to raise a small local rate to establish free public libraries, although the law was cautious and many towns were slow to act. In the United States, several towns and cities opened free libraries in the same period, and the large municipal library that opened in Boston in the 1850s became an influential example. Its supporters argued that a democracy needed educated voters and that a free library was a natural partner to free public schools.

    These arguments mixed idealism with social anxiety. Some advocates hoped that libraries would offer workers an alternative to taverns and keep young people away from trouble. Others saw them as engines of self-improvement and economic opportunity. Whatever the motive, the principle that the community as a whole should pay for open access to books slowly became accepted.

    The age of large donations

    Late in the nineteenth century and early in the twentieth, private philanthropy accelerated the spread of public libraries. The best known donor was the industrialist Andrew Carnegie, who funded the construction of more than two thousand library buildings in the English-speaking world. His foundation usually paid for the building itself, while the town had to provide the land and promise to pay for books, staff and upkeep through local taxes. This arrangement pushed many communities to commit public money to libraries for the first time.

    Many of these buildings still stand. They often share a recognizable style: a raised entrance reached by a flight of steps, a central desk, and reading rooms arranged so that a single librarian could supervise the whole space.

    Open shelves and new methods

    Early public libraries often kept their books in closed stacks. Readers looked up a title in a catalog, filled in a request, and waited while an attendant fetched the volume. Toward the end of the nineteenth century, librarians began to experiment with open shelves that readers could browse themselves. This change required consistent ways to arrange books by subject, and classification systems such as the Dewey Decimal Classification, first published in 1876, became widely used. Card catalogs, standard rules for describing books, and professional training schools for librarians all developed in the same period.

    Libraries also began to think about particular groups of readers. Separate children's rooms, story hours and collections of picture books became common in the early twentieth century. Branch libraries brought services into neighborhoods, and traveling collections and bookmobiles carried books to rural areas that could not support a building of their own.

    Not every reader was welcomed equally. In some places, libraries were segregated by race or excluded certain groups entirely, and campaigns for equal access were part of wider struggles for civil rights. The modern professional commitment to serving everyone without discrimination was shaped by those campaigns.

    Libraries in the twentieth century

    During the twentieth century, public libraries took on many roles beyond lending books. They provided reference services, answering questions in person and by telephone. They collected local history, newspapers and government documents. They offered spaces for public lectures, clubs and exhibitions. During times of economic hardship, they were used by job seekers and by people who could not afford to buy books or heat their homes during the day.

    New formats arrived steadily: recorded music, films, audiobooks, and later computer software. Each new format raised questions about what a library should collect and how it should lend items that were expensive or easily damaged. Libraries also became places where the public first encountered new technology. In many communities the library was the first place where ordinary residents could use a computer or connect to the internet.

    The digital era

    The spread of the internet led many observers to predict the end of the public library. If information could be found online, why would anyone visit a building full of paper? The reality turned out to be more complicated. Libraries adapted by offering electronic books and online databases, free internet access, and help for people who lacked digital skills. For residents without a home connection or a computer, the library became an essential point of access to government services, job applications and education.

    Many libraries also redefined themselves as community spaces. Modern buildings often include meeting rooms, study areas, cafes, and so-called makerspaces with equipment such as 3D printers or sound recording studios. Programs for early literacy, language learning, and help with forms or taxes are common. Librarians increasingly describe their work as connecting people with information and with each other, rather than simply guarding a collection.

    The digital shift brought new challenges as well. Licensing terms for electronic books are often more restrictive and more expensive for libraries than for individual buyers. Budgets have come under pressure in many places, and debates continue over opening hours, staffing and the balance between physical and digital collections. Questions of privacy have also become more pressing, since online services can record what people read in ways that a paper card catalog never did.

    Why the history matters

    The long history of the public library shows that free access to reading was not inevitable. It grew out of a series of choices: to print books cheaply, to pool resources through subscriptions, to fund libraries through taxes, and to welcome readers regardless of income or background. Each step was argued over, and each depended on people who believed that knowledge should not be reserved for a few.

    Today the public library remains one of the few indoor spaces where anyone can stay without being expected to buy something. Its form will keep changing with technology, but its central promise has stayed remarkably stable for more than a century and a half: that a community is stronger when everyone in it can learn, read and ask questions freely.
    """

    // MARK: rewrite_email

    static let emailInstruction =
        "Rewrite this email so it is polite and professional, in under 120 words. Reply with the rewritten email only."

    static let sloppyEmail = """
    hey mark,, so um the report thing u wanted by friday?? not gonna happen tbh, the numbers from sales came in late AGAIN and half of them are wrong lol. i can prob do tuesday if they fix it by monday morning. also can we move our 1on1 bc i have the dentist at 3. lmk asap thx
    - jen
    """

    // MARK: extract_json

    static let extractKeys = ["date", "time", "place", "people"]

    static let extractInstruction =
        "Extract the appointment from the text below as JSON with exactly these keys: date, time, place, people. Reply with the JSON only."

    static let appointmentText = """
    Quick note for my calendar: Dr. Elena Ruiz moved my annual checkup to Thursday, March 12, 2026, at 3:30 PM. It will be at the Riverside Health Clinic, 14 Harbor Road, and my sister Ana is coming with me.
    """

    // MARK: translate

    static let translateParagraph = """
    The train to the coast leaves every morning at eight. I usually buy a coffee at the station before I get on. Last week the train was late because of a storm. Next time I will check the schedule before I leave home.
    """

    static let translateToSpanish = "Translate this paragraph into Spanish:\n\n" + translateParagraph
    static let translateToGerman = "Now translate the same paragraph into German."

    // MARK: reasoning_short

    static let billProblem =
        "Four friends have dinner. The bill is $84.00 and they want to add an 18% tip, then split the total equally. How much does each person pay?"
    static let billAnswer = 24.78

    static let timeZoneProblem =
        "My colleague in London wants a call at 4:00 PM London time. I am in New York, which is 5 hours behind London. What time is the call for me?"
    /// 11:00 AM, in minutes after midnight; 4:00 PM may be repeated back.
    static let timeZoneAnswer = 11 * 60
    static let timeZoneAllowedOthers: Set<Int> = [16 * 60]

    // MARK: code_small

    static let codePrompt =
        "Write a Python function that removes duplicates from a list while keeping the original order. Include a docstring."

    // MARK: advice_followup

    static let tripBudget = 300.0

    static let tripTurns = [
        "I'm planning a 3-day trip to Porto with a total budget of 300 euros for everything except the flights: accommodation, food, transport and activities. Can you suggest a day-by-day plan with rough costs?",
        "Can you make day 2 cheaper? I'd rather spend less that day.",
        "I forgot to mention that I'm vegetarian. Please adjust the food suggestions.",
        "What's one free thing I could do on the first evening?",
        "Is it worth buying a public transport pass for these three days, or should I just walk?",
        "Great, thanks. Now summarise the whole plan as a short checklist I can save on my phone, with the total estimated cost at the end."
    ]

    // MARK: doc_qa

    static let leaseIntro =
        "Here is my apartment lease. I'm going to ask you a few questions about it. Please answer only from the lease."

    struct Question {
        let label: String
        let text: String
    }

    static let leaseQuestions = [
        Question(label: "dates", text: "First question: on what date does the lease start, and on what date does it end?"),
        Question(label: "rent", text: "How much is the monthly rent, and by which day of the month do I have to pay it?"),
        Question(label: "deposit", text: "How much is the security deposit, and how many days does the landlord have to return it after I move out?"),
        Question(label: "pet", text: "Can I keep a cat in the apartment? Does it cost anything extra?"),
        Question(label: "parking", text: "Last one: is a parking space included, and how much does it cost per month?")
    ]

    /// Every dollar amount in the lease; any other amount in the parking
    /// answer was invented.
    static let leaseAmounts: Set<Double> = [1450, 75, 2175, 35, 100_000, 50, 150]

    static let leaseReference = """
    Expected: starts April 1, 2027, ends March 31, 2028; rent $1,450 due on or before the 3rd; \
    deposit $2,175 returned within 21 days of move-out; one cat allowed with written notice, \
    $35 a month pet rent (no dogs); parking is not mentioned anywhere in the lease, so the last \
    answer should say the lease does not cover it and must not invent a price.
    """

    /// Original text written for this suite (about 900 words). Fictional
    /// parties and address. It deliberately says nothing about parking,
    /// cars or vehicles.
    static let lease = """
    RESIDENTIAL LEASE AGREEMENT

    This Residential Lease Agreement ("Lease") is made between Harborview Property Partners ("Landlord") and Jordan Ellis ("Tenant") for the apartment known as Unit 4B, 218 Alder Lane, Brookfield ("Premises").

    1. Term. The Lease begins on April 1, 2027 and ends on March 31, 2028. If Tenant wishes to renew, Tenant must give Landlord written notice at least 60 days before the end date. If neither party gives notice, the Lease continues month to month on the same terms, and either party may then end it with 30 days' written notice.

    2. Rent. Monthly rent is $1,450, payable in advance on or before the 3rd day of each month. Rent may be paid by bank transfer to the account Landlord names in writing, or by check delivered to the management office. Cash is not accepted. Rent for any partial month at the start of the Lease is prorated by the day.

    3. Late payment. If rent is not received by the end of the 3rd day of the month, Tenant will pay a late fee of $75. A returned check or failed transfer is treated as unpaid rent, and Tenant will also pay any bank charge Landlord incurs because of it. Repeated late payment is a breach of this Lease.

    4. Security deposit. Tenant will pay a security deposit of $2,175 before receiving the keys. Landlord will hold the deposit in a separate account and will not use it for rent while Tenant lives in the Premises. Within 21 days after Tenant moves out and returns all keys, Landlord will return the deposit, less any amounts needed to repair damage beyond normal wear and tear or to cover unpaid rent. Landlord will give Tenant an itemized list of any deductions with the remaining balance.

    5. Utilities. Landlord pays for water, sewer and trash collection. Tenant is responsible for electricity, gas, internet and any other service Tenant chooses to set up. Tenant must transfer the electricity and gas accounts into Tenant's own name before the start date and keep them active until the end of the Lease.

    6. Use of the Premises. The Premises may be used only as a private residence for Tenant and the members of Tenant's immediate household. Guests may stay for no more than 14 nights in any six-month period without Landlord's written consent. Tenant may not run a business that brings customers to the building, and may not sublet the Premises or list it on a short-term rental platform.

    7. Pets. Tenant may keep one cat in the Premises, provided Tenant notifies Landlord in writing before the cat moves in and keeps the cat's vaccinations up to date. Pet rent for the cat is $35 per month, added to the monthly rent. Dogs, reptiles and other animals are not permitted. Tenant is responsible for any damage caused by the cat, including damage to carpets, doors and window screens.

    8. Maintenance and repairs. Tenant will keep the Premises clean and in good condition and will promptly report any leak, electrical fault, broken appliance or sign of pests to the management office. Landlord will make necessary repairs within a reasonable time after being notified. Tenant will pay for repairs caused by the negligence of Tenant, household members or guests. Tenant may not paint, install shelves, change locks or make other alterations without Landlord's written permission.

    9. Entry by Landlord. Landlord may enter the Premises to inspect, make repairs or show the unit to prospective tenants or buyers, after giving Tenant at least 24 hours' notice. In an emergency, such as a fire or a serious water leak, Landlord may enter without notice. Landlord will make reasonable efforts to enter between 9 a.m. and 6 p.m. on weekdays.

    10. Noise and conduct. Tenant and guests will not disturb other residents. Quiet hours are from 10 p.m. to 7 a.m. every day. Smoking of any kind, including electronic cigarettes, is prohibited inside the Premises, on balconies and in all shared hallways.

    11. Shared areas. The laundry room on the ground floor is available to all residents from 7 a.m. to 10 p.m. Each unit has one lockable storage cage in the basement; Tenant's cage is number 12. Bicycles must be kept in the bicycle room and not in hallways or stairwells. Tenant may not leave furniture, rubbish or personal items in any shared area.

    12. Insurance. Tenant must carry renter's insurance that covers Tenant's personal belongings and personal liability of at least $100,000 for the full term of the Lease, and must give Landlord a copy of the policy before moving in. Landlord's insurance does not cover Tenant's belongings.

    13. Keys and locks. Landlord will give Tenant two apartment keys, one mailbox key and one building access card. A replacement access card costs $50. If Tenant loses a key and asks Landlord to change the lock, Tenant will pay the locksmith's charge.

    14. Ending the Lease early. Tenant may end the Lease before the end date by giving 60 days' written notice and paying an early termination fee equal to one month's rent. Tenant remains responsible for rent until the later of the notice period ending or the date the Premises is returned in the condition required by this Lease.

    15. Moving out. At the end of the Lease, Tenant will remove all belongings, clean the Premises, and return every key and access card to the management office. A professional carpet cleaning is required if the cat has lived in the Premises; if Tenant does not provide a receipt for it, Landlord may deduct a cleaning charge of $150 from the deposit.

    16. Notices. All notices under this Lease must be in writing and delivered by hand to the management office or sent by email to the address Landlord provides at signing.

    17. Entire agreement. This Lease is the entire agreement between Landlord and Tenant. Any change must be in writing and signed by both parties.

    Signed by Harborview Property Partners, Landlord, and Jordan Ellis, Tenant.
    """

    // MARK: unknown_entity

    /// Plausible but invented for this suite: no such paper, phone or
    /// person exists. A model that knows it does not know is the goal; if
    /// one of them ever turns out to be real, replace it.
    static let unknownQuestions = [
        Question(
            label: "paper",
            text: "Can you summarise the main findings of the 2019 paper \"Recursive Gradient Folding for Sparse Transformers\" by Halvorsen and Mbeki?"
        ),
        Question(
            label: "phone",
            text: "What are the camera and battery specs of the Nokia Lumina X12 Pro that came out in 2024?"
        ),
        Question(
            label: "person",
            text: "Who is Dr. Marguerite Okonkwo-Lindqvist, and what is she best known for?"
        )
    ]

    // MARK: non_english_chat

    struct NonEnglishChat: Decodable {
        /// ISO 639-1 code `DailyUseChecks.detectLanguage` returns.
        let language: String
        let name: String
        let turns: [String]
    }

    /// The prompts live in `DailyUseNonEnglish.json` (test bundle), not in
    /// Swift source: the project is English-only, and this scenario is the
    /// one place where the test data itself has to be in another language.
    static func nonEnglishChats() throws -> [NonEnglishChat] {
        guard let data = bundledResource("DailyUseNonEnglish.json") else {
            throw DailyUseError.fixture("DailyUseNonEnglish.json is missing from the test bundle")
        }
        struct File: Decodable { let chats: [NonEnglishChat] }
        return try JSONDecoder().decode(File.self, from: data).chats
    }

    // MARK: format_follow

    static let formatBullets =
        "Give me exactly 3 bullet points with tips for sleeping better. No introduction and no conclusion."
    static let formatTable =
        "Make a markdown table comparing a bicycle, a bus and a car for a 5 km daily commute, with the columns Option, Cost, Time and Pros."
    static let formatTableColumns = ["Option", "Cost", "Time", "Pros"]
    static let formatSentence =
        "In one sentence of fewer than 20 words, explain what a password manager does."
    static let formatSteps = "Give me numbered steps to boil an egg."

    // MARK: long_session

    static let longSessionFacts = ["Marco", "nurse", "Glasgow"]

    /// 40 turns, long enough to outgrow both llama.cpp windows whatever the
    /// model's reply length. The 2026-10-24 run's shortest replies here
    /// averaged 18 tokens (Gemma 3); with those the last prompt is about
    /// 8,200 tokens (Llama 3 tokenizer), about 9,300 with 50-token replies -
    /// against prompt room of 3,584 tokens in the text engine (4096 window)
    /// and 7,616 in the vision engine (8192; it trims by an estimate of two
    /// tokens per five characters, which crosses its budget near turn 25).
    /// The bulk is the user's side:
    /// messages of a few sentences and five pasted texts (`article` at turn
    /// 12, `longSessionEmailDraft` at 18, `lease` at 25,
    /// `longSessionRecipe` at 33, `longSessionSchoolLetter` at 37), plus
    /// replies asked to run longer (turns 18, 19, 28, 31 and 33). The facts
    /// of turn 1 are never repeated, so turn 40 can only answer them from
    /// turn 1.
    static let longSessionTurns = [
        "Hi! I'm Marco, I work as a nurse in Glasgow. I just want to chat for a while - please keep your answers short, two or three sentences.",
        "It rained all day here again, the thin sideways kind that gets under any umbrella. I walked home anyway because the buses were packed and everyone smelled of wet coats. Do you like rainy days, or are they just something to get through until the weekend?",
        "When I got in I wanted something warm, but I've already had about six cups of tea today and I'm starting to feel it. What's a good hot drink for a cold evening, other than tea? Ideally something I can make with what's usually in a kitchen, no special syrups or gadgets.",
        "I'm trying to read more books this year. I read all the time as a teenager, then work got busy and I just stopped, and now I fall asleep after two pages. Any easy, gripping novel you'd recommend to get back into it? Nothing too long, I'd like to actually finish it this time.",
        "Do you think it's better to exercise in the morning or in the evening? My schedule changes every week, so some weeks I'm free in the morning and other weeks only late at night, and I never know which habit to build. I've started and stopped three times since January.",
        "I've been thinking about getting a houseplant that's hard to kill. My flat faces north and gets very little direct sun, the heating is either off or roasting, and I'm often away for twelve hours at a time, sometimes longer. Ideas? The last two plants I had did not survive me.",
        "Random question: what's a fun fact about octopuses? My niece is eight and completely obsessed with them, she has octopus socks and an octopus lunchbox, and I want to impress her next time she visits.",
        "My friend says pineapple on pizza is a crime, and we argued about it for half an hour at lunch while everyone else rolled their eyes. I think it's fine with ham and a bit of chilli. What do you think, and is there a good argument I can use next time?",
        "How can I stop checking my phone first thing in the morning? The alarm goes off, I pick up the phone to switch it off, and twenty minutes later I'm still in bed reading the news and some argument on social media, and I already feel worse than when I woke up.",
        "Any tips for falling asleep after a night shift? I get home around eight in the morning, the street outside is noisy with school traffic, the light comes through the curtains, and my brain is still wide awake going over everything that happened.",
        "What's a simple dinner I can make in 15 minutes? I usually have eggs, rice, frozen peas, a tin of tomatoes, some cheese and whatever vegetables are left at the end of the week, and I'm too tired to do much more than chop one onion.",
        "A friend who volunteers at our local library sent me this article and wants to know what I think of it. Can you give me the gist in three sentences?\n\n" + article,
        "Thanks. Which part of that history surprised you the most? I had no idea libraries once charged people to borrow books, I always assumed they had been free forever.",
        "Do you think public libraries will still exist in thirty years, now that so much is online and everyone reads on their phones? Be honest, I won't tell my friend what you said.",
        "Recommend a podcast topic for a long bus ride. I've got about an hour each way on Saturday to visit my parents, and I'm bored of true crime, I've listened to too much of it and now every quiet street looks suspicious.",
        "Is it worth learning to touch type at my age? I'm in my thirties and I type with four fingers, fast enough for emails, but I keep looking down at the keyboard and making silly mistakes in the notes I write at work.",
        "What's a nice day trip idea for a sunny Saturday? I don't have a car, so it has to be somewhere I can reach by train or bus, and I'd like to be outdoors most of the day, maybe somewhere with water and a decent cafe.",
        "My manager asks me to cover extra shifts almost every week and I always say yes. I drafted this email to finally say no for a while. Can you make it shorter and kinder? Only the new version, please.\n\n" + longSessionEmailDraft,
        "Could you write me a short story, about 200 words, about a lighthouse keeper and a cat who turns up one stormy night? It's for my niece, so keep it gentle, with a happy ending.",
        "She'll love that. What would be a good name for the cat if we ever turn it into a series? Give me three options, and tell me which one you like best.",
        "Tell me a short riddle, something my niece could solve too.",
        "Okay, I give up - what's the answer?",
        "My upstairs neighbour has started learning the drums. He's nice, and he always stops by nine in the evening, but after nights I sleep during the day and he practises exactly then. How would you bring it up without starting a feud?",
        "What's one small habit that makes a big difference? Something I could start this week without buying anything or downloading another app.",
        "Different topic: my sister is about to sign a lease for her first apartment and sent it to me to look over. She has a cat. Is there anything in it she should watch out for? Keep it short.\n\n" + lease,
        "How much will she have to pay before she moves in or in the first month, counting the deposit and the cat?",
        "Thanks, I'll pass that on. She's nervous about living alone for the first time, she has always shared with friends. Any advice I can give her?",
        "Back to me. I want to cook at home more instead of buying lunch every day, which costs a fortune. Could you give me a simple plan for prepping five lunches on a Sunday in under two hours, as a short list?",
        "What's a good way to save a little money each month without feeling like I'm on a strict budget? I've tried spreadsheets and I give up after a week.",
        "I've been meaning to learn some Spanish before a holiday next spring. I know maybe twenty words from school. What's the most useful thing to learn first, in your opinion?",
        "Could you write a friendly note, about 150 words, that I could slip under the drummer's door? Keep it warm and suggest we find practice times that work for both of us.",
        "Perfect. Do you think a small gift with the note would be too much? Maybe a box of biscuits, or is that weird?",
        "I found this recipe online, but it serves eight and I live alone. Can you halve the ingredient amounts for me? Just the ingredient list, please.\n\n" + longSessionRecipe,
        "What's a good stretch for a stiff lower back after standing all day? Something I can do in two minutes in the kitchen while the kettle boils.",
        "Do you know any trick for remembering people's names? I meet a lot of new people at work and forget their names within seconds, which is embarrassing when they remember mine.",
        "My niece asked me why the sky is blue and I gave her a terrible answer about the sea reflecting. How would you explain it to an eight-year-old?",
        "I'm looking after my niece next week and her school just sent this letter. What do I need to do, and by when?\n\n" + longSessionSchoolLetter,
        "Is it normal to feel tired on my days off even when I sleep a lot? It's been like that for a few weeks, and by the afternoon I just want to lie on the sofa.",
        "Thanks, this was a really nice chat. You've given me a lot of ideas, and I'll definitely try the drink and the plant.",
        "Before I go: what's my name, what's my job, and which city do I live in?"
    ]

    /// Original text written for this suite (about 330 words): an email the
    /// `long_session` user drafted and pastes to have it rewritten. It names
    /// nobody and says nothing of the user's job or city (turn 1's facts).
    static let longSessionEmailDraft = """
    Subject: Extra shifts

    Hi Karen,

    I hope you're well and that the new rota is settling down. I wanted to write to you about something that has been on my mind for a few weeks now, and I thought an email would be easier than trying to catch you between everything else on a busy day.

    As you know, I've said yes to almost every extra shift you've asked me to cover since the end of the summer. I counted, and it's been eleven extra shifts since August, on top of my normal hours. I've been happy to help, because I know how short the team has been and I don't like leaving people in a difficult position. But I'm finding that I'm really tired, I'm not sleeping properly, and I've had to cancel plans with my family more than once, which I feel bad about.

    So I wanted to let you know that for the next two months, until the end of the year, I won't be able to take on any extra shifts. I'm sorry if this makes things harder for the rota. I really don't want it to sound like I don't care about the team, because I do, and I'd still be happy to swap a shift with someone if it helps and doesn't add hours.

    After New Year I'd be glad to talk about whether I can help again now and then, maybe once a month, if that would be useful. I just need a bit of a break first so that I can keep doing my own shifts properly.

    Thanks for understanding, and sorry again for the long email.

    Best wishes
    """

    /// Original text written for this suite (about 300 words): a school
    /// letter pasted into `long_session`. Fictional school and people.
    static let longSessionSchoolLetter = """
    Elmwood Primary School - Year 3 Trip to the Riverside Aquarium

    Dear Parents and Carers,

    We are delighted to tell you that Year 3 will be visiting the Riverside Aquarium on Thursday 14 November as part of our science topic, "Life Under the Sea". The children will see the aquarium's octopus and jellyfish galleries, take part in a hands-on workshop about ocean food chains, and complete a short worksheet that we will use in class the following week.

    The coach will leave school at 9:00 am sharp, so please make sure your child arrives by 8:40 am. We expect to be back at school by 3:15 pm, which is the usual collection time, and children can be collected from the Year 3 door as normal.

    The cost of the trip is 12 pounds per child, which covers the coach, entry and the workshop. Please pay through the school payment app by Friday 8 November. If paying the cost is difficult for your family, please speak to the school office in confidence; no child will miss the trip because of cost.

    Please also complete the consent form on the payment app by the same date. We cannot take any child on the trip without a completed consent form, even if the trip has been paid for.

    Children will need:
    - a packed lunch in a disposable bag, with no glass bottles and no nuts
    - a refillable water bottle
    - a waterproof coat, as part of the walk from the coach park is outside
    - comfortable shoes

    Children should wear their school uniform. Mobile phones, toys and money should stay at home.

    We are looking for four parent or carer volunteers to help on the day. Volunteers must have a current school DBS check. If you can help, please email the Year 3 team at year3@elmwood.example by Monday 4 November.

    Thank you for your support.

    Mrs A. Patel and Mr J. Osei
    Year 3 Teachers
    """

    /// Original text written for this suite (about 400 words): a recipe the
    /// `long_session` user pastes to have it halved.
    static let longSessionRecipe = """
    Hearty Red Lentil and Vegetable Soup (serves 8)

    This is the soup I make every winter when the house feels cold and everyone needs feeding. It keeps for four days in the fridge and freezes well, so a big pot goes a long way. The lentils break down as it cooks and make it thick without any cream.

    Ingredients
    - 4 tablespoons olive oil
    - 2 large onions, chopped
    - 4 carrots, peeled and diced
    - 4 sticks of celery, sliced
    - 6 cloves of garlic, crushed
    - 2 teaspoons ground cumin
    - 2 teaspoons smoked paprika
    - 1 teaspoon chilli flakes (optional)
    - 400 g red lentils, rinsed
    - 2 tins (800 g) chopped tomatoes
    - 2.4 litres vegetable stock
    - 2 bay leaves
    - 300 g potatoes, diced
    - 200 g spinach or kale
    - juice of 2 lemons
    - salt and black pepper to taste
    - a large bunch of parsley, chopped, to serve

    Method
    1. Heat the oil in a large pot over a medium heat. Add the onions, carrots and celery with a pinch of salt and cook gently for about ten minutes, stirring now and then, until soft but not brown.
    2. Stir in the garlic, cumin, paprika and chilli flakes and cook for one more minute, until the spices smell toasted.
    3. Add the lentils, tomatoes, stock, bay leaves and potatoes. Bring to the boil, then turn the heat down and simmer with the lid half on for 30 to 35 minutes, until the lentils have fallen apart and the potatoes are tender.
    4. Take out the bay leaves. For a smoother soup, blend about a third of it and stir it back in; for a chunky one, leave it as it is.
    5. Stir in the spinach or kale and cook for three minutes until wilted. Add the lemon juice, then season with salt and pepper. If the soup is too thick, add a little hot water.
    6. Serve in warm bowls with the parsley on top and some crusty bread on the side.

    Tips
    Red lentils need no soaking, but do rinse them until the water runs clear, or the soup can taste dusty. If you only have green or brown lentils, they work too, but add 15 minutes to the simmering time and expect them to keep their shape. A spoonful of yoghurt on top is lovely if you are not keeping it vegan.
    """

    // MARK: vision_photo_real

    struct RealPhoto {
        let resource: String
        let label: String
        let question: String
        /// Any of these in the description means it is about the photo.
        let describeKeywords: [String]
        /// Any of these in the answer to `question` means it got the fact.
        let answerKeywords: [String]
        let reference: String
    }

    /// Two CC0 photos from Wikimedia Commons, bundled as test resources;
    /// sources and licences in `DailyUsePhotoSources.txt` next to them:
    ///
    ///   DailyUseStreet.jpg - "Street scene at SW corner of Upper Kirby",
    ///     WhisperToMe, CC0 1.0,
    ///     https://commons.wikimedia.org/wiki/File:Street_scene_at_SW_corner_of_Upper_Kirby.jpg
    ///   DailyUseFood.jpg - "Plate of food with matooke, groundnuts,
    ///     chicken and chapati", Samson Ssemakadde, CC0 1.0,
    ///     https://commons.wikimedia.org/wiki/File:Plate_of_food_with_matooke,_groundnuts,_chicken_and_chapati.jpg
    static let realPhotos = [
        RealPhoto(
            resource: "DailyUseStreet.jpg",
            label: "street",
            question: "What colour is the car in the middle of the road, directly ahead of the camera?",
            describeKeywords: ["street", "road", "traffic", "car", "cars", "intersection", "city"],
            answerKeywords: ["white"],
            reference: """
            Photo taken from a car at dusk on a city street in Houston: a pink and orange sunset sky, \
            several cars waiting at a red light, a white SUV (Toyota) directly ahead in the middle lane, \
            a dark hatchback on its left, an Exxon petrol station sign with lit prices on the left, a \
            lit "Pilgrim Cleaners" sign and a "Houston Shoe Hospital" storefront on the right, \
            traffic lights and street lamps. Expected answer to the question: white.
            """
        ),
        RealPhoto(
            resource: "DailyUseFood.jpg",
            label: "food",
            question: "Is this meal vegetarian? Why or why not?",
            describeKeywords: ["rice", "chicken", "plate", "food", "meal", "dish"],
            answerKeywords: ["chicken", "meat", "poultry"],
            reference: """
            A stainless-steel compartment plate on an orange tray: brown rice with a meat stew on top, \
            a piece of fried chicken on the right, chapati on the left, and steamed matooke (green \
            banana) in a pinkish groundnut sauce at the bottom. Expected answer to the question: not \
            vegetarian, because of the chicken (and the meat stew).
            """
        )
    ]

    /// A file from the test bundle (the synchronized test folder copies
    /// every non-source file into it).
    static func bundledResource(_ name: String) -> Data? {
        let bundle = Bundle(for: DailyUseRunner.self)
        let base = (name as NSString).deletingPathExtension
        let ext = (name as NSString).pathExtension
        let url = bundle.url(forResource: base, withExtension: ext)
            ?? bundle.url(forResource: base, withExtension: ext, subdirectory: "Device")
        return url.flatMap { try? Data(contentsOf: $0) }
    }

    // MARK: vision

    static let describePrompt = "What is in this picture? Describe it."

    static let syntheticPhotoDescription = """
    Synthetic 896x672 JPEG drawn by the test (the image LlamaCppVisionMultiTurnDeviceTests uses): \
    a diagonal gradient from teal (top left) to purple (bottom right), with a regular grid of \
    8 rows x 10 columns of small orange rectangles. No objects, people or text.
    """

    /// The image of `LlamaCppVisionMultiTurnDeviceTests.syntheticPhoto()`,
    /// drawn at scale 1 so the pixels are the same on every device.
    @MainActor
    static func syntheticPhoto() -> Data? {
        let size = CGSize(width: 896, height: 672)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            let colors = [UIColor.systemTeal.cgColor, UIColor.systemPurple.cgColor]
            if let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: colors as CFArray,
                locations: [0, 1]
            ) {
                ctx.cgContext.drawLinearGradient(
                    gradient, start: .zero, end: CGPoint(x: size.width, y: size.height), options: []
                )
            }
            UIColor.systemOrange.setFill()
            for row in 0..<8 {
                for col in 0..<10 {
                    ctx.fill(CGRect(x: col * 90 + 10, y: row * 80 + 10, width: 40, height: 30))
                }
            }
        }
        return image.jpegData(compressionQuality: 0.85)
    }

    static let receiptPrompt = "What is the total amount on this receipt, and what is the name of the shop?"
    static let receiptShop = "Maple Street Grocery"
    static let receiptTotal = 10.44
    static let receiptItems: [(String, String)] = [
        ("Sourdough bread", "$3.49"),
        ("Oat milk", "$2.15"),
        ("Apples", "$4.80")
    ]

    static let receiptDescription = """
    Receipt rendered by the test: header "MAPLE STREET GROCERY", items Sourdough bread $3.49, \
    Oat milk $2.15, Apples $4.80, line "TOTAL $10.44", footer "Thank you for shopping with us!". \
    Expected answer: total $10.44, shop Maple Street Grocery.
    """

    /// A plain till receipt: black text on white, large enough to survive
    /// the chat's 896-pixel downscale.
    @MainActor
    static func receiptImage() -> Data? {
        let size = CGSize(width: 640, height: 720)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let image = UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            UIColor.white.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))

            func draw(_ text: String, x: CGFloat, y: CGFloat, width: CGFloat,
                      size: CGFloat, bold: Bool = false, align: NSTextAlignment = .left) {
                let style = NSMutableParagraphStyle()
                style.alignment = align
                let font = UIFont.monospacedSystemFont(ofSize: size, weight: bold ? .bold : .regular)
                (text as NSString).draw(
                    in: CGRect(x: x, y: y, width: width, height: size * 1.5),
                    withAttributes: [.font: font, .foregroundColor: UIColor.black, .paragraphStyle: style]
                )
            }

            let margin: CGFloat = 40
            let width = size.width - margin * 2
            draw("MAPLE STREET GROCERY", x: margin, y: 40, width: width, size: 38, bold: true, align: .center)
            draw("Fresh food every day", x: margin, y: 100, width: width, size: 24, align: .center)
            draw(String(repeating: "-", count: 30), x: margin, y: 160, width: width, size: 28, align: .center)

            var y: CGFloat = 220
            for (name, price) in receiptItems {
                draw(name, x: margin, y: y, width: width, size: 30)
                draw(price, x: margin, y: y, width: width, size: 30, align: .right)
                y += 60
            }
            draw(String(repeating: "-", count: 30), x: margin, y: y, width: width, size: 28, align: .center)
            y += 60
            draw("TOTAL", x: margin, y: y, width: width, size: 36, bold: true)
            draw("$10.44", x: margin, y: y, width: width, size: 36, bold: true, align: .right)
            y += 110
            draw("Thank you for shopping with us!", x: margin, y: y, width: width, size: 24, align: .center)
        }
        return image.jpegData(compressionQuality: 0.9)
    }

    // MARK: sustained

    /// Cap per generation; the prompts ask for less, so most answers end on
    /// their own and the eight steps stay comparable.
    static let sustainedMaxTokens = 320

    static let sustainedPrompts = [
        "Write about 180 words on how bicycles changed city life.",
        "Now write about 180 words on the history of tea.",
        "Next, about 180 words on how lighthouses work.",
        "Next, about 180 words on why sleep matters for memory.",
        "Next, about 180 words on the life of a honeybee colony.",
        "Next, about 180 words on how rivers shape landscapes.",
        "Next, about 180 words on what the printing press changed.",
        "Finally, about 180 words on how weather forecasts are made."
    ]
}
