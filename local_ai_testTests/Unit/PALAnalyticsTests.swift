//
//  PALAnalyticsTests.swift
//
//  Pins the measurement telemetry contract: the per-event allowlist (which
//  must equal APP_EVENTS in the server's services/openpanel_pal_routes.py),
//  the guarantee that no key can carry what a person wrote, model naming for
//  imported models, battery bucketing, error kinds, the chat meter, and the
//  flattening of a BenchmarkReport.
//

import Foundation
import Testing
@testable import local_ai_test

struct PALAnalyticsSchemaTests {

    /// The server's allowlist, copied by hand on purpose: a change on one side
    /// fails here until the other side is changed too.
    static let serverKeys: [PALEvent: Set<String>] = {
        let model: Set<String> = ["model_id", "backend"]
        let load = model.union(["quantization", "parameter_size", "file_size_mb", "cold", "load_seconds",
                                "peak_memory_mb", "memory_budget_mb", "thermal_before", "low_power_mode"])
        let generation = model.union(["kind", "backend_rev", "quantization", "context_size", "outcome", "error_kind"])
        return [
            .appOpened: [],
            .modelDownloadFinished: model.union(["size_mb", "seconds"]),
            .modelDownloadFailed: model.union(["size_mb", "seconds", "error_kind", "http_status"]),
            .modelLoadMeasured: load,
            .modelLoadFailed: load.union(["error_kind"]),
            .generationMeasured: generation.union([
                "prompt_tokens", "tokens_generated", "generation_tps", "ttft_ms", "wall_clock_seconds",
                "peak_memory_mb", "memory_budget_mb", "thermal_before", "thermal_after", "low_power_mode",
                "battery_bucket", "has_image", "has_audio"]),
            .generationFailed: generation,
            .benchmarkMeasured: model.union([
                "kind", "backend_rev", "quantization", "parameter_size", "load_seconds", "prefill_tps",
                "generation_tps", "generation_tps_stddev", "generation_repetitions", "peak_memory_mb",
                "memory_budget_mb", "thermal_before", "thermal_after", "peak_thermal_state",
                "thermally_constrained", "throttle_drop_percent", "warmup_performed", "metrics_version",
                "tokens_generated", "wall_clock_seconds", "pp512_tps", "tg128_tps", "tg128_tps_stddev",
                "suite_id", "suite_version", "quality_passed", "quality_total", "quality_completed",
                "quality_truncated", "score", "app_version", "app_build", "report_id"]),
        ]
    }()

    @Test func theAllowlistEqualsTheServers() {
        #expect(Set(PALEvent.allCases.map(\.rawValue)) == [
            "app_opened", "model_download_finished", "model_download_failed", "model_load_measured",
            "model_load_failed", "generation_measured", "generation_failed", "benchmark_measured",
        ])
        for event in PALEvent.allCases {
            #expect(PALAnalyticsSchema.keys[event] == Self.serverKeys[event], "\(event.rawValue)")
        }
    }

    /// The server's value rules (APP_EVENTS in openpanel_pal_routes.py),
    /// copied by hand like the key sets above: same types, bounds, enums and
    /// patterns, `kind` fixed per event.
    static func serverRule(_ key: String, in event: PALEvent) -> PALAnalyticsSchema.Rule? {
        typealias R = PALAnalyticsSchema.Rule
        let thermal = R.oneOf(["nominal", "fair", "serious", "critical", "unknown"])
        let seconds = R.number(max: 1e7), megabytes = R.number(max: 1e7), rate = R.number(max: 1e6)
        let tokens = R.integer(max: 10_000_000), count = R.integer(max: 100_000)
        let version = R.slug(#"^[0-9A-Za-z.+-]{1,32}$"#)
        let table: [String: R] = [
            "model_id": .slug(#"^(import:[a-z0-9-]{1,32}|[A-Za-z0-9][A-Za-z0-9._/-]{0,119})$"#),
            "backend": .oneOf(["appleIntelligence", "mlx", "mlxVision", "mlxAudio", "coreML", "coreMLWhisper",
                               "coreMLKokoro", "llamaCpp", "llamaCppVision", "whisperCpp", "other"]),
            "backend_rev": .slug(#"^[a-z0-9][a-z0-9.+-]{0,39}$"#),
            "quantization": .slug(#"^[A-Za-z0-9_.+-]{1,32}$"#),
            "parameter_size": .slug(#"^[A-Za-z0-9_.+-]{1,16}$"#),
            "error_kind": .slug(#"^[a-z0-9][a-z0-9_.:-]{0,63}$"#),
            "http_status": .integer(min: 100, max: 599),
            "size_mb": megabytes, "seconds": seconds, "file_size_mb": megabytes, "cold": .bool,
            "load_seconds": seconds, "peak_memory_mb": megabytes, "memory_budget_mb": megabytes,
            "thermal_before": thermal, "thermal_after": thermal, "peak_thermal_state": thermal,
            "low_power_mode": .bool, "context_size": tokens, "prompt_tokens": tokens, "tokens_generated": tokens,
            "generation_tps": rate, "ttft_ms": .number(max: 1e9), "wall_clock_seconds": seconds,
            "battery_bucket": .oneOf(["0-20", "20-50", "50-80", "80-100"]), "has_image": .bool, "has_audio": .bool,
            "outcome": .oneOf(["ok", "cancelled", "context_full", "oom", "error"]),
            "prefill_tps": rate, "generation_tps_stddev": rate, "generation_repetitions": .integer(max: 1000),
            "thermally_constrained": .bool, "throttle_drop_percent": .number(max: 100), "warmup_performed": .bool,
            "metrics_version": .integer(max: 1000), "pp512_tps": rate, "tg128_tps": rate, "tg128_tps_stddev": rate,
            "suite_id": .oneOf(["built-in", "imported"]), "suite_version": .integer(max: 1_000_000),
            "quality_passed": count, "quality_total": count, "quality_completed": count, "quality_truncated": .bool,
            "score": .number(max: 100), "app_version": version, "app_build": version,
            "report_id": .slug(#"^[0-9A-Fa-f-]{36}$"#),
        ]
        if key == "kind" {
            switch event {
            case .generationMeasured, .generationFailed: return .oneOf(["chat"])
            case .benchmarkMeasured: return .oneOf(["benchmark"])
            default: return nil
            }
        }
        return serverKeys[event]?.contains(key) == true ? table[key] : nil
    }

    @Test func theValueRulesEqualTheServers() {
        for event in PALEvent.allCases {
            for key in Self.serverKeys[event] ?? [] {
                #expect(PALAnalyticsSchema.rule(for: key, in: event) == Self.serverRule(key, in: event),
                        "\(event.rawValue).\(key)")
            }
        }
        #expect(PALAnalyticsSchema.coerce("chat", to: PALAnalyticsSchema.rule(for: "kind", in: .generationMeasured)!) != nil)
        #expect(PALAnalyticsSchema.coerce("benchmark", to: PALAnalyticsSchema.rule(for: "kind", in: .generationMeasured)!) == nil)
        #expect(PALAnalyticsSchema.rule(for: "kind", in: .modelLoadMeasured) == nil)
    }

    @Test func requiredKeysMatchTheServer() {
        for event in PALEvent.allCases {
            var expected: Set<String> = event == .appOpened ? [] : ["model_id", "backend"]
            if event == .generationMeasured || event == .generationFailed { expected.formUnion(["kind", "outcome"]) }
            if event == .benchmarkMeasured { expected.insert("kind") }
            #expect(PALAnalyticsSchema.required[event] == expected, "\(event.rawValue)")
        }
        let missing = PALAnalyticsSchema.missingRequired(
            PALAnalyticsSchema.sanitize(["model_id": "a/b", "kind": "benchmark"], for: .benchmarkMeasured),
            for: .benchmarkMeasured)
        #expect(missing == ["backend"])
    }

    @Test func everyAllowedKeyHasARule() {
        for (event, keys) in PALAnalyticsSchema.keys {
            for key in keys {
                #expect(PALAnalyticsSchema.rule(for: key, in: event) != nil, "\(event.rawValue).\(key)")
            }
        }
    }

    /// The guard of the whole design: no event can carry a field for what a
    /// person wrote or said. A key that mentions such a word (prompt_tokens)
    /// must be a number, so the thing itself can never fit.
    @Test func noKeyCanCarryWhatAPersonWrote() {
        let banned: Set<String> = ["prompt", "prompts", "text", "answer", "answers", "title", "filename",
                                   "name", "message", "content", "transcript", "speech", "repo", "path",
                                   "url", "attachment", "question"]
        for (event, keys) in PALAnalyticsSchema.keys {
            for key in keys {
                let words = key.split(separator: "_").map(String.init)
                #expect(!banned.contains(key), "\(event.rawValue).\(key)")
                #expect(!banned.contains(words.last ?? ""), "\(event.rawValue).\(key)")
                #expect(!key.contains("filename"), "\(event.rawValue).\(key)")
                if !banned.isDisjoint(with: words), let rule = PALAnalyticsSchema.rule(for: key, in: event) {
                    #expect(PALAnalyticsSchema.coerce("a sentence", to: rule) == nil, "\(event.rawValue).\(key)")
                }
            }
        }
        for key in ["prompt", "text", "answer", "title", "filename", "chat_title", "file_name"] {
            for event in PALEvent.allCases {
                #expect(PALAnalyticsSchema.rule(for: key, in: event) == nil, "\(event.rawValue).\(key)")
            }
        }
    }

    @Test func sanitizeKeepsOnlyAllowlistedTypedValues() {
        let out = PALAnalyticsSchema.sanitize([
            "model_id": "unsloth/Qwen3.5-2B-GGUF",
            "backend": "llamaCpp",
            "kind": "chat",
            "outcome": "ok",
            "prompt": "tell me a secret",
            "title": "My chat",
            "tokens_generated": 120,
            "generation_tps": 41.25,
            "has_image": true,
            "prompt_tokens": "312",
            "ttft_ms": Double.nan,
            "battery_bucket": "55",
        ], for: .generationMeasured)
        #expect(out["model_id"] == .string("unsloth/Qwen3.5-2B-GGUF"))
        #expect(out["tokens_generated"] == .int(120))
        #expect(out["generation_tps"] == .double(41.25))
        #expect(out["has_image"] == .bool(true))
        #expect(out["prompt"] == nil)
        #expect(out["title"] == nil)
        #expect(out["prompt_tokens"] == nil, "a string is not a count")
        #expect(out["ttft_ms"] == nil)
        #expect(out["battery_bucket"] == nil)
        #expect(PALAnalyticsSchema.sanitize(["screen": "chat"], for: .appOpened).isEmpty)
    }

    @Test func stringsAreSlugsNotSentences() {
        let rule = PALAnalyticsSchema.rules["model_id"]!
        #expect(PALAnalyticsSchema.coerce("mlx-community/gemma-3-4b-it-4bit", to: rule) != nil)
        #expect(PALAnalyticsSchema.coerce("import:gguf", to: rule) != nil)
        #expect(PALAnalyticsSchema.coerce("my notes about Bob", to: rule) == nil)
        #expect(PALAnalyticsSchema.coerce("bartowski/repo name.gguf", to: rule) == nil)
        let kind = PALAnalyticsSchema.rules["error_kind"]!
        #expect(PALAnalyticsSchema.coerce("The model said: hello", to: kind) == nil)
        #expect(PALAnalyticsSchema.coerce(true, to: PALAnalyticsSchema.rules["prompt_tokens"]!) == nil)
        #expect(PALAnalyticsSchema.coerce(-1.0, to: PALAnalyticsSchema.rules["generation_tps"]!) == nil)
    }

    @Test func categoryScoresOnlyOnTheBenchmark() {
        #expect(PALAnalyticsSchema.rule(for: "score_math", in: .benchmarkMeasured) != nil)
        #expect(PALAnalyticsSchema.rule(for: "score_math", in: .generationMeasured) == nil)
        #expect(PALAnalyticsSchema.rule(for: "score_My Category", in: .benchmarkMeasured) == nil)
    }
}

struct PALAnalyticsGateTests {

    @Test func onlyAReleaseDeviceBuildOutsideTestsSends() {
        func send(_ env: [String: String] = [:], simulator: Bool = false, debug: Bool = false,
                  override: Bool = false) -> Bool {
            PALAnalyticsGate.shouldSend(environment: env, isSimulator: simulator, isDebug: debug, debugOverride: override)
        }
        #expect(send())
        #expect(!send(["XCTestConfigurationFilePath": "/x"]))
        #expect(!send(["XCTestBundlePath": "/x"]))
        #expect(!send([PALAnalyticsGate.disableEnvironmentKey: "1"]), "a UI-test launch")
        #expect(!send(simulator: true))
        #expect(!send(simulator: true, debug: true, override: true))
        #expect(!send(debug: true))
        #expect(send(debug: true, override: true), "the documented end-to-end override")
        #expect(!send([PALAnalyticsGate.disableEnvironmentKey: "1"], debug: true, override: true))
    }
}

struct PALImportedModelPrivacyTests {

    private let importKinds = Set(ModelBackend.allCases.map { PALModelIdentity.importKind($0) })

    /// A Hugging Face GGUF without a quant tag: the import flow names the
    /// quantization after the file, and that name must never leave.
    @MainActor
    @Test func aTaglessFileNameIsNeverSent() {
        let label = ImportAnalyzer.quantLabel(for: "diary-bot-v2.gguf")
        #expect(label == "diary-bot-v2", "the hazard: the stem becomes the label")
        #expect(PALModelIdentity.quantization(label, source: .imported) == "other")
        #expect(PALModelIdentity.quantization(label, source: nil) == "other")
        let tagged = ImportAnalyzer.quantLabel(for: "someone-Llama-3-8B-Q4_K_M.gguf")
        #expect(PALModelIdentity.quantization(tagged, source: .imported) == "Q4_K_M")
        for known in ["IQ2_XXS", "Q8_0", "q5_k_s", "BF16", "4bit", "int8", "gguf", "as-published", "MXFP4"] {
            #expect(PALModelIdentity.quantization(known, source: .imported) == known, "\(known)")
        }
        for leaky in ["q4_bob", "Q4_K_M-diary", "my-notes", "Q4 K M"] {
            #expect(PALModelIdentity.quantization(leaky, source: .imported) == "other", "\(leaky)")
        }
        #expect(PALModelIdentity.parameterSize("7B", source: .imported) == "7B")
        #expect(PALModelIdentity.parameterSize("1.5B", source: .imported) == "1.5B")
        #expect(PALModelIdentity.parameterSize("diary", source: .imported) == "other")
        #expect(PALModelIdentity.parameterSize("?", source: .imported) == "other")
        // A catalog entry's own labels are ours and pass unchanged.
        #expect(PALModelIdentity.quantization("UD-Q4_K_XL", source: .curated) == "UD-Q4_K_XL")
    }

    /// Every string in every record built from an imported model is an enum
    /// value, a slug from a fixed set, "other", or the app's own version and
    /// report id. Nothing from the model's repository, file or display name.
    @MainActor
    @Test func everyStringFromAnImportedModelIsFixed() {
        let tagless = ImportAnalyzer.quantLabel(for: "diary-bot-v2.gguf")
        let model = AIModel(id: "someone/diary-bot-GGUF#\(tagless)", name: "Diary Bot", family: "diary",
                            parameterSize: "diary", quantization: tagless, backend: .llamaCppVision,
                            ramRequiredGB: 2, diskSizeGB: 1, huggingFaceRepo: "someone/diary-bot-GGUF",
                            contextLength: 4096, quality: .good, preferredGGUFFilename: "diary-bot-v2.gguf",
                            source: .imported)
        let conditions = PALConditions(thermalBefore: "nominal", thermalAfter: "fair", lowPowerMode: false,
                                       batteryBucket: "50-80", memoryBudgetMB: 4000, peakMemoryMB: 900)
        let start = Date()
        var meter = PALGenerationMeter(start: start)
        meter.observe(.textDelta("a"), at: start.addingTimeInterval(0.2))
        meter.observe(.textDelta("b"), at: start.addingTimeInterval(0.4))
        let measured = PALRecords.generation(model: model, contextSize: 8192, meter: meter, end: start.addingTimeInterval(1),
                                             outcome: "ok", errorKind: nil, conditions: conditions,
                                             hasImage: true, hasAudio: false)
        let failed = PALRecords.generation(model: model, contextSize: 8192, meter: PALGenerationMeter(start: start),
                                           end: start, outcome: "error", errorKind: "generation_failed",
                                           conditions: conditions, hasImage: false, hasAudio: false)
        let report = BenchmarkReport(
            appVersion: "1.3", appBuild: "42", modelId: model.id, modelName: model.name,
            backend: model.backend.displayName, quantization: model.quantization, parameterSize: model.parameterSize,
            device: DeviceSnapshot(identifier: "iPhone18,4", marketingName: "iPhone Air", chip: "A19 Pro",
                                   ramGB: 12, iOSVersion: "26.1"),
            performance: PerformanceMetrics(loadSeconds: 1, prefillTokensPerSecond: 100, generationTokensPerSecond: 20,
                                            peakMemoryMB: 900, memoryBudgetMB: 4000, thermalStateBefore: "nominal",
                                            thermalStateAfter: "fair", tokensGenerated: 100, wallClockSeconds: 10),
            quality: QualityResults(suiteVersion: 1, categories: [CategoryScore(category: "Diary Questions", tasks: [
                TaskResult(taskId: "d1", passed: true, latencySeconds: 1, answer: "dear diary")])],
                suiteID: "imported-diary", suiteName: "Diary exam"))
        let records: [(PALEvent, [String: Any])] = [
            (.modelDownloadFinished, PALRecords.download(model: model, seconds: 3)),
            (.modelDownloadFailed, PALRecords.download(model: model, seconds: 3, errorKind: "http", httpStatus: 404)),
            (.modelLoadMeasured, PALRecords.load(model: model, cold: true, loadSeconds: 1, conditions: conditions)),
            (.modelLoadFailed, PALRecords.load(model: model, cold: true, loadSeconds: 1, conditions: conditions,
                                               errorKind: "memory_preflight")),
            measured, failed,
            (.benchmarkMeasured, PALRecords.benchmark(report, source: .imported)),
        ]
        let revisions = Set(ModelBackend.allCases.map { PALBackendRevision.revision(for: $0) })
        let errorKinds: Set<String> = ["http", "memory_preflight", "generation_failed"]
        for (event, raw) in records {
            let props = PALAnalyticsSchema.sanitize(raw, for: event)
            #expect(props.count == raw.count, "\(event.rawValue) dropped \(Set(raw.keys).subtracting(props.keys))")
            for (key, value) in props {
                guard case .string(let s) = value else { continue }
                #expect(!s.lowercased().contains("diary") && !s.contains("someone"), "\(event.rawValue).\(key)=\(s)")
                switch key {
                case "model_id":
                    #expect(s.hasPrefix("import:") && importKinds.contains(String(s.dropFirst(7))), "\(s)")
                case "quantization", "parameter_size":
                    #expect(s == "other", "\(event.rawValue).\(key)=\(s)")
                case "backend_rev":
                    #expect(revisions.contains(s), "\(s)")
                case "error_kind":
                    #expect(errorKinds.contains(s), "\(s)")
                case "app_version", "app_build", "report_id":
                    #expect([report.appVersion, report.appBuild, report.id].contains(s), "\(key)=\(s)")
                default:
                    if case .oneOf(let allowed)? = PALAnalyticsSchema.rule(for: key, in: event) {
                        #expect(allowed.contains(s), "\(event.rawValue).\(key)=\(s)")
                    } else {
                        Issue.record("\(event.rawValue).\(key) is a free string: \(s)")
                    }
                }
            }
            #expect(!props.keys.contains { $0.contains("diary") }, "\(event.rawValue)")
        }
    }

    @Test func anUnresolvableBackendIsOtherNotMissing() {
        let report = BenchmarkReport(
            appVersion: "1.3", appBuild: "42", modelId: "x/y", modelName: "Y", backend: "Some Future Engine",
            quantization: "Q4_K_M", parameterSize: "1B",
            device: DeviceSnapshot(identifier: "iPhone18,4", marketingName: "iPhone Air", chip: "A19 Pro",
                                   ramGB: 12, iOSVersion: "26.1"),
            performance: PerformanceMetrics(loadSeconds: 1, prefillTokensPerSecond: 100, generationTokensPerSecond: 20,
                                            peakMemoryMB: 900, memoryBudgetMB: 4000, thermalStateBefore: "nominal",
                                            thermalStateAfter: "fair", tokensGenerated: 100, wallClockSeconds: 10),
            quality: nil)
        let props = PALAnalyticsSchema.sanitize(PALRecords.benchmark(report, source: nil), for: .benchmarkMeasured)
        #expect(props["backend"] == .string("other"))
        #expect(PALAnalyticsSchema.missingRequired(props, for: .benchmarkMeasured).isEmpty)
    }
}

struct PALAnalyticsHelperTests {

    private func model(id: String, source: ModelSource, backend: ModelBackend = .llamaCpp) -> AIModel {
        AIModel(id: id, name: "Model", family: "f", parameterSize: "2B", quantization: "Q4_K_M",
                backend: backend, ramRequiredGB: 2, diskSizeGB: 1.5, huggingFaceRepo: id,
                contextLength: 4096, quality: .good, source: source)
    }

    @Test func catalogModelsSendTheirCatalogId() {
        #expect(PALModelIdentity.modelId(for: model(id: "unsloth/Qwen3.5-2B-GGUF", source: .curated))
                == "unsloth/Qwen3.5-2B-GGUF")
        #expect(PALModelIdentity.modelId(for: model(id: "apple-intelligence", source: .curated,
                                                    backend: .appleIntelligence)) == "apple-intelligence")
    }

    @Test func importedModelsNeverSendTheirRepositoryOrFileName() {
        let hf = model(id: "someone/private-finetune-GGUF#Q5_K_M", source: .imported)
        #expect(PALModelIdentity.modelId(for: hf) == "import:gguf")
        let mlx = model(id: "someone/thing-4bit", source: .imported, backend: .mlxVision)
        #expect(PALModelIdentity.modelId(for: mlx) == "import:mlx-vision")
        // A report whose model can no longer be resolved counts as imported.
        #expect(PALModelIdentity.modelId(id: "someone/gone", source: nil, backend: .mlx) == "import:mlx")
        #expect(PALModelIdentity.modelId(id: "x", source: nil, backend: nil) == "import:unknown")
        // A catalog id that would not pass the server's alphabet is not sent as is.
        #expect(PALModelIdentity.modelId(id: "weird id with spaces", source: .curated, backend: .mlx) == "import:mlx")
        for backend in ModelBackend.allCases {
            let id = PALModelIdentity.modelId(id: "a/b", source: .imported, backend: backend)
            #expect(PALAnalyticsSchema.matches(id, PALAnalyticsSchema.modelIdPattern), "\(id)")
        }
    }

    @Test func backendRevisions() {
        #expect(PALBackendRevision.revision(for: .llamaCpp) == "llama.cpp-b8882")
        #expect(PALBackendRevision.revision(for: .llamaCppVision) == "llama.cpp-b8882")
        #expect(PALBackendRevision.revision(for: .mlx) == "mlx-swift-0.29.1")
        #expect(PALBackendRevision.revision(for: .coreML) == "coreml")
        #expect(PALBackendRevision.revision(for: .appleIntelligence) == "apple-intelligence")
        #expect(PALBackendRevision.revision(for: .whisperCpp) == "whisper")
        let rule = PALAnalyticsSchema.rules["backend_rev"]!
        for backend in ModelBackend.allCases {
            #expect(PALAnalyticsSchema.coerce(PALBackendRevision.revision(for: backend), to: rule) != nil)
        }
    }

    @Test func batteryBuckets() {
        #expect(PALDeviceState.batteryBucket(level: -1) == nil)
        #expect(PALDeviceState.batteryBucket(level: 0) == "0-20")
        #expect(PALDeviceState.batteryBucket(level: 0.19) == "0-20")
        #expect(PALDeviceState.batteryBucket(level: 0.2) == "20-50")
        #expect(PALDeviceState.batteryBucket(level: 0.5) == "50-80")
        #expect(PALDeviceState.batteryBucket(level: 0.79) == "50-80")
        #expect(PALDeviceState.batteryBucket(level: 0.8) == "80-100")
        #expect(PALDeviceState.batteryBucket(level: 1) == "80-100")
        #expect(PALDeviceState.batteryBucket(level: 1.5) == nil)
    }

    @Test func errorKindsComeFromTypesNeverFromMessages() {
        let secret = "my private prompt about Bob"
        let cases: [(Error, String)] = [
            (InferenceError.generationFailed("llama.cpp: the context window (4096 tokens) leaves no room. \(secret)"), "context_full"),
            (InferenceError.generationFailed("decode failed: \(secret)"), "generation_failed"),
            (InferenceError.generationFailed("out of memory"), "out_of_memory"),
            (InferenceError.backendUnavailable(secret), "backend_unavailable"),
            (InferenceError.cancelled, "cancelled"),
            (CancellationError(), "cancelled"),
            (URLError(.notConnectedToInternet), "url:-1009"),
            (ModelDownloadError.preferredPackageMissing(package: secret, repo: secret), "package_missing"),
        ]
        for (error, expected) in cases {
            let kind = PALErrorClassifier.kind(of: error, phase: .generate)
            #expect(kind == expected)
            #expect(!kind.contains("bob"))
        }
        #expect(PALErrorClassifier.kind(of: InferenceError.generationFailed("x"), phase: .load) == "engine_load_failed")
        let ns = NSError(domain: "com.apple.CoreML", code: 7, userInfo: [NSLocalizedDescriptionKey: secret])
        #expect(PALErrorClassifier.kind(of: ns, phase: .load) == "com.apple.coreml:7")
        for (error, _) in cases {
            let kind = PALErrorClassifier.kind(of: error, phase: .download)
            #expect(PALAnalyticsSchema.coerce(kind, to: PALAnalyticsSchema.rules["error_kind"]!) != nil, "\(kind)")
        }
    }

    @Test func generationOutcomes() {
        #expect(PALErrorClassifier.outcome(of: nil, cancelled: false, memoryPressureUnload: false) == ("ok", nil))
        #expect(PALErrorClassifier.outcome(of: nil, cancelled: true, memoryPressureUnload: false).outcome == "cancelled")
        #expect(PALErrorClassifier.outcome(of: InferenceError.generationFailed("context window full"),
                                           cancelled: false, memoryPressureUnload: false).outcome == "context_full")
        #expect(PALErrorClassifier.outcome(of: InferenceError.modelNotLoaded, cancelled: false,
                                           memoryPressureUnload: true) == ("oom", "memory_pressure_unload"))
        #expect(PALErrorClassifier.outcome(of: InferenceError.modelNotLoaded, cancelled: false,
                                           memoryPressureUnload: false) == ("error", "model_not_loaded"))
    }
}

struct PALGenerationMeterTests {

    private let start = Date(timeIntervalSince1970: 1_000)

    @Test func measuresLikeTheBenchmark() {
        var meter = PALGenerationMeter(start: start)
        meter.observe(.textDelta("Hel"), at: start.addingTimeInterval(0.5))
        meter.observe(.textDelta("lo"), at: start.addingTimeInterval(0.6))
        meter.observe(.textDelta("!"), at: start.addingTimeInterval(1.5))
        meter.observe(.diagnostic(.promptProcessed(tokenCount: 312, timeMs: 400)), at: start.addingTimeInterval(1.5))
        #expect(meter.ttftMs == 500)
        #expect(meter.tokensGenerated == 3, "delta count without an engine report")
        // (3 - 1) tokens over the 1.0 s between the first and the last token.
        #expect(abs((meter.decodeTokensPerSecond(end: start.addingTimeInterval(2)) ?? 0) - 2.0) < 1e-9)
        meter.observe(.diagnostic(.finished(tokensGenerated: 11, tokensPerSecond: 9)), at: start.addingTimeInterval(1.6))
        #expect(meter.tokensGenerated == 11, "the engine's own count wins")
        #expect(abs((meter.decodeTokensPerSecond(end: start.addingTimeInterval(2)) ?? 0) - 10.0) < 1e-9)
        #expect(meter.promptTokens == 312)
    }

    @Test func noTokenMeansNoRates() {
        let meter = PALGenerationMeter(start: start)
        #expect(meter.ttftMs == nil)
        #expect(meter.decodeTokensPerSecond(end: start.addingTimeInterval(3)) == nil)
    }

    private func conditions() -> PALConditions {
        PALConditions(thermalBefore: "nominal", thermalAfter: "fair", lowPowerMode: false,
                      batteryBucket: "50-80", memoryBudgetMB: 7000, peakMemoryMB: 1890.5)
    }

    private let model = AIModel(id: "unsloth/Qwen3.5-2B-GGUF", name: "Qwen", family: "qwen", parameterSize: "2B",
                                quantization: "Q4_K_M", backend: .llamaCpp, ramRequiredGB: 2, diskSizeGB: 1.3,
                                huggingFaceRepo: "unsloth/Qwen3.5-2B-GGUF", contextLength: 4096, quality: .good)

    @Test func aFinishedGenerationIsMeasured() {
        var meter = PALGenerationMeter(start: start)
        meter.observe(.textDelta("a"), at: start.addingTimeInterval(0.25))
        meter.observe(.textDelta("b"), at: start.addingTimeInterval(1.25))
        let (event, record) = PALRecords.generation(
            model: model, contextSize: 4096, meter: meter, end: start.addingTimeInterval(2), outcome: "ok",
            errorKind: nil, conditions: conditions(), hasImage: true, hasAudio: false)
        #expect(event == .generationMeasured)
        let props = PALAnalyticsSchema.sanitize(record, for: event)
        #expect(props.count == record.count, "nothing the builder makes is dropped by the schema")
        #expect(props["kind"] == .string("chat"))
        #expect(props["ttft_ms"] == .double(250))
        #expect(props["wall_clock_seconds"] == .double(2))
        #expect(props["backend_rev"] == .string("llama.cpp-b8882"))
        #expect(props["has_image"] == .bool(true))
        #expect(props["outcome"] == .string("ok"))
        #expect(props["error_kind"] == nil)
    }

    @Test func anErrorBeforeTheFirstTokenIsAFailure() {
        let meter = PALGenerationMeter(start: start)
        let (event, record) = PALRecords.generation(
            model: model, contextSize: 4096, meter: meter, end: start.addingTimeInterval(1), outcome: "context_full",
            errorKind: "context_full", conditions: conditions(), hasImage: false, hasAudio: false)
        #expect(event == .generationFailed)
        #expect(Set(record.keys) == ["kind", "model_id", "backend", "backend_rev", "quantization",
                                     "context_size", "outcome", "error_kind"])
        #expect(PALAnalyticsSchema.sanitize(record, for: event).count == record.count)
    }
}

struct PALBenchmarkFlatteningTests {

    private func report(suiteID: String? = nil, quality: Bool = true, plannedTasks: Int? = nil,
                        truncated: Bool = false, chance: Double = 0) -> BenchmarkReport {
        BenchmarkReport(
            id: "0B7E2C55-1D7A-4C61-9C1B-3D7E2B1A0F00",
            appVersion: "1.3",
            appBuild: "42",
            modelId: "unsloth/Qwen3.5-2B-GGUF",
            modelName: "Qwen 3.5 2B",
            backend: "llama.cpp",
            quantization: "Q4_K_M",
            parameterSize: "2B",
            device: DeviceSnapshot(identifier: "iPhone18,4", marketingName: "iPhone Air", chip: "A19 Pro",
                                   ramGB: 12, iOSVersion: "26.1"),
            performance: PerformanceMetrics(
                loadSeconds: 2.5,
                prefillTokensPerSecond: 410.2,
                generationTokensPerSecond: 38.1,
                peakMemoryMB: 2100,
                memoryBudgetMB: 7000,
                thermalStateBefore: "nominal",
                thermalStateAfter: "serious",
                tokensGenerated: 600,
                wallClockSeconds: 240.3,
                generationTokensPerSecondStdDev: 0.8,
                generationRunsTokensPerSecond: [38.9, 38.0, 37.4],
                warmupPerformed: true,
                thermallyConstrained: false,
                throttleDropPercent: 3.9,
                peakThermalState: "serious",
                standardProtocol: StandardProtocolMetrics(
                    promptId: "pp512", promptTokensEstimated: 512, generationMaxTokens: 128,
                    prefillTokensPerSecond: 420, generationTokensPerSecond: 37.5,
                    generationTokensPerSecondStdDev: 0.4, repetitions: 1)
            ),
            quality: quality ? QualityResults(
                suiteVersion: 3,
                categories: [
                    CategoryScore(category: "math", tasks: [
                        TaskResult(taskId: "m1", passed: true, latencySeconds: 1, answer: "forty-two"),
                        TaskResult(taskId: "m2", passed: false, latencySeconds: 9, answer: "",
                                   error: "timed out after 60 s", chance: chance),
                    ]),
                    CategoryScore(category: "My Secret Category", tasks: [
                        TaskResult(taskId: "s1", passed: true, latencySeconds: 1,
                                   answer: String(repeating: "x", count: 900)),
                    ]),
                ],
                suiteID: suiteID,
                suiteName: suiteID == nil ? nil : "Private exam",
                total: plannedTasks,
                truncated: truncated
            ) : nil
        )
    }

    @Test func aBuiltInReportFlattensCompletely() {
        let record = PALRecords.benchmark(report(), source: .curated)
        let props = PALAnalyticsSchema.sanitize(record, for: .benchmarkMeasured)
        #expect(props.count == record.count, "dropped: \(Set(record.keys).subtracting(props.keys))")
        #expect(props["kind"] == .string("benchmark"))
        #expect(props["model_id"] == .string("unsloth/Qwen3.5-2B-GGUF"))
        #expect(props["backend"] == .string("llamaCpp"))
        #expect(props["backend_rev"] == .string("llama.cpp-b8882"))
        #expect(props["load_seconds"] == .double(2.5))
        #expect(props["prefill_tps"] == .double(410.2))
        #expect(props["generation_tps"] == .double(38.1))
        #expect(props["generation_tps_stddev"] == .double(0.8))
        #expect(props["generation_repetitions"] == .int(3))
        #expect(props["peak_memory_mb"] == .double(2100))
        #expect(props["memory_budget_mb"] == .double(7000))
        #expect(props["thermal_before"] == .string("nominal"))
        #expect(props["thermal_after"] == .string("serious"))
        #expect(props["peak_thermal_state"] == .string("serious"))
        #expect(props["tokens_generated"] == .int(600))
        #expect(props["wall_clock_seconds"] == .double(240.3))
        #expect(props["pp512_tps"] == .double(420))
        #expect(props["tg128_tps"] == .double(37.5))
        #expect(props["suite_id"] == .string("built-in"))
        #expect(props["suite_version"] == .int(3))
        #expect(props["quality_passed"] == .int(2))
        #expect(props["quality_total"] == .int(3))
        #expect(props["quality_completed"] == .int(3))
        #expect(props["quality_truncated"] == .bool(false))
        #expect(props["score_math"] == .double(50))
        #expect(props["score_my_secret_category"] == .double(100))
        #expect(props["app_version"] == .string("1.3"))
        #expect(props["report_id"] == .string("0B7E2C55-1D7A-4C61-9C1B-3D7E2B1A0F00"))
        if case .double(let score)? = props["score"] { #expect(abs(score - 66.667) < 0.001) } else { Issue.record("no score") }
    }

    /// A run the time budget cut short: the plan, what was graded, the flag,
    /// and a score over the graded tasks only.
    @Test func aTruncatedRunSaysSo() {
        let props = PALAnalyticsSchema.sanitize(
            PALRecords.benchmark(report(plannedTasks: 118, truncated: true), source: .curated), for: .benchmarkMeasured)
        #expect(props["quality_total"] == .int(118))
        #expect(props["quality_completed"] == .int(3))
        #expect(props["quality_truncated"] == .bool(true))
        if case .double(let score)? = props["score"] { #expect(abs(score - 66.667) < 0.001) } else { Issue.record("no score") }
    }

    /// Scores are the chance-normalised ones the report itself shows, not raw
    /// pass rates.
    @Test func scoresAreChanceNormalised() {
        let r = report(chance: 0.5)
        let record = PALRecords.benchmark(r, source: .curated)
        let math = r.quality!.categories.first { $0.category == "math" }!
        #expect(math.normalizedScore != math.passRate)
        #expect(record["score_math"] as? Double == math.normalizedScore * 100)
        #expect(record["score"] as? Double == r.quality!.normalizedScore * 100)
    }

    @Test func anImportedSuiteSendsNoCategoryNames() {
        let record = PALRecords.benchmark(report(suiteID: "imported-abc"), source: .curated)
        #expect(record["suite_id"] as? String == "imported")
        #expect(!record.keys.contains { $0.hasPrefix("score_") })
        #expect(record["score"] != nil)
    }

    @Test func anImportedModelIsAnonymisedAndAPerformanceOnlyReportHasNoScore() {
        let record = PALRecords.benchmark(report(quality: false), source: .imported)
        #expect(record["model_id"] as? String == "import:gguf")
        #expect(record["score"] == nil)
        #expect(record["suite_id"] == nil)
        #expect(PALRecords.benchmark(report(), source: nil)["model_id"] as? String == "import:gguf")
    }

    @Test func noValueCarriesTheModelsWordsOrTheSuiteName() {
        let r = report(suiteID: "imported-abc")
        let props = PALAnalyticsSchema.sanitize(PALRecords.benchmark(r, source: .curated), for: .benchmarkMeasured)
        let answers = Set(r.quality!.allTasks.map(\.answer))
        for value in props.values {
            if case .string(let s) = value {
                #expect(!answers.contains(s), "\(s)")
                #expect(!s.contains("Private exam") && !s.contains("Secret"), "\(s)")
            }
        }
        #expect(!props.keys.contains { $0.contains("answer") || $0.contains("task") || $0.contains("secret") })
    }
}
