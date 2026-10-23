//
//  PALAnalyticsRecords.swift
//  local_ai_test
//
//  Pure builders for the measurement records. Everything here is a function
//  of its arguments (no clocks, no device reads, no singletons), so the unit
//  tests pin exactly what each event carries. The probes in
//  PALMeasurements.swift collect the readings and hand them to these.
//

import Foundation

// MARK: - Model identity

nonisolated enum PALModelIdentity {

    /// The id a record carries. A catalog model sends its catalog id; a model
    /// the user imported (from a Hugging Face link or from Files) sends only
    /// `import:<kind>`, never its repository or file name. Anything that is
    /// not provably a catalog entry counts as imported.
    static func modelId(id: String, source: ModelSource?, backend: ModelBackend?) -> String {
        if source == .curated, PALAnalyticsSchema.matches(id, PALAnalyticsSchema.modelIdPattern),
           !id.hasPrefix("import:") {
            return id
        }
        return "import:" + importKind(backend)
    }

    static func modelId(for model: AIModel) -> String {
        modelId(id: model.id, source: model.source, backend: model.backend)
    }

    static func importKind(_ backend: ModelBackend?) -> String {
        switch backend {
        case .llamaCpp: return "gguf"
        case .llamaCppVision: return "gguf-vision"
        case .whisperCpp: return "whisper-ggml"
        case .mlx: return "mlx"
        case .mlxVision: return "mlx-vision"
        case .mlxAudio: return "mlx-audio"
        case .coreML: return "coreml"
        case .coreMLWhisper: return "coreml-whisper"
        case .coreMLKokoro: return "coreml-kokoro"
        case .appleIntelligence: return "apple"
        case nil: return "unknown"
        }
    }

    /// Quantization labels a non-catalog model may send as they are. The
    /// import flow names a quantization after the file (`ImportAnalyzer.quantLabel`
    /// returns the whole stem of a file without a quant tag), so anything
    /// outside this fixed vocabulary is sent as "other".
    static let knownQuantizationPattern =
        #"(?i)^(q[2-8](_(k|0|1|s|m|l|xs|xxs|xl|nl)){0,3}|iq[1-4](_(k|0|1|s|m|l|xs|xxs|xl|nl)){0,3}|f16|bf16|f32|int[48]|[2-8]bit|mxfp4|tq[12]_0|as-published|coreml|ggml|gguf)$"#

    /// Parameter counts a non-catalog model may send ("7B", "1.5B", "82M").
    static let knownParameterSizePattern = #"^[0-9]+(\.[0-9]+)?[BM]$"#

    /// A catalog model's labels are ours and go as they are; a model the user
    /// brought gets the fixed vocabulary or "other".
    static func quantization(_ raw: String, source: ModelSource?) -> String {
        if source == .curated { return raw }
        return PALAnalyticsSchema.matches(raw, knownQuantizationPattern) ? raw : "other"
    }

    static func parameterSize(_ raw: String, source: ModelSource?) -> String {
        if source == .curated { return raw }
        return PALAnalyticsSchema.matches(raw, knownParameterSizePattern) ? raw : "other"
    }

    /// Reports store `ModelBackend.displayName`; records carry the raw value.
    static func backend(displayName: String) -> ModelBackend? {
        ModelBackend.allCases.first { $0.displayName == displayName }
    }
}

// MARK: - Engine revision

nonisolated enum PALBackendRevision {

    /// llama.cpp pin of the shipped xcframework. MUST equal `rev=` in
    /// Vendor/llama.xcframework/LLAMA_REV (gitignored, so it cannot be read at
    /// build time without a project change): bump both together.
    static let llamaCpp = "b8882"

    /// mlx-swift version from local_ai_test.xcodeproj/.../Package.resolved.
    static let mlxSwift = "0.29.1"

    static func revision(for backend: ModelBackend) -> String {
        switch backend {
        case .llamaCpp, .llamaCppVision: return "llama.cpp-" + llamaCpp
        case .mlx, .mlxVision, .mlxAudio: return "mlx-swift-" + mlxSwift
        case .coreML, .coreMLWhisper, .coreMLKokoro: return "coreml"
        case .appleIntelligence: return "apple-intelligence"
        case .whisperCpp: return "whisper"
        }
    }
}

// MARK: - Device state

nonisolated enum PALDeviceState {

    /// `UIDevice.batteryLevel` in 0…1 → coarse bucket; nil when unknown (-1).
    static func batteryBucket(level: Float) -> String? {
        guard level >= 0, level <= 1 else { return nil }
        switch level {
        case ..<0.2: return "0-20"
        case ..<0.5: return "20-50"
        case ..<0.8: return "50-80"
        default: return "80-100"
        }
    }

    static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        state.reportName
    }

    /// Thermal spellings from reports ("nominal", …) pass through; anything
    /// else becomes "unknown" rather than an invalid value.
    static func thermalName(_ reportName: String) -> String {
        PALAnalyticsSchema.thermalStates.contains(reportName) ? reportName : "unknown"
    }
}

// MARK: - Errors

nonisolated enum PALErrorClassifier {

    enum Phase { case download, load, generate }

    /// A short machine kind for an error. Built from types, enum cases,
    /// domains and codes only — never from a message, which can contain a
    /// path or a model's own words.
    static func kind(of error: Error, phase: Phase) -> String {
        if error is CancellationError { return "cancelled" }
        if let inference = error as? InferenceError {
            switch inference {
            case .cancelled: return "cancelled"
            case .modelNotLoaded: return "model_not_loaded"
            case .unsupportedAttachment: return "unsupported_attachment"
            case .backendUnavailable: return "backend_unavailable"
            case .generationFailed(let detail):
                // Matched, not forwarded: the message never leaves the device.
                let lower = detail.lowercased()
                if lower.contains("context window") || lower.contains("no kv slot") { return "context_full" }
                if lower.contains("memory") { return "out_of_memory" }
                return phase == .load ? "engine_load_failed" : "generation_failed"
            }
        }
        if let download = error as? ModelDownloadError {
            switch download {
            case .preferredPackageMissing: return "package_missing"
            }
        }
        if error is DecodingError { return "decoding" }
        let ns = error as NSError
        if ns.domain == NSURLErrorDomain { return "url:\(ns.code)" }
        let domain = slug(ns.domain)
        if !domain.isEmpty, ns.domain != "\(type(of: error))" {
            return String("\(domain):\(ns.code)".prefix(64))
        }
        return String(("type:" + slug(String(describing: type(of: error)))).prefix(64))
    }

    /// Generation outcome for the record.
    static func outcome(of error: Error?, cancelled: Bool, memoryPressureUnload: Bool) -> (outcome: String, kind: String?) {
        if memoryPressureUnload { return ("oom", "memory_pressure_unload") }
        guard let error else { return cancelled ? ("cancelled", "cancelled") : ("ok", nil) }
        let kind = self.kind(of: error, phase: .generate)
        switch kind {
        case "cancelled": return ("cancelled", kind)
        case "context_full": return ("context_full", kind)
        case "out_of_memory": return ("oom", kind)
        default: return ("error", kind)
        }
    }

    static func slug(_ raw: String) -> String {
        var out = ""
        var lastSeparator = false
        for scalar in raw.lowercased().unicodeScalars {
            if ("a"..."z").contains(scalar) || ("0"..."9").contains(scalar) || scalar == "." {
                out.unicodeScalars.append(scalar)
                lastSeparator = false
            } else if !lastSeparator, !out.isEmpty {
                out.append("_")
                lastSeparator = true
            }
        }
        while out.hasSuffix("_") { out.removeLast() }
        return String(out.prefix(48))
    }
}

// MARK: - Chat generation meter

/// Accumulates what one chat generation reveals, the same way the benchmark
/// measures (`BenchmarkEngine.generate`): time to first token from the start
/// of the stream, decode rate from the first token to the last over the
/// tokens after the first, and the engine's own token count when it reports
/// one (a text chunk can carry more than one token).
nonisolated struct PALGenerationMeter: Sendable {
    let start: Date
    private(set) var firstTokenAt: Date?
    private(set) var lastTokenAt: Date?
    private(set) var deltaCount = 0
    private(set) var reportedTokens: Int?
    private(set) var promptTokens: Int?

    init(start: Date) { self.start = start }

    mutating func observe(_ output: ChatOutput, at now: Date) {
        switch output {
        case .textDelta:
            if firstTokenAt == nil { firstTokenAt = now }
            lastTokenAt = now
            deltaCount += 1
        case .diagnostic(.finished(let tokens, _)):
            reportedTokens = tokens
        case .diagnostic(.promptProcessed(let count, _)):
            promptTokens = count
        case .diagnostic, .audioChunk:
            break
        }
    }

    var tokensGenerated: Int { reportedTokens ?? deltaCount }

    var ttftMs: Double? {
        firstTokenAt.map { $0.timeIntervalSince(start) * 1000 }
    }

    func decodeTokensPerSecond(end: Date) -> Double? {
        guard let first = firstTokenAt else { return nil }
        let window = (lastTokenAt ?? end).timeIntervalSince(first)
        let counted = tokensGenerated
        guard window > 0, counted > 1 else { return nil }
        return Double(counted - 1) / window
    }
}

// MARK: - Records

/// Readings a probe takes around a load or a generation.
nonisolated struct PALConditions: Sendable {
    var thermalBefore: String
    var thermalAfter: String?
    var lowPowerMode: Bool
    var batteryBucket: String?
    var memoryBudgetMB: Int64
    var peakMemoryMB: Double
}

nonisolated enum PALRecords {

    static func download(model: AIModel, seconds: Double?, errorKind: String? = nil,
                         httpStatus: Int? = nil) -> [String: Any] {
        var out: [String: Any] = [
            "model_id": PALModelIdentity.modelId(for: model),
            "backend": model.backend.rawValue,
            "size_mb": model.diskSizeGB * 1024,
        ]
        out["seconds"] = seconds
        out["error_kind"] = errorKind
        out["http_status"] = httpStatus
        return out
    }

    static func load(model: AIModel, cold: Bool, loadSeconds: Double, conditions: PALConditions,
                     errorKind: String? = nil) -> [String: Any] {
        var out = modelFields(model)
        out["parameter_size"] = PALModelIdentity.parameterSize(model.parameterSize, source: model.source)
        out["file_size_mb"] = model.diskSizeGB * 1024
        out["cold"] = cold
        out["load_seconds"] = loadSeconds
        out["peak_memory_mb"] = conditions.peakMemoryMB
        out["memory_budget_mb"] = conditions.memoryBudgetMB
        out["thermal_before"] = conditions.thermalBefore
        out["low_power_mode"] = conditions.lowPowerMode
        out["error_kind"] = errorKind
        return out
    }

    /// `generation_measured` when the generation produced tokens or ended
    /// normally; `generation_failed` (identity, outcome and kind only) when it
    /// ended in an error or a cancel before the first token.
    static func generation(model: AIModel, contextSize: Int, meter: PALGenerationMeter, end: Date,
                           outcome: String, errorKind: String?, conditions: PALConditions,
                           hasImage: Bool, hasAudio: Bool) -> (PALEvent, [String: Any]) {
        var out = modelFields(model)
        out["kind"] = "chat"
        out["backend_rev"] = PALBackendRevision.revision(for: model.backend)
        out["context_size"] = contextSize
        out["outcome"] = outcome
        out["error_kind"] = errorKind
        if outcome != "ok", meter.firstTokenAt == nil {
            return (.generationFailed, out)
        }
        out["prompt_tokens"] = meter.promptTokens
        out["tokens_generated"] = meter.tokensGenerated
        out["generation_tps"] = meter.decodeTokensPerSecond(end: end)
        out["ttft_ms"] = meter.ttftMs
        out["wall_clock_seconds"] = end.timeIntervalSince(meter.start)
        out["peak_memory_mb"] = conditions.peakMemoryMB
        out["memory_budget_mb"] = conditions.memoryBudgetMB
        out["thermal_before"] = conditions.thermalBefore
        out["thermal_after"] = conditions.thermalAfter
        out["low_power_mode"] = conditions.lowPowerMode
        out["battery_bucket"] = conditions.batteryBucket
        out["has_image"] = hasImage
        out["has_audio"] = hasAudio
        return (.generationMeasured, out)
    }

    /// One stored benchmark report, flattened. `source` is where the model
    /// came from (nil when it can no longer be resolved, which counts as
    /// imported). Per-category scores are sent for the built-in suite only:
    /// an imported suite's categories are the user's own words.
    static func benchmark(_ report: BenchmarkReport, source: ModelSource?) -> [String: Any] {
        let backend = PALModelIdentity.backend(displayName: report.backend)
        let perf = report.performance
        var out: [String: Any] = [
            "kind": "benchmark",
            "model_id": PALModelIdentity.modelId(id: report.modelId, source: source, backend: backend),
            "quantization": PALModelIdentity.quantization(report.quantization, source: source),
            "parameter_size": PALModelIdentity.parameterSize(report.parameterSize, source: source),
            "load_seconds": perf.loadSeconds,
            "prefill_tps": perf.prefillTokensPerSecond,
            "generation_tps": perf.generationTokensPerSecond,
            "generation_repetitions": perf.generationRepetitions,
            "peak_memory_mb": perf.peakMemoryMB,
            "memory_budget_mb": perf.memoryBudgetMB,
            "thermal_before": PALDeviceState.thermalName(perf.thermalStateBefore),
            "thermal_after": PALDeviceState.thermalName(perf.thermalStateAfter),
            "thermally_constrained": perf.thermallyConstrained,
            "warmup_performed": perf.warmupPerformed,
            "metrics_version": perf.metricsVersion,
            "tokens_generated": perf.tokensGenerated,
            "wall_clock_seconds": perf.wallClockSeconds,
            "app_version": report.appVersion,
            "app_build": report.appBuild,
            "report_id": report.id,
        ]
        // Total: a backend name this build no longer knows is "other", never a
        // missing required key.
        out["backend"] = backend?.rawValue ?? "other"
        if let backend {
            out["backend_rev"] = PALBackendRevision.revision(for: backend)
        }
        out["generation_tps_stddev"] = perf.generationTokensPerSecondStdDev
        out["throttle_drop_percent"] = perf.throttleDropPercent
        out["peak_thermal_state"] = perf.peakThermalState.map(PALDeviceState.thermalName)
        if let standard = perf.standardProtocol {
            out["pp512_tps"] = standard.prefillTokensPerSecond
            out["tg128_tps"] = standard.generationTokensPerSecond
            out["tg128_tps_stddev"] = standard.generationTokensPerSecondStdDev
        }
        if let quality = report.quality {
            let builtIn = (quality.suiteID ?? BenchmarkSuiteDocument.builtInID) == BenchmarkSuiteDocument.builtInID
            out["suite_id"] = builtIn ? BenchmarkSuiteDocument.builtInID : "imported"
            out["suite_version"] = quality.suiteVersion
            out["quality_passed"] = quality.passed
            // `total` is what the run set out to grade, `completed` what it
            // graded; `truncated` means the time budget cut it short, and then
            // every score below is over the completed tasks only.
            out["quality_total"] = quality.total
            out["quality_completed"] = quality.completed
            out["quality_truncated"] = quality.truncated
            if quality.completed > 0 {
                out["score"] = quality.normalizedScore * 100
            }
            if builtIn {
                for category in quality.categories where category.total > 0 {
                    let key = "score_" + PALErrorClassifier.slug(category.category).replacingOccurrences(of: ".", with: "_")
                    out[key] = category.normalizedScore * 100
                }
            }
        }
        return out
    }

    private static func modelFields(_ model: AIModel) -> [String: Any] {
        [
            "model_id": PALModelIdentity.modelId(for: model),
            "backend": model.backend.rawValue,
            "quantization": PALModelIdentity.quantization(model.quantization, source: model.source),
        ]
    }
}
