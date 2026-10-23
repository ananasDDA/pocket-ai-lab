//
//  PALAnalyticsSchema.swift
//  local_ai_test
//
//  The contract of the measurement telemetry: which events exist and which
//  property keys each may carry, with the type and alphabet of every value.
//  The server (services/openpanel_pal_routes.py, APP_EVENTS) holds the same
//  list and refuses a whole batch with 422 when anything differs, so the two
//  must change together.
//
//  Only measurements of how a model and an engine behave on this device go
//  out, plus one app_opened per launch to count installations. Nothing a
//  person wrote or said ever has a key here: no prompt, answer, chat title,
//  attachment, file name, transcription or imported suite content. Strings
//  are short identifiers without spaces, so a sentence cannot fit anywhere.
//

import Foundation

nonisolated enum PALEvent: String, CaseIterable, Sendable {
    case appOpened = "app_opened"
    case modelDownloadFinished = "model_download_finished"
    case modelDownloadFailed = "model_download_failed"
    case modelLoadMeasured = "model_load_measured"
    case modelLoadFailed = "model_load_failed"
    case generationMeasured = "generation_measured"
    case generationFailed = "generation_failed"
    case benchmarkMeasured = "benchmark_measured"
}

nonisolated enum PALAnalyticsSchema {

    /// What a value may look like. Mirrors the server's rules.
    enum Rule: Sendable, Equatable {
        case number(max: Double)
        case integer(min: Int64 = 0, max: Int64)
        case bool
        /// Anchored regular expression the whole string must match.
        case slug(String)
        case oneOf(Set<String>)
    }

    static let modelIdPattern = #"^(import:[a-z0-9-]{1,32}|[A-Za-z0-9][A-Za-z0-9._/-]{0,119})$"#
    static let thermalStates: Set<String> = ["nominal", "fair", "serious", "critical", "unknown"]
    static let outcomes: Set<String> = ["ok", "cancelled", "context_full", "oom", "error"]
    static let batteryBuckets: Set<String> = ["0-20", "20-50", "50-80", "80-100"]
    /// Every backend raw value, plus "other" for a report whose backend name
    /// no longer resolves.
    static let backends: Set<String> = Set(ModelBackend.allCases.map(\.rawValue)).union(["other"])

    /// Every key any event may carry, with its rule. Per-event membership is
    /// in `keys`; `score_<category>` is matched separately.
    static let rules: [String: Rule] = {
        let seconds = Rule.number(max: 1e7)
        let megabytes = Rule.number(max: 1e7)
        let rate = Rule.number(max: 1e6)
        let tokens = Rule.integer(max: 10_000_000)
        let thermal = Rule.oneOf(thermalStates)
        let count = Rule.integer(max: 100_000)
        let version = Rule.slug(#"^[0-9A-Za-z.+-]{1,32}$"#)
        return [
            "model_id": .slug(modelIdPattern),
            "backend": .oneOf(backends),
            "backend_rev": .slug(#"^[a-z0-9][a-z0-9.+-]{0,39}$"#),
            "quantization": .slug(#"^[A-Za-z0-9_.+-]{1,32}$"#),
            "parameter_size": .slug(#"^[A-Za-z0-9_.+-]{1,16}$"#),
            "error_kind": .slug(#"^[a-z0-9][a-z0-9_.:-]{0,63}$"#),
            "http_status": .integer(min: 100, max: 599),
            "size_mb": megabytes,
            "seconds": seconds,
            "file_size_mb": megabytes,
            "cold": .bool,
            "load_seconds": seconds,
            "peak_memory_mb": megabytes,
            "memory_budget_mb": megabytes,
            "thermal_before": thermal,
            "thermal_after": thermal,
            "peak_thermal_state": thermal,
            "low_power_mode": .bool,
            "context_size": tokens,
            "prompt_tokens": tokens,
            "tokens_generated": tokens,
            "generation_tps": rate,
            "ttft_ms": .number(max: 1e9),
            "wall_clock_seconds": seconds,
            "battery_bucket": .oneOf(batteryBuckets),
            "has_image": .bool,
            "has_audio": .bool,
            "outcome": .oneOf(outcomes),
            "prefill_tps": rate,
            "generation_tps_stddev": rate,
            "generation_repetitions": .integer(max: 1000),
            "thermally_constrained": .bool,
            "throttle_drop_percent": .number(max: 100),
            "warmup_performed": .bool,
            "metrics_version": .integer(max: 1000),
            "pp512_tps": rate,
            "tg128_tps": rate,
            "tg128_tps_stddev": rate,
            "suite_id": .oneOf(["built-in", "imported"]),
            "suite_version": .integer(max: 1_000_000),
            "quality_passed": count,
            "quality_total": count,
            "quality_completed": count,
            "quality_truncated": .bool,
            "score": .number(max: 100),
            "app_version": version,
            "app_build": version,
            "report_id": .slug(#"^[0-9A-Fa-f-]{36}$"#),
        ]
    }()

    private static let modelKeys: Set<String> = ["model_id", "backend"]
    private static let loadKeys: Set<String> = modelKeys.union([
        "quantization", "parameter_size", "file_size_mb", "cold", "load_seconds",
        "peak_memory_mb", "memory_budget_mb", "thermal_before", "low_power_mode",
    ])
    private static let generationIdentityKeys: Set<String> = modelKeys.union([
        "kind", "backend_rev", "quantization", "context_size", "outcome", "error_kind",
    ])

    /// The allowlist, event by event.
    static let keys: [PALEvent: Set<String>] = [
        .appOpened: [],
        .modelDownloadFinished: modelKeys.union(["size_mb", "seconds"]),
        .modelDownloadFailed: modelKeys.union(["size_mb", "seconds", "error_kind", "http_status"]),
        .modelLoadMeasured: loadKeys,
        .modelLoadFailed: loadKeys.union(["error_kind"]),
        .generationMeasured: generationIdentityKeys.union([
            "prompt_tokens", "tokens_generated", "generation_tps", "ttft_ms", "wall_clock_seconds",
            "peak_memory_mb", "memory_budget_mb", "thermal_before", "thermal_after", "low_power_mode",
            "battery_bucket", "has_image", "has_audio",
        ]),
        .generationFailed: generationIdentityKeys,
        .benchmarkMeasured: modelKeys.union([
            "kind", "backend_rev", "quantization", "parameter_size", "load_seconds", "prefill_tps",
            "generation_tps", "generation_tps_stddev", "generation_repetitions", "peak_memory_mb",
            "memory_budget_mb", "thermal_before", "thermal_after", "peak_thermal_state",
            "thermally_constrained", "throttle_drop_percent", "warmup_performed", "metrics_version",
            "tokens_generated", "wall_clock_seconds", "pp512_tps", "tg128_tps", "tg128_tps_stddev",
            "suite_id", "suite_version", "quality_passed", "quality_total", "quality_completed",
            "quality_truncated", "score", "app_version", "app_build", "report_id",
        ]),
    ]

    /// Per-category scores of the built-in suite: `score_math`, `score_retrieval`, …
    static let scoreKeyPattern = #"^score_[a-z0-9_]{1,40}$"#

    /// `kind` is fixed per event, exactly as on the server.
    static let kindByEvent: [PALEvent: String] = [
        .generationMeasured: "chat", .generationFailed: "chat", .benchmarkMeasured: "benchmark",
    ]

    /// Keys without which the server refuses an event (REQUIRED there). An
    /// event that lacks one is dropped on its own in `PALAnalytics.track`, so
    /// it can never cost the other events of its batch a 422.
    static let required: [PALEvent: Set<String>] = {
        var out: [PALEvent: Set<String>] = [:]
        for event in PALEvent.allCases {
            var keys: Set<String> = event == .appOpened ? [] : ["model_id", "backend"]
            if event == .generationMeasured || event == .generationFailed { keys.formUnion(["kind", "outcome"]) }
            if event == .benchmarkMeasured { keys.insert("kind") }
            out[event] = keys
        }
        return out
    }()

    static func rule(for key: String, in event: PALEvent) -> Rule? {
        if key == "kind" { return kindByEvent[event].map { .oneOf([$0]) } }
        if keys[event]?.contains(key) == true { return rules[key] }
        if event == .benchmarkMeasured, matches(key, scoreKeyPattern) { return .number(max: 100) }
        return nil
    }

    /// Keeps only what the server will accept for this event: allowlisted keys
    /// whose value has the right type and alphabet. Anything else is dropped
    /// here rather than costing the whole batch a 422 there.
    /// The required keys a sanitized event is missing; empty when it may go.
    static func missingRequired(_ props: [String: PALAnalyticsValue], for event: PALEvent) -> Set<String> {
        (required[event] ?? []).subtracting(props.keys)
    }

    static func sanitize(_ raw: [String: Any], for event: PALEvent) -> [String: PALAnalyticsValue] {
        var out: [String: PALAnalyticsValue] = [:]
        for (key, value) in raw {
            guard let rule = rule(for: key, in: event), let typed = coerce(value, to: rule) else { continue }
            out[key] = typed
        }
        return out
    }

    static func coerce(_ value: Any, to rule: Rule) -> PALAnalyticsValue? {
        switch rule {
        case .bool:
            return (value as? Bool).map(PALAnalyticsValue.bool)
        case .integer(let lo, let hi):
            guard !(value is Bool) else { return nil }
            let int: Int64?
            switch value {
            case let v as Int: int = Int64(v)
            case let v as Int64: int = v
            case let v as Int32: int = Int64(v)
            case let v as UInt32: int = Int64(v)
            default: int = nil
            }
            guard let int, int >= lo, int <= hi else { return nil }
            return .int(int)
        case .number(let hi):
            guard !(value is Bool) else { return nil }
            let number: Double?
            switch value {
            case let v as Double: number = v
            case let v as Float: number = Double(v)
            case let v as Int: number = Double(v)
            case let v as Int64: number = Double(v)
            default: number = nil
            }
            guard let number, number.isFinite, number >= 0, number <= hi else { return nil }
            // Three decimals are plenty for any measurement here and keep the
            // payload readable.
            return .double((number * 1000).rounded() / 1000)
        case .slug(let pattern):
            guard let string = value as? String, matches(string, pattern) else { return nil }
            return .string(string)
        case .oneOf(let allowed):
            guard let string = value as? String, allowed.contains(string) else { return nil }
            return .string(string)
        }
    }

    static func matches(_ string: String, _ pattern: String) -> Bool {
        string.range(of: pattern, options: .regularExpression) != nil
    }
}

/// One property value on the wire: strings, numbers and booleans only.
nonisolated enum PALAnalyticsValue: Codable, Equatable, Sendable {
    case string(String), int(Int64), double(Double), bool(Bool)

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Int64.self) { self = .int(v) }
        else if let v = try? c.decode(Double.self) { self = .double(v) }
        else { self = .string(try c.decode(String.self)) }
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .string(let v): try c.encode(v)
        case .int(let v): try c.encode(v)
        case .double(let v): try c.encode(v)
        case .bool(let v): try c.encode(v)
        }
    }
}
