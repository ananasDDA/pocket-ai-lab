//
//  RunConditionsGate.swift
//
//  The decisions behind the waits a long unattended run makes before it
//  measures anything: is the phone cool enough, is it on power (or has
//  enough battery), is the network there, has a download stalled, and in
//  which order should the models run. Pure functions and small value types
//  only - no UIKit, no clocks, no sleeping - so every decision is unit
//  tested. The device side (reading the battery, NWPathMonitor, the polling
//  loop) is in `RunConditionsProbe.swift`.
//
//  Shared by the catalog sweep (`BuildTestRunner`, driven by
//  `CatalogBenchmarkTests`) and the daily-use suite (`DailyUseDeviceTests`,
//  driven by `Scripts/daily_use_night.sh`).
//
//  Why: overnight runs on an iPhone Air left the phone at `serious` after
//  every model. Waiting for `fair` with a five-minute cap proceeded hot
//  twice a night, and one night the phone dropped to a flat ~10x lower speed
//  (Low Power Mode, or a battery that stopped charging while hot) and the
//  harness kept producing numbers for hours. The gate now waits for
//  `nominal` plus a settle period, refuses to measure on a draining battery,
//  and flags - or stops - instead of silently proceeding.
//

import Foundation

// MARK: - Device conditions

/// `UIDevice.BatteryState`, without UIKit.
nonisolated enum BatteryCharge: String, Sendable, Codable, Hashable {
    case charging, full, unplugged, unknown
}

/// One reading of everything the gate looks at.
nonisolated struct RunConditions: Sendable, Hashable {
    var thermal: ProcessInfo.ThermalState
    var lowPowerMode: Bool
    var batteryCharge: BatteryCharge
    /// 0...1, `nil` when the OS does not know (simulator, monitoring off).
    var batteryLevel: Double?

    init(
        thermal: ProcessInfo.ThermalState,
        lowPowerMode: Bool = false,
        batteryCharge: BatteryCharge = .charging,
        batteryLevel: Double? = 1
    ) {
        self.thermal = thermal
        self.lowPowerMode = lowPowerMode
        self.batteryCharge = batteryCharge
        self.batteryLevel = batteryLevel
    }

    /// "battery 23%, unplugged".
    var batteryDescription: String {
        let level = batteryLevel.map { String(format: "%.0f%%", $0 * 100) } ?? "?"
        return "battery \(level), \(batteryCharge.rawValue)"
    }

    /// "thermal fair, low power off, battery 80%, charging".
    var summary: String {
        "thermal \(thermal.reportName), low power \(lowPowerMode ? "on" : "off"), \(batteryDescription)"
    }
}

// MARK: - Policy

/// How long and for what a run waits before each model (and, in the daily
/// suite, before each scenario).
nonisolated struct RunConditionsPolicy: Codable, Sendable, Hashable {

    enum ThermalTarget: String, Codable, Sendable, Hashable {
        /// Wait for `nominal`: the phone is actually cool.
        case nominal
        /// Wait for `fair` (the old between-model rule).
        case fair
        /// No thermal wait at all.
        case off

        /// Highest acceptable `severityRank`, `nil` for `.off`.
        var maxRank: Int? {
            switch self {
            case .nominal: return 0
            case .fair: return 1
            case .off: return nil
            }
        }
    }

    var thermalTarget: ThermalTarget
    /// Longest one wait may last, in seconds. When it runs out the run is
    /// flagged (or, for bad power with `stopOnBadPower`, stopped).
    var maxWait: TimeInterval
    /// Extra idle time once the target is reached, when the gate had to wait
    /// for it: a state that has only just dropped under the threshold
    /// describes a SoC that is still warm.
    var settle: TimeInterval
    /// Require Low Power Mode off and (charging or full, or the battery at
    /// `minBatteryUnplugged` or more).
    var powerGate: Bool
    /// 0...1.
    var minBatteryUnplugged: Double
    /// When power is still bad at the cap: stop the run instead of measuring
    /// garbage for every model that is left.
    var stopOnBadPower: Bool
    /// Fail a download that moves no bytes for this long, and wait for a
    /// lost network path before the next download. `nil` keeps the old
    /// behaviour (only the 30-minute download budget).
    var downloadStallTimeout: TimeInterval?
    /// Seconds between readings while waiting.
    var pollInterval: TimeInterval

    init(
        thermalTarget: ThermalTarget,
        maxWait: TimeInterval,
        settle: TimeInterval,
        powerGate: Bool,
        minBatteryUnplugged: Double = 0.40,
        stopOnBadPower: Bool,
        downloadStallTimeout: TimeInterval?,
        pollInterval: TimeInterval = 10
    ) {
        self.thermalTarget = thermalTarget
        self.maxWait = max(0, maxWait)
        self.settle = max(0, settle)
        self.powerGate = powerGate
        self.minBatteryUnplugged = min(max(minBatteryUnplugged, 0), 1)
        self.stopOnBadPower = stopOnBadPower
        self.downloadStallTimeout = downloadStallTimeout.flatMap { $0 > 0 ? $0 : nil }
        self.pollInterval = max(1, pollInterval)
    }

    /// Unattended runs (the Xcode harness, the daily-use night).
    static let night = RunConditionsPolicy(
        thermalTarget: .nominal,
        maxWait: 20 * 60,
        settle: 60,
        powerGate: true,
        stopOnBadPower: true,
        downloadStallTimeout: 5 * 60
    )

    /// The in-app build test: the rule it always had - cool to `fair`
    /// between models for at most five minutes, no power or network gate.
    /// Someone is holding the phone and can see what is going on.
    static let interactive = RunConditionsPolicy(
        thermalTarget: .fair,
        maxWait: 5 * 60,
        settle: 0,
        powerGate: false,
        stopOnBadPower: false,
        downloadStallTimeout: nil
    )

    /// Reads `<prefix>COOLDOWN_TARGET=nominal|fair|off`,
    /// `<prefix>COOLDOWN_MAX_SEC`, `<prefix>COOLDOWN_SETTLE_SEC`,
    /// `<prefix>STOP_ON_BAD_POWER=0|1` and `<prefix>DOWNLOAD_STALL_SEC`
    /// (0 turns stall detection off) on top of `base`. Garbage keeps the
    /// base value.
    static func fromEnvironment(
        _ environment: [String: String],
        prefix: String,
        base: RunConditionsPolicy = .night
    ) -> RunConditionsPolicy {
        func value(_ key: String) -> String? {
            guard let raw = environment[prefix + key]?
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased(), !raw.isEmpty else { return nil }
            return raw
        }
        func seconds(_ key: String) -> TimeInterval? {
            value(key).flatMap(TimeInterval.init).flatMap { $0.isFinite && $0 >= 0 ? $0 : nil }
        }

        var policy = base
        if let target = value("COOLDOWN_TARGET").flatMap(ThermalTarget.init(rawValue:)) {
            policy.thermalTarget = target
        }
        if let cap = seconds("COOLDOWN_MAX_SEC") { policy.maxWait = cap }
        if let settle = seconds("COOLDOWN_SETTLE_SEC") { policy.settle = settle }
        switch value("STOP_ON_BAD_POWER") {
        case "1", "true", "yes": policy.stopOnBadPower = true
        case "0", "false", "no": policy.stopOnBadPower = false
        default: break
        }
        if let stall = seconds("DOWNLOAD_STALL_SEC") {
            policy.downloadStallTimeout = stall > 0 ? stall : nil
        }
        return policy
    }

    /// One line for a log header.
    var summaryLine: String {
        var parts: [String] = []
        if thermalTarget == .off {
            parts.append("no cooldown")
        } else {
            parts.append("cool to \(thermalTarget.rawValue)"
                + (settle > 0 ? " + \(Int(settle)) s settle" : ""))
        }
        parts.append("cap \(Int(maxWait)) s")
        if powerGate {
            parts.append("power gate (no low power, charging or >= \(Int(minBatteryUnplugged * 100))%"
                + (stopOnBadPower ? ", stop when not met" : ", flag when not met") + ")")
        }
        if let downloadStallTimeout { parts.append("download stall \(Int(downloadStallTimeout)) s") }
        return parts.joined(separator: ", ")
    }
}

// MARK: - The gate

nonisolated enum RunConditionsDecision: Sendable, Hashable {
    /// Measure now.
    case proceed
    /// Keep waiting; the reasons are for the log.
    case wait([String])
    /// The cap ran out: measure anyway, with these issues recorded on the
    /// run (`measurementIssues`).
    case flag([String])
    /// The cap ran out with the power still bad and `stopOnBadPower` set:
    /// do not measure this or any later model.
    case stop([String])
}

/// Turns a series of readings into a decision. A struct with a little state
/// (did it ever see the phone hot, since when has it been cool) so the
/// settle period can be decided from readings and elapsed time alone.
nonisolated struct RunConditionsGate: Sendable {

    let policy: RunConditionsPolicy
    /// False for a gate that only cares about power (before a download).
    let checkThermal: Bool

    /// True once a reading was above the target: only then is there
    /// something to settle.
    private(set) var sawHot = false
    /// Elapsed time of the first reading of the current cool stretch.
    private(set) var coolSince: TimeInterval?

    init(policy: RunConditionsPolicy, checkThermal: Bool = true) {
        self.policy = policy
        self.checkThermal = checkThermal
    }

    /// Why the power is not good enough, empty when it is (or when the gate
    /// does not check power).
    static func powerProblems(_ conditions: RunConditions, policy: RunConditionsPolicy) -> [String] {
        guard policy.powerGate else { return [] }
        var problems: [String] = []
        if conditions.lowPowerMode { problems.append("low power mode on") }
        switch conditions.batteryCharge {
        case .charging, .full:
            break
        case .unplugged, .unknown:
            // An unknown level is not held against the run: nothing can be
            // done about a phone that does not report its battery.
            if let level = conditions.batteryLevel, level < policy.minBatteryUnplugged {
                problems.append(conditions.batteryDescription)
            }
        }
        return problems
    }

    mutating func observe(_ conditions: RunConditions, elapsed: TimeInterval) -> RunConditionsDecision {
        let maxRank = checkThermal ? policy.thermalTarget.maxRank : nil
        let thermalOK = maxRank.map { conditions.thermal.severityRank <= $0 } ?? true
        if thermalOK {
            if coolSince == nil { coolSince = elapsed }
        } else {
            sawHot = true
            coolSince = nil
        }

        let power = Self.powerProblems(conditions, policy: policy)
        var reasons: [String] = []
        if !thermalOK {
            reasons.append("thermal \(conditions.thermal.reportName) (target \(policy.thermalTarget.rawValue))")
        }
        reasons += power

        if reasons.isEmpty {
            if sawHot, policy.settle > 0, let since = coolSince,
               elapsed - since < policy.settle, elapsed < policy.maxWait {
                return .wait([String(
                    format: "settling at %@, %.0f of %.0f s",
                    conditions.thermal.reportName, elapsed - since, policy.settle
                )])
            }
            return .proceed
        }

        guard elapsed >= policy.maxWait else { return .wait(reasons) }

        var issues: [String] = []
        if !thermalOK {
            issues.append(String(
                format: "cooldown cap: still %@ after %.0f s (target %@)",
                conditions.thermal.reportName, elapsed, policy.thermalTarget.rawValue
            ))
        }
        if !power.isEmpty {
            issues.append(String(
                format: "power gate cap: %@ after %.0f s", power.joined(separator: ", "), elapsed
            ))
        }
        return !power.isEmpty && policy.stopOnBadPower ? .stop(issues) : .flag(issues)
    }
}

// MARK: - Network

nonisolated enum NetworkGateDecision: Sendable, Hashable {
    case proceed
    case wait
    /// The path did not come back within the cap: try the download anyway
    /// (it will fail fast on the stall rule).
    case giveUp
}

nonisolated enum NetworkGate {
    static func decide(satisfied: Bool, elapsed: TimeInterval, maxWait: TimeInterval) -> NetworkGateDecision {
        if satisfied { return .proceed }
        return elapsed >= maxWait ? .giveUp : .wait
    }
}

/// Detects a download that stopped moving bytes. Fed the downloaded byte
/// count at every poll; `nil` (manifest fetch, idle) counts as no progress.
nonisolated struct DownloadStallWatch: Sendable {
    let timeout: TimeInterval
    private(set) var bestBytes: Int64 = 0
    private(set) var lastProgressAt: TimeInterval

    init(timeout: TimeInterval, startedAt elapsed: TimeInterval = 0) {
        self.timeout = timeout
        self.lastProgressAt = elapsed
    }

    /// True when no new byte arrived for `timeout` seconds.
    mutating func observe(bytes: Int64?, elapsed: TimeInterval) -> Bool {
        if let bytes, bytes > bestBytes {
            bestBytes = bytes
            lastProgressAt = elapsed
        }
        return elapsed - lastProgressAt >= timeout
    }

    /// Restarts the clock - for a phase that legitimately moves no bytes
    /// (on-device Core ML compilation).
    mutating func reset(at elapsed: TimeInterval) {
        lastProgressAt = elapsed
    }

    /// Seconds since the last new byte.
    func idle(at elapsed: TimeInterval) -> TimeInterval { elapsed - lastProgressAt }
}

// MARK: - Ordering

nonisolated enum RunOrdering {
    /// Heaviest, lightest, second heaviest, second lightest, ... so every
    /// heavy model is followed by a light one that lets the phone recover.
    /// Ties keep their original order. The first model of a night runs on a
    /// cool phone, so it is the heaviest.
    static func interleaveHeavyLight<T>(_ items: [T], weight: (T) -> Double) -> [T] {
        let sorted = items.indices.sorted { a, b in
            let wa = weight(items[a]), wb = weight(items[b])
            return wa != wb ? wa > wb : a < b
        }
        var result: [T] = []
        result.reserveCapacity(items.count)
        var low = 0, high = sorted.count - 1
        var takeHeavy = true
        while low <= high {
            if takeHeavy {
                result.append(items[sorted[low]])
                low += 1
            } else {
                result.append(items[sorted[high]])
                high -= 1
            }
            takeHeavy.toggle()
        }
        return result
    }
}

// MARK: - Checklist

/// Printed at the start of a night; the owner reads the log in the morning
/// and should see first whether the phone was set up for it.
nonisolated enum NightChecklist {
    static let lines: [String] = [
        "Night checklist (the gates wait for these, but cannot fix them):",
        "  - Auto-Lock: Never (a locked phone sleeps Wi-Fi and blocks xcodebuild)",
        "  - cable to the Mac, charging",
        "  - Low Power Mode off",
        "  - no case",
        "  - phone on a cool hard surface, not on the laptop",
        "  - screen brightness at minimum",
    ]
}
