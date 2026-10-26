//
//  RunConditionsProbe.swift
//
//  The device side of `RunConditionsGate`: reads the thermal state, Low
//  Power Mode, the battery and the network path, and runs the polling loops
//  that wait on them. Every decision is made by the pure types in
//  `RunConditionsGate.swift`; this file only measures, sleeps and logs.
//

import Foundation
import Network
import UIKit

// MARK: - Readings

@MainActor
enum RunConditionsProbe {

    static func current() -> RunConditions {
        let device = UIDevice.current
        if !device.isBatteryMonitoringEnabled { device.isBatteryMonitoringEnabled = true }
        let charge: BatteryCharge
        switch device.batteryState {
        case .charging: charge = .charging
        case .full: charge = .full
        case .unplugged: charge = .unplugged
        case .unknown: charge = .unknown
        @unknown default: charge = .unknown
        }
        return RunConditions(
            thermal: ProcessInfo.processInfo.thermalState,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            batteryCharge: charge,
            batteryLevel: device.batteryLevel >= 0 ? Double(device.batteryLevel) : nil
        )
    }
}

/// The current network path, from one long-lived `NWPathMonitor`.
@MainActor
final class NetworkPathWatcher {

    static let shared = NetworkPathWatcher()

    private let monitor = NWPathMonitor()
    private var started = false

    private init() {}

    /// Idempotent. The first path arrives asynchronously, so a reading taken
    /// in the same instant may still be the monitor's initial unsatisfied
    /// placeholder; `waitForNetwork` polls, which covers that.
    func start() {
        guard !started else { return }
        started = true
        monitor.start(queue: DispatchQueue(label: "pal.network-path-watcher", qos: .utility))
    }

    var isSatisfied: Bool {
        start()
        return monitor.currentPath.status == .satisfied
    }

    /// "satisfied (wifi)", "unsatisfied", "requiresConnection (cellular, expensive)".
    var stateDescription: String {
        start()
        let path = monitor.currentPath
        let status: String
        switch path.status {
        case .satisfied: status = "satisfied"
        case .unsatisfied: status = "unsatisfied"
        case .requiresConnection: status = "requiresConnection"
        @unknown default: status = "unknown"
        }
        var details: [String] = []
        let types: [(NWInterface.InterfaceType, String)] = [
            (.wifi, "wifi"), (.cellular, "cellular"), (.wiredEthernet, "wired"),
            (.loopback, "loopback"), (.other, "other"),
        ]
        for (type, name) in types where path.usesInterfaceType(type) { details.append(name) }
        if path.isExpensive { details.append("expensive") }
        if path.isConstrained { details.append("constrained") }
        return details.isEmpty ? status : "\(status) (\(details.joined(separator: ", ")))"
    }
}

// MARK: - Waiting

/// What one gate wait came to.
nonisolated struct RunConditionsOutcome: Sendable, Hashable {
    /// Seconds spent waiting.
    var waited: TimeInterval
    /// Non-empty when the cap ran out: record these on the run.
    var issues: [String]
    /// True when the run must stop (power still bad at the cap).
    var stop: Bool

    static let clear = RunConditionsOutcome(waited: 0, issues: [], stop: false)
}

@MainActor
enum RunConditionsWaiter {

    /// How often a long wait repeats its log line.
    static let heartbeat: TimeInterval = 120

    /// Waits until the conditions `policy` asks for hold, the cap runs out,
    /// or the task is cancelled. Logs the start of every wait, a heartbeat,
    /// and how it ended, with the elapsed time and the state reached.
    ///
    /// - Parameters:
    ///   - checkThermal: false for a power-only check (before a download).
    ///   - onPoll: called after every reading (the runner refreshes its
    ///     thermal chip from it).
    /// - Throws: `CancellationError`.
    static func wait(
        policy: RunConditionsPolicy,
        checkThermal: Bool,
        log: (String) -> Void,
        onPoll: () -> Void = {}
    ) async throws -> RunConditionsOutcome {
        var gate = RunConditionsGate(policy: policy, checkThermal: checkThermal)
        let start = Date()
        var waitingSince: Date?
        var lastLog = Date.distantPast
        var lastKind = ""

        while true {
            try Task.checkCancellation()
            let conditions = RunConditionsProbe.current()
            onPoll()
            let elapsed = Date().timeIntervalSince(start)

            switch gate.observe(conditions, elapsed: elapsed) {
            case .proceed:
                if waitingSince != nil {
                    log(String(format: "[conditions] ready after %.0f s: %@", elapsed, conditions.summary))
                }
                return RunConditionsOutcome(waited: elapsed, issues: [], stop: false)

            case .wait(let reasons):
                // The first word of each reason ("thermal", "settling",
                // "battery", "low") is its kind; a change of kind is worth a
                // line, a counter ticking is not.
                let kind = reasons.map { $0.split(separator: " ").first.map(String.init) ?? $0 }
                    .joined(separator: ",")
                if waitingSince == nil {
                    waitingSince = Date()
                    lastLog = Date()
                    lastKind = kind
                    log("[conditions] waiting: \(reasons.joined(separator: "; ")) - "
                        + "up to \(Int(policy.maxWait)) s (\(conditions.summary))")
                } else if kind != lastKind || Date().timeIntervalSince(lastLog) >= heartbeat {
                    lastLog = Date()
                    lastKind = kind
                    log(String(format: "[conditions] waiting %.0f s: %@", elapsed, reasons.joined(separator: "; ")))
                }
                let remaining = max(policy.maxWait - elapsed, 0)
                let nap = max(min(policy.pollInterval, remaining), 1)
                try await Task.sleep(nanoseconds: UInt64(nap * 1_000_000_000))

            case .flag(let issues):
                log(String(format: "[conditions] cap of %.0f s reached - %@ - proceeding, the run is flagged (%@)",
                           policy.maxWait, issues.joined(separator: "; "), conditions.summary))
                return RunConditionsOutcome(waited: elapsed, issues: issues, stop: false)

            case .stop(let issues):
                log(String(format: "[conditions] cap of %.0f s reached - %@ - STOPPING: every number from "
                    + "here on would describe the power state, not the model (%@)",
                           policy.maxWait, issues.joined(separator: "; "), conditions.summary))
                return RunConditionsOutcome(waited: elapsed, issues: issues, stop: true)
            }
        }
    }

    /// Waits for the network path to be satisfied, up to `maxWait`. Returns
    /// true when it is. Silent when the path is already there.
    ///
    /// - Throws: `CancellationError`.
    static func waitForNetwork(
        maxWait: TimeInterval,
        pollInterval: TimeInterval = 10,
        log: (String) -> Void
    ) async throws -> Bool {
        let watcher = NetworkPathWatcher.shared
        watcher.start()
        let start = Date()
        var announced = false
        var lastLog = Date()
        while true {
            try Task.checkCancellation()
            let elapsed = Date().timeIntervalSince(start)
            switch NetworkGate.decide(satisfied: watcher.isSatisfied, elapsed: elapsed, maxWait: maxWait) {
            case .proceed:
                if announced {
                    log(String(format: "[network] back after %.0f s: %@", elapsed, watcher.stateDescription))
                }
                return true
            case .giveUp:
                log(String(format: "[network] still %@ after %.0f s - trying the download anyway",
                           watcher.stateDescription, elapsed))
                return false
            case .wait:
                // The monitor's very first path can be a placeholder; give it
                // two seconds before calling the network gone.
                if !announced, elapsed >= 2 {
                    announced = true
                    lastLog = Date()
                    log("[network] path \(watcher.stateDescription) - waiting for it to come back "
                        + "(up to \(Int(maxWait)) s)")
                } else if announced, Date().timeIntervalSince(lastLog) >= heartbeat {
                    lastLog = Date()
                    log(String(format: "[network] still %@ after %.0f s", watcher.stateDescription, elapsed))
                }
                let nap = elapsed < 2 ? 0.5 : pollInterval
                try await Task.sleep(nanoseconds: UInt64(nap * 1_000_000_000))
            }
        }
    }
}
