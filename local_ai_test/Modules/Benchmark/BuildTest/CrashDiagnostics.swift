//
//  CrashDiagnostics.swift
//  local_ai_test
//
//  What the OS is willing to say about the last death of this process.
//
//  MetricKit hands an app its own crash diagnostics on the launch after the
//  crash: signal, Mach exception type and code, the kernel's termination
//  reason (which is how a jetsam kill or a watchdog shows up), and the build
//  it happened in. That is exactly what a `crashed` build-test entry lacks
//  when it is reconstructed from the journal alone, so the summaries are
//  kept in a small file next to the reports and `BuildTestRecovery` picks
//  the one that fits the window in which the model was in flight.
//
//  Delivery is asynchronous and usually lands seconds after launch - after
//  the recovery has already run - so a payload that arrives late is also
//  attached to the most recent interrupted report by `BuildTestRunner`.
//  The full call-stack tree MetricKit provides is NOT stored: it is large,
//  unsymbolicated, and nothing here can read it.
//

import Foundation
import MetricKit

/// The compact record of one crash, as this app understands it.
nonisolated struct CrashSummary: Codable, Sendable, Hashable {
    let timeStampEnd: Date
    let exceptionType: Int?
    let exceptionCode: Int?
    let signal: Int?
    let terminationReason: String?
    let virtualMemoryRegionInfo: String?
    let appBuild: String?

    /// One line for an entry's error text.
    var text: String {
        var parts: [String] = []
        if let terminationReason, !terminationReason.isEmpty {
            parts.append("termination reason \(terminationReason)")
        }
        if let signal { parts.append("signal \(signal)\(Self.signalName(signal).map { " (\($0))" } ?? "")") }
        if let exceptionType { parts.append("exception type \(exceptionType)") }
        if let exceptionCode { parts.append("code \(exceptionCode)") }
        if let virtualMemoryRegionInfo, !virtualMemoryRegionInfo.isEmpty {
            parts.append(virtualMemoryRegionInfo)
        }
        if let appBuild { parts.append("build \(appBuild)") }
        return parts.isEmpty ? "crash reported, no details" : parts.joined(separator: ", ")
    }

    static func signalName(_ signal: Int) -> String? {
        switch signal {
        case 4:  return "SIGILL"
        case 5:  return "SIGTRAP"
        case 6:  return "SIGABRT - an assertion, e.g. GGML_ASSERT"
        case 9:  return "SIGKILL - the system killed it: memory or watchdog"
        case 10: return "SIGBUS"
        case 11: return "SIGSEGV"
        default: return nil
        }
    }
}

/// Subscribes to MetricKit for the life of the process and persists what
/// arrives. Start it once, early - `ContentView` does - so a payload
/// delivered right after launch is not missed.
nonisolated final class CrashDiagnosticsCollector: NSObject, MXMetricManagerSubscriber, Sendable {

    static let shared = CrashDiagnosticsCollector()

    static let filename = ".crash-diagnostics.json"
    static let maxSummaries = 20

    private static var url: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("BuildTestReports", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent(filename)
    }

    func start() {
        MXMetricManager.shared.add(self)
    }

    /// Everything stored so far, oldest first.
    static func summaries() -> [CrashSummary] {
        guard let data = try? Data(contentsOf: url),
              let stored = try? BenchmarkReportStore.makeDecoder().decode([CrashSummary].self, from: data)
        else { return [] }
        return stored
    }

    static func store(_ new: [CrashSummary]) {
        guard !new.isEmpty else { return }
        var all = summaries()
        for summary in new where !all.contains(summary) { all.append(summary) }
        all.sort { $0.timeStampEnd < $1.timeStampEnd }
        if all.count > maxSummaries { all.removeFirst(all.count - maxSummaries) }
        if let data = try? BenchmarkReportStore.makeEncoder().encode(all) {
            try? data.write(to: url, options: .atomic)
        }
    }

    // MARK: MXMetricManagerSubscriber

    func didReceive(_ payloads: [MXDiagnosticPayload]) {
        var summaries: [CrashSummary] = []
        for payload in payloads {
            for crash in payload.crashDiagnostics ?? [] {
                summaries.append(CrashSummary(
                    timeStampEnd: payload.timeStampEnd,
                    exceptionType: crash.exceptionType?.intValue,
                    exceptionCode: crash.exceptionCode?.intValue,
                    signal: crash.signal?.intValue,
                    terminationReason: crash.terminationReason,
                    virtualMemoryRegionInfo: crash.virtualMemoryRegionInfo,
                    appBuild: crash.metaData.applicationBuildVersion
                ))
            }
        }
        guard !summaries.isEmpty else { return }
        Self.store(summaries)
        let delivered = summaries
        Task { @MainActor in
            BuildTestRunner.shared.attachCrashDiagnostics(delivered)
        }
    }

    func didReceive(_ payloads: [MXMetricPayload]) {
        // Performance metrics are not this collector's business.
    }
}
