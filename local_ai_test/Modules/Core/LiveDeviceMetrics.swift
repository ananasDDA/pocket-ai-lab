//
//  LiveDeviceMetrics.swift
//  local_ai_test
//
//  Lightweight system telemetry sampler used by the live-metrics overlay
//  attached at the top of `ChatView`. Samples once per second while
//  `isActive` is true and keeps a small ring buffer so sparkline charts
//  can be rendered without re-reading mach APIs on every redraw.
//
//  Metrics collected (all on-device, no network):
//    • CPU: aggregate user+system across cores, via `host_processor_info()`.
//      The first call just primes the diff, so the first emitted value is 0.
//    • App memory: `phys_footprint` from `task_info(TASK_VM_INFO)` — this is
//      the same figure Xcode's memory gauge reports.
//    • System memory: `host_statistics64(HOST_VM_INFO64)` for total/used RAM.
//    • Thermal state: `ProcessInfo.thermalState`.
//
//  Battery level is deliberately NOT sampled: `UIDevice.batteryLevel` is
//  quantised and refreshes on its own schedule, so it routinely disagrees
//  with the status-bar indicator by a percent — a worse experience than
//  just letting the user read the system indicator right above the panel.
//
//  This class is intentionally `@MainActor` — it's only used by SwiftUI
//  views, and the sampling rate (1 Hz) is negligible overhead on the main
//  loop. If a heavier rate is ever needed, move the mach calls off the
//  main queue and hop back to publish.
//

import Foundation
import Darwin
import MachO

@MainActor
@Observable
final class LiveDeviceMetrics {
    static let shared = LiveDeviceMetrics()

    struct Snapshot: Identifiable {
        let id = UUID()
        let timestamp: Date
        let cpuPercent: Double           // 0…100, total across all cores
        let appMemoryMB: Double          // phys_footprint of this process
        let systemUsedMemoryMB: Double   // active + wired + compressed
        let systemTotalMemoryMB: Double  // physical RAM installed
        let thermal: ProcessInfo.ThermalState
    }

    private(set) var history: [Snapshot] = []
    private(set) var isActive = false

    /// Size of the rolling window. 60 s @ 1 Hz ≈ one-minute sparklines.
    private let capacity = 60

    private var timer: Timer?
    private var lastCPUTicks: (user: UInt32, system: UInt32, idle: UInt32, nice: UInt32)?

    private init() {}

    // MARK: - Lifecycle

    /// Begins 1 Hz sampling. Safe to call repeatedly; duplicate calls are no-ops.
    func start() {
        guard !isActive else { return }
        isActive = true
        // Prime the CPU baseline so the first real sample can diff against it.
        _ = readCPUUsage()
        // Emit an immediate snapshot so the overlay shows something before 1 s.
        appendSnapshot()
        // Capture `self` via the singleton so the closure stays `Sendable` under
        // Swift 6's stricter actor isolation — the instance is always alive.
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { _ in
            Task { @MainActor in
                LiveDeviceMetrics.shared.appendSnapshot()
            }
        }
    }

    /// Stops sampling and frees the timer. Keeps `history` so a re-open of
    /// the overlay still shows the previous window briefly.
    func stop() {
        timer?.invalidate()
        timer = nil
        isActive = false
    }

    // MARK: - Sampling

    private func appendSnapshot() {
        let snap = Snapshot(
            timestamp: .now,
            cpuPercent: readCPUUsage(),
            appMemoryMB: readAppMemoryMB(),
            systemUsedMemoryMB: readSystemMemory().used,
            systemTotalMemoryMB: readSystemMemory().total,
            thermal: ProcessInfo.processInfo.thermalState
        )
        history.append(snap)
        if history.count > capacity {
            history.removeFirst(history.count - capacity)
        }
    }

    // MARK: - mach probes

    /// Aggregate CPU usage across all online cores in %.
    /// Based on the difference between two `host_processor_info` snapshots —
    /// the first call returns 0 because there is nothing to diff against yet.
    private func readCPUUsage() -> Double {
        var numCPUs: natural_t = 0
        var info: processor_info_array_t?
        var infoCount: mach_msg_type_number_t = 0

        let kr = host_processor_info(
            mach_host_self(),
            PROCESSOR_CPU_LOAD_INFO,
            &numCPUs,
            &info,
            &infoCount
        )
        guard kr == KERN_SUCCESS, let info else { return 0 }
        defer {
            let size = vm_size_t(infoCount) * vm_size_t(MemoryLayout<integer_t>.size)
            _ = vm_deallocate(mach_task_self_, vm_address_t(bitPattern: info), size)
        }

        var user: UInt32 = 0, system: UInt32 = 0, idle: UInt32 = 0, nice: UInt32 = 0
        for i in 0..<Int(numCPUs) {
            let base = i * Int(CPU_STATE_MAX)
            user   &+= UInt32(bitPattern: info[base + Int(CPU_STATE_USER)])
            system &+= UInt32(bitPattern: info[base + Int(CPU_STATE_SYSTEM)])
            idle   &+= UInt32(bitPattern: info[base + Int(CPU_STATE_IDLE)])
            nice   &+= UInt32(bitPattern: info[base + Int(CPU_STATE_NICE)])
        }

        guard let last = lastCPUTicks else {
            lastCPUTicks = (user, system, idle, nice)
            return 0
        }
        let dUser   = Double(user   &- last.user)
        let dSystem = Double(system &- last.system)
        let dIdle   = Double(idle   &- last.idle)
        let dNice   = Double(nice   &- last.nice)
        lastCPUTicks = (user, system, idle, nice)

        let busy = dUser + dSystem + dNice
        let total = busy + dIdle
        guard total > 0 else { return 0 }
        return min(100, max(0, busy / total * 100))
    }

    /// Current process `phys_footprint` in MB — matches Xcode's memory gauge.
    private func readAppMemoryMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Double(info.phys_footprint) / 1_048_576
    }

    /// System-wide RAM usage in MB (used, total). "Used" adds active, wired
    /// and compressed pages — the same split the Activity Monitor shows.
    private func readSystemMemory() -> (used: Double, total: Double) {
        let total = Double(ProcessInfo.processInfo.physicalMemory) / 1_048_576

        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return (0, total) }

        let pageSize = Double(vm_kernel_page_size)
        let activeMB     = Double(stats.active_count)     * pageSize / 1_048_576
        let wiredMB      = Double(stats.wire_count)       * pageSize / 1_048_576
        let compressedMB = Double(stats.compressor_page_count) * pageSize / 1_048_576
        return (activeMB + wiredMB + compressedMB, total)
    }
}

// MARK: - Display helpers

extension ProcessInfo.ThermalState {
    var displayName: String {
        switch self {
        case .nominal:  return "Nominal"
        case .fair:     return "Fair"
        case .serious:  return "Serious"
        case .critical: return "Critical"
        @unknown default: return "Unknown"
        }
    }

    /// 1…4 step used by the compact thermal gauge. iOS exposes no actual
    /// temperature reading to third-party apps, so this four-step pressure
    /// scale is the finest granularity available.
    var severityStep: Int {
        switch self {
        case .nominal:  return 1
        case .fair:     return 2
        case .serious:  return 3
        case .critical: return 4
        @unknown default: return 1
        }
    }
}
