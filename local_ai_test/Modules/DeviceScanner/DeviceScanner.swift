//
//  DeviceScanner.swift
//  local_ai_test
//

import UIKit
import Darwin
import FoundationModels
import Metal

// MARK: - Models

struct DeviceInfo {
    let identifier: String        // "iPhone17,1"
    let marketingName: String     // "iPhone 16 Pro"
    let chip: String              // "A18 Pro"
    let totalRAM: UInt64          // bytes
    /// The process's jetsam LIMIT — the `phys_footprint` at which the kernel
    /// kills us. Measured as `os_proc_available_memory()` (what we may still
    /// allocate) plus the footprint we were already holding when the scan
    /// ran, because those two always sum to the same constant limit.
    ///
    /// Storing the limit rather than the raw allowance is what makes the
    /// budget stable: a rescan performed while a 4 GB model is loaded used to
    /// report a 4 GB smaller allowance and silently demote every model to
    /// "not enough resources". The limit already accounts for the
    /// increased-memory-limit entitlement, unlike any `totalRAM` heuristic.
    /// 0 when the kernel APIs are unavailable (simulator, unit tests).
    let jetsamLimit: Int64        // bytes
    /// `MTLDevice.recommendedMaxWorkingSetSize` (~0.67 × RAM on iPhone).
    /// 0 when Metal is unavailable.
    let metalWorkingSet: Int64    // bytes
    let iOSVersion: String        // "26.0"
    let freeDiskSpace: Int64      // bytes
    let appleIntelligence: AppleIntelligenceStatus

    /// The jetsam limit in GB, with a `totalRAM - 3` fallback for contexts
    /// (unit tests, simulator) where the kernel API reports nothing.
    var jetsamLimitGB: Double {
        if jetsamLimit > 0 {
            return Double(jetsamLimit) / 1_073_741_824
        }
        return Double(totalRAM) / 1_073_741_824 - 3.0
    }

    /// Headroom kept free inside the jetsam limit. The limit is where we get
    /// killed, not where we can comfortably live: the SwiftUI stack, the
    /// tokenizer, image buffers and — above all — a KV cache that keeps
    /// growing with the conversation all have to fit in the same envelope.
    ///
    /// Only small devices carry a reserve: a 6 GB phone gets a ~3 GB limit,
    /// where every model in reach fits "barely" and the KV cache tips it
    /// over. On 8 GB and up the 1.0 budgets already worked in the field, so
    /// no reserve — shrinking them would strand models users run today.
    /// (Device RAM tiers report ~5.5 GB for "6 GB" and ~7.3 GB for "8 GB",
    /// so 6.5 cleanly separates the two.)
    var dirtyReserveGB: Double {
        let ramGB = Double(totalRAM) / 1_073_741_824
        return ramGB < 6.5 ? 1.0 : 0.0
    }

    /// Budget for DIRTY memory — MLX and Core ML copy weights into anonymous
    /// Metal buffers, every byte of which jetsam charges to the process
    /// (`phys_footprint`). Derived from the stable limit, never from the
    /// instantaneous remainder, so the verdict does not move when a model is
    /// already resident.
    var usableRAMGB: Double {
        max(jetsamLimitGB - dirtyReserveGB, 0)
    }

    /// The budget for mmap-backed weights (llama.cpp / whisper.cpp GGUF).
    /// Clean file-backed pages are NOT charged to `phys_footprint`, so jetsam
    /// is not the binding constraint — the Metal working set (~0.67 × RAM)
    /// and raw physical-RAM pressure are.
    ///
    /// Physical RAM is shared with the OS, SpringBoard and our own dirty
    /// allocations, so the working set alone is too generous on small
    /// devices: 0.67 × 6 GB = 4 GB, and a 4 GB GGUF on a 6 GB phone thrashes
    /// the pager until the watchdog kills the app ("freezes, then crashes").
    /// So on small devices (< 6.5 GB measured, i.e. the 3-6 GB tiers) cap it
    /// at `totalRAM - systemReserve`, with the reserve scaled down on the
    /// tiniest devices so it never swallows most of the machine. 8 GB and up
    /// keep the pure Metal ceiling — those budgets already worked in 1.0.
    var mmapCeilingGB: Double {
        let ramGB = Double(totalRAM) / 1_073_741_824
        let metalCeiling = metalWorkingSet > 0
            ? min(Double(metalWorkingSet) / 1_073_741_824, ramGB * 0.70)
            : ramGB * 0.67
        guard ramGB < 6.5 else { return metalCeiling }
        let systemReserveGB = min(3.0, ramGB * 0.45)
        return max(min(metalCeiling, ramGB - systemReserveGB), 0)
    }

    /// Budget for a given model: mmap-backed backends get the Metal
    /// working-set / physical-RAM ceiling, everything else the dirty-memory
    /// budget carved out of the stable jetsam limit.
    func budgetGB(for backend: ModelBackend) -> Double {
        switch backend {
        case .llamaCpp, .llamaCppVision, .whisperCpp:
            return mmapCeilingGB
        default:
            return usableRAMGB
        }
    }
}

enum AppleIntelligenceStatus {
    case available
    case notEnabled       // device supports it, but the user has not enabled it
    case notSupported     // device does not support it
    case modelNotReady    // the model is still being installed
    case unknown(String)
}

// MARK: - DeviceScanner

@MainActor
final class DeviceScanner {

    static let shared = DeviceScanner()
    private init() {}

    func scan() -> DeviceInfo {
        let identifier = machineIdentifier()
        let ram = ProcessInfo.processInfo.physicalMemory
        let iosVersion = UIDevice.current.systemVersion
        let disk = freeDiskSpace()
        let aiStatus = appleIntelligenceStatus()
        let name = DeviceNameMapper.marketingName(for: identifier)
        let chip = DeviceNameMapper.chip(for: identifier)

        // allowance + footprint = the constant jetsam limit. Adding the
        // footprint back is what keeps a rescan taken with a model loaded
        // from reporting a smaller device than a rescan taken at launch.
        let allowance = Int64(os_proc_available_memory())
        let limit = allowance > 0 ? allowance + currentFootprint() : 0

        return DeviceInfo(
            identifier: identifier,
            marketingName: name,
            chip: chip,
            totalRAM: ram,
            jetsamLimit: limit,
            metalWorkingSet: Int64(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0),
            iOSVersion: iosVersion,
            freeDiskSpace: disk,
            appleIntelligence: aiStatus
        )
    }

    // MARK: - Private

    /// Current `phys_footprint` of this process in bytes — the number jetsam
    /// charges and Xcode's memory gauge shows. 0 when the call fails.
    private func currentFootprint() -> Int64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return 0 }
        return Int64(info.phys_footprint)
    }

    private func machineIdentifier() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var machine = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &machine, &size, nil, 0)
        return String(cString: machine)
    }

    private func freeDiskSpace() -> Int64 {
        guard let attrs = try? FileManager.default.attributesOfFileSystem(
            forPath: NSHomeDirectory()
        ) else { return 0 }
        return (attrs[.systemFreeSize] as? Int64) ?? 0
    }

    private func appleIntelligenceStatus() -> AppleIntelligenceStatus {
        guard #available(iOS 26, *) else { return .notSupported }

        switch SystemLanguageModel.default.availability {
        case .available:
            return .available
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible:
                return .notSupported
            case .appleIntelligenceNotEnabled:
                return .notEnabled
            case .modelNotReady:
                return .modelNotReady
            default:
                return .unknown(String(describing: reason))
            }
        }
    }
}

// MARK: - Helpers

extension DeviceInfo {
    /// Older call sites (test fixtures) build a device from a raw
    /// `os_proc_available_memory` reading. Measured with a negligible
    /// footprint that reading *is* the limit, so it maps straight onto
    /// `jetsamLimit`; 0 keeps them on the `totalRAM - 3` fallback.
    init(
        identifier: String,
        marketingName: String,
        chip: String,
        totalRAM: UInt64,
        processAllowance: Int64,
        metalWorkingSet: Int64,
        iOSVersion: String,
        freeDiskSpace: Int64,
        appleIntelligence: AppleIntelligenceStatus
    ) {
        self.init(
            identifier: identifier,
            marketingName: marketingName,
            chip: chip,
            totalRAM: totalRAM,
            jetsamLimit: processAllowance,
            metalWorkingSet: metalWorkingSet,
            iOSVersion: iOSVersion,
            freeDiskSpace: freeDiskSpace,
            appleIntelligence: appleIntelligence
        )
    }

    var totalRAMFormatted: String {
        let gb = Double(totalRAM) / 1_073_741_824
        return String(format: "%.0f GB", gb.rounded())
    }

    var freeDiskFormatted: String {
        let gb = Double(freeDiskSpace) / 1_073_741_824
        return String(format: "%.1f GB", gb)
    }
}
