//
//  DeviceScanner.swift
//  local_ai_test
//

import UIKit
import FoundationModels
import Metal

// MARK: - Models

struct DeviceInfo {
    let identifier: String        // "iPhone17,1"
    let marketingName: String     // "iPhone 16 Pro"
    let chip: String              // "A18 Pro"
    let totalRAM: UInt64          // bytes
    /// What the process may still allocate before jetsam kills it —
    /// `os_proc_available_memory` at scan time. This is the honest ceiling
    /// for "will this model fit": it already accounts for the
    /// increased-memory-limit entitlement, unlike any heuristic derived
    /// from `totalRAM`. 0 when the API is unavailable.
    let processAllowance: Int64   // bytes
    /// `MTLDevice.recommendedMaxWorkingSetSize` (~0.67 × RAM on iPhone).
    /// 0 when Metal is unavailable.
    let metalWorkingSet: Int64    // bytes
    let iOSVersion: String        // "26.0"
    let freeDiskSpace: Int64      // bytes
    let appleIntelligence: AppleIntelligenceStatus

    /// Allowance in GB with a `totalRAM - 3` fallback for contexts (unit
    /// tests, simulator) where the kernel API reports nothing.
    ///
    /// This is the budget for DIRTY memory — MLX and Core ML copy weights
    /// into anonymous Metal buffers, every byte of which jetsam charges to
    /// the process (`phys_footprint`).
    var usableRAMGB: Double {
        if processAllowance > 0 {
            return Double(processAllowance) / 1_073_741_824
        }
        return Double(totalRAM) / 1_073_741_824 - 3.0
    }

    /// The budget for mmap-backed weights (llama.cpp / whisper.cpp GGUF).
    /// Clean file-backed pages are NOT charged to `phys_footprint`, so the
    /// binding ceiling is the Metal working set (~0.67 × RAM) and overall
    /// physical-RAM pressure — noticeably above the jetsam allowance.
    var mmapCeilingGB: Double {
        let ramGB = Double(totalRAM) / 1_073_741_824
        if metalWorkingSet > 0 {
            return min(Double(metalWorkingSet) / 1_073_741_824, ramGB * 0.70)
        }
        return ramGB * 0.67
    }

    /// Budget for a given model: mmap-backed backends get the higher Metal
    /// working-set ceiling, everything else the strict jetsam allowance.
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

        return DeviceInfo(
            identifier: identifier,
            marketingName: name,
            chip: chip,
            totalRAM: ram,
            processAllowance: Int64(os_proc_available_memory()),
            metalWorkingSet: Int64(MTLCreateSystemDefaultDevice()?.recommendedMaxWorkingSetSize ?? 0),
            iOSVersion: iosVersion,
            freeDiskSpace: disk,
            appleIntelligence: aiStatus
        )
    }

    // MARK: - Private

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
    var totalRAMFormatted: String {
        let gb = Double(totalRAM) / 1_073_741_824
        return String(format: "%.0f GB", gb.rounded())
    }

    var freeDiskFormatted: String {
        let gb = Double(freeDiskSpace) / 1_073_741_824
        return String(format: "%.1f GB", gb)
    }
}
