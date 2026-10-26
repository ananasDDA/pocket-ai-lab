//
//  ProblemReportContext+Device.swift
//  local_ai_test
//
//  Reads the device, the app and the model state into a ProblemReportContext.
//  The analytics installation id is deliberately not read here.
//

import Foundation
import UIKit

extension ProblemReportContext {

    /// `model` defaults to the model selected in the chat, `failure` to the
    /// last engine error the app showed.
    @MainActor
    static func current(model: AIModel? = nil, failure: EngineFailure? = nil) -> ProblemReportContext {
        #if DEBUG
        let debug = true
        #else
        let debug = false
        #endif
        #if targetEnvironment(simulator)
        let simulator = true
        #else
        let simulator = false
        #endif
        let info = Bundle.main.infoDictionary ?? [:]
        let identifier = PALAnalyticsDevice.machineIdentifier
        let disk = (try? FileManager.default.attributesOfFileSystem(forPath: NSHomeDirectory()))?[.systemFreeSize]
            as? Int64 ?? 0
        let selected = model ?? InstalledModelsStore.shared.selectedModel
        let lastError = failure ?? InferenceManager.shared.lastError
        return ProblemReportContext(
            appVersion: info["CFBundleShortVersionString"] as? String ?? "?",
            appBuild: info["CFBundleVersion"] as? String ?? "",
            debugBuild: debug || simulator,
            osVersion: UIDevice.current.systemVersion,
            deviceIdentifier: identifier,
            deviceName: DeviceNameMapper.marketingName(for: identifier),
            ramBytes: ProcessInfo.processInfo.physicalMemory,
            freeDiskBytes: disk,
            model: selected.map(Self.model(from:)),
            lastError: lastError.map(Self.failure(from:)),
            thermalState: Self.thermalName(ProcessInfo.processInfo.thermalState),
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            locale: Locale.current.identifier
        )
    }

    static func model(from model: AIModel) -> Model {
        let imported = model.source != .curated
        return Model(id: model.id, name: model.name, isImported: imported,
                     repo: imported ? model.huggingFaceRepo : "",
                     file: imported ? model.preferredGGUFFilename : nil,
                     backend: model.backend.rawValue)
    }

    static func failure(from failure: EngineFailure) -> Failure {
        Failure(summary: failure.summary, detail: failure.detail, phase: failure.phase.rawValue,
                type: failure.underlyingType, backend: failure.backend, modelName: failure.modelName,
                modelId: failure.modelId)
    }

    nonisolated static func thermalName(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }
}

/// Lets a screen present the report sheet with `.sheet(item:)`, reading the
/// device at the moment the user asks to report.
extension ProblemReportContext: Identifiable {
    nonisolated var id: String { "\(deviceIdentifier)|\(model?.id ?? "")|\(lastError?.summary ?? "")" }
}
