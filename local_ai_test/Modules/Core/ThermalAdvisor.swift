//
//  ThermalAdvisor.swift
//  local_ai_test
//
//  Why replies got slow, said once and calmly.
//
//  Every model in the overnight daily-use run fell to 5-11 tok/s once the
//  phone reached thermal state "serious", which takes two to five long
//  answers. Nothing on screen said so: the chat just looked broken. The
//  advisor decides what the chat shows; `ThermalMonitor` feeds it the state.
//

import Foundation
import Observation

/// What the chat shows while the phone is hot.
struct ThermalAdvice: Equatable, Sendable {
    let state: ProcessInfo.ThermalState
    /// A lighter installed model worth offering, or nil. Only a suggestion:
    /// the app never switches models on its own.
    let lighterModel: AIModel?

    /// The one line shown on tap.
    nonisolated static let explanation =
        "The phone is hot, so replies are slower. A lighter model or a short break helps."
}

enum ThermalAdvisor {

    /// Models at or above this many GB of weights are "heavy" for the
    /// purpose of suggesting another one. The catalog's 3-4B models sit at
    /// 2.0-2.4 GB; the 1-2B ones below 2.
    nonisolated static let heavyModelGB = 2.0

    /// Backends whose heat comes from running the weights on the GPU here
    /// in the app - the ones a smaller download actually cools down.
    nonisolated static let localBackends: Set<ModelBackend> = [
        .llamaCpp, .llamaCppVision, .mlx, .mlxVision,
    ]

    /// nil while the phone is `nominal` or `fair`. At `serious` and
    /// `critical` the indicator shows; when the loaded model is a heavy
    /// local one, the largest installed chat model that is below the
    /// threshold (and smaller than the loaded one) is offered.
    nonisolated static func advice(
        thermal: ProcessInfo.ThermalState,
        loaded: AIModel?,
        installed: [AIModel]
    ) -> ThermalAdvice? {
        guard thermal == .serious || thermal == .critical else { return nil }
        return ThermalAdvice(state: thermal, lighterModel: lighterModel(than: loaded, among: installed))
    }

    nonisolated static func lighterModel(than loaded: AIModel?, among installed: [AIModel]) -> AIModel? {
        guard let loaded,
              localBackends.contains(loaded.backend),
              loaded.diskSizeGB >= heavyModelGB
        else { return nil }
        return installed
            .filter { candidate in
                candidate.id != loaded.id
                    && localBackends.contains(candidate.backend)
                    && SystemPrompt.applies(to: candidate.capabilities)
                    && candidate.diskSizeGB > 0
                    && candidate.diskSizeGB < heavyModelGB
                    && candidate.diskSizeGB < loaded.diskSizeGB
            }
            // The best of the light ones: the largest still under the bar.
            .max { $0.diskSizeGB < $1.diskSizeGB }
    }
}

/// `ProcessInfo.thermalState`, observable. Updated from
/// `thermalStateDidChangeNotification`, which iOS posts on an arbitrary
/// thread whenever the state changes.
@MainActor
@Observable
final class ThermalMonitor {

    static let shared = ThermalMonitor()

    private(set) var state: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState

    @ObservationIgnored private var observer: NSObjectProtocol?

    private init() {
        observer = NotificationCenter.default.addObserver(
            forName: ProcessInfo.thermalStateDidChangeNotification,
            object: nil,
            queue: .main
        ) { _ in
            let current = ProcessInfo.processInfo.thermalState
            Task { @MainActor in ThermalMonitor.shared.state = current }
        }
    }
}
