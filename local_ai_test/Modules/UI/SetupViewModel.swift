//
//  SetupViewModel.swift
//  local_ai_test
//

import SwiftUI

@MainActor
@Observable
final class SetupViewModel {

    enum State {
        case scanning
        case ready(DeviceInfo, RecommendationResult)
        case error(String)
    }

    private(set) var state: State = .scanning

    func scan() async {
        state = .scanning
        // brief pause so the scanning animation is actually visible
        try? await Task.sleep(for: .milliseconds(600))

        let device = DeviceScanner.shared.scan()
        let result = ModelRecommender.shared.recommend(for: device)

        if device.appleIntelligence == .available,
           let aiModel = ModelCatalog.all.first(where: { $0.backend == .appleIntelligence }) {
            InstalledModelsStore.shared.markInstalled(aiModel)
        }

        state = .ready(device, result)
    }
}
