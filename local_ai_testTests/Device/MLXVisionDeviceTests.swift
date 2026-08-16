//
//  MLXVisionDeviceTests.swift
//
//  VLM smoke test with Gemma 3 4B. Draws a solid red square and asks the
//  model to describe the dominant color — a robust property that reliable
//  VLMs pass.
//

import Testing
import Foundation
import UIKit
@testable import local_ai_test

@Suite(.tags(.device, .mlx))
struct MLXVisionDeviceTests {

    private let modelId = "mlx-community/gemma-3-4b-it-4bit"

    @Test @MainActor func redSquareIsRecognized() async throws {
        guard let model = ModelCatalog.model(id: modelId) else { return }
        guard InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else { return }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else { return }

        let size = CGSize(width: 256, height: 256)
        let redImage = UIGraphicsImageRenderer(size: size).image { ctx in
            UIColor.red.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
        }
        guard let jpeg = redImage.jpegData(compressionQuality: 0.9) else { return }

        var out = ""
        let stream = InferenceManager.shared.generate(
            turns: [.user("What color is this image? Respond with one word.", attachments: [.image(jpeg)])],
            parameters: .deterministic
        )
        for try await output in stream {
            if case .textDelta(let s) = output { out += s }
            if out.count > 100 { break }
        }
        #expect(out.lowercased().contains("red"))
    }
}
