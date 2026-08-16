//
//  LlamaCppVisionDeviceTests.swift
//
//  End-to-end smoke test for the libmtmd bridge. Uses the GGUF flavour of
//  Gemma 3 4B (text model + mmproj projector) and asks the model to
//  identify the dominant color of a synthetically generated solid square
//  — same property-style assertion as MLXVisionDeviceTests to keep the
//  coverage of the two backends comparable.
//
//  Only passes on a real device where:
//    • the `llama` module (with libmtmd) is linked, and
//    • Gemma 3 4B GGUF + mmproj have been downloaded via the Models tab.
//
//  Otherwise the test records an Issue and returns — matches the
//  "soft skip" convention used by the sibling device tests.
//

import Testing
import Foundation
import UIKit
@testable import local_ai_test

@Suite(.tags(.device, .llamaCpp))
struct LlamaCppVisionDeviceTests {

    private let modelId = "unsloth/gemma-3-4b-it-GGUF"

    @Test @MainActor func gemma3VisionDescribesSolidColor() async throws {
        guard BackendAvailability.hasLlama else {
            Issue.record("llama module not linked; skipping")
            return
        }
        guard let model = ModelCatalog.model(id: modelId) else {
            Issue.record("model not in catalog: \(modelId)")
            return
        }
        guard InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) else {
            Issue.record("Install \(modelId) via the Models tab first")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("llama.cpp vision load failed: \(InferenceManager.shared.state)")
            return
        }

        // Solid red 256×256 JPEG — trivial content the projector should
        // handle at any resolution bucket.
        let size = CGSize(width: 256, height: 256)
        let redImage = UIGraphicsImageRenderer(size: size).image { ctx in
            UIColor.red.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
        }
        guard let jpeg = redImage.jpegData(compressionQuality: 0.9) else {
            Issue.record("failed to encode test image to JPEG")
            return
        }

        let start = Date()
        var firstTokenMs: Double?
        var tokensPerSecond: Double?
        var tokensGenerated = 0
        var out = ""

        let stream = InferenceManager.shared.generate(
            turns: [.user(
                "What color is this image? Respond with one word.",
                attachments: [.image(jpeg)]
            )],
            parameters: .deterministic
        )

        for try await output in stream {
            switch output {
            case .textDelta(let piece):
                if firstTokenMs == nil {
                    firstTokenMs = Date().timeIntervalSince(start) * 1000
                }
                tokensGenerated += 1
                out += piece
                if out.count > 100 { break }
            case .diagnostic(.firstTokenLatency(let ms)):
                firstTokenMs = ms
            case .diagnostic(.finished(let n, let tps)):
                tokensGenerated = n
                tokensPerSecond = tps
            default:
                break
            }
        }

        // Assertions: first token within 60s, sane tokens/s, correct answer.
        if let ftl = firstTokenMs {
            #expect(ftl < 60_000, "first-token latency too high: \(ftl) ms")
        }
        #expect(tokensGenerated >= 1)
        if let tps = tokensPerSecond {
            #expect(tps > 0)
        }
        #expect(out.lowercased().contains("red"), "model output did not contain 'red': \(out)")
    }
}
