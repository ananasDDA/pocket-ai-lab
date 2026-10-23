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

// MARK: - Multi-turn regression

/// The chat UI re-sends the WHOLE conversation on every message and only
/// resets the engine on "Clear chat" or a model switch. This suite pins the
/// behaviour that has to follow from that contract: the KV cache is rebuilt
/// from the prompt, so a long chat stays inside the context window instead
/// of growing by one transcript per message.
///
/// Reproduces the 1.0.1 report "Generation failed - llama.cpp: context
/// window full" a couple of messages into a vision chat.
@Suite(.tags(.device, .llamaCpp))
struct LlamaCppVisionMultiTurnDeviceTests {

    private let modelId = "unsloth/gemma-3-4b-it-GGUF"

    /// A photo-shaped image at the size `ImagePreprocessor` produces, busy
    /// enough that the JPEG does not collapse to a few bytes.
    private func syntheticPhoto() -> Data? {
        let size = CGSize(width: 896, height: 672)
        let image = UIGraphicsImageRenderer(size: size).image { ctx in
            let colors = [UIColor.systemTeal.cgColor, UIColor.systemPurple.cgColor]
            let gradient = CGGradient(
                colorsSpace: CGColorSpaceCreateDeviceRGB(),
                colors: colors as CFArray,
                locations: [0, 1]
            )!
            ctx.cgContext.drawLinearGradient(
                gradient,
                start: .zero,
                end: CGPoint(x: size.width, y: size.height),
                options: []
            )
            UIColor.systemOrange.setFill()
            for row in 0..<8 {
                for col in 0..<10 {
                    ctx.fill(CGRect(x: col * 90 + 10, y: row * 80 + 10, width: 40, height: 30))
                }
            }
        }
        return image.jpegData(compressionQuality: 0.85)
    }

    @Test @MainActor func longChatAfterAnImageKeepsGenerating() async throws {
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
        guard let jpeg = syntheticPhoto() else {
            Issue.record("failed to encode test image to JPEG")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("llama.cpp vision load failed: \(InferenceManager.shared.state)")
            return
        }
        await InferenceManager.shared.resetConversation()

        // Long answers in a non-Latin script: the tokens-per-character ratio
        // that made the reported chat fill its window fastest.
        let questions = [
            "Describe in detail what you see in the picture.",
            "How are you? Tell me a few sentences about yourself.",
            "Which colours dominate the image?",
            "Make up a short story inspired by this picture.",
            "What else can be said about this image?",
            "Thanks! Sum up our conversation."
        ]
        // Twelve exchanges: with the pre-fix engine the transcript was
        // decoded on top of the previous turns, so the window ran out
        // somewhere in the middle of a chat this long.
        let followUps = (0..<12).map { questions[$0 % questions.count] }

        var turns: [ChatTurn] = [
            .user("What is this? Describe it in detail.", attachments: [.image(jpeg)])
        ]

        for (index, followUp) in followUps.enumerated() {
            var reply = ""
            do {
                for try await output in InferenceManager.shared.generate(
                    turns: turns,
                    parameters: .default
                ) {
                    if case .textDelta(let piece) = output { reply += piece }
                }
            } catch {
                Issue.record("turn \(index + 1) of \(followUps.count) failed: \(error)")
                return
            }
            #expect(!reply.isEmpty, "turn \(index + 1) produced no text")
            print(String(
                format: "[multiturn-vision] turn %d: reply %d chars, %lld MB allowance left",
                index + 1, reply.count, InferenceManager.availableMemoryMB()
            ))
            turns.append(.assistant(reply))
            turns.append(.user(followUp))
        }
    }
}
