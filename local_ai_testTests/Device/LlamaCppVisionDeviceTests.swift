//
//  LlamaCppVisionDeviceTests.swift
//
//  End-to-end smoke test for the libmtmd bridge. Uses a GGUF vision model
//  (text model + mmproj projector; Gemma 3 4B unless configured) and asks
//  the model to identify the dominant color of a synthetically generated
//  solid square — same property-style assertion as MLXVisionDeviceTests to
//  keep the coverage of the two backends comparable.
//
//  Only passes on a real device where:
//    • the `llama` module (with libmtmd) is linked, and
//    • the model's GGUF + mmproj have been downloaded via the Models tab.
//
//  Otherwise the test records an Issue and returns — matches the
//  "soft skip" convention used by the sibling device tests.
//
//  Choosing the model (one per invocation), as for the daily-use suite the
//  variable MUST be an environment variable placed BEFORE `xcodebuild`:
//
//      TEST_RUNNER_VISION_MODEL_ID=unsloth/gemma-4-E2B-it-GGUF \
//      xcodebuild test \
//        -project local_ai_test.xcodeproj -scheme local_ai_test \
//        -destination 'platform=iOS,id=<hardware UDID>' \
//        -only-testing:local_ai_testTests/LlamaCppVisionMultiTurnDeviceTests
//
//  Unset, the suites run `LlamaCppVisionTestConfig.defaultModelID`.
//

import Testing
import Foundation
import UIKit
@testable import local_ai_test

/// Which catalog model the llama.cpp vision suites run. Xcode copies
/// `TEST_RUNNER_VISION_MODEL_ID` from its environment into the runner's as
/// `VISION_MODEL_ID`.
enum LlamaCppVisionTestConfig {
    static let defaultModelID = "unsloth/gemma-3-4b-it-GGUF"

    static func modelID(
        _ environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> String {
        let raw = environment["VISION_MODEL_ID"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (raw?.isEmpty == false) ? raw! : defaultModelID
    }
}

// `.serialized`: every case drives the one shared engine through
// `InferenceManager.shared`. Swift Testing runs a suite's tests in parallel
// by default, and two conversations interleaved on one KV cache make each
// turn a cache miss for the other (and, before the vision context's
// helpers were made nonisolated, corrupted each other's prompts).
@Suite(.tags(.device, .llamaCpp), .serialized)
struct LlamaCppVisionDeviceTests {

    private let modelId = LlamaCppVisionTestConfig.modelID()

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
/// behaviour that has to follow from that contract: the KV cache always
/// describes exactly the prompt (reusing the prefix it shares with the last
/// one), so a long chat stays inside the context window instead of growing
/// by one transcript per message.
///
/// Reproduces the 1.0.1 report "Generation failed - llama.cpp: context
/// window full" a couple of messages into a vision chat.
@Suite(.tags(.device, .llamaCpp), .serialized)
struct LlamaCppVisionMultiTurnDeviceTests {

    private let modelId = LlamaCppVisionTestConfig.modelID()

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
            let turn: VisionTurnResult
            do {
                turn = try await runVisionTurn(turns, parameters: .default)
            } catch {
                Issue.record("turn \(index + 1) of \(followUps.count) failed: \(error)")
                return
            }
            let reply = turn.reply
            #expect(!reply.isEmpty, "turn \(index + 1) produced no text")
            print(String(
                format: "[multiturn-vision] turn %d: reply %d chars, %lld MB allowance left",
                index + 1, reply.count, InferenceManager.availableMemoryMB()
            ))
            turn.log(model: modelId, turn: index + 1)
            turns.append(.assistant(reply))
            turns.append(.user(followUp))
        }
    }
    // MARK: KV prefix reuse

    /// The vision context reuses the KV prefix a follow-up shares with the
    /// previous turn: after the first turn only the new tokens (the reply
    /// re-tokenized past the cache, the template glue, the new question)
    /// are decoded, and the photo is not encoded again. A new photo is
    /// encoded once; a different photo at the place of a cached one is
    /// never served from the cache.
    @Test @MainActor func followUpTurnsEvaluateOnlyTheNewTokens() async throws {
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
        guard let red = solidSquare(.red), let blue = solidSquare(.blue), let green = solidSquare(.green) else {
            Issue.record("failed to encode test images to JPEG")
            return
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("llama.cpp vision load failed: \(InferenceManager.shared.state)")
            return
        }
        await InferenceManager.shared.resetConversation()

        var parameters = GenerationParameters.deterministic
        parameters.maxTokens = 96
        // Tokens the template may legitimately re-decode at the seam of the
        // previous turn: Qwen 3 / 3.5 renders history without the four-token
        // empty reasoning block its generation prompt ends with.
        let seamSlack = 8

        let question = "What colour is this picture? Answer with one word."
        var turns: [ChatTurn] = [.user(question, attachments: [.image(red)])]
        let first = try await runVisionTurn(turns, parameters: parameters)
        first.log(model: modelId, turn: 1)
        guard let firstCache = first.cache else {
            Issue.record("the engine reported no prompt cache figures")
            return
        }
        #expect(firstCache.reusedTokens == 0, "a fresh conversation reused \(firstCache.reusedTokens) tokens")
        #expect(firstCache.evaluatedMedia == 1)
        #expect(first.reply.lowercased().contains("red"), "turn 1 did not say red: \(first.reply)")
        turns.append(.assistant(first.reply))

        let followUps = [
            "Name three things that often have that colour.",
            "Write one sentence about why that colour is easy to notice.",
            "How do you say the colour name in French?",
            "Thanks! Sum up our chat in one sentence."
        ]
        var previous = firstCache
        for (index, followUp) in followUps.enumerated() {
            turns.append(.user(followUp))
            let result = try await runVisionTurn(turns, parameters: parameters)
            result.log(model: modelId, turn: index + 2)
            turns.append(.assistant(result.reply))
            guard let cache = result.cache else {
                Issue.record("turn \(index + 2): no prompt cache figures")
                return
            }
            let newTokens = cache.promptTokens - previous.promptTokens
            #expect(
                cache.rewind == .prefix || cache.rewind == .checkpoint,
                "turn \(index + 2): cache was \(cache.rewind.rawValue)"
            )
            #expect(
                cache.reusedTokens >= previous.promptTokens - seamSlack,
                "turn \(index + 2): reused \(cache.reusedTokens) of the previous \(previous.promptTokens)-token prompt"
            )
            #expect(
                cache.evaluatedTokens <= newTokens + seamSlack,
                "turn \(index + 2): evaluated \(cache.evaluatedTokens) tokens, \(newTokens) are new"
            )
            #expect(cache.reusedMedia == 1 && cache.evaluatedMedia == 0,
                    "turn \(index + 2): the photo was encoded again")
            #expect(!result.reply.isEmpty, "turn \(index + 2) produced no text")
            previous = cache
        }

        // A second photo later in the chat: the first stays cached, only
        // the new one is encoded.
        turns.append(.user("And this one? One word.", attachments: [.image(blue)]))
        let second = try await runVisionTurn(turns, parameters: parameters)
        second.log(model: modelId, turn: followUps.count + 2)
        if let cache = second.cache {
            #expect(cache.reusedMedia == 1 && cache.evaluatedMedia == 1,
                    "new photo: reused \(cache.reusedMedia), encoded \(cache.evaluatedMedia)")
            #expect(cache.reusedTokens >= previous.promptTokens - seamSlack)
        } else {
            Issue.record("new photo: no prompt cache figures")
        }
        #expect(second.reply.lowercased().contains("blue"), "the second photo was not read as blue: \(second.reply)")

        // The first question again with a different picture in the same
        // place. The text before the picture may come from the cache; the
        // picture must not.
        let replaced = try await runVisionTurn(
            [.user(question, attachments: [.image(green)])],
            parameters: parameters
        )
        replaced.log(model: modelId, turn: followUps.count + 3)
        if let cache = replaced.cache {
            #expect(cache.reusedMedia == 0 && cache.evaluatedMedia == 1,
                    "replaced photo: reused \(cache.reusedMedia), encoded \(cache.evaluatedMedia)")
        } else {
            Issue.record("replaced photo: no prompt cache figures")
        }
        #expect(replaced.reply.lowercased().contains("green"),
                "the replaced photo was not read as green (stale image in the cache?): \(replaced.reply)")
    }

    private func solidSquare(_ color: UIColor) -> Data? {
        let size = CGSize(width: 448, height: 448)
        let image = UIGraphicsImageRenderer(size: size).image { ctx in
            color.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
        }
        return image.jpegData(compressionQuality: 0.9)
    }
}

/// One generation through `InferenceManager`, with what the engine
/// reported about it.
struct VisionTurnResult {
    var reply = ""
    var ttftMs: Double?
    var cache: PromptCacheReport?
    var prefillMs: Double?

    /// One greppable line per turn, for before/after comparisons.
    func log(model: String, turn: Int) {
        let c = cache
        print("[vision-cache] \(model) turn \(turn): ttft \(ttftMs.map { String(format: "%.0f", $0) } ?? "-") ms, "
            + "prompt \(c?.promptTokens ?? -1), reused \(c?.reusedTokens ?? -1), "
            + "evaluated \(c?.evaluatedTokens ?? -1), media \(c?.reusedMedia ?? -1)/\(c?.evaluatedMedia ?? -1), "
            + "rewind \(c?.rewind.rawValue ?? "-"), prefill \(prefillMs.map { String(format: "%.0f", $0) } ?? "-") ms, "
            + "reply \(reply.count) chars")
    }
}

@MainActor
func runVisionTurn(_ turns: [ChatTurn], parameters: GenerationParameters) async throws -> VisionTurnResult {
    var result = VisionTurnResult()
    let start = Date()
    for try await output in InferenceManager.shared.generate(turns: turns, parameters: parameters) {
        switch output {
        case .textDelta(let piece):
            if result.ttftMs == nil { result.ttftMs = Date().timeIntervalSince(start) * 1000 }
            result.reply += piece
        case .diagnostic(.promptCache(let report)):
            result.cache = report
        case .diagnostic(.promptProcessed(_, let ms)):
            result.prefillMs = ms
        default:
            break
        }
    }
    return result
}
