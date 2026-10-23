//
//  LlamaCppTextMultiTurnDeviceTests.swift
//
//  A long chat on the llama.cpp text backend.
//
//  WHY THIS EXISTS: the chat UI re-sends the whole conversation on every
//  message and only resets the engine on "Clear chat" or a model switch.
//  Until 1.0.2 the llama.cpp contexts appended that transcript to the KV
//  cache instead of rebuilding from it, so `n_past` grew by a full
//  transcript per message and a normal chat ran out of context window after
//  a handful of turns - reported from the field as "Generation failed -
//  llama.cpp: context window full".
//
//  The vision twin lives in LlamaCppVisionDeviceTests.swift.
//
//  Needs the GGUF text model on the device. `LLAMA_TEXT_MULTITURN_DOWNLOAD=1`
//  lets the test fetch it and delete it again afterwards.
//

import Foundation
import Testing
@testable import local_ai_test

@Suite(.tags(.device, .llamaCpp))
struct LlamaCppTextMultiTurnDeviceTests {

    private let modelId = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    private static let downloadTimeout: TimeInterval = 30 * 60

    private func environmentFlag(_ key: String) -> Bool {
        ["1", "true", "yes"].contains(
            (ProcessInfo.processInfo.environment[key] ?? "").lowercased()
        )
    }

    @Test @MainActor func longChatStaysInsideTheContextWindow() async throws {
        guard BackendAvailability.hasLlama else {
            Issue.record("llama module not linked; skipping")
            return
        }
        guard let model = ModelCatalog.model(id: modelId) else {
            Issue.record("model not in catalog: \(modelId)")
            return
        }

        let downloader = ModelDownloader.shared
        var fetchedHere = false
        if !InstalledModelsStore.shared.installedModels.contains(where: { $0.id == modelId }) {
            guard environmentFlag("LLAMA_TEXT_MULTITURN_DOWNLOAD") else {
                Issue.record("Install \(modelId) first, or set LLAMA_TEXT_MULTITURN_DOWNLOAD=1")
                return
            }
            try await download(model, using: downloader)
            fetchedHere = true
        }
        defer {
            if fetchedHere {
                print("[multiturn-test] deleting \(model.id)")
                downloader.delete(model)
            }
        }

        await InferenceManager.shared.prepare(model: model)
        guard case .ready = InferenceManager.shared.state else {
            Issue.record("llama.cpp load failed: \(InferenceManager.shared.state)")
            return
        }
        await InferenceManager.shared.resetConversation()

        // Long answers in a non-Latin script: the tokens-per-character ratio
        // that exhausts a 4096-token window fastest.
        let questions = [
            "Explain in detail how an internal combustion engine works.",
            "Now explain the same thing in simple words for a child.",
            "What alternatives to such an engine exist?",
            "Make up a short story about a mechanic.",
            "What is important to remember when servicing a car?",
            "Sum up our conversation."
        ]
        var turns: [ChatTurn] = [.user(questions[0])]

        for turn in 1...12 {
            var reply = ""
            do {
                for try await output in InferenceManager.shared.generate(
                    turns: turns,
                    parameters: .default
                ) {
                    if case .textDelta(let piece) = output { reply += piece }
                }
            } catch {
                Issue.record("turn \(turn) of 12 failed: \(error)")
                await InferenceManager.shared.unloadCurrent()
                return
            }
            #expect(!reply.isEmpty, "turn \(turn) produced no text")
            print(String(
                format: "[multiturn-test] turn %d: reply %d chars, %lld MB allowance left",
                turn, reply.count, InferenceManager.availableMemoryMB()
            ))
            turns.append(.assistant(reply))
            turns.append(.user(questions[turn % questions.count]))
        }

        await InferenceManager.shared.unloadCurrent()
    }

    @MainActor
    private func download(_ model: AIModel, using downloader: ModelDownloader) async throws {
        print(String(format: "[multiturn-test] downloading %@ (~%.1f GB)", model.id, model.diskSizeGB))
        downloader.download(model)
        let deadline = Date().addingTimeInterval(Self.downloadTimeout)
        var lastPrint = Date.distantPast

        while Date() < deadline {
            switch downloader.state(for: model) {
            case .installed:
                print("[multiturn-test] installed")
                return
            case .error(let message):
                throw MultiTurnDeviceTestError.download(message)
            case .downloading(let fraction, let done, let total):
                if Date().timeIntervalSince(lastPrint) >= 30 {
                    lastPrint = Date()
                    print(String(
                        format: "[multiturn-test] download %.0f%% (%.2f / %.2f GB)",
                        fraction * 100,
                        Double(done) / 1_073_741_824, Double(total) / 1_073_741_824
                    ))
                }
            case .compiling, .fetchingManifest, .idle:
                break
            }
            try await Task.sleep(nanoseconds: 2_000_000_000)
        }
        throw MultiTurnDeviceTestError.download("timed out after \(Int(Self.downloadTimeout / 60)) min")
    }
}

private enum MultiTurnDeviceTestError: Error, CustomStringConvertible {
    case download(String)
    var description: String {
        switch self {
        case .download(let message): return "download failed: \(message)"
        }
    }
}
