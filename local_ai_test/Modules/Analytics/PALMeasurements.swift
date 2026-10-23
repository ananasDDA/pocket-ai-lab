//
//  PALMeasurements.swift
//  local_ai_test
//
//  The probes behind the one-line hooks in InferenceManager and
//  ChatViewModel. A probe is created when a load or a generation starts,
//  takes the readings that need a "before" (thermal state, Low Power Mode,
//  battery, memory budget), samples phys_footprint for the peak while the
//  work runs, and queues one record when told how it ended.
//

import Foundation
import UIKit

@MainActor
enum PALReadings {

    static func conditionsNow(budgetMB: Int64) -> PALConditions {
        let device = UIDevice.current
        if !device.isBatteryMonitoringEnabled { device.isBatteryMonitoringEnabled = true }
        return PALConditions(
            thermalBefore: ProcessInfo.processInfo.thermalState.reportName,
            thermalAfter: nil,
            lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
            batteryBucket: PALDeviceState.batteryBucket(level: device.batteryLevel),
            memoryBudgetMB: budgetMB,
            peakMemoryMB: 0
        )
    }

    /// Context window the engine actually runs with: llama.cpp ignores the
    /// catalog's figure and uses its own fixed window.
    static func contextSize(for model: AIModel) -> Int {
        switch model.backend {
        #if canImport(llama)
        case .llamaCpp: return Int(LlamaContext.contextSize)
        case .llamaCppVision: return Int(LlamaVisionContext.contextTokens)
        #endif
        default: return model.contextLength
        }
    }
}

/// One model load, from the moment the previous engine is gone to ready or
/// failure.
@MainActor
final class PALLoadProbe {
    private let model: AIModel
    private let start = Date()
    private let sampler = MemorySampler()
    private var conditions: PALConditions
    private var done = false

    init(_ model: AIModel) {
        self.model = model
        // Measured before the load, the same figure the preflight compares
        // against: for dirty-memory backends it shrinks once weights are in.
        conditions = PALReadings.conditionsNow(budgetMB: InferenceManager.memoryBudgetMB(for: model.backend))
        if PALAnalytics.isEnabled { sampler.start() }
    }

    func finished() { report(errorKind: nil) }

    func failed(_ errorKind: String) { report(errorKind: errorKind) }

    func failed(_ error: Error) { report(errorKind: PALErrorClassifier.kind(of: error, phase: .load)) }

    private func report(errorKind: String?) {
        guard !done, PALAnalytics.isEnabled else { return }
        done = true
        sampler.stop()
        conditions.peakMemoryMB = sampler.peakMB
        let cold = errorKind == nil ? PALAnalytics.shared.markLoaded(model.id) : true
        let record = PALRecords.load(model: model, cold: cold, loadSeconds: Date().timeIntervalSince(start),
                                     conditions: conditions, errorKind: errorKind)
        PALAnalytics.shared.track(errorKind == nil ? .modelLoadMeasured : .modelLoadFailed, record)
    }
}

/// One chat generation, from the request to the end of the stream.
@MainActor
final class PALChatProbe {
    private let model: AIModel?
    private var meter: PALGenerationMeter
    private let sampler = MemorySampler()
    private var conditions: PALConditions
    private let hasImage: Bool
    private let hasAudio: Bool
    private var sourceEnded: Date?
    private var done = false

    init(model: AIModel?, attachments: [ChatAttachment]) {
        self.model = model
        let now = Date()
        meter = PALGenerationMeter(start: now)
        conditions = PALReadings.conditionsNow(
            budgetMB: model.map { InferenceManager.memoryBudgetMB(for: $0.backend) } ?? 0)
        hasImage = attachments.contains { $0.kind == .image || $0.kind == .video }
        hasAudio = attachments.contains { $0.kind == .audio }
        if PALAnalytics.isEnabled, model != nil { sampler.start() }
    }

    /// Passes the engine's stream through unchanged while timing it. Tapped
    /// before the speech pipeline on purpose: Kokoro holds the consumer up per
    /// sentence, and timing there would measure speech, not generation.
    func tap(_ source: AsyncThrowingStream<ChatOutput, Error>) -> AsyncThrowingStream<ChatOutput, Error> {
        AsyncThrowingStream { continuation in
            let task = Task { @MainActor [weak self] in
                do {
                    for try await output in source {
                        self?.meter.observe(output, at: Date())
                        continuation.yield(output)
                    }
                    self?.sourceDidEnd()
                    continuation.finish()
                } catch {
                    self?.sourceDidEnd()
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    private func sourceDidEnd() {
        guard sourceEnded == nil else { return }
        sourceEnded = Date()
        sampler.stop()
    }

    /// `error` is nil for a stream that ended normally. A stream whose task was
    /// cancelled also ends normally (the iterator just returns nil), so the
    /// task's own cancellation flag decides that case.
    func finish(error: Error?) {
        guard !done else { return }
        done = true
        if sourceEnded == nil { sampler.stop() }
        guard PALAnalytics.isEnabled, let model else { return }
        let end = sourceEnded ?? Date()
        conditions.peakMemoryMB = sampler.peakMB
        conditions.thermalAfter = ProcessInfo.processInfo.thermalState.reportName
        let (outcome, kind) = PALErrorClassifier.outcome(
            of: error,
            cancelled: Task.isCancelled,
            memoryPressureUnload: PALAnalytics.shared.memoryPressureUnload(since: meter.start)
        )
        let (event, record) = PALRecords.generation(
            model: model, contextSize: PALReadings.contextSize(for: model), meter: meter, end: end,
            outcome: outcome, errorKind: kind, conditions: conditions, hasImage: hasImage, hasAudio: hasAudio)
        PALAnalytics.shared.track(event, record)
    }
}

extension PALAnalytics {

    /// A benchmark report was written to disk. The model is looked up only to
    /// learn whether it is a catalog entry; an unresolvable one counts as
    /// imported.
    @MainActor
    func benchmarkSaved(_ report: BenchmarkReport) {
        guard Self.isEnabled else { return }
        let model = ModelCatalog.model(id: report.modelId)
            ?? InstalledModelsStore.shared.knownModel(id: report.modelId)
            ?? ModelCatalog.builtIn.first { $0.id == report.modelId }
        track(.benchmarkMeasured, PALRecords.benchmark(report, source: model?.source))
    }
}
