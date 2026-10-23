//
//  BuildTestRecovery.swift
//  local_ai_test
//
//  Turns a journal the previous process left behind into the report that
//  process never got to write. Pure: given the journal, the catalog and the
//  clock, it produces the report - the runner does the file handling.
//
//  The story a recovered report tells is deliberately blunt. The entries
//  that finished are exactly as the run recorded them. The model that was
//  in flight gets a `crashed` entry whose text names the phase, the
//  benchmark stage and the memory left at the last journal write, plus
//  whatever MetricKit delivered about the death. Everything the sweep had
//  still planned is recorded as skipped WITH a reason - unlike a
//  cancellation, nobody chose to stop, so an absent model would read as
//  "fine" when it was never looked at.
//

import Foundation

nonisolated enum BuildTestRecovery {

    /// Below this much allowance at the last journal write, a silent death
    /// is far more likely to be jetsam than a bug in the model.
    static let lowMemoryThresholdMB: Int64 = 700

    static let interruptedReason = "not run: the run was interrupted by a crash before reaching this model"

    static func report(
        from journal: BuildTestJournal,
        catalog: [AIModel],
        diagnostics: [CrashSummary],
        now: Date
    ) -> BuildTestReport {
        var entries = journal.entries
        var reached = Set(entries.map(\.modelId))

        if let inFlight = journal.inFlight {
            entries.append(.crashed(
                modelId: inFlight.modelId,
                modelName: inFlight.modelName,
                backend: inFlight.backend,
                error: describe(inFlight, journal: journal, diagnostics: diagnostics),
                log: journal.logTail
            ))
            reached.insert(inFlight.modelId)
        }

        for id in journal.plannedModelIDs where !reached.contains(id) {
            if let model = catalog.first(where: { $0.id == id }) {
                entries.append(.skipped(model, reason: interruptedReason))
            } else {
                entries.append(.skipped(unknownModelId: id, reason: interruptedReason))
            }
            reached.insert(id)
        }

        return BuildTestReport(
            id: journal.runID,
            date: now,
            appVersion: journal.appVersion,
            appBuild: journal.appBuild,
            stages: journal.options.stages,
            device: journal.device,
            options: journal.options,
            entries: entries,
            durationSeconds: max(journal.updatedAt.timeIntervalSince(journal.startedAt), 0),
            cancelled: false,
            interrupted: true
        )
    }

    /// The models the interrupted run never reached - what a "resume" runs.
    /// The crashed model is NOT among them: it gets a run of its own, or a
    /// bug report, not a second chance to take the sweep down.
    static func remainingModelIDs(in journal: BuildTestJournal) -> [String] {
        var reached = Set(journal.entries.map(\.modelId))
        if let inFlight = journal.inFlight { reached.insert(inFlight.modelId) }
        return journal.plannedModelIDs.filter { !reached.contains($0) }
    }

    /// The crashed entry's text: what was happening, how the device looked,
    /// what the OS reported.
    static func describe(
        _ inFlight: BuildTestJournal.InFlight,
        journal: BuildTestJournal,
        diagnostics: [CrashSummary]
    ) -> String {
        var doing = inFlight.phase.lowercased()
        if let stage = inFlight.stage { doing += " / \(stage)" }
        if let label = inFlight.label, !label.isEmpty { doing += " (\(label))" }

        var text = "The app was terminated while \(doing)."
        if inFlight.availableMemoryMB >= 0 {
            text += " Memory allowance at the last journal write: \(inFlight.availableMemoryMB) MB"
            text += inFlight.availableMemoryMB < lowMemoryThresholdMB
                ? " - low, so a jetsam kill is likely."
                : "."
        }
        text += " Thermal state: \(inFlight.thermalState)."
        let seconds = Int(journal.updatedAt.timeIntervalSince(inFlight.since))
        text += " \(seconds) s into this model."

        if let match = matchingDiagnostic(diagnostics, after: inFlight.since, journal: journal) {
            text += " MetricKit: \(match.text)"
        }
        return text
    }

    /// The crash MetricKit reported for the window this model was in
    /// flight: at or after pickup, and not long after the last journal
    /// write. Anything else is some other death.
    static func matchingDiagnostic(
        _ diagnostics: [CrashSummary],
        after since: Date,
        journal: BuildTestJournal
    ) -> CrashSummary? {
        let deadline = journal.updatedAt.addingTimeInterval(30 * 60)
        return diagnostics
            .filter { $0.timeStampEnd >= since.addingTimeInterval(-60) && $0.timeStampEnd <= deadline }
            .max { $0.timeStampEnd < $1.timeStampEnd }
    }
}
