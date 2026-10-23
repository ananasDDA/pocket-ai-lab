//
//  BuildTestRecoveryTests.swift
//
//  The journal-to-report reconstruction, and the wire contract around it.
//

import Foundation
import Testing
@testable import local_ai_test

struct BuildTestRecoveryTests {

    private let device = DeviceSnapshot(
        identifier: "iPhone18,4", marketingName: "iPhone Air", chip: "A19",
        ramGB: 11.5, iOSVersion: "26.6.1"
    )

    private var catalog: [AIModel] { ModelCatalog.builtIn }

    private func model(_ index: Int) -> AIModel {
        catalog.filter { $0.capabilities.contains(.textIn) && !$0.huggingFaceRepo.isEmpty }[index]
    }

    private func journal(inFlight: BuildTestJournal.InFlight?, entries: [BuildTestEntry] = []) -> BuildTestJournal {
        let started = Date(timeIntervalSince1970: 1_000_000)
        return BuildTestJournal(
            runID: "RUN-1",
            startedAt: started,
            appVersion: "1.0.1",
            appBuild: "2",
            device: device,
            options: BuildTestOptions(maxModels: 4),
            plannedModelIDs: [model(0).id, model(1).id, model(2).id, model(3).id],
            entries: entries,
            inFlight: inFlight,
            logTail: ["[build-test] (2/4) something", "  [quality] task 12 of 46"],
            updatedAt: started.addingTimeInterval(3_600)
        )
    }

    private func inFlight(_ m: AIModel, memoryMB: Int64 = 3_500) -> BuildTestJournal.InFlight {
        BuildTestJournal.InFlight(
            modelId: m.id, modelName: m.name, backend: m.backend.displayName,
            phase: "Benchmarking", stage: "quality", label: "task 12 of 46",
            since: Date(timeIntervalSince1970: 1_003_000),
            availableMemoryMB: memoryMB, thermalState: "serious"
        )
    }

    @Test func theInFlightModelBecomesTheCrashedEntryAndTheRestAreSkipped() {
        let done = BuildTestEntry.failed(model(0), error: "download failed")
        let j = journal(inFlight: inFlight(model(1)), entries: [done])
        let report = BuildTestRecovery.report(from: j, catalog: catalog, diagnostics: [], now: Date())

        #expect(report.interrupted)
        #expect(!report.cancelled)
        #expect(report.id == "RUN-1")
        #expect(report.entries.map(\.status) == [.failed, .crashed, .skipped, .skipped])
        #expect(report.crashedEntry?.modelId == model(1).id)
        #expect(report.crashedEntry?.log?.count == 2)
        #expect(report.entries[2].reason == BuildTestRecovery.interruptedReason)
        #expect(report.tested == 2)
        #expect(report.durationSeconds == 3_600)
        #expect(report.summaryLine.contains("interrupted"))
        #expect(report.summaryLine.contains("crashed on \(model(1).name)"))
    }

    @Test func aDeathBetweenModelsCrashesNothingButStillSkipsTheRest() {
        let j = journal(inFlight: nil, entries: [.completedFixture(model(0))])
        let report = BuildTestRecovery.report(from: j, catalog: catalog, diagnostics: [], now: Date())
        #expect(report.crashed == 0)
        #expect(report.skipped == 3)
        #expect(report.interrupted)
    }

    @Test func theCrashedTextNamesPhaseStageMemoryAndThermalState() {
        let j = journal(inFlight: inFlight(model(1), memoryMB: 412))
        let text = BuildTestRecovery.describe(j.inFlight!, journal: j, diagnostics: [])
        #expect(text.contains("benchmarking / quality (task 12 of 46)"))
        #expect(text.contains("412 MB"))
        #expect(text.contains("jetsam kill is likely"))
        #expect(text.contains("serious"))
    }

    @Test func plentyOfMemoryDoesNotBlameJetsam() {
        let j = journal(inFlight: inFlight(model(1), memoryMB: 3_500))
        let text = BuildTestRecovery.describe(j.inFlight!, journal: j, diagnostics: [])
        #expect(!text.contains("jetsam"))
    }

    @Test func aDiagnosticInsideTheWindowIsAttachedAndOneOutsideIsNot() {
        let j = journal(inFlight: inFlight(model(1)))
        let inside = CrashSummary(
            timeStampEnd: j.updatedAt.addingTimeInterval(20), exceptionType: 1, exceptionCode: nil,
            signal: 6, terminationReason: nil, virtualMemoryRegionInfo: nil, appBuild: "2"
        )
        let earlier = CrashSummary(
            timeStampEnd: j.startedAt.addingTimeInterval(-3_600), exceptionType: 1, exceptionCode: nil,
            signal: 11, terminationReason: nil, virtualMemoryRegionInfo: nil, appBuild: "2"
        )
        let text = BuildTestRecovery.describe(j.inFlight!, journal: j, diagnostics: [earlier, inside])
        #expect(text.contains("MetricKit: signal 6 (SIGABRT"))
        #expect(!text.contains("signal 11"))
    }

    @Test func remainingModelsExcludeTheFinishedAndTheCrashedOnes() {
        let j = journal(inFlight: inFlight(model(1)), entries: [.completedFixture(model(0))])
        #expect(BuildTestRecovery.remainingModelIDs(in: j) == [model(2).id, model(3).id])
    }

    @Test func journalRoundTripsThroughJSON() throws {
        let j = journal(inFlight: inFlight(model(1)), entries: [.failed(model(0), error: "x")])
        let data = try BenchmarkReportStore.makeEncoder().encode(j)
        let back = try BenchmarkReportStore.makeDecoder().decode(BuildTestJournal.self, from: data)
        #expect(back == j)
    }

    // A 1.0.1 report has no `interrupted` key and must still open.
    @Test func reportsWithoutTheInterruptedKeyStillDecode() throws {
        let report = BuildTestReport(
            appVersion: "1.0.1", appBuild: "2", stages: .full, device: device,
            options: .default, entries: [], durationSeconds: 1
        )
        var object = try JSONSerialization.jsonObject(
            with: BenchmarkReportStore.makeEncoder().encode(report)
        ) as! [String: Any]
        object.removeValue(forKey: "interrupted")
        let data = try JSONSerialization.data(withJSONObject: object)
        let decoded = try BenchmarkReportStore.makeDecoder().decode(BuildTestReport.self, from: data)
        #expect(decoded.interrupted == false)
    }

    @Test func crashedIsPartOfTheWireContract() throws {
        let entry = BuildTestEntry.crashed(
            modelId: "a/b", modelName: "B", backend: "Core ML", error: "died", log: ["l1"]
        )
        let data = try BenchmarkReportStore.makeEncoder().encode(entry)
        let text = String(decoding: data, as: UTF8.self)
        #expect(text.contains("\"status\" : \"crashed\"") || text.contains("\"status\":\"crashed\""))
        let back = try BenchmarkReportStore.makeDecoder().decode(BuildTestEntry.self, from: data)
        #expect(back.status == .crashed)
        #expect(back.log == ["l1"])
    }
}

private extension BuildTestEntry {
    /// A completed entry without a real benchmark: the recovery code only
    /// looks at ids and statuses.
    static func completedFixture(_ model: AIModel) -> BuildTestEntry {
        BuildTestEntry(
            modelId: model.id, modelName: model.name,
            backend: model.backend.displayName, status: .completed
        )
    }
}
