//
//  ThrottleVerdictTests.swift
//
//  Pins `PerformanceMetrics.throttleVerdict`, the rule behind the
//  `thermallyConstrained` flag. The numbers come from a real 12-model sweep
//  on an iPhone Air: every model drove the device to `.serious` within
//  minutes, so only the measured slowdown separates a model that held its
//  speed from one that did not.
//

import Foundation
import Testing
@testable import local_ai_test

struct ThrottleVerdictTests {

    // MARK: - Not throttled

    @Test func flatRunsAreNotThrottled() {
        // Gemma 2 2B: 30.5 / 30.4 / 28.1 tok/s - 7.9% spread, well inside the
        // 20% threshold even though the phone was at .serious throughout.
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [30.5, 30.4, 28.1],
            protocolRate: nil,
            probeMean: 29.666
        )
        #expect(verdict.constrained == false)
        // The drop is still reported, so a reader can see what was measured.
        #expect(verdict.dropPercent != nil)
        #expect(abs((verdict.dropPercent ?? 0) - 7.87) < 0.1)
    }

    // MARK: - Throttled by the repetition spread

    @Test func collapsingRunsAreThrottled() {
        // Llama 3.1 8B: 11.5 / 9.1 / 4.0 tok/s - a 65% collapse.
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [11.5, 9.1, 4.0],
            protocolRate: nil,
            probeMean: 8.2
        )
        #expect(verdict.constrained)
        #expect(abs((verdict.dropPercent ?? 0) - 65.2) < 0.5)
    }

    @Test func borderlineDropJustOverThresholdIsThrottled() {
        // Llama 3.2 3B on MLX: 28.5 / 22.9 / 21.6 - 24%, the smallest real
        // throttle in the sweep, and the case a 20% line has to catch.
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [28.5, 22.9, 21.6],
            protocolRate: nil,
            probeMean: 24.333
        )
        #expect(verdict.constrained)
        #expect(abs((verdict.dropPercent ?? 0) - 24.2) < 0.5)
    }

    @Test func dropExactlyAtThresholdIsNotThrottled() {
        // 10.0 -> 8.0 is exactly 20%: the rule is "more than", so this run
        // stays comparable.
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [10.0, 8.0],
            protocolRate: nil,
            probeMean: 9.0
        )
        #expect(verdict.constrained == false)
        #expect(abs((verdict.dropPercent ?? 0) - 20.0) < 0.001)
    }

    // MARK: - Too little data for the rule to fire

    @Test func singleRunCannotFireTheRule() {
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [12.4],
            protocolRate: nil,
            probeMean: 12.4
        )
        #expect(verdict.constrained == false)
        // nil, not zero: one measurement has no spread to report.
        #expect(verdict.dropPercent == nil)
    }

    @Test func noRunsAtAllIsNotThrottled() {
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: nil,
            protocolRate: nil,
            probeMean: nil
        )
        #expect(verdict.constrained == false)
        #expect(verdict.dropPercent == nil)
    }

    @Test func zeroRatesAreIgnoredRatherThanReadAsACollapse() {
        // A repetition that produced too few tokens to time must not look
        // like a 100% drop.
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [28.2, 0],
            protocolRate: nil,
            probeMean: 28.2
        )
        #expect(verdict.constrained == false)
        #expect(verdict.dropPercent == nil)
    }

    // MARK: - Throttled by the standard-protocol shortfall

    @Test func protocolRateWellBelowProbeMeanIsThrottled() {
        // tg128 runs last, on the hottest phone of the run: steady decode
        // repetitions followed by a 25% slower tg128 still mean the numbers
        // are not comparable.
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [20.0, 19.5, 20.5],
            protocolRate: 15.0,
            probeMean: 20.0
        )
        #expect(verdict.constrained)
        // The verdict came from the protocol check, so the spread reported is
        // still the (small) one the repetitions showed.
        #expect(abs((verdict.dropPercent ?? 0) - 4.88) < 0.1)
    }

    @Test func protocolRateCloseToProbeMeanIsNotThrottled() {
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [20.0, 19.5, 20.5],
            protocolRate: 18.5,
            probeMean: 20.0
        )
        #expect(verdict.constrained == false)
    }

    @Test func protocolCheckAloneCanFlagASingleRepetitionRun() {
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [20.0],
            protocolRate: 10.0,
            probeMean: 20.0
        )
        #expect(verdict.constrained)
        #expect(verdict.dropPercent == nil)
    }

    @Test func missingProtocolRateLeavesTheVerdictToTheRepetitions() {
        let verdict = PerformanceMetrics.throttleVerdict(
            runs: [28.2, 27.1, 26.4],
            protocolRate: nil,
            probeMean: 27.233
        )
        #expect(verdict.constrained == false)
    }

    // MARK: - Old reports

    @Test func reportWithoutTheNewFieldKeepsTheStoredFlag() throws {
        // A file written before the measured rule existed: no
        // `throttleDropPercent`, and `thermallyConstrained` is read as stored.
        let json = """
        {
          "loadSeconds": 3.1,
          "prefillTokensPerSecond": 180.0,
          "generationTokensPerSecond": 11.5,
          "peakMemoryMB": 4100.0,
          "memoryBudgetMB": 6000,
          "thermalStateBefore": "fair",
          "thermalStateAfter": "serious",
          "tokensGenerated": 600,
          "wallClockSeconds": 240.0,
          "thermallyConstrained": true
        }
        """
        let metrics = try JSONDecoder().decode(
            PerformanceMetrics.self,
            from: Data(json.utf8)
        )
        #expect(metrics.thermallyConstrained)
        #expect(metrics.throttleDropPercent == nil)
        #expect(metrics.measurementIssues == nil)
        #expect(metrics.throughputIsComparable == false)
    }

    // MARK: - Measurement conditions

    @Test func aSteadyRunTakenInLowPowerModeIsInvalidNotSlow() throws {
        // The night-sweep shape: three repetitions that agree to 0.02 tok/s,
        // ten times slower than the model really is.
        let metrics = PerformanceMetrics(
            loadSeconds: 5.1,
            prefillTokensPerSecond: 90,
            generationTokensPerSecond: 3.2,
            peakMemoryMB: 2169,
            memoryBudgetMB: 6000,
            thermalStateBefore: "nominal",
            thermalStateAfter: "nominal",
            tokensGenerated: 600,
            wallClockSeconds: 900,
            generationRunsTokensPerSecond: [3.21, 3.19, 3.20],
            thermallyConstrained: true,
            throttleDropPercent: 0.6,
            measurementIssues: ["low power mode"]
        )
        #expect(metrics.throughputIsComparable == false)

        let data = try JSONEncoder().encode(metrics)
        let decoded = try JSONDecoder().decode(PerformanceMetrics.self, from: data)
        #expect(decoded.measurementIssues == ["low power mode"])
        #expect(decoded.thermallyConstrained)

        let report = BenchmarkReport(
            appVersion: "1.2.0",
            appBuild: "4",
            modelId: "unsloth/gemma-4-E2B-it-GGUF",
            modelName: "Gemma 4 E2B",
            backend: "llama.cpp Vision",
            quantization: "Q4_K_M",
            parameterSize: "2B",
            device: DeviceSnapshot(
                identifier: "iPhone18,4", marketingName: "iPhone Air",
                chip: "A19", ramGB: 11.5, iOSVersion: "26.6.1"
            ),
            performance: metrics,
            quality: nil
        )
        #expect(report.summaryLine.contains("[invalid: low power mode - not comparable]"))
        #expect(!report.summaryLine.contains("throttled"))
    }
}
