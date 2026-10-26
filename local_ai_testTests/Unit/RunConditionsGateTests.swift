//
//  RunConditionsGateTests.swift
//
//  Pins the decisions of the conditions gate that unattended runs (the
//  catalog sweep and the daily-use night) make before every model and
//  scenario: state + battery + Low Power Mode + elapsed time -> wait,
//  proceed, flag or stop. Also the download stall rule, the network wait
//  and the heavy/light ordering. All pure: no device, no clock.
//

import Foundation
import Testing
@testable import local_ai_test

struct RunConditionsGateTests {

    private let night = RunConditionsPolicy.night

    private func reading(
        _ thermal: ProcessInfo.ThermalState,
        lowPower: Bool = false,
        charge: BatteryCharge = .charging,
        level: Double? = 0.8
    ) -> RunConditions {
        RunConditions(thermal: thermal, lowPowerMode: lowPower, batteryCharge: charge, batteryLevel: level)
    }

    // MARK: - Defaults

    @Test func nightDefaults() {
        #expect(night.thermalTarget == .nominal)
        #expect(night.maxWait == 1200)
        #expect(night.settle == 60)
        #expect(night.powerGate)
        #expect(night.minBatteryUnplugged == 0.40)
        #expect(night.stopOnBadPower)
        #expect(night.downloadStallTimeout == 300)
    }

    @Test func interactiveKeepsTheOldInAppRule() {
        let policy = RunConditionsPolicy.interactive
        #expect(policy.thermalTarget == .fair)
        #expect(policy.maxWait == 300)
        #expect(policy.settle == 0)
        #expect(!policy.powerGate)
        #expect(!policy.stopOnBadPower)
        #expect(policy.downloadStallTimeout == nil)
    }

    // MARK: - Thermal

    @Test func coolPhoneOnPowerProceedsAtOnce() {
        var gate = RunConditionsGate(policy: night)
        let decision1 = gate.observe(reading(.nominal), elapsed: 0)
        #expect(decision1 == .proceed)
    }

    @Test func fairIsNotEnoughForTheNightTarget() {
        var gate = RunConditionsGate(policy: night)
        guard case .wait(let reasons) = gate.observe(reading(.fair), elapsed: 0) else {
            Issue.record("expected a wait")
            return
        }
        #expect(reasons == ["thermal fair (target nominal)"])
    }

    @Test func fairTargetAcceptsFair() {
        var policy = night
        policy.thermalTarget = .fair
        var gate = RunConditionsGate(policy: policy)
        let decision2 = gate.observe(reading(.fair), elapsed: 0)
        #expect(decision2 == .proceed)
    }

    @Test func thermalTargetOffNeverWaitsForTemperature() {
        var policy = night
        policy.thermalTarget = .off
        var gate = RunConditionsGate(policy: policy)
        let decision3 = gate.observe(reading(.critical), elapsed: 0)
        #expect(decision3 == .proceed)
    }

    @Test func powerOnlyGateIgnoresTemperature() {
        var gate = RunConditionsGate(policy: night, checkThermal: false)
        let decision4 = gate.observe(reading(.serious), elapsed: 0)
        #expect(decision4 == .proceed)
    }

    @Test func settlesAfterCoolingThenProceeds() {
        var gate = RunConditionsGate(policy: night)
        let decision5 = gate.observe(reading(.serious), elapsed: 0)
        #expect(decision5 != .proceed)
        let decision6 = gate.observe(reading(.fair), elapsed: 300)
        #expect(decision6 != .proceed)
        // Reached nominal at 400 s: 60 s of settling before measuring.
        guard case .wait(let reasons) = gate.observe(reading(.nominal), elapsed: 400) else {
            Issue.record("expected the settle wait")
            return
        }
        #expect(reasons.first?.hasPrefix("settling at nominal") == true)
        let decision7 = gate.observe(reading(.nominal), elapsed: 450)
        #expect(decision7 != .proceed)
        let decision8 = gate.observe(reading(.nominal), elapsed: 460)
        #expect(decision8 == .proceed)
    }

    @Test func settleRestartsWhenThePhoneWarmsAgain() {
        var gate = RunConditionsGate(policy: night)
        _ = gate.observe(reading(.fair), elapsed: 0)
        _ = gate.observe(reading(.nominal), elapsed: 100)
        _ = gate.observe(reading(.fair), elapsed: 140)   // back over the line
        _ = gate.observe(reading(.nominal), elapsed: 150)
        let decision9 = gate.observe(reading(.nominal), elapsed: 200)  // only 50 s
        #expect(decision9 != .proceed)
        let decision10 = gate.observe(reading(.nominal), elapsed: 210)
        #expect(decision10 == .proceed)
    }

    @Test func noSettleWhenThePhoneWasCoolFromTheStart() {
        var gate = RunConditionsGate(policy: night)
        let decision11 = gate.observe(reading(.nominal), elapsed: 0)
        #expect(decision11 == .proceed)
    }

    @Test func zeroSettleProceedsTheMomentTheTargetIsReached() {
        var policy = night
        policy.settle = 0
        var gate = RunConditionsGate(policy: policy)
        _ = gate.observe(reading(.serious), elapsed: 0)
        let decision12 = gate.observe(reading(.nominal), elapsed: 100)
        #expect(decision12 == .proceed)
    }

    @Test func thermalCapFlagsAndProceeds() {
        var gate = RunConditionsGate(policy: night)
        let decision13 = gate.observe(reading(.serious), elapsed: 1199)
        #expect(decision13 != .proceed)
        let decision = gate.observe(reading(.serious), elapsed: 1200)
        #expect(decision == .flag(["cooldown cap: still serious after 1200 s (target nominal)"]))
    }

    @Test func settlingWhenTheCapRunsOutProceedsUnflagged() {
        var gate = RunConditionsGate(policy: night)
        _ = gate.observe(reading(.fair), elapsed: 0)
        _ = gate.observe(reading(.nominal), elapsed: 1180)
        let decision14 = gate.observe(reading(.nominal), elapsed: 1200)
        #expect(decision14 == .proceed)
    }

    // MARK: - Power

    @Test func lowPowerModeWaits() {
        var gate = RunConditionsGate(policy: night)
        let decision15 = gate.observe(reading(.nominal, lowPower: true), elapsed: 0)
        #expect(decision15 == .wait(["low power mode on"]))
    }

    @Test func lowBatteryUnpluggedWaits() {
        var gate = RunConditionsGate(policy: night)
        let decision = gate.observe(reading(.nominal, charge: .unplugged, level: 0.23), elapsed: 0)
        #expect(decision == .wait(["battery 23%, unplugged"]))
    }

    @Test func enoughBatteryUnpluggedProceeds() {
        var gate = RunConditionsGate(policy: night)
        let decision16 = gate.observe(reading(.nominal, charge: .unplugged, level: 0.40), elapsed: 0)
        #expect(decision16 == .proceed)
    }

    @Test func lowBatteryWhileChargingOrFullProceeds() {
        var gate = RunConditionsGate(policy: night)
        let decision17 = gate.observe(reading(.nominal, charge: .charging, level: 0.05), elapsed: 0)
        #expect(decision17 == .proceed)
        let decision18 = gate.observe(reading(.nominal, charge: .full, level: 0.05), elapsed: 0)
        #expect(decision18 == .proceed)
    }

    @Test func unknownBatteryLevelIsNotHeldAgainstTheRun() {
        var gate = RunConditionsGate(policy: night)
        let decision19 = gate.observe(reading(.nominal, charge: .unknown, level: nil), elapsed: 0)
        #expect(decision19 == .proceed)
    }

    @Test func powerGateOffIgnoresLowPowerAndBattery() {
        var gate = RunConditionsGate(policy: .interactive)
        let bad = reading(.nominal, lowPower: true, charge: .unplugged, level: 0.1)
        let decision20 = gate.observe(bad, elapsed: 0)
        #expect(decision20 == .proceed)
    }

    @Test func badPowerAtTheCapStops() {
        var gate = RunConditionsGate(policy: night)
        let decision = gate.observe(reading(.nominal, lowPower: true, charge: .unplugged, level: 0.23), elapsed: 1200)
        #expect(decision == .stop(["power gate cap: low power mode on, battery 23%, unplugged after 1200 s"]))
    }

    @Test func badPowerAtTheCapFlagsWhenStoppingIsOff() {
        var policy = night
        policy.stopOnBadPower = false
        var gate = RunConditionsGate(policy: policy)
        let decision = gate.observe(reading(.nominal, lowPower: true), elapsed: 1500)
        #expect(decision == .flag(["power gate cap: low power mode on after 1500 s"]))
    }

    @Test func hotAndBadPowerAtTheCapStopsWithBothIssues() {
        var gate = RunConditionsGate(policy: night)
        let decision = gate.observe(reading(.serious, charge: .unplugged, level: 0.3), elapsed: 1200)
        #expect(decision == .stop([
            "cooldown cap: still serious after 1200 s (target nominal)",
            "power gate cap: battery 30%, unplugged after 1200 s",
        ]))
    }

    @Test func powerRecoveringBeforeTheCapProceeds() {
        var gate = RunConditionsGate(policy: night, checkThermal: false)
        let decision21 = gate.observe(reading(.nominal, charge: .unplugged, level: 0.3), elapsed: 0)
        #expect(decision21 != .proceed)
        let decision22 = gate.observe(reading(.nominal, charge: .charging, level: 0.3), elapsed: 600)
        #expect(decision22 == .proceed)
    }

    // MARK: - Environment

    @Test func environmentOverridesTheNightPolicy() {
        let policy = RunConditionsPolicy.fromEnvironment([
            "BENCH_COOLDOWN_TARGET": "fair",
            "BENCH_COOLDOWN_MAX_SEC": "600",
            "BENCH_COOLDOWN_SETTLE_SEC": "0",
            "BENCH_STOP_ON_BAD_POWER": "0",
            "BENCH_DOWNLOAD_STALL_SEC": "0",
        ], prefix: "BENCH_")
        #expect(policy.thermalTarget == .fair)
        #expect(policy.maxWait == 600)
        #expect(policy.settle == 0)
        #expect(!policy.stopOnBadPower)
        #expect(policy.downloadStallTimeout == nil)
        #expect(policy.powerGate)
    }

    @Test func environmentPrefixAndGarbage() {
        // DAILY_ knobs do not leak into the BENCH_ policy, and garbage keeps
        // the defaults.
        let env = [
            "DAILY_COOLDOWN_TARGET": "fair",
            "BENCH_COOLDOWN_TARGET": "lukewarm",
            "BENCH_COOLDOWN_MAX_SEC": "-5",
            "BENCH_STOP_ON_BAD_POWER": "maybe",
        ]
        #expect(RunConditionsPolicy.fromEnvironment(env, prefix: "BENCH_") == .night)
        #expect(RunConditionsPolicy.fromEnvironment(env, prefix: "DAILY_").thermalTarget == .fair)
    }

    // MARK: - Network and downloads

    @Test func networkGate() {
        #expect(NetworkGate.decide(satisfied: true, elapsed: 0, maxWait: 1200) == .proceed)
        #expect(NetworkGate.decide(satisfied: false, elapsed: 600, maxWait: 1200) == .wait)
        #expect(NetworkGate.decide(satisfied: false, elapsed: 1200, maxWait: 1200) == .giveUp)
    }

    @Test func downloadAtZeroBytesFailsAfterTheStallTimeout() {
        var watch = DownloadStallWatch(timeout: 300)
        let decision23 = watch.observe(bytes: nil, elapsed: 10)      // manifest
        #expect(!decision23)
        let decision24 = watch.observe(bytes: 0, elapsed: 299)
        #expect(!decision24)
        let decision25 = watch.observe(bytes: 0, elapsed: 300)
        #expect(decision25)
    }

    @Test func movingBytesKeepTheDownloadAlive() {
        var watch = DownloadStallWatch(timeout: 300)
        let decision26 = watch.observe(bytes: 1_000, elapsed: 200)
        #expect(!decision26)
        let decision27 = watch.observe(bytes: 2_000, elapsed: 450)
        #expect(!decision27)
        let decision28 = watch.observe(bytes: 2_000, elapsed: 749)
        #expect(!decision28)
        let decision29 = watch.observe(bytes: 2_000, elapsed: 750)
        #expect(decision29)
    }

    @Test func compileResetsTheStallClock() {
        var watch = DownloadStallWatch(timeout: 300)
        _ = watch.observe(bytes: 5_000, elapsed: 100)
        watch.reset(at: 1_000)
        let decision30 = watch.observe(bytes: nil, elapsed: 1_200)
        #expect(!decision30)
        #expect(watch.idle(at: 1_200) == 200)
    }

    // MARK: - Ordering

    @Test func interleaveHeavyLight() {
        let weights: [(String, Double)] = [
            ("gemma4", 5.0), ("lfm26", 2.2), ("lfm12", 1.4), ("qwen2", 2.8),
            ("qwen4", 4.4), ("gemma3", 5.0), ("llama3", 2.5), ("apple", 0),
        ]
        let order = RunOrdering.interleaveHeavyLight(weights) { $0.1 }.map(\.0)
        #expect(order == ["gemma4", "apple", "gemma3", "lfm12", "qwen4", "lfm26", "qwen2", "llama3"])
    }

    @Test func interleaveSmallInputs() {
        #expect(RunOrdering.interleaveHeavyLight([Double]()) { $0 }.isEmpty)
        #expect(RunOrdering.interleaveHeavyLight([1.0]) { $0 } == [1.0])
        #expect(RunOrdering.interleaveHeavyLight([1.0, 3.0, 2.0]) { $0 } == [3.0, 1.0, 2.0])
    }
}
