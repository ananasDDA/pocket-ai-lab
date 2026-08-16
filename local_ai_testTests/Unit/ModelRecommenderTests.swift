//
//  ModelRecommenderTests.swift
//

import Testing
@testable import local_ai_test

struct ModelRecommenderTests {

    private func device(ramGB: Double, diskGB: Double, ai: AppleIntelligenceStatus = .notSupported) -> DeviceInfo {
        DeviceInfo(
            identifier: "iPhone15,2",
            marketingName: "iPhone Test",
            chip: "A17 Pro",
            totalRAM: UInt64(ramGB * 1_073_741_824),
            // 0 → tests exercise the `totalRAM - 3` fallback, same math the
            // suite was written against.
            processAllowance: 0,
            metalWorkingSet: 0,
            iOSVersion: "26.1",
            freeDiskSpace: Int64(diskGB * 1_073_741_824),
            appleIntelligence: ai
        )
    }

    @Test func recommendsAppleIntelligenceWhenAvailable() {
        let result = ModelRecommender.shared.recommend(for: device(ramGB: 6, diskGB: 100, ai: .available))
        #expect(result.allCompatible.contains(where: { $0.backend == .appleIntelligence }))
    }

    @Test func filtersOutTooLargeModels() {
        let result = ModelRecommender.shared.recommend(for: device(ramGB: 4, diskGB: 5))
        for model in result.allCompatible {
            #expect(model.ramRequiredGB < 4.0)
        }
    }

    @Test func filtersOutWhenDiskIsFull() {
        let result = ModelRecommender.shared.recommend(for: device(ramGB: 16, diskGB: 0.5))
        for model in result.allCompatible {
            #expect(model.effectiveDiskSizeGB <= 0.5 || model.backend == .appleIntelligence)
        }
    }

    @Test func doublesCoreMLDiskForCompilation() {
        let smallDiskDevice = device(ramGB: 16, diskGB: 5.0)
        let result = ModelRecommender.shared.recommend(for: smallDiskDevice)
        for model in result.allCompatible where model.requiresCompilation {
            #expect(model.effectiveDiskSizeGB <= 5.0)
            #expect(model.diskSizeGB <= 2.5)
        }
    }

    @Test func higherQualitySortedFirst() {
        let result = ModelRecommender.shared.recommend(for: device(ramGB: 16, diskGB: 100))
        let qualities = result.allCompatible.map(\.quality.rawValue)
        for i in 0..<(qualities.count - 1) {
            #expect(qualities[i] >= qualities[i + 1])
        }
    }
}
