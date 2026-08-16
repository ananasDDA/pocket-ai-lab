//
//  DeviceNameMapper.swift
//  local_ai_test
//

enum DeviceNameMapper {

    static func marketingName(for identifier: String) -> String {
        nameMap[identifier] ?? identifier
    }

    static func chip(for identifier: String) -> String {
        chipMap[identifier] ?? "Unknown"
    }

    // MARK: - iPhone identifier → Marketing name
    // Source: https://www.theiphonewiki.com/wiki/Models

    private static let nameMap: [String: String] = [
        // iPhone 15
        "iPhone15,4": "iPhone 15",
        "iPhone15,5": "iPhone 15 Plus",
        "iPhone16,1": "iPhone 15 Pro",
        "iPhone16,2": "iPhone 15 Pro Max",
        // iPhone 16
        "iPhone17,1": "iPhone 16 Pro",
        "iPhone17,2": "iPhone 16 Pro Max",
        "iPhone17,3": "iPhone 16",
        "iPhone17,4": "iPhone 16 Plus",
        // iPhone 16e
        "iPhone17,5": "iPhone 16e",
        // iPhone 17
        "iPhone18,1": "iPhone 17 Pro",
        "iPhone18,2": "iPhone 17 Pro Max",
        "iPhone18,3": "iPhone 17",
        "iPhone18,4": "iPhone Air",
        // iPad (M-series)
        "iPad13,18": "iPad Pro 11\" (M2)",
        "iPad13,19": "iPad Pro 11\" (M2)",
        "iPad14,3": "iPad Pro 11\" (M2)",
        "iPad14,4": "iPad Pro 11\" (M2)",
        "iPad14,5": "iPad Pro 12.9\" (M2)",
        "iPad14,6": "iPad Pro 12.9\" (M2)",
        "iPad16,3": "iPad Pro 11\" (M4)",
        "iPad16,4": "iPad Pro 11\" (M4)",
        "iPad16,5": "iPad Pro 13\" (M4)",
        "iPad16,6": "iPad Pro 13\" (M4)",
    ]

    private static let chipMap: [String: String] = [
        // iPhone 15
        "iPhone15,4": "A16 Bionic",
        "iPhone15,5": "A16 Bionic",
        "iPhone16,1": "A17 Pro",
        "iPhone16,2": "A17 Pro",
        // iPhone 16
        "iPhone17,1": "A18 Pro",
        "iPhone17,2": "A18 Pro",
        "iPhone17,3": "A18",
        "iPhone17,4": "A18",
        // iPhone 16e
        "iPhone17,5": "A18",
        // iPhone 17
        "iPhone18,1": "A19 Pro",
        "iPhone18,2": "A19 Pro",
        "iPhone18,3": "A19",
        "iPhone18,4": "A19",
        // iPad
        "iPad13,18": "M2",
        "iPad13,19": "M2",
        "iPad14,3": "M2",
        "iPad14,4": "M2",
        "iPad14,5": "M2",
        "iPad14,6": "M2",
        "iPad16,3": "M4",
        "iPad16,4": "M4",
        "iPad16,5": "M4",
        "iPad16,6": "M4",
    ]
}
