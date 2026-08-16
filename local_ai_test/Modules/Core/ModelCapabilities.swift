//
//  ModelCapabilities.swift
//  local_ai_test
//
//  Describes what a given model / engine can ingest and produce.
//  UI uses this to show / hide attachment buttons, the recommender to
//  match against device capabilities, the inference manager to route
//  multimodal requests to the right engine.
//

import Foundation

nonisolated struct ModelCapabilities: OptionSet, Sendable, Hashable, Codable {
    let rawValue: Int

    init(rawValue: Int) { self.rawValue = rawValue }

    static let textIn   = ModelCapabilities(rawValue: 1 << 0)
    static let imageIn  = ModelCapabilities(rawValue: 1 << 1)
    static let audioIn  = ModelCapabilities(rawValue: 1 << 2)
    static let videoIn  = ModelCapabilities(rawValue: 1 << 3)
    static let textOut  = ModelCapabilities(rawValue: 1 << 4)
    static let audioOut = ModelCapabilities(rawValue: 1 << 5)

    static let textOnly: ModelCapabilities = [.textIn, .textOut]
    static let vision: ModelCapabilities   = [.textIn, .imageIn, .textOut]
    static let videoVision: ModelCapabilities = [.textIn, .imageIn, .videoIn, .textOut]
    static let speechToText: ModelCapabilities = [.audioIn, .textOut]
    static let textToSpeech: ModelCapabilities = [.textIn, .audioOut]
    static let audioLLM: ModelCapabilities = [.textIn, .audioIn, .textOut]

    var acceptsAttachments: Bool {
        !intersection([.imageIn, .audioIn, .videoIn]).isEmpty
    }
}

// MARK: - Codable

nonisolated extension ModelCapabilities {

    /// Named flags, in bit order. Used for the JSON representation so that
    /// `catalog.json` stays hand-editable instead of carrying a bitmask.
    private static let namedFlags: [(name: String, flag: ModelCapabilities)] = [
        ("textIn", .textIn), ("imageIn", .imageIn), ("audioIn", .audioIn),
        ("videoIn", .videoIn), ("textOut", .textOut), ("audioOut", .audioOut)
    ]

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Raw bitmask is still accepted: registry files written before the
        // readable form existed must keep decoding.
        if let names = try? container.decode([String].self) {
            var result = ModelCapabilities()
            for name in names {
                if let flag = Self.namedFlags.first(where: { $0.name == name })?.flag {
                    result.insert(flag)
                }
            }
            self = result
        } else {
            self.init(rawValue: try container.decode(Int.self))
        }
    }

    func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(Self.namedFlags.filter { contains($0.flag) }.map(\.name))
    }
}

extension ModelCapabilities: CustomStringConvertible {
    var description: String {
        var parts: [String] = []
        if contains(.textIn) { parts.append("text-in") }
        if contains(.imageIn) { parts.append("image-in") }
        if contains(.audioIn) { parts.append("audio-in") }
        if contains(.videoIn) { parts.append("video-in") }
        if contains(.textOut) { parts.append("text-out") }
        if contains(.audioOut) { parts.append("audio-out") }
        return parts.joined(separator: ", ")
    }
}
