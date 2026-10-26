//
//  DailySampling.swift
//
//  The daily-use suite's DAILY_SAMPLING switch, kept apart from the UIKit
//  runner so it can be unit-tested:
//
//    catalog  (default) what the chat screen sends: the global chat
//             parameters with the model's catalog `sampling` merged over them
//             (`ChatViewModel.parameters(for:)`).
//    global   the global chat parameters, ignoring any catalog `sampling`.
//    authors  the global chat parameters with the model authors' values from
//             `DailyUseAuthorsSampling.json` (test bundle) merged over them,
//             whether or not the bundled catalog carries any. A model with
//             no entry there runs with the global parameters.
//

import Foundation
@testable import local_ai_test

enum DailySamplingMode: String, CaseIterable, Codable {
    case catalog
    case global
    case authors

    /// Unset or blank means `catalog`; an unknown value is nil, so a typo
    /// fails the run instead of quietly running the default.
    static func parse(_ raw: String?) -> DailySamplingMode? {
        let value = raw?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
        if value.isEmpty { return .catalog }
        return DailySamplingMode(rawValue: value)
    }
}

/// `DailyUseAuthorsSampling.json`: catalog id -> the authors' sampling.
struct DailyAuthorsSampling: Decodable {
    let models: [String: ModelSampling]

    static let resourceName = "DailyUseAuthorsSampling.json"

    static func decode(_ data: Data) throws -> DailyAuthorsSampling {
        try JSONDecoder().decode(DailyAuthorsSampling.self, from: data)
    }
}

enum DailySampling {

    /// Parameters for one daily-use run before its own tweaks (seed, the
    /// sustained scenario's cap). `global` is `ChatViewModel.parameters`;
    /// `catalogSampling` is what the chat would merge for this model
    /// (`ModelSampling.effective`).
    static func parameters(
        mode: DailySamplingMode,
        global: GenerationParameters,
        catalogSampling: ModelSampling?,
        authorsSampling: ModelSampling?
    ) -> GenerationParameters {
        switch mode {
        case .catalog: return global.applying(catalogSampling)
        case .global: return global
        case .authors: return global.applying(authorsSampling)
        }
    }
}
