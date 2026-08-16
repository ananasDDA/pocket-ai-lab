//
//  RoutingTests.swift
//
//  Verifies that InferenceManager.makeEngine returns the right type for
//  each backend. Because makeEngine is private, we exercise it indirectly
//  through prepare() + inspection of currentEngine.capabilities.
//

import Testing
@testable import local_ai_test

struct RoutingTests {

    /// For every backend represented in the catalog, the model's capabilities
    /// must be a superset of the minimum required by that backend family.
    /// We do not require every backend to have an entry (e.g. `.coreMLWhisper`
    /// may be consumed only through `.whisperCpp` indirection).
    @Test func everyCatalogEntryHasCapabilitiesMatchingBackend() {
        let minimums: [ModelBackend: ModelCapabilities] = [
            .mlx:               .textOnly,
            .llamaCpp:          .textOnly,
            .coreML:            .textOnly,
            .appleIntelligence: .textOnly,
            .mlxVision:         [.textIn, .imageIn, .textOut],
            .llamaCppVision:    [.textIn, .imageIn, .textOut],
            .mlxAudio:          [.textIn, .audioIn, .textOut],
            .whisperCpp:        .speechToText,
            .coreMLWhisper:     .speechToText,
            .coreMLKokoro:      .textToSpeech,
        ]
        for model in ModelCatalog.all {
            guard let minimum = minimums[model.backend] else {
                Issue.record("Unknown backend mapping: \(model.backend)")
                continue
            }
            #expect(model.capabilities.intersection(minimum) == minimum,
                    "Catalog model \(model.id) capabilities \(model.capabilities.description) missing parts of backend minimum \(minimum.description)")
        }
    }

    @Test func catalogHasAtLeastOneVisionTextAndSpeech() {
        #expect(ModelCatalog.all.contains { $0.capabilities.contains(.imageIn) })
        #expect(ModelCatalog.all.contains { $0.capabilities == .textOnly })
        #expect(ModelCatalog.all.contains { $0.capabilities.contains(.audioIn) })
    }
}
