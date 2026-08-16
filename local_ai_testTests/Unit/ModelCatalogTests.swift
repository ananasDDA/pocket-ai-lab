//
//  ModelCatalogTests.swift
//  local_ai_testTests
//
//  Validates the static model catalog: uniqueness, URL sanity, ram/disk
//  coherence, capabilities matching backend type.
//

import Testing
@testable import local_ai_test

struct ModelCatalogTests {

    @Test func allIDsAreUnique() {
        let ids = ModelCatalog.all.map(\.id)
        #expect(ids.count == Set(ids).count, "Duplicate model ids in catalog")
    }

    @Test func allHaveNonEmptyNames() {
        for model in ModelCatalog.all {
            #expect(!model.name.isEmpty, "Model \(model.id) has empty name")
            #expect(!model.family.isEmpty, "Model \(model.id) has empty family")
        }
    }

    @Test func ramRequirementIsReasonable() {
        for model in ModelCatalog.all where model.backend != .appleIntelligence {
            #expect(model.ramRequiredGB > 0, "\(model.id) declares zero RAM")
            #expect(model.ramRequiredGB <= 16, "\(model.id) RAM \(model.ramRequiredGB) looks wrong")
        }
    }

    @Test func diskSizeCoherent() {
        for model in ModelCatalog.all where model.backend != .appleIntelligence {
            #expect(model.diskSizeGB > 0)
            if model.requiresCompilation {
                #expect(model.effectiveDiskSizeGB == model.diskSizeGB * 2)
            } else {
                #expect(model.effectiveDiskSizeGB == model.diskSizeGB)
            }
        }
    }

    @Test func capabilitiesMatchBackend() {
        for model in ModelCatalog.all {
            switch model.backend {
            case .mlxVision, .llamaCppVision:
                #expect(model.capabilities.contains(.imageIn))
            case .mlxAudio:
                #expect(model.capabilities.contains(.audioIn))
                #expect(model.capabilities.contains(.textOut))
            case .whisperCpp, .coreMLWhisper:
                #expect(model.capabilities.contains(.audioIn))
                #expect(model.capabilities.contains(.textOut))
            case .coreMLKokoro:
                #expect(model.capabilities.contains(.audioOut))
            case .mlx, .llamaCpp, .coreML, .appleIntelligence:
                #expect(model.capabilities.contains(.textIn))
                #expect(model.capabilities.contains(.textOut))
            }
        }
    }

    @Test func fileLayoutConsistency() {
        for model in ModelCatalog.all {
            switch model.backend {
            case .llamaCpp:
                #expect(model.fileLayout == .singleGGUF)
            case .llamaCppVision:
                #expect(model.fileLayout == .ggufWithMmproj)
                #expect(model.mmprojFilename != nil, "\(model.id) has no mmproj filename")
            case .whisperCpp:
                #expect(model.fileLayout == .whisperGGML)
            case .coreML, .coreMLKokoro:
                #expect(model.fileLayout == .coreMLPackage)
                #expect(model.requiresCompilation == true)
            case .mlx, .mlxVision, .mlxAudio:
                #expect(model.fileLayout == .huggingFaceMLX)
            case .appleIntelligence, .coreMLWhisper:
                break
            }
        }
    }

    @Test func hfRepoLooksValid() {
        for model in ModelCatalog.all where model.backend != .appleIntelligence {
            let r = model.huggingFaceRepo
            #expect(!r.isEmpty)
            #expect(r.contains("/") || r.contains("ggerganov"))
        }
    }

    @Test func modelCatalogLookupById() {
        for model in ModelCatalog.all {
            #expect(ModelCatalog.model(id: model.id)?.id == model.id)
        }
        #expect(ModelCatalog.model(id: "nonsense/does-not-exist") == nil)
    }

    @Test func catalogHasAtLeastOneOfEachBackendFamily() {
        for family in BackendFamily.allCases where family != .apple {
            let models = ModelCatalog.models(forBackendFamily: family)
            #expect(!models.isEmpty, "No models for backend family \(family.rawValue)")
        }
    }
}
