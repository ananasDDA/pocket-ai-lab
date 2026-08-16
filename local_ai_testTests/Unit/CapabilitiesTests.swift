//
//  CapabilitiesTests.swift
//

import Testing
import Foundation
@testable import local_ai_test

struct CapabilitiesTests {

    @Test func textOnlyIsInAndOut() {
        let caps: ModelCapabilities = .textOnly
        #expect(caps.contains(.textIn))
        #expect(caps.contains(.textOut))
        #expect(!caps.contains(.imageIn))
        #expect(!caps.contains(.audioIn))
        #expect(!caps.acceptsAttachments)
    }

    @Test func visionAcceptsAttachments() {
        #expect(ModelCapabilities.vision.acceptsAttachments)
        #expect(ModelCapabilities.videoVision.acceptsAttachments)
        #expect(ModelCapabilities.audioLLM.acceptsAttachments)
        #expect(ModelCapabilities.speechToText.acceptsAttachments)
    }

    @Test func textToSpeechHasAudioOut() {
        let caps: ModelCapabilities = .textToSpeech
        #expect(caps.contains(.audioOut))
        #expect(caps.contains(.textIn))
        #expect(!caps.contains(.textOut))
    }

    @Test func intersectionAndUnion() {
        let a: ModelCapabilities = [.textIn, .imageIn]
        let b: ModelCapabilities = [.imageIn, .audioIn]
        #expect(a.intersection(b) == .imageIn)
        #expect(a.union(b) == [.textIn, .imageIn, .audioIn])
    }

    @Test func codableRoundTrip() throws {
        let caps: ModelCapabilities = [.textIn, .imageIn, .audioOut]
        let data = try JSONEncoder().encode(caps)
        let back = try JSONDecoder().decode(ModelCapabilities.self, from: data)
        #expect(back == caps)
    }
}
