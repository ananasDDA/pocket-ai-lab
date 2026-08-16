//
//  ChatAttachment.swift
//  local_ai_test
//
//  Non-textual content that can be attached to a chat turn.
//  Carried through the engine pipeline together with the user message.
//

import Foundation

enum ChatAttachment: Sendable, Hashable {
    /// Encoded image bytes (JPEG, PNG, HEIC). Engines normalize internally.
    case image(Data)

    /// Raw PCM audio plus sample rate. 16 kHz mono f32 or s16 is the
    /// preferred format; engines resample if needed.
    case audio(Data, sampleRate: Int, channels: Int = 1)

    /// A set of frames sampled from a video asset. Each frame is JPEG/PNG
    /// bytes. Duration is informational, used for prompt formatting.
    case videoFrames([Data], durationSeconds: Double)

    var kind: Kind {
        switch self {
        case .image: return .image
        case .audio: return .audio
        case .videoFrames: return .video
        }
    }

    enum Kind: String, Sendable { case image, audio, video }

    var approximateBytes: Int {
        switch self {
        case .image(let data): return data.count
        case .audio(let data, _, _): return data.count
        case .videoFrames(let frames, _): return frames.reduce(0) { $0 + $1.count }
        }
    }
}
