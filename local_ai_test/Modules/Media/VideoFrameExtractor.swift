//
//  VideoFrameExtractor.swift
//  local_ai_test
//
//  Turns a video asset into N evenly-sampled JPEG frames for consumption by
//  VLMs. Used as a poor-man's "video understanding" on iOS where no true
//  on-device video-LLM exists. The number of frames is caller-controlled,
//  typical values 4 / 8 / 16.
//

import Foundation
import AVFoundation
import UIKit

enum VideoFrameExtractor {

    struct ExtractResult {
        let frames: [Data]
        let durationSeconds: Double
    }

    /// Default `frameCount` is 4 (not 8) because each frame on Gemma 3
    /// expands to ~256 tokens — 8 frames would consume 2048 KV slots from
    /// an 8192-token window, leaving little room for chat history.
    /// Bump back to 8 if you have a 16K+ context model or only short turns.
    static func extract(from url: URL, frameCount: Int = 4, jpegQuality: CGFloat = 0.85) async throws -> ExtractResult {
        let asset = AVURLAsset(url: url)
        let duration = try await asset.load(.duration)
        let durationSeconds = CMTimeGetSeconds(duration)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        generator.maximumSize = CGSize(width: 768, height: 768)

        var times: [CMTime] = []
        for i in 0..<frameCount {
            let t = (Double(i) + 0.5) / Double(frameCount) * durationSeconds
            times.append(CMTime(seconds: t, preferredTimescale: 600))
        }

        var frames: [Data] = []
        for time in times {
            do {
                let result = try await generator.image(at: time)
                let cg = result.image
                let uiImage = UIImage(cgImage: cg)
                if let data = uiImage.jpegData(compressionQuality: jpegQuality) {
                    frames.append(data)
                }
            } catch {
                continue
            }
        }

        return ExtractResult(frames: frames, durationSeconds: durationSeconds)
    }
}
