//
//  AudioPlayer.swift
//  local_ai_test
//
//  Streaming playback of PCM chunks produced by a TTS engine (e.g. Kokoro).
//  The public API accepts Float PCM + sample rate; chunks are scheduled on
//  an `AVAudioPlayerNode` so the UI can enqueue sentences progressively as
//  the LLM generates text.
//

import Foundation
import AVFoundation
import Observation

@MainActor
@Observable
final class AudioPlayer {

    static let shared = AudioPlayer()

    private(set) var isPlaying = false

    @ObservationIgnored
    private let engine = AVAudioEngine()
    @ObservationIgnored
    private let player = AVAudioPlayerNode()
    @ObservationIgnored
    private var format: AVAudioFormat?

    private init() {
        engine.attach(player)
    }

    /// Configures the engine for a given sample rate / channels.
    func prepare(sampleRate: Double, channels: AVAudioChannelCount = 1) throws {
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: channels,
            interleaved: false
        ) else {
            throw InferenceError.generationFailed(
                "AudioPlayer: unsupported PCM format (sr=\(sampleRate), ch=\(channels))"
            )
        }
        if self.format?.sampleRate != sampleRate {
            engine.disconnectNodeOutput(player)
            engine.connect(player, to: engine.mainMixerNode, format: format)
            self.format = format
        }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)

        if !engine.isRunning { try engine.start() }
    }

    /// Schedules a PCM chunk. Automatically starts playback on the first enqueue.
    func enqueue(pcm: [Float]) {
        guard let format else { return }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(pcm.count)) else { return }
        buffer.frameLength = AVAudioFrameCount(pcm.count)
        if let channel = buffer.floatChannelData?[0] {
            _ = pcm.withUnsafeBufferPointer { src in
                memcpy(channel, src.baseAddress, pcm.count * MemoryLayout<Float>.size)
            }
        }
        player.scheduleBuffer(buffer) { [weak self] in
            Task { @MainActor [weak self] in self?.checkPlaying() }
            _ = self
        }
        if !player.isPlaying { player.play(); isPlaying = true }
    }

    /// Pushes raw PCM bytes (as produced by `ChatOutput.audioChunk`).
    func enqueue(data: Data, sampleRate: Int, channels: Int) {
        // Data is not guaranteed to be Float-aligned; baseAddress is nil for
        // empty buffers. Copy into a properly-aligned array instead.
        let floats = [Float](unsafeUninitializedCapacity: data.count / MemoryLayout<Float>.size) { dest, initializedCount in
            let byteCount = dest.count * MemoryLayout<Float>.size
            if byteCount > 0 {
                data.copyBytes(to: UnsafeMutableRawBufferPointer(dest), count: byteCount)
            }
            initializedCount = dest.count
        }
        guard !floats.isEmpty else { return }
        do {
            try prepare(sampleRate: Double(sampleRate), channels: AVAudioChannelCount(channels))
            enqueue(pcm: floats)
        } catch {
            print("[AudioPlayer] prepare error: \(error)")
        }
    }

    func stop() {
        player.stop()
        isPlaying = false
    }

    private func checkPlaying() {
        // No direct API for "queue empty" — the completion handler is called
        // per-buffer. We keep `isPlaying` true until the caller explicitly
        // stops, or the engine is torn down. UI typically does not care.
    }
}
