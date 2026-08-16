//
//  AudioRecorder.swift
//  local_ai_test
//
//  Captures microphone audio as 16 kHz mono Float32 PCM — the format both
//  Whisper and Qwen2-Audio expect. Publishes an input level (RMS) for UI
//  waveform rendering.
//
//  Integration point: `record()` starts buffering; `stop()` returns the
//  accumulated PCM. The buffer is owned by the recorder, so the caller
//  should copy it immediately if it needs to survive a new recording.
//

import Foundation
@preconcurrency import AVFoundation
import Observation

@MainActor
@Observable
final class AudioRecorder {

    static let shared = AudioRecorder()

    private(set) var isRecording = false
    private(set) var level: Float = 0.0 // 0…1 normalized RMS

    @ObservationIgnored
    private let engine = AVAudioEngine()
    @ObservationIgnored
    private var buffer: [Float] = []
    @ObservationIgnored
    private let targetSampleRate: Double = 16_000
    @ObservationIgnored
    private var converter: AVAudioConverter?

    private init() {}

    // MARK: - Public

    func requestPermission() async -> Bool {
        await withCheckedContinuation { cont in
            AVAudioApplication.requestRecordPermission { granted in
                cont.resume(returning: granted)
            }
        }
    }

    func record() async throws {
        guard !isRecording else { return }
        guard await requestPermission() else {
            throw NSError(domain: "AudioRecorder", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied"])
        }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)

        buffer.removeAll(keepingCapacity: true)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw NSError(domain: "AudioRecorder", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Unsupported target audio format"])
        }
        converter = AVAudioConverter(from: inputFormat, to: targetFormat)

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] pcmBuffer, _ in
            guard let self else { return }
            Task { @MainActor in
                self.handleBuffer(pcmBuffer, targetFormat: targetFormat)
            }
        }

        try engine.start()
        isRecording = true
    }

    func stop() -> [Float] {
        guard isRecording else { return [] }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRecording = false
        level = 0
        try? AVAudioSession.sharedInstance().setActive(false, options: [.notifyOthersOnDeactivation])
        return buffer
    }

    // MARK: - Private

    private func handleBuffer(_ source: AVAudioPCMBuffer, targetFormat: AVAudioFormat) {
        guard let converter else { return }

        let ratio = targetFormat.sampleRate / source.format.sampleRate
        let capacity = AVAudioFrameCount(Double(source.frameLength) * ratio) + 1024
        guard let out = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: capacity) else { return }

        var consumed = false
        var error: NSError?
        converter.convert(to: out, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return source
        }

        if let error {
            print("[AudioRecorder] conversion error: \(error)")
            return
        }

        guard let channelData = out.floatChannelData?[0] else { return }
        let frames = Int(out.frameLength)
        var rms: Float = 0
        for i in 0..<frames {
            let v = channelData[i]
            rms += v * v
        }
        rms = sqrt(rms / Float(max(frames, 1)))

        buffer.append(contentsOf: UnsafeBufferPointer(start: channelData, count: frames))
        self.level = min(1.0, rms * 5)
    }

    var recordedData: Data {
        buffer.withUnsafeBufferPointer { Data(buffer: $0) }
    }
}
