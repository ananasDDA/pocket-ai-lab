//
//  DeviceTestTags.swift
//
//  Suite-level tags for swift-testing. Tests in `Device/` can only pass on
//  a real iPhone with the referenced models installed — they are excluded
//  from CI by default. Run them manually via:
//
//      xcodebuild test -scheme local_ai_test \
//          -only-testing:local_ai_testTests/DeviceTests \
//          -destination 'platform=iOS,id=...'
//

import Testing

extension Tag {
    @Tag static var device: Self
    @Tag static var performance: Self
    @Tag static var mlx: Self
    @Tag static var llamaCpp: Self
    @Tag static var coreML: Self
    @Tag static var whisper: Self
    @Tag static var kokoro: Self
    @Tag static var e2e: Self
    @Tag static var stress: Self
}
