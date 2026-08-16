//
//  ImagePreprocessorTests.swift
//

import Testing
import Foundation
import UIKit
@testable import local_ai_test

struct ImagePreprocessorTests {

    private func makeTestImage(size: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        return UIGraphicsImageRenderer(size: size, format: format).image { ctx in
            UIColor.red.setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
        }
    }

    @Test func largeImageIsScaledDown() throws {
        let original = makeTestImage(size: CGSize(width: 4000, height: 3000))
        let data = ImagePreprocessor.normalize(original, maxDimension: 896)
        #expect(data != nil)
        let resized = UIImage(data: data!)!
        #expect(max(resized.size.width, resized.size.height) <= 896)
    }

    @Test func smallImageIsLeftAlone() throws {
        let original = makeTestImage(size: CGSize(width: 200, height: 100))
        let data = ImagePreprocessor.normalize(original, maxDimension: 896)
        let resized = UIImage(data: data!)!
        #expect(resized.size.width == 200)
        #expect(resized.size.height == 100)
    }

    @Test func dataRoundTrip() {
        let original = makeTestImage(size: CGSize(width: 1024, height: 1024))
        let jpegInput = original.jpegData(compressionQuality: 0.9)!
        let out = ImagePreprocessor.normalize(jpegInput, maxDimension: 512)
        #expect(out != nil)
        #expect(out!.count < jpegInput.count)
    }
}
