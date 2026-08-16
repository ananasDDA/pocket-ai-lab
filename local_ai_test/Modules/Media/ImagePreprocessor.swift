//
//  ImagePreprocessor.swift
//  local_ai_test
//
//  Downscales and re-encodes images to cut down attachment size before they
//  hit the inference engine. Most VLMs internally resize to ~336-768 px
//  anyway, so sending 4K originals is pure overhead.
//

import Foundation
import UIKit

enum ImagePreprocessor {

    /// Rescales `source` so its longest side equals `maxDimension`, preserving
    /// aspect ratio. Encodes as JPEG to keep size down. Returns nil on failure.
    static func normalize(_ source: UIImage, maxDimension: CGFloat = 896, quality: CGFloat = 0.85) -> Data? {
        let w = source.size.width
        let h = source.size.height
        let longSide = max(w, h)
        let scale: CGFloat = longSide > maxDimension ? maxDimension / longSide : 1.0
        let targetSize = CGSize(width: floor(w * scale), height: floor(h * scale))

        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        format.opaque = false
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let resized = renderer.image { _ in
            source.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        return resized.jpegData(compressionQuality: quality)
    }

    static func normalize(_ data: Data, maxDimension: CGFloat = 896) -> Data? {
        guard let ui = UIImage(data: data) else { return nil }
        return normalize(ui, maxDimension: maxDimension)
    }
}
