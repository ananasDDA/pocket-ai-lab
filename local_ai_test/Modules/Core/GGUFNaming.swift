//
//  GGUFNaming.swift
//  local_ai_test
//
//  What a `.gguf` filename says about the file.
//
//  A multimodal GGUF repo ships two files: the language weights and a vision
//  projector. Nothing in the format marks which is which - the convention is
//  the word "mmproj" in the filename, and the ecosystem writes it at either
//  end:
//
//      mmproj-F16.gguf                     bartowski, unsloth, ggml-org
//      PaddleOCR-VL-1.6-GGUF-mmproj.gguf   PaddlePaddle's own release
//      PaddleOCR-VL-1.6-mmproj.gguf        community requants
//
//  This app used to match the prefix only, so a repo naming its projector the
//  second way imported as a TEXT model: the projector was never found, the
//  model was registered `llamaCpp` with text-only capabilities, and the user
//  could not attach an image to a vision model. The projector was also
//  offered as a weight file to download, being 840 MB of "not mmproj-*".
//

import Foundation

nonisolated enum GGUFNaming {

    /// The last path component of a repo-relative filename.
    static func basename(_ path: String) -> String {
        path.split(separator: "/").last.map(String.init) ?? path
    }

    /// True when this `.gguf` is a vision/audio projector rather than model
    /// weights. Matched anywhere in the basename - see the file comment for
    /// why a prefix test is not enough.
    ///
    /// The extension is not required: callers filter by `.gguf` first, and a
    /// projector in a `.bin` (older LLaVA repos) is the same thing.
    static func isProjector(_ path: String) -> Bool {
        basename(path).lowercased().contains("mmproj")
    }

    /// True for a `.gguf` that holds model weights.
    static func isWeights(_ path: String) -> Bool {
        path.lowercased().hasSuffix(".gguf") && !isProjector(path)
    }
}
