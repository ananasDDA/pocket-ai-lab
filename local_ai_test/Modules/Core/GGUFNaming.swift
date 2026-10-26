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
        path.lowercased().hasSuffix(".gguf") && !isProjector(path) && !isImportanceMatrix(path)
    }

    /// True for an importance matrix saved as GGUF (`imatrix.gguf`,
    /// `imatrix_unsloth.gguf`). Quantizers publish it next to the weights;
    /// it is calibration data, not a model, and cannot be loaded.
    static func isImportanceMatrix(_ path: String) -> Bool {
        basename(path).lowercased().contains("imatrix")
    }

    // MARK: - Split GGUF

    /// One file of a model that `gguf-split` cut into several parts:
    /// `<prefix>-00002-of-00004.gguf`. llama.cpp (b8882 included) loads such
    /// a model when it is handed the FIRST part and every other part sits
    /// next to it under exactly this naming; a single part on its own does
    /// not load.
    struct SplitPart: Sendable, Equatable, Hashable {
        /// Everything before `-0000N-of-0000M.gguf`, directories included.
        let prefix: String
        /// 1-based part number.
        let index: Int
        /// Number of parts in the set.
        let count: Int
    }

    /// Parses the split naming llama.cpp itself expects (`llama_split_path`:
    /// five-digit index and count). nil for an ordinary single-file GGUF.
    static func splitPart(_ path: String) -> SplitPart? {
        let lower = path.lowercased()
        guard lower.hasSuffix(".gguf") else { return nil }
        let stem = path.dropLast(5)
        // "-00001-of-00003" is exactly 15 characters.
        guard stem.count > 15 else { return nil }
        let tail = stem.suffix(15)
        let chars = Array(tail)
        guard chars[0] == "-", String(chars[6...9]).lowercased() == "-of-" else { return nil }
        let indexDigits = String(chars[1...5])
        let countDigits = String(chars[10...14])
        let isDigit: (Character) -> Bool = { $0.isASCII && $0.isNumber }
        guard indexDigits.allSatisfy(isDigit), countDigits.allSatisfy(isDigit),
              let index = Int(indexDigits), let count = Int(countDigits),
              index >= 1, count >= 1, index <= count
        else { return nil }
        return SplitPart(prefix: String(stem.dropLast(15)), index: index, count: count)
    }

    /// The repo-relative file of part `index` in the same set as `part`.
    static func splitFilename(prefix: String, index: Int, count: Int) -> String {
        prefix + String(format: "-%05d-of-%05d.gguf", index, count)
    }

    /// Every part of the set `first` belongs to that is present in `names`,
    /// in part order. Empty when `first` is not a split file.
    static func splitSiblings(of first: String, in names: [String]) -> [String] {
        guard let part = splitPart(first) else { return [] }
        return names
            .compactMap { name -> (Int, String)? in
                guard let other = splitPart(name),
                      other.prefix == part.prefix, other.count == part.count
                else { return nil }
                return (other.index, name)
            }
            .sorted { $0.0 < $1.0 }
            .map(\.1)
    }
}
