//
//  GGUFHeaderReader.swift
//  local_ai_test
//
//  Minimal reader for the front of a GGUF file: the magic, the container
//  version and the metadata key/value block. The tensor data that follows is
//  never read — these files run to several gigabytes and everything the
//  import flow wants sits in the first few hundred kilobytes.
//
//  Format reference: the GGUF spec in ggml-org/ggml (docs/gguf.md).
//

import Foundation

// MARK: - Parsed header

nonisolated struct GGUFHeader: Sendable, Equatable {
    /// Container version. 2 and 3 are what converters emit today; 1 is the
    /// pre-release layout that used 32-bit lengths.
    let version: UInt32
    /// `general.architecture` — "llama", "qwen2", "gemma3", …
    let architecture: String?
    /// `general.name`, the name the converter stamped into the file.
    let name: String?
    /// Quantization read from `general.file_type`, e.g. "Q4_K_M". nil when the
    /// key is missing or carries a value this build does not know.
    let fileTypeLabel: String?
}

// MARK: - Reader

nonisolated enum GGUFHeaderReader {

    /// "GGUF" in file order.
    private static let magic: [UInt8] = [0x47, 0x47, 0x55, 0x46]

    /// How far into the file the metadata walk may go before giving up. The
    /// tokenizer vocabulary alone can be tens of megabytes, and a corrupt
    /// length prefix must not turn into an unbounded loop.
    private static let maxHeaderBytes: UInt64 = 64 * 1024 * 1024

    /// `general.file_type` values, from llama.cpp's `llama_ftype`. Only the
    /// ones a converter is likely to stamp into a shipped file are listed;
    /// anything else falls back to the filename.
    private static let fileTypeLabels: [UInt32: String] = [
        0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
        9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
        14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
        19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS",
        23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
        28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
        36: "TQ1_0", 37: "TQ2_0"
    ]

    /// Reads what it can from the header. Throws `GGUFReadError.unreadable`
    /// when the file itself could not be opened (an undownloaded iCloud
    /// placeholder, a sandbox permissions error, …) — distinct from a file
    /// that opened fine but was not a GGUF container, which returns nil.
    /// A file whose metadata block is truncated or uses an unknown value
    /// type still comes back non-nil, with nil fields, so the caller can
    /// fall back to the filename.
    static func read(fileURL: URL) throws -> GGUFHeader? {
        guard let handle = try? FileHandle(forReadingFrom: fileURL) else {
            throw GGUFReadError.unreadable
        }
        defer { try? handle.close() }

        var cursor = ByteCursor(handle: handle)
        guard let magicBytes = try? cursor.read(4), magicBytes == magic else { return nil }
        guard let version = try? cursor.readUInt32() else { return nil }

        // v1 wrote 32-bit counts and string lengths; v2 widened them to 64
        // bit and v3 only changed tensor quantization types. Anything newer
        // is walked as if it were v3 — at worst the walk bails out early and
        // the filename supplies the metadata.
        let wide = version >= 2

        var architecture: String?
        var name: String?
        var fileType: UInt32?

        // A partially-walked block is still useful: converters write the
        // `general.*` keys first, so the interesting ones land well before
        // anything can go wrong.
        do {
            _ = try cursor.readLength(wide: wide)                // tensor count
            var remaining = try cursor.readLength(wide: wide)    // KV count

            while remaining > 0, cursor.consumed < maxHeaderBytes {
                remaining -= 1
                let key = try cursor.readString(wide: wide)
                let rawType = try cursor.readUInt32()
                guard let type = GGUFValueType(rawValue: rawType) else {
                    // An unknown type has an unknown width, so the walk
                    // cannot step over its value. Stop with what is in hand.
                    break
                }

                switch (key, type) {
                case ("general.architecture", .string):
                    architecture = try cursor.readString(wide: wide)
                case ("general.name", .string):
                    name = try cursor.readString(wide: wide)
                // Written as either signed or unsigned depending on the
                // converter; the four bytes are the same either way and every
                // real value is a small positive.
                case ("general.file_type", .uint32), ("general.file_type", .int32):
                    fileType = try cursor.readUInt32()
                default:
                    try skipValue(type, in: &cursor, wide: wide)
                }

                if architecture != nil, name != nil, fileType != nil { break }
            }
        } catch {
            // Truncated or malformed metadata. Keep whatever was collected.
        }

        return GGUFHeader(
            version: version,
            architecture: architecture,
            name: name,
            fileTypeLabel: fileType.flatMap { fileTypeLabels[$0] }
        )
    }

    /// Nesting only grows through arrays-of-arrays; a hostile file can chain
    /// these arbitrarily deep purely in the header (no tensor data needed),
    /// which without a cap would recurse until the call stack overflows.
    private static let maxArrayDepth = 8

    /// Steps over a value without materialising it. Fixed-width types and
    /// arrays of them are a single seek; strings and string arrays have to be
    /// walked one length prefix at a time.
    private static func skipValue(
        _ type: GGUFValueType, in cursor: inout ByteCursor, wide: Bool, depth: Int = 0
    ) throws {
        switch type {
        case .string:
            let length = try cursor.readLength(wide: wide)
            try cursor.skip(length)

        case .array:
            guard depth < maxArrayDepth else { throw GGUFReadError.malformed }
            let rawElement = try cursor.readUInt32()
            guard let element = GGUFValueType(rawValue: rawElement) else {
                throw GGUFReadError.malformed
            }
            let count = try cursor.readLength(wide: wide)
            guard count <= maxHeaderBytes else { throw GGUFReadError.malformed }
            if let width = element.fixedWidth {
                try cursor.skip(count * UInt64(width))
            } else {
                for _ in 0 ..< count {
                    guard cursor.consumed < maxHeaderBytes else {
                        throw GGUFReadError.malformed
                    }
                    try skipValue(element, in: &cursor, wide: wide, depth: depth + 1)
                }
            }

        default:
            guard let width = type.fixedWidth else { throw GGUFReadError.malformed }
            try cursor.skip(UInt64(width))
        }
    }
}

// MARK: - Wire types

nonisolated private enum GGUFValueType: UInt32 {
    case uint8 = 0
    case int8 = 1
    case uint16 = 2
    case int16 = 3
    case uint32 = 4
    case int32 = 5
    case float32 = 6
    case bool = 7
    case string = 8
    case array = 9
    case uint64 = 10
    case int64 = 11
    case float64 = 12

    /// Bytes the value occupies, nil for the two variable-length types.
    var fixedWidth: Int? {
        switch self {
        case .uint8, .int8, .bool:      return 1
        case .uint16, .int16:           return 2
        case .uint32, .int32, .float32: return 4
        case .uint64, .int64, .float64: return 8
        case .string, .array:           return nil
        }
    }
}

nonisolated enum GGUFReadError: Error {
    case truncated
    case malformed
    /// The file could not be opened at all — see `GGUFHeaderReader.read`.
    case unreadable
}

// MARK: - Buffered cursor

/// Forward-only cursor over the front of a file. Reads in chunks and seeks
/// past values it does not care about, so walking a 40 MB tokenizer array
/// costs a handful of syscalls rather than one per entry.
nonisolated private struct ByteCursor {

    private static let chunkSize = 256 * 1024

    /// Longest string the header may contain. Real keys and values are tiny;
    /// anything larger means the length prefix was misread.
    private static let maxStringBytes: UInt64 = 1 << 20

    private let handle: FileHandle
    private var buffer: [UInt8] = []
    private var index = 0

    /// Bytes stepped over so far, used as the walk's budget.
    private(set) var consumed: UInt64 = 0

    init(handle: FileHandle) {
        self.handle = handle
    }

    private var available: Int { buffer.count - index }

    private mutating func refill(minimum: Int) throws {
        while available < minimum {
            guard let chunk = try handle.read(upToCount: Self.chunkSize),
                  !chunk.isEmpty
            else { throw GGUFReadError.truncated }
            if index > 0 {
                buffer.removeFirst(index)
                index = 0
            }
            buffer.append(contentsOf: chunk)
        }
    }

    mutating func read(_ count: Int) throws -> [UInt8] {
        guard count >= 0 else { throw GGUFReadError.malformed }
        try refill(minimum: count)
        let bytes = Array(buffer[index ..< (index + count)])
        index += count
        consumed += UInt64(count)
        return bytes
    }

    mutating func skip(_ count: UInt64) throws {
        let buffered = UInt64(available)
        if count <= buffered {
            index += Int(count)
        } else {
            // The handle is already `available` bytes ahead of the logical
            // position, so dropping the buffer first makes the two agree.
            let remainder = count - buffered
            buffer.removeAll(keepingCapacity: true)
            index = 0
            let target = try handle.offset() + remainder
            try handle.seek(toOffset: target)
        }
        consumed += count
    }

    mutating func readUInt32() throws -> UInt32 {
        let b = try read(4)
        return UInt32(b[0]) | UInt32(b[1]) << 8 | UInt32(b[2]) << 16 | UInt32(b[3]) << 24
    }

    mutating func readUInt64() throws -> UInt64 {
        let b = try read(8)
        var value: UInt64 = 0
        for byte in b.reversed() { value = value << 8 | UInt64(byte) }
        return value
    }

    /// Counts and string lengths: 64 bit from GGUF v2 on, 32 bit in v1.
    mutating func readLength(wide: Bool) throws -> UInt64 {
        if wide { return try readUInt64() }
        return UInt64(try readUInt32())
    }

    mutating func readString(wide: Bool) throws -> String {
        let length = try readLength(wide: wide)
        guard length <= Self.maxStringBytes else { throw GGUFReadError.malformed }
        let bytes = try read(Int(length))
        return String(decoding: bytes, as: UTF8.self)
    }
}
