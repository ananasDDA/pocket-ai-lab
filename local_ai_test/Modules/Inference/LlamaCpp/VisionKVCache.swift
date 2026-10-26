//
//  VisionKVCache.swift
//  local_ai_test
//
//  The bookkeeping behind KV prefix reuse in `LlamaVisionContext`: what the
//  cache holds, described as the chunk list libmtmd produced for it, and
//  the arithmetic that finds how much of a new prompt the cache already
//  holds.
//
//  The chat re-sends the whole conversation on every turn. Decoding it from
//  scratch made every reply slower than the last (7-13 s to the first token
//  by turn ten on an iPhone Air) while the text engine, which reuses its
//  prefix, stayed under 400 ms. Here the same idea is applied to a prompt
//  that is not a flat token list: libmtmd splits it into text runs and
//  media chunks, and a media chunk is an opaque block of embeddings that
//  can only be kept or dropped whole.
//
//  Positions, not tokens, are the unit everything is measured in. A text
//  token takes one position. A media chunk takes `mtmd_input_chunk_get_n_pos`
//  positions, which equals its token count for most models but not for the
//  M-RoPE ones (Qwen-VL family), where an image spans max(rows, columns)
//  positions however many embeddings it has. `llama_memory_seq_rm` cuts by
//  position, so the cut points computed here are positions too.
//
//  Pure Swift on purpose: no dependency on the `llama` module, so the
//  matcher is unit-testable on any target. Token ids are plain `Int32`,
//  which is what `llama_token` is.
//
//  Every type here is `nonisolated`. The app target defaults to MainActor
//  isolation, and `LlamaVisionContext` (its own actor) calls these from
//  inside `generate`: as main-actor code each call was an implicit hop to
//  the main actor, i.e. a suspension point in the middle of a generation.
//  Another `generate` then ran in between on the same KV cache - two
//  conversations tokenized, rewound and decoded interleaved - which showed
//  up on the device as cache misses, empty replies and answers about the
//  other conversation's picture.
//

import Foundation
import CryptoKit

/// A media chunk (one image, or one tile of an image) as the cache sees it.
nonisolated struct KVMedia: Equatable, Sendable {
    /// Content hash of the decoded bitmap (see `KVPrefix.mediaID`). Two
    /// chunks with the same id, token count and position count are the same
    /// embeddings.
    let id: String
    /// Embedding tokens the chunk decodes to.
    let tokens: Int
    /// KV positions the chunk advances by.
    let positions: Int
}

/// One run of the prompt as it sits in the KV cache, in prompt order.
nonisolated enum KVChunk: Equatable, Sendable {
    case text([Int32])
    case media(KVMedia)

    var positions: Int {
        switch self {
        case .text(let tokens): return tokens.count
        case .media(let media): return media.positions
        }
    }

    var tokens: Int {
        switch self {
        case .text(let tokens): return tokens.count
        case .media(let media): return media.tokens
        }
    }

    var isMedia: Bool {
        if case .media = self { return true }
        return false
    }
}

/// Where a position falls in a chunk list: the chunk that holds the first
/// position after the cut, and how many of that chunk's tokens come before
/// the cut (always 0 for a media chunk - media is never split).
nonisolated struct KVCut: Equatable, Sendable {
    let chunk: Int
    let offset: Int
}

nonisolated enum KVPrefix {

    // MARK: Measuring

    static func totalPositions(_ chunks: [KVChunk]) -> Int {
        chunks.reduce(0) { $0 + $1.positions }
    }

    static func totalTokens(_ chunks: [KVChunk]) -> Int {
        chunks.reduce(0) { $0 + $1.tokens }
    }

    static func mediaCount(_ chunks: [KVChunk]) -> Int {
        chunks.reduce(0) { $0 + ($1.isMedia ? 1 : 0) }
    }

    // MARK: Matching

    /// Positions at the start of `cached` and `new` that hold the same
    /// thing. Text is compared token by token, whatever the run boundaries
    /// (the cache holds the previous prompt and the reply as one run; the
    /// new prompt tokenizes the reply as part of a longer one). A media
    /// chunk matches only the same media at the same place: same id, same
    /// token and position count.
    static func commonPositions(_ cached: [KVChunk], _ new: [KVChunk]) -> Int {
        var walkCached = Walker(cached)
        var walkNew = Walker(new)
        var shared = 0
        while let a = walkCached.current, let b = walkNew.current {
            switch (a, b) {
            case (.token(let x), .token(let y)) where x == y:
                shared += 1
            case (.media(let x), .media(let y)) where x == y:
                shared += x.positions
            default:
                return shared
            }
            walkCached.advance()
            walkNew.advance()
        }
        return shared
    }

    /// The part of `new` that can be taken from a cache holding `cached`.
    ///
    /// It is the common prefix, except that at least the last item of the
    /// prompt is always evaluated again: sampling needs the logits of the
    /// final prompt position, and those only exist for something decoded in
    /// this call. A trailing media chunk is re-evaluated whole.
    static func reusablePositions(cached: [KVChunk], new: [KVChunk]) -> Int {
        let common = commonPositions(cached, new)
        let total = totalPositions(new)
        guard total > 0, common >= total else { return common }
        guard let last = new.last(where: { $0.positions > 0 }) else { return 0 }
        return total - (last.isMedia ? last.positions : 1)
    }

    /// The chunk and token offset where `position` falls, or nil when it is
    /// past the end or inside a media chunk (media cannot be split).
    static func cut(_ chunks: [KVChunk], atPosition position: Int) -> KVCut? {
        guard position >= 0 else { return nil }
        var start = 0
        for (index, chunk) in chunks.enumerated() {
            let end = start + chunk.positions
            if position < end || (position == start && chunk.positions == 0) {
                switch chunk {
                case .text:
                    return KVCut(chunk: index, offset: position - start)
                case .media:
                    return position == start ? KVCut(chunk: index, offset: 0) : nil
                }
            }
            start = end
        }
        return position == start ? KVCut(chunk: chunks.count, offset: 0) : nil
    }

    /// The first `positions` positions of `chunks`, or nil when that cut
    /// would split a media chunk.
    static func prefix(_ chunks: [KVChunk], positions: Int) -> [KVChunk]? {
        guard let cut = cut(chunks, atPosition: positions) else { return nil }
        var out = Array(chunks[..<cut.chunk])
        if cut.offset > 0, case .text(let tokens) = chunks[cut.chunk] {
            out.append(.text(Array(tokens[..<cut.offset])))
        }
        return out
    }

    /// `chunks` with `tokens` added at the end, merged into a trailing text
    /// run so the record stays in the shape libmtmd produces.
    static func appending(_ tokens: [Int32], to chunks: [KVChunk]) -> [KVChunk] {
        guard !tokens.isEmpty else { return chunks }
        var out = chunks
        if case .text(let existing)? = out.last {
            out[out.count - 1] = .text(existing + tokens)
        } else {
            out.append(.text(tokens))
        }
        return out
    }

    // MARK: Checkpoints (recurrent / hybrid memory)

    /// The newest checkpoint a rewind to `reusable` can start from: one at
    /// or before the cut, never at 0 (an empty state is a full clear).
    static func checkpoint(atOrBefore reusable: Int, among positions: [Int]) -> Int? {
        positions.filter { $0 > 0 && $0 <= reusable }.max()
    }

    /// Where to snapshot a recurrent state while evaluating `new` from
    /// `reused`: `offsetFromEnd` positions before the end of the prompt,
    /// when that point lies inside a text run that this call decodes.
    ///
    /// Four is what llama-server uses. It used to be exactly the next turn's
    /// divergence for Qwen 3 / 3.5, whose history rendered the reply without
    /// the four-token empty reasoning block the prompt ended in;
    /// `ChatTemplateResolver.historyAsGenerated` now keeps that block, so the
    /// snapshot is a fallback for any template that still re-renders the end
    /// of the prompt differently on the next turn.
    static func checkpointPosition(in new: [KVChunk], reused: Int, offsetFromEnd: Int) -> Int? {
        let target = totalPositions(new) - offsetFromEnd
        guard offsetFromEnd > 0, target > reused, target > 0,
              let cut = cut(new, atPosition: target),
              cut.chunk < new.count,
              cut.offset > 0,
              case .text = new[cut.chunk]
        else { return nil }
        return target
    }

    // MARK: Tokens and identity

    /// Drops one of two leading BOS tokens. `mtmd_tokenize` with
    /// `add_special` prepends BOS unconditionally, so a rendered prompt
    /// that already starts with the literal BOS marker would decode it
    /// twice - the text engine guards against the same thing.
    static func droppingDuplicateLeadingBOS(_ tokens: [Int32], bos: Int32) -> [Int32] {
        guard tokens.count >= 2, tokens[0] == bos, tokens[1] == bos else { return tokens }
        return Array(tokens.dropFirst())
    }

    /// Stable identity for a decoded bitmap: SHA-256 over its size and its
    /// pixel bytes. The same photo re-sent on a later turn hashes the same;
    /// any other picture - even one at the same place in the conversation -
    /// does not.
    static func mediaID(width: UInt32, height: UInt32, bytes: UnsafeRawBufferPointer) -> String {
        var hasher = SHA256()
        withUnsafeBytes(of: width.littleEndian) { hasher.update(bufferPointer: $0) }
        withUnsafeBytes(of: height.littleEndian) { hasher.update(bufferPointer: $0) }
        hasher.update(bufferPointer: bytes)
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: Walking

    private enum Item {
        case token(Int32)
        case media(KVMedia)
    }

    /// Steps through a chunk list one token or one whole media at a time,
    /// skipping empty text runs.
    private struct Walker {
        let chunks: [KVChunk]
        var chunk = 0
        var offset = 0

        init(_ chunks: [KVChunk]) {
            self.chunks = chunks
            skipEmpty()
        }

        var current: Item? {
            guard chunk < chunks.count else { return nil }
            switch chunks[chunk] {
            case .text(let tokens): return .token(tokens[offset])
            case .media(let media): return .media(media)
            }
        }

        mutating func advance() {
            guard chunk < chunks.count else { return }
            switch chunks[chunk] {
            case .text(let tokens):
                offset += 1
                if offset >= tokens.count { chunk += 1; offset = 0 }
            case .media:
                chunk += 1
                offset = 0
            }
            skipEmpty()
        }

        private mutating func skipEmpty() {
            while chunk < chunks.count, case .text(let tokens) = chunks[chunk], tokens.isEmpty {
                chunk += 1
            }
        }
    }
}
