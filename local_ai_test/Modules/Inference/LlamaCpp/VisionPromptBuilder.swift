//
//  VisionPromptBuilder.swift
//  local_ai_test
//
//  Turns a conversation into the (prompt, images) pair libmtmd expects: a
//  formatted transcript carrying one `<__media__>` marker per image, the
//  bitmaps in the same order the markers appear, and the whole thing cut
//  down to something the context window can hold.
//
//  Media is replayed for EVERY turn that carries it, not just the last
//  one. The KV cache only keeps the prefix a new prompt shares with the
//  previous one (see `LlamaVisionContext.generate`), so an image that is
//  not in the prompt is an image the model can no longer see - "what
//  colour was that car?" two messages later has to re-send the picture. A
//  re-sent picture that is still in the cache at the same place costs
//  nothing: it is recognised by content and not encoded again.
//
//  Pure Swift on purpose: no dependency on the `llama` module, so it is
//  unit-testable on any target.
//

import Foundation

/// The literal token libmtmd looks for in the prompt to splice in an
/// image/audio chunk. Hard-coded on the C side (see `mtmd_default_marker`
/// in `mtmd.cpp`); mirrored here to avoid a round-trip just to read it.
let kMediaMarker = "<__media__>"

enum VisionPromptBuilder {

    /// What one image costs in the text context. Gemma 3 maps an image to
    /// 256 tokens, Qwen2-VL and LLaVA land in the same neighbourhood at the
    /// resolutions `ImagePreprocessor` produces; the rest is slack.
    static let imageTokenEstimate = 320

    /// Template tags around a turn ("<start_of_turn>user\n" and friends).
    static let perTurnOverhead = 8

    /// How many images a prompt may carry, newest first.
    ///
    /// Images in the cached prefix are not encoded again, but every image
    /// still occupies the window, and whenever the prefix changes (the
    /// oldest turns are trimmed, or the oldest image falls out of this cap)
    /// everything after the change is decoded again, images included. An
    /// unbounded replay crowds the window with pictures the current
    /// question is not about and makes those rebuilds ever more expensive.
    /// Four keeps "compare these two" and "what was in the first photo"
    /// working while putting a ceiling on the cost; a document parser, where
    /// each page is a separate job, is better served by Clear chat between
    /// pages.
    static let maxReplayedImages = 4

    struct Prepared {
        let prompt: String
        /// The conversation with the media markers spliced into the turns
        /// they belong to, for a caller that renders with the model's own
        /// chat template instead of `prompt`.
        let turns: [ChatTurn]
        let images: [Data]
        /// How many older turns were dropped to make the prompt fit.
        let droppedTurns: Int
        /// Whether the first user turn was among them.
        var firstUserTurnDropped = false
    }

    static func build(
        turns: [ChatTurn],
        family: PromptFamily,
        budgetTokens: Int
    ) -> Prepared {
        // Cap the media BEFORE measuring: the estimate has to describe the
        // prompt that will actually be built, or it trims history to make
        // room for images this prompt is not going to send.
        let capped = capMedia(turns: turns, limit: maxReplayedImages)
        let fitted = fitDetailed(turns: capped, budgetTokens: budgetTokens)
        let kept = fitted.kept

        var withMarkers: [ChatTurn] = []
        var images: [Data] = []
        withMarkers.reserveCapacity(kept.count)

        for turn in kept {
            var markerCount = 0
            for attachment in turn.attachments {
                switch attachment {
                case .image(let data):
                    images.append(data)
                    markerCount += 1
                case .videoFrames(let frames, _):
                    // libmtmd has no video chunk: every sampled frame is an
                    // image, and the models read a run of them as a
                    // sequence in time.
                    images.append(contentsOf: frames)
                    markerCount += frames.count
                case .audio:
                    // Needs mtmd_bitmap_init_from_audio + PCM decoding.
                    continue
                }
            }
            guard markerCount > 0 else {
                withMarkers.append(turn)
                continue
            }
            // Markers go after the text so the model reads the question
            // before the picture, which is what mtmd-cli does and what most
            // architectures expect. PaddleOCR-VL is the exception: its own
            // chat template emits the image block first and the instruction
            // after it, and an OCR model reading its instruction before the
            // page it applies to is the wrong way round.
            let content: String
            if imagesLeadTheTurn(in: family) {
                content = String(repeating: kMediaMarker + "\n", count: markerCount) + turn.content
            } else {
                content = turn.content + String(repeating: "\n" + kMediaMarker, count: markerCount)
            }
            withMarkers.append(ChatTurn(role: turn.role, content: content))
        }

        return Prepared(
            prompt: PromptFormatter.format(turns: withMarkers, family: family),
            turns: withMarkers,
            images: images,
            droppedTurns: fitted.droppedTurns,
            firstUserTurnDropped: !fitted.firstUserTurnIntact
        )
    }

    /// Keeps the newest `limit` images and strips the attachments of every
    /// older turn, leaving that turn's text in place: the assistant's reply
    /// about the picture is usually what the thread needs later, not the
    /// picture itself.
    static func capMedia(turns: [ChatTurn], limit: Int) -> [ChatTurn] {
        guard limit >= 0 else { return turns }
        var budget = limit
        var out: [ChatTurn] = []
        out.reserveCapacity(turns.count)

        for turn in turns.reversed() {
            let count = imageCount(in: turn)
            if count == 0 {
                out.append(turn)
                continue
            }
            if count <= budget {
                budget -= count
                out.append(turn)
            } else {
                // Partial turns are not worth the confusion of a message
                // whose pictures are half there: drop this turn's media
                // whole, and every older turn's with it.
                budget = 0
                out.append(ChatTurn(role: turn.role, content: turn.content))
            }
        }
        return out.reversed()
    }

    static func imageCount(in turn: ChatTurn) -> Int {
        turn.attachments.reduce(0) { total, attachment in
            switch attachment {
            case .image: return total + 1
            case .videoFrames(let frames, _): return total + frames.count
            case .audio: return total
            }
        }
    }

    /// True for families whose chat template puts the image before the text
    /// of the same turn.
    static func imagesLeadTheTurn(in family: PromptFamily) -> Bool {
        family == .ernie
    }

    /// Drops older exchanges until the estimate fits `budgetTokens`, by the
    /// rules both llama.cpp engines share (`HistoryTrim`): whole exchanges,
    /// oldest first, in batches of about a quarter of the budget so the KV
    /// prefix cache keeps hitting for the next turns, and the first
    /// exchange kept while it is small.
    ///
    /// The system turn (the model's instructions) and the last turn (the
    /// question being asked) are never dropped - if those alone do not fit,
    /// the prompt is handed over as-is and llama.cpp reports a window it
    /// cannot serve, which is a truthful error rather than a silently
    /// mangled conversation.
    static func fit(
        turns: [ChatTurn],
        budgetTokens: Int
    ) -> (kept: [ChatTurn], dropped: Int) {
        let fitted = fitDetailed(turns: turns, budgetTokens: budgetTokens)
        return (fitted.kept, fitted.droppedTurns)
    }

    static func fitDetailed(turns: [ChatTurn], budgetTokens: Int) -> HistoryTrim.Fit {
        HistoryTrim.fitEstimated(turns: turns, room: budgetTokens, cost: estimatedTokens(of:))
    }

    /// Rough token cost of one turn. Deliberately pessimistic: over-counting
    /// drops one turn too many, under-counting hands llama.cpp a prompt it
    /// cannot decode. 2.5 characters per token is about right for Cyrillic
    /// on a Gemma/Qwen vocabulary and generous for English.
    static func estimatedTokens(of turn: ChatTurn) -> Int {
        var media = 0
        for attachment in turn.attachments {
            switch attachment {
            case .image:
                media += imageTokenEstimate
            case .videoFrames(let frames, _):
                media += frames.count * imageTokenEstimate
            case .audio:
                continue
            }
        }
        return (turn.content.count * 2) / 5 + media + perTurnOverhead
    }
}
