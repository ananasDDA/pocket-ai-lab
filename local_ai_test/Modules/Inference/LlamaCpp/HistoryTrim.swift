//
//  HistoryTrim.swift
//  local_ai_test
//
//  How a conversation that outgrows the context window is cut down, shared
//  by both llama.cpp engines (the text engine measures with the real
//  tokenizer, the vision builder with its estimates).
//
//  The rules, in order of priority:
//
//  1. Whole turns go, never parts of them, so every message the model sees
//     still starts with its role header. The system turn (the model's
//     instructions; Gemma-family templates fold it into the first user turn
//     they render) and the turn being answered - the last user turn and
//     anything after it - always stay.
//
//  2. Older exchanges (a user turn and the replies that follow it) go oldest
//     first, in batches of about a quarter of the room: the overflow is
//     rounded up to whole batches. Dropping exactly as much as needed would
//     move the start of the history on every message, and the KV prefix
//     cache misses whenever its start moves: a long chat then re-reads the
//     whole window each turn (23 s to the first token on Phi-3.5 by turn
//     eight). Rounded up, the same exchanges stay dropped until the
//     conversation has grown by another batch, and in between the prompt
//     only grows at its end.
//
//  3. The first exchange is kept while it costs at most a quarter of the
//     room. That is where users state the facts they come back to ("my name
//     is...", "I'm planning a trip to..."), a short one is cheap, and keeping
//     it also keeps the part of the prompt a Gemma template folds the system
//     text into unchanged. A long one - a pasted article - is not worth a
//     permanent quarter of the window and goes first like any other.
//     If the prompt cannot fit even with every other exchange gone, the
//     first exchange goes too.
//
//  4. Only when what remains is still too long is a single turn cut inside:
//     the longest one loses its middle (its start and its end, where the
//     instruction of a pasted document usually sits, survive), so its role
//     header stays.
//
//  Pure Swift: no dependency on the `llama` module, unit-testable anywhere.
//

import Foundation

enum HistoryTrim {

    /// Batch size and pinning limit, as a fraction of the prompt room.
    static let batchDivisor = 4
    static let pinDivisor = 4
    /// Marker left where a turn lost its middle.
    static let truncationMarker = "\n[...]\n"

    // MARK: - Layout

    /// A conversation split into what always stays and what may go.
    struct Layout: Equatable {
        let turns: [ChatTurn]
        /// Droppable exchanges, oldest first, as index ranges into `turns`.
        let units: [Range<Int>]
        /// Index of the first user turn, if any.
        let firstUserIndex: Int?

        /// Whether the oldest unit is a real exchange (starts with a user
        /// turn), which is what may be pinned.
        var firstUnitIsExchange: Bool {
            guard let first = units.first else { return false }
            return turns[first.lowerBound].role == .user
        }

        /// The turns left after dropping the given units, in order.
        func kept(dropping dropped: [Int]) -> [ChatTurn] {
            guard !dropped.isEmpty else { return turns }
            var gone = Set<Int>()
            for unit in dropped { gone.formUnion(units[unit]) }
            return turns.indices.filter { !gone.contains($0) }.map { turns[$0] }
        }

        func droppedTurnCount(_ dropped: [Int]) -> Int {
            dropped.reduce(0) { $0 + units[$1].count }
        }

        func dropsFirstUserTurn(_ dropped: [Int]) -> Bool {
            guard let firstUserIndex else { return false }
            return dropped.contains { units[$0].contains(firstUserIndex) }
        }
    }

    /// Splits `turns` into protected turns and droppable exchanges. System
    /// turns anywhere are protected; so is the tail from the last user turn
    /// on (or the last turn alone when there is no user turn). Everything in
    /// between is grouped into units that start at a user turn and run to
    /// the next one; non-user turns before the first user turn form units of
    /// their own.
    static func layout(_ turns: [ChatTurn]) -> Layout {
        let firstUser = turns.firstIndex { $0.role == .user }
        guard !turns.isEmpty else { return Layout(turns: turns, units: [], firstUserIndex: nil) }
        let tailStart = turns.lastIndex { $0.role == .user } ?? (turns.count - 1)

        var units: [Range<Int>] = []
        var current: [Int] = []
        func close() {
            if let first = current.first, let last = current.last { units.append(first..<(last + 1)) }
            current = []
        }
        for index in 0..<tailStart {
            let turn = turns[index]
            if turn.role == .system {
                // A system turn splits units but never joins one.
                close()
                continue
            }
            if turn.role == .user { close() }
            current.append(index)
        }
        close()
        return Layout(turns: turns, units: units, firstUserIndex: firstUser)
    }

    // MARK: - Schedule

    /// The drop lists to try, in order of preference. A caller takes the
    /// first one whose prompt fits.
    ///
    /// The amount to drop is the overflow rounded UP to whole batches:
    /// `ceil((total - room) / batch) * batch`. It stays the same while the
    /// conversation grows by up to a batch, and the exchanges that cover it
    /// (oldest first) depend only on the costs of the oldest exchanges,
    /// which a growing conversation never changes - so the same exchanges
    /// stay dropped, and the prompt keeps its start, for about a batch's
    /// worth of new turns. Dropping just the overflow instead would move
    /// the start on every message. Each list after the first drops one
    /// more exchange (for when the costs underestimated the prompt); when
    /// the first exchange was pinned, the last list drops it as well.
    static func schedule(
        unitCosts: [Int],
        firstUnitIsExchange: Bool,
        room: Int,
        total: Int
    ) -> [[Int]] {
        let need = total - room
        guard need > 0, !unitCosts.isEmpty else { return [[]] }
        let batch = max(1, room / batchDivisor)
        // A sixteenth of a batch on top of the overflow absorbs the error in
        // the exchange costs (the tokenizer does not split a transcript
        // exactly at turn boundaries); without it a prompt that overflows
        // by just under a batch point would fall short, need one more
        // exchange, and move its start a turn early.
        let target = ((need + batch / 16 + batch - 1) / batch) * batch
        let pinned = firstUnitIsExchange && unitCosts[0] <= max(0, room / pinDivisor)
        let order = Array((pinned ? 1 : 0)..<unitCosts.count)

        var first = order.count
        var cumulative = 0
        for (k, unit) in order.enumerated() {
            cumulative += unitCosts[unit]
            if cumulative >= target { first = k + 1; break }
        }
        var lists = (max(first, order.isEmpty ? 0 : 1)...max(order.count, 1))
            .filter { $0 <= order.count }
            .map { Array(order.prefix($0)) }
        if lists.isEmpty { lists = [[]] }
        if pinned { lists.append(order + [0]) }
        return lists
    }

    // MARK: - Fitting

    /// What fitting did to a conversation.
    struct Fit: Equatable {
        var kept: [ChatTurn]
        /// Whole turns left out.
        var droppedTurns: Int
        /// Tokens removed from inside a turn that was cut (0 when none was).
        var truncatedTokens: Int
        /// The first user turn reached the model whole.
        var firstUserTurnIntact: Bool
        /// Measured size of the prompt that was kept.
        var promptTokens: Int
    }

    /// Fits `turns` into `room` tokens.
    ///
    /// - Parameters:
    ///   - count: tokens of one turn's content (used to cost exchanges and
    ///     to size a cut; it does not have to include template glue).
    ///   - measure: tokens of the whole prompt rendered from these turns,
    ///     exactly as it will be decoded.
    ///   - truncate: whether a turn may be cut inside when dropping whole
    ///     exchanges is not enough. Without it the shortest prompt the
    ///     schedule allows is returned as is.
    static func fit(
        turns: [ChatTurn],
        room: Int,
        truncate: Bool = true,
        count: (String) async -> Int,
        measure: ([ChatTurn]) async -> Int
    ) async -> Fit {
        let firstUser = turns.firstIndex { $0.role == .user }
        let full = await measure(turns)
        guard full > room else {
            return Fit(kept: turns, droppedTurns: 0, truncatedTokens: 0,
                       firstUserTurnIntact: true, promptTokens: full)
        }

        let layout = layout(turns)
        // Cost an exchange as its contents plus the template's share per
        // turn, calibrated on this very prompt: what the measured whole
        // costs beyond its contents, spread over its turns. A guessed
        // overhead over- or under-states what dropping saves, and then the
        // batch boundary lands one exchange off from turn to turn.
        var contentTokens: [Int] = []
        for turn in turns { contentTokens.append(await count(turn.content)) }
        let overhead = max(0, Double(full - contentTokens.reduce(0, +)) / Double(max(turns.count, 1)))
        let unitCosts = layout.units.map { unit in
            Int((unit.reduce(0.0) { $0 + Double(contentTokens[$1]) + overhead }).rounded())
        }

        var chosen: [Int] = []
        var kept = turns
        var size = full
        let lists = schedule(
            unitCosts: unitCosts,
            firstUnitIsExchange: layout.firstUnitIsExchange,
            room: room,
            total: full
        )
        for drops in lists where !drops.isEmpty {
            chosen = drops
            kept = layout.kept(dropping: drops)
            size = await measure(kept)
            if size <= room { break }
        }
        var intact = !layout.dropsFirstUserTurn(chosen)
        var fit = Fit(kept: kept, droppedTurns: layout.droppedTurnCount(chosen), truncatedTokens: 0,
                      firstUserTurnIntact: intact, promptTokens: size)
        guard size > room, truncate else { return fit }

        // Every droppable exchange is gone and the rest is still too long:
        // cut inside the longest turn left, a few times at most (the
        // character-per-token ratio is only an estimate).
        let firstUserTurn = firstUser.map { turns[$0] }
        var removed = 0
        for _ in 0..<6 where size > room {
            var longest = -1
            var longestTokens = 0
            for (index, turn) in kept.enumerated() {
                let tokens = await count(turn.content)
                if tokens > longestTokens { longest = index; longestTokens = tokens }
            }
            guard longest >= 0, longestTokens > 0 else { break }
            let turn = kept[longest]
            let excess = size - room
            let keepTokens = max(0, longestTokens - excess - truncationSlack)
            let charsPerToken = Double(turn.content.count) / Double(longestTokens)
            let keepChars = Int(Double(keepTokens) * charsPerToken * 0.95)
            guard keepChars < turn.content.count else { break }
            if turn == firstUserTurn, intact { intact = false }
            let cut = ChatTurn(
                role: turn.role,
                content: truncatingMiddle(turn.content, keepingCharacters: keepChars),
                attachments: turn.attachments
            )
            removed += longestTokens - (await count(cut.content))
            kept[longest] = cut
            size = await measure(kept)
        }
        fit.kept = kept
        fit.truncatedTokens = max(0, removed)
        fit.firstUserTurnIntact = intact
        fit.promptTokens = size
        return fit
    }

    /// Fits `turns` into `room` by dropping whole exchanges only, costing
    /// every turn with `cost` (the vision builder's estimate). The prompt's
    /// size is the sum of its turns' costs, so no rendering is needed.
    /// Never cuts inside a turn: when nothing more can go, the shortest
    /// prompt the schedule allows is returned and llama.cpp reports the
    /// window it cannot serve.
    static func fitEstimated(turns: [ChatTurn], room: Int, cost: (ChatTurn) -> Int) -> Fit {
        let costs = turns.map(cost)
        let total = costs.reduce(0, +)
        guard room > 0, total > room else {
            return Fit(kept: turns, droppedTurns: 0, truncatedTokens: 0,
                       firstUserTurnIntact: true, promptTokens: total)
        }
        let layout = layout(turns)
        let unitCosts = layout.units.map { unit in unit.reduce(0) { $0 + costs[$1] } }
        var chosen: [Int] = []
        var size = total
        let lists = schedule(
            unitCosts: unitCosts,
            firstUnitIsExchange: layout.firstUnitIsExchange,
            room: room,
            total: total
        )
        for drops in lists where !drops.isEmpty {
            chosen = drops
            size = total - drops.reduce(0) { $0 + unitCosts[$1] }
            if size <= room { break }
        }
        return Fit(
            kept: layout.kept(dropping: chosen),
            droppedTurns: layout.droppedTurnCount(chosen),
            truncatedTokens: 0,
            firstUserTurnIntact: !layout.dropsFirstUserTurn(chosen),
            promptTokens: size
        )
    }

    /// Extra tokens taken off a cut turn so one pass usually suffices.
    static let truncationSlack = 16

    /// `text` with its middle replaced by the truncation marker, keeping
    /// about `keepingCharacters` characters: three fifths from the start,
    /// two fifths from the end.
    static func truncatingMiddle(_ text: String, keepingCharacters keep: Int) -> String {
        guard keep < text.count else { return text }
        guard keep > 0 else { return truncationMarker.trimmingCharacters(in: .whitespacesAndNewlines) }
        let head = keep * 3 / 5
        let tail = keep - head
        return String(text.prefix(head)) + truncationMarker + String(text.suffix(tail))
    }
}
