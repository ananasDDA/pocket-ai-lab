//
//  ContextBudget.swift
//  local_ai_test
//
//  How a fixed context window is split between the prompt (system turn plus
//  history) and the reply, and how the app tells that a reply was cut off
//  by its token cap rather than finished.
//

import Foundation

enum ContextBudget {

    /// Tokens kept free for the reply: the caller's cap, but never more than
    /// half the window, so a large cap cannot starve a small window of
    /// history. At least one token. This is the arithmetic the llama.cpp
    /// text engine and `CoreMLTextEngine.trimToContext` apply.
    nonisolated static func reservedForCompletion(maxTokens: Int, contextSize: Int) -> Int {
        max(1, min(maxTokens, max(contextSize, 1) / 2))
    }

    /// Tokens left for the prompt once the reply's share is reserved.
    nonisolated static func promptRoom(maxTokens: Int, contextSize: Int) -> Int {
        max(1, max(contextSize, 1) - reservedForCompletion(maxTokens: maxTokens, contextSize: contextSize))
    }
}

enum ReplyLimit {

    /// Whether a finished generation stopped because it reached its token
    /// cap. Engines stop at exactly `maxTokens` in that case; a model that
    /// ends its own turn stops earlier. Unknown counts are never a cut.
    nonisolated static func wasCut(tokensGenerated: Int?, maxTokens: Int) -> Bool {
        guard let tokensGenerated, maxTokens > 0 else { return false }
        return tokensGenerated >= maxTokens
    }

    /// What the "Continue" button sends. A plain user turn: every engine
    /// re-sends the whole conversation, so the model sees its own cut reply
    /// right above it.
    nonisolated static let continuePrompt = "Continue."
}
