//
//  EmptyReply.swift
//  local_ai_test
//
//  A reply with nothing in it. In the first field data from 1.1 an imported
//  F16 GGUF answered seven messages in a row with zero tokens, and the chat
//  showed "(Empty response)" each time - true, and no help. Zero tokens means
//  the model emitted its end-of-turn token first, which almost always comes
//  down to the prompt format: no chat template in the file, a template that
//  does not match the model, or a base model that was never tuned to chat.
//

import Foundation

enum EmptyReply {

    enum Kind: Sendable, Equatable {
        /// The model ended its turn before writing a single token.
        case endedImmediately
        /// Tokens were generated, but all of them were blank or stripped.
        case blank
    }

    /// nil when the reply has visible text. `tokensGenerated` is the engine's
    /// own count when it reports one; without it an empty reply counts as
    /// having ended immediately.
    nonisolated static func detect(text: String, tokensGenerated: Int?) -> Kind? {
        guard text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        if let tokensGenerated, tokensGenerated > 0 { return .blank }
        return .endedImmediately
    }

    /// What the chat shows in place of the reply.
    nonisolated static func message(for kind: Kind) -> String {
        switch kind {
        case .endedImmediately:
            return "The model ended its reply immediately, without writing anything. This "
                + "often means it has no chat template or is a base model rather than a chat "
                + "model. Open the template menu (… > Chat template) to check or set one."
        case .blank:
            return "The model replied with only blank space. This often means its chat "
                + "template does not match the model. Open the template menu (… > Chat "
                + "template) to check or set one, then try again."
        }
    }
}
