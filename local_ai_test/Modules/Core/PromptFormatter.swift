//
//  PromptFormatter.swift
//  local_ai_test
//
//  Last-resort fallback chat templates, used when the underlying engine
//  cannot pull a `chat_template` from tokenizer_config.json (mostly relevant
//  for llama.cpp where the template is usually embedded in the GGUF, and for
//  Core ML when a repo omits tokenizer_config.json). Engines should prefer
//  their native templater and fall back here by family.
//

import Foundation

enum PromptFamily: String, Sendable {
    case llama3
    case llama2
    case gemma
    case qwen
    case mistral
    case phi
    case chatml
    case plain
}

enum PromptFormatter {

    static func detectFamily(modelId: String) -> PromptFamily {
        let lowered = modelId.lowercased()
        if lowered.contains("llama-3") || lowered.contains("llama3") { return .llama3 }
        if lowered.contains("llama-2") || lowered.contains("llama2") { return .llama2 }
        if lowered.contains("gemma") { return .gemma }
        if lowered.contains("qwen") { return .qwen }
        if lowered.contains("mistral") { return .mistral }
        if lowered.contains("phi") { return .phi }
        if lowered.contains("chatml") { return .chatml }
        return .plain
    }

    static func format(turns: [ChatTurn], family: PromptFamily) -> String {
        switch family {
        case .llama3:  return formatLlama3(turns)
        case .llama2:  return formatLlama2(turns)
        case .gemma:   return formatGemma(turns)
        case .qwen:    return formatChatML(turns)
        case .chatml:  return formatChatML(turns)
        case .mistral: return formatMistral(turns)
        case .phi:     return formatPhi(turns)
        case .plain:   return formatPlain(turns)
        }
    }

    // MARK: - Templates

    private static func formatLlama3(_ turns: [ChatTurn]) -> String {
        var out = "<|begin_of_text|>"
        for turn in turns {
            out += "<|start_header_id|>\(turn.role.rawValue)<|end_header_id|>\n\n"
            out += turn.content
            out += "<|eot_id|>"
        }
        out += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return out
    }

    private static func formatLlama2(_ turns: [ChatTurn]) -> String {
        var out = ""
        var system = ""
        for turn in turns where turn.role == .system { system = turn.content }
        var userTurns: [ChatTurn] = []
        for turn in turns where turn.role != .system { userTurns.append(turn) }

        for (idx, turn) in userTurns.enumerated() {
            if turn.role == .user {
                if idx == 0, !system.isEmpty {
                    out += "<s>[INST] <<SYS>>\n\(system)\n<</SYS>>\n\n\(turn.content) [/INST]"
                } else {
                    out += "<s>[INST] \(turn.content) [/INST]"
                }
            } else {
                out += " \(turn.content) </s>"
            }
        }
        return out
    }

    private static func formatGemma(_ turns: [ChatTurn]) -> String {
        // Gemma has no system role. Mapping a system turn onto "user" would
        // produce two consecutive user turns and break the strict
        // user/model alternation — fold it into the first user turn instead.
        let (system, rest) = extractSystem(turns)
        var out = ""
        var systemPending = system
        for turn in rest {
            let role = turn.role == .assistant ? "model" : "user"
            var content = turn.content
            if turn.role == .user, let sys = systemPending {
                content = sys + "\n\n" + content
                systemPending = nil
            }
            out += "<start_of_turn>\(role)\n\(content)<end_of_turn>\n"
        }
        out += "<start_of_turn>model\n"
        return out
    }

    private static func formatChatML(_ turns: [ChatTurn]) -> String {
        var out = ""
        for turn in turns {
            out += "<|im_start|>\(turn.role.rawValue)\n\(turn.content)<|im_end|>\n"
        }
        out += "<|im_start|>assistant\n"
        return out
    }

    private static func formatMistral(_ turns: [ChatTurn]) -> String {
        // Mistral v0.3 has no system role either; emitting the system turn as
        // its own [INST] block makes the model answer the instructions
        // instead of the first question. Fold it into the first user turn.
        let (system, rest) = extractSystem(turns)
        var out = "<s>"
        var systemPending = system
        for turn in rest {
            switch turn.role {
            case .user, .system:
                var content = turn.content
                if let sys = systemPending {
                    content = sys + "\n\n" + content
                    systemPending = nil
                }
                out += "[INST] \(content) [/INST]"
            case .assistant:
                out += "\(turn.content)</s>"
            }
        }
        return out
    }

    /// Pulls the first system turn out of the list (Gemma / Mistral have no
    /// native system role and need it merged into the first user message).
    private static func extractSystem(_ turns: [ChatTurn]) -> (String?, [ChatTurn]) {
        let system = turns.first(where: { $0.role == .system })?.content
        let rest = turns.filter { $0.role != .system }
        return (system?.isEmpty == false ? system : nil, rest)
    }

    private static func formatPhi(_ turns: [ChatTurn]) -> String {
        var out = ""
        for turn in turns {
            out += "<|\(turn.role.rawValue)|>\n\(turn.content)<|end|>\n"
        }
        out += "<|assistant|>\n"
        return out
    }

    private static func formatPlain(_ turns: [ChatTurn]) -> String {
        turns.map { "\($0.role.rawValue): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
    }
}
