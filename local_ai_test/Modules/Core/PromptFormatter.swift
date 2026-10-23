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
    /// Gemma 4 dropped Gemma 3's `<start_of_turn>` markers for `<|turn>` /
    /// `<turn|>` and gained a real system role. llama.cpp b8882 cannot render
    /// its embedded Jinja template, so for Gemma 4 this fallback is the
    /// format the model actually sees.
    case gemma4
    case qwen
    case mistral
    case phi
    case chatml
    /// ERNIE-4.5 and the PaddleOCR-VL document models built on it:
    /// `<|begin_of_sentence|>` then plain "User:" / "Assistant:" turns.
    case ernie
    case plain
}

enum PromptFormatter {

    static func detectFamily(modelId: String) -> PromptFamily {
        let lowered = modelId.lowercased()
        // Before the rest: PaddleOCR-VL is an ERNIE-4.5 model whose id says
        // neither, and it would otherwise fall through to `.plain`, which
        // writes a transcript the model was never trained on.
        if lowered.contains("paddleocr") || lowered.contains("ernie") { return .ernie }
        if lowered.contains("llama-3") || lowered.contains("llama3") { return .llama3 }
        if lowered.contains("llama-2") || lowered.contains("llama2") { return .llama2 }
        // Before plain "gemma": a Gemma 4 id also contains "gemma", and
        // Gemma 3's markers mean nothing to it.
        if lowered.contains("gemma-4") || lowered.contains("gemma4") { return .gemma4 }
        if lowered.contains("gemma") { return .gemma }
        if lowered.contains("qwen") { return .qwen }
        if lowered.contains("mistral") { return .mistral }
        if lowered.contains("phi") { return .phi }
        if lowered.contains("chatml") { return .chatml }
        // Liquid AI's LFM2 is trained on plain ChatML.
        if lowered.contains("lfm2") || lowered.contains("lfm-2") { return .chatml }
        return .plain
    }

    static func format(turns: [ChatTurn], family: PromptFamily) -> String {
        switch family {
        case .llama3:  return formatLlama3(turns)
        case .llama2:  return formatLlama2(turns)
        case .gemma:   return formatGemma(turns)
        case .gemma4:  return formatGemma4(turns)
        case .qwen:    return formatChatML(turns)
        case .chatml:  return formatChatML(turns)
        case .ernie:   return formatErnie(turns)
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

    /// Mirrors the non-thinking, no-tools path of Gemma 4's
    /// `chat_template.jinja`: an optional system turn, then
    /// `<|turn>user` / `<|turn>model` turns closed by `<turn|>`, ending on
    /// an open model turn. BOS is not written here; the engine prepends it.
    private static func formatGemma4(_ turns: [ChatTurn]) -> String {
        let (system, rest) = extractSystem(turns)
        var out = ""
        if let system {
            out += "<|turn>system\n\(system.trimmingCharacters(in: .whitespacesAndNewlines))<turn|>\n"
        }
        for turn in rest {
            let role = turn.role == .assistant ? "model" : "user"
            let content = turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
            out += "<|turn>\(role)\n\(content)<turn|>\n"
        }
        out += "<|turn>model\n"
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

    /// Mirrors `chat_template.jinja` in PaddlePaddle's own GGUF release:
    /// a `<|begin_of_sentence|>` prefix, the system turn as bare text, then
    /// "User: ...\n" and "Assistant:\n...</s>" pairs, ending on an open
    /// "Assistant:\n". The image markers belong at the START of the user
    /// content for this family - see `VisionPromptBuilder`.
    private static func formatErnie(_ turns: [ChatTurn]) -> String {
        var out = "<|begin_of_sentence|>"
        for turn in turns {
            switch turn.role {
            case .system:
                out += turn.content + "\n"
            case .user:
                out += "User: " + turn.content + "\n"
            case .assistant:
                out += "Assistant:\n" + turn.content + "</s>"
            }
        }
        out += "Assistant:\n"
        return out
    }

    private static func formatPlain(_ turns: [ChatTurn]) -> String {
        turns.map { "\($0.role.rawValue): \($0.content)" }.joined(separator: "\n\n") + "\n\nassistant:"
    }
}
