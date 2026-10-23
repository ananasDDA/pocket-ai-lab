//
//  LlamaChatTemplate.swift
//  local_ai_test
//
//  Renders a conversation with llama.cpp's own template engine.
//
//  `llama_chat_apply_template` is not a Jinja interpreter: it sniffs the
//  template string against the formats llama.cpp implements natively and
//  writes the transcript for the one it recognises. That is exactly what we
//  want here - it covers every mainstream family, needs no dependency, and
//  it reports failure rather than guessing, which is what makes the
//  fallback chain honest.
//

import Foundation

#if canImport(llama)

import llama

enum LlamaChatTemplate {

    /// The template stored in the model file, if it has one.
    static func embedded(in model: OpaquePointer) -> String? {
        guard let raw = llama_model_chat_template(model, nil) else { return nil }
        let template = String(cString: raw)
        return template.isEmpty ? nil : template
    }

    /// Renders `turns` with `template`, ending on an open assistant turn.
    /// Returns nil when llama.cpp does not recognise the template - the
    /// caller then tries the next source rather than sending the model a
    /// transcript in a format nobody verified.
    static func render(turns: [ChatTurn], template: String) -> String? {
        guard !turns.isEmpty else { return nil }

        // `llama_chat_message` borrows both strings for the duration of the
        // call, so they are heap copies freed on the way out - taking the
        // address of a Swift Array's storage would hand C a pointer that is
        // valid only inside the closure that produced it.
        var allocated: [UnsafeMutablePointer<CChar>] = []
        defer { allocated.forEach { free($0) } }

        var messages: [llama_chat_message] = []
        messages.reserveCapacity(turns.count)
        for turn in turns {
            guard let role = strdup(turn.role.rawValue),
                  let content = strdup(turn.content) else { return nil }
            allocated.append(role)
            allocated.append(content)
            messages.append(llama_chat_message(
                role: UnsafePointer(role),
                content: UnsafePointer(content)
            ))
        }

        // A negative result means llama.cpp does not implement this
        // template; a result larger than the buffer is the size to retry
        // with.
        var buffer = [CChar](repeating: 0, count: 8192)
        var written = template.withCString { tmpl in
            llama_chat_apply_template(
                tmpl, messages, messages.count, true, &buffer, Int32(buffer.count)
            )
        }
        guard written >= 0 else { return nil }
        if Int(written) > buffer.count {
            buffer = [CChar](repeating: 0, count: Int(written) + 1)
            written = template.withCString { tmpl in
                llama_chat_apply_template(
                    tmpl, messages, messages.count, true, &buffer, Int32(buffer.count)
                )
            }
            guard written >= 0 else { return nil }
        }

        let bytes = buffer.prefix(Int(written)).map { UInt8(bitPattern: $0) }
        let text = String(decoding: bytes, as: UTF8.self)
        return text.isEmpty ? nil : text
    }

    /// Whether llama.cpp can render with this template at all. Used by the
    /// UI to reject a pasted template while the user can still fix it,
    /// rather than at the next message.
    static func isSupported(_ template: String) -> Bool {
        render(turns: [.user("probe")], template: template) != nil
    }
}

#else

enum LlamaChatTemplate {
    static func render(turns: [ChatTurn], template: String) -> String? { nil }
    static func isSupported(_ template: String) -> Bool { false }
}

#endif
