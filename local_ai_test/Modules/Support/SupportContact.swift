//
//  SupportContact.swift
//  local_ai_test
//
//  Where "Report a problem" sends its mail, and the privacy page the report
//  screen links to. The only place to change either.
//

import Foundation

nonisolated enum SupportContact {
    static let email = "p.r.o.devel001@gmail.com"

    static let privacyURL = URL(string: "https://ananasdda.github.io/pocket-ai-lab/privacy/")!
}
