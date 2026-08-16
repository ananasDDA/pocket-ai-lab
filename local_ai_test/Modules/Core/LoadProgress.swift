//
//  LoadProgress.swift
//  local_ai_test
//
//  Emitted by engines during load() to drive UI progress indicators for
//  long, multi-stage operations (compile + weight load).
//

import Foundation

enum LoadProgress: Sendable, Equatable {
    case downloading(Double)
    case compiling(Double)
    case loadingWeights(Double)
    case ready

    var fraction: Double {
        switch self {
        case .downloading(let p), .compiling(let p), .loadingWeights(let p): return p
        case .ready: return 1.0
        }
    }
}
