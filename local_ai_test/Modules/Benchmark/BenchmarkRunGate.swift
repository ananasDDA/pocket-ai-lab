//
//  BenchmarkRunGate.swift
//  local_ai_test
//
//  One flag: "a benchmark is being measured right now".
//
//  A benchmark measures tokens per second, time to first token and peak
//  memory. A background refresh that wakes the radio, decodes a few hundred
//  kilobytes of JSON and allocates while the decode probe is running does not
//  break the run — it quietly shifts the numbers, which is worse, because
//  the report still looks publishable. The app's best-effort refreshes
//  (catalog, task suite, release feed) therefore check this gate and skip a
//  cycle instead; they are best-effort by design and losing one costs
//  nothing.
//
//  It also means an imported suite cannot be fetched from a URL mid-run.
//
//  Deliberately NOT wired into `ModelDownloader`: a download is something the
//  user or the release harness started on purpose, and silently pausing it
//  because a benchmark happens to be running would look like a stuck
//  download. The build-test sweep downloads BETWEEN runs, never during one.
//

import Foundation

nonisolated final class BenchmarkRunGate: @unchecked Sendable {

    static let shared = BenchmarkRunGate()

    private let lock = NSLock()
    /// A count rather than a bool: the release harness runs benchmarks
    /// back to back, and a nested or overlapping run must not clear the gate
    /// for the one still going.
    private var activeRuns = 0

    private init() {}

    var isRunning: Bool {
        lock.withLock { activeRuns > 0 }
    }

    func begin() {
        lock.withLock { activeRuns += 1 }
    }

    func end() {
        lock.withLock { activeRuns = max(0, activeRuns - 1) }
    }
}
