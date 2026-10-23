//
//  PALAnalytics.swift
//  local_ai_test
//
//  Measurement telemetry: how models and engines behave on real devices.
//  Records go into a small queue on disk and are sent in batches to our own
//  ingest (INGEST below), which forwards them to OpenPanel; the app never
//  talks to OpenPanel and holds no secret. The queue survives restarts and
//  offline periods; a failed send backs off and retries, and a batch is only
//  dropped when the server refuses it as malformed or it is older than seven
//  days. Same design as the P.R.O. app's PROAnalytics, without an account.
//
//  What goes out is fixed by PALAnalyticsSchema: download, load, chat
//  generation and benchmark measurements, plus one app_opened per launch.
//  Every record is filtered through that allowlist before it is queued.
//
//  The device (model identifier, chip, RAM, iOS and app version, locale,
//  time zone and a random installation id created once per install) rides
//  with every batch.
//

import Foundation
import UIKit

nonisolated struct PALAnalyticsEvent: Codable, Equatable, Sendable {
    let event_id: String
    let name: String
    let properties: [String: PALAnalyticsValue]
    let occurred_at_ms: Int64
    let sequence: Int64
}

nonisolated enum PALAnalyticsDevice {
    private static let installationKey = "pal.analytics.installationId"

    /// Created once per install. `static let` runs its initialiser exactly
    /// once even when two threads ask at the same moment on first launch.
    static let installationId: String = {
        if let id = UserDefaults.standard.string(forKey: installationKey) { return id }
        let id = UUID().uuidString.lowercased()
        UserDefaults.standard.set(id, forKey: installationKey)
        return id
    }()

    static var machineIdentifier: String {
        var info = utsname()
        uname(&info)
        return withUnsafeBytes(of: &info.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
    }

    /// The device block of every batch. Values that the server would refuse
    /// are left out rather than costing a batch.
    static func fields(chip: String, osVersion: String) -> [String: PALAnalyticsValue] {
        let bundle = Bundle.main.infoDictionary ?? [:]
        var out: [String: PALAnalyticsValue] = [
            "platform": .string("ios"),
            "installation_id": .string(installationId),
            "locale": .string(String(Locale.current.identifier.prefix(35))),
            "timezone": .string(String(TimeZone.current.identifier.prefix(64))),
            "ram_gb": .double((Double(ProcessInfo.processInfo.physicalMemory) / 1_073_741_824 * 10).rounded() / 10),
        ]
        let version = #"^[0-9A-Za-z.+-]{1,32}$"#
        if let v = bundle["CFBundleShortVersionString"] as? String, PALAnalyticsSchema.matches(v, version) {
            out["app_version"] = .string(v)
        }
        if let v = bundle["CFBundleVersion"] as? String, PALAnalyticsSchema.matches(v, version) {
            out["app_build"] = .string(v)
        }
        if PALAnalyticsSchema.matches(osVersion, #"^[0-9A-Za-z.+_-]{1,32}$"#) { out["os_version"] = .string(osVersion) }
        let model = machineIdentifier
        if PALAnalyticsSchema.matches(model, #"^[A-Za-z0-9][A-Za-z0-9 .,()_+/-]{0,79}$"#) { out["device_model"] = .string(model) }
        if PALAnalyticsSchema.matches(chip, #"^[A-Za-z0-9][A-Za-z0-9 .+-]{0,39}$"#) { out["chip"] = .string(chip) }
        return out
    }
}

/// Whether this process may send anything. Pure, so the rule is tested.
nonisolated enum PALAnalyticsGate {

    /// Set in the launch environment of an app process that must stay silent.
    /// The UI tests set it (XCUIApplication.launchEnvironment); XCTest's own
    /// variables do not reach the app a UI test launches.
    static let disableEnvironmentKey = "PAL_ANALYTICS_DISABLED"

    /// Debug builds send nothing unless this UserDefaults key is true, for an
    /// end-to-end check of a development build. Set it from Xcode with the
    /// launch argument `-PALAnalyticsSendFromDebug YES` (arguments land in the
    /// UserDefaults argument domain) or with `defaults write`. It has no
    /// effect in Release, on the simulator or under tests.
    static let debugOverrideKey = "PALAnalyticsSendFromDebug"

    static func shouldSend(environment: [String: String], isSimulator: Bool, isDebug: Bool,
                           debugOverride: Bool) -> Bool {
        let testMarkers = ["XCTestConfigurationFilePath", "XCTestBundlePath", "XCTestSessionIdentifier",
                           disableEnvironmentKey]
        if testMarkers.contains(where: { environment[$0] != nil }) { return false }
        if isSimulator { return false }
        if isDebug { return debugOverride }
        return true
    }

    static var current: Bool {
        #if targetEnvironment(simulator)
        let simulator = true
        #else
        let simulator = false
        #endif
        #if DEBUG
        let debug = true
        #else
        let debug = false
        #endif
        return shouldSend(environment: ProcessInfo.processInfo.environment, isSimulator: simulator,
                          isDebug: debug, debugOverride: UserDefaults.standard.bool(forKey: debugOverrideKey))
    }
}

nonisolated final class PALAnalytics: @unchecked Sendable {
    static let shared = PALAnalytics()

    /// The one place that names the ingest. Its own subdomain, proxied to a
    /// single route by our Caddy.
    static let ingest = URL(string: "https://ingest.proapp.uk/analytics/pal/app")!

    /// Tests, UI-test launches, the simulator and Debug builds (without the
    /// override) measure the developer, not the product: they queue nothing.
    static let isEnabled = PALAnalyticsGate.current

    // The queue and the file behind it. Everything here runs on `queue`,
    // which the main actor never waits for.
    private let queue = DispatchQueue(label: "pal.analytics")
    private var events: [PALAnalyticsEvent] = []
    private var device: [String: PALAnalyticsValue] = [:]
    private var sequence: Int64 = 0
    private var sending = false
    private var failures = 0
    private var nextAttempt = Date.distantPast
    private var timer: DispatchSourceTimer?
    private var saveScheduled = false
    private let maxQueue = 2000
    private let maxAge: TimeInterval = 7 * 86_400
    private let batchSize = 50

    // Small measurement state the hooks read synchronously, behind its own
    // lock so a hook never waits for a file write.
    private let stateLock = NSLock()
    private var started = false
    private var loadedModelIds: Set<String> = []
    private var lastMemoryPressureUnload: Date?
    private var downloadStarts: [String: Date] = [:]
    private let downloadStartsKey = "pal.analytics.downloadStarts"

    private var fileURL: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("pal-analytics-queue.json")
    }

    private init() {
        guard Self.isEnabled else { return }
        let cutoff = Date().addingTimeInterval(-maxAge)
        if let raw = UserDefaults.standard.dictionary(forKey: downloadStartsKey) as? [String: Double] {
            downloadStarts = raw.mapValues { Date(timeIntervalSince1970: $0) }.filter { $0.value > cutoff }
        }
        // First on the serial queue, so every later operation sees the
        // restored events; the main actor does not wait for the read.
        queue.async { [weak self] in self?.load() }
    }

    /// Call once at launch, on the main thread: reads the device block, counts
    /// this launch and starts the periodic flush.
    @MainActor
    func start() {
        guard Self.isEnabled else { return }
        let first: Bool = stateLock.withLock {
            defer { started = true }
            return !started
        }
        guard first else { return }
        let identifier = PALAnalyticsDevice.machineIdentifier
        let fields = PALAnalyticsDevice.fields(chip: DeviceNameMapper.chip(for: identifier),
                                               osVersion: UIDevice.current.systemVersion)
        queue.async { [weak self] in self?.device = fields }
        track(.appOpened, [:])

        NotificationCenter.default.addObserver(forName: UIApplication.didEnterBackgroundNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            MainActor.assumeIsolated { self?.enteredBackground() }
        }
        queue.async { [weak self] in
            guard let self else { return }
            let timer = DispatchSource.makeTimerSource(queue: self.queue)
            timer.schedule(deadline: .now() + 20, repeating: 20)
            timer.setEventHandler { [weak self] in self?.send(force: false) {} }
            timer.resume()
            self.timer = timer
        }
    }

    /// Queues one record. Keys and values outside the schema are dropped; an
    /// event that then lacks a key the server requires is dropped on its own,
    /// so it never costs the rest of a batch a 422.
    func track(_ event: PALEvent, _ properties: [String: Any]) {
        guard Self.isEnabled else { return }
        let props = PALAnalyticsSchema.sanitize(properties, for: event)
        let missing = PALAnalyticsSchema.missingRequired(props, for: event)
        guard missing.isEmpty else {
            print("[analytics] dropped \(event.rawValue): missing \(missing.sorted().joined(separator: ", "))")
            return
        }
        let now = Int64(Date().timeIntervalSince1970 * 1000)
        queue.async { [weak self] in
            guard let self else { return }
            self.sequence += 1
            self.events.append(PALAnalyticsEvent(event_id: UUID().uuidString.lowercased(), name: event.rawValue,
                                                 properties: props, occurred_at_ms: now, sequence: self.sequence))
            if self.events.count > self.maxQueue { self.events.removeFirst(self.events.count - self.maxQueue) }
            self.saveSoon()
            if self.events.count >= 10 { self.send(force: false) {} }
        }
    }

    // MARK: - Hook state (lock, never the queue)

    /// True the first time a model loads in this process.
    func markLoaded(_ modelId: String) -> Bool {
        stateLock.withLock { loadedModelIds.insert(modelId).inserted }
    }

    /// The memory-warning handler dropped the engine to avoid a jetsam kill.
    func noteMemoryPressureUnload() {
        let now = Date()
        stateLock.withLock { lastMemoryPressureUnload = now }
    }

    func memoryPressureUnload(since start: Date) -> Bool {
        stateLock.withLock { (lastMemoryPressureUnload ?? .distantPast) >= start }
    }

    func downloadStarted(_ model: AIModel) {
        guard Self.isEnabled else { return }
        let now = Date()
        let snapshot: [String: Double] = stateLock.withLock {
            downloadStarts[model.id] = now
            return downloadStarts.mapValues { $0.timeIntervalSince1970 }
        }
        UserDefaults.standard.set(snapshot, forKey: downloadStartsKey)
    }

    /// The download reached a terminal state. Only a download this process
    /// (or an earlier launch) saw start is reported, and only once.
    func downloadEnded(_ model: AIModel, errorKind: String? = nil, httpStatus: Int? = nil) {
        guard Self.isEnabled else { return }
        let now = Date()
        let (started, snapshot): (Date?, [String: Double]) = stateLock.withLock {
            let value = downloadStarts.removeValue(forKey: model.id)
            return (value, downloadStarts.mapValues { $0.timeIntervalSince1970 })
        }
        guard let started else { return }
        UserDefaults.standard.set(snapshot, forKey: downloadStartsKey)
        let record = PALRecords.download(model: model, seconds: now.timeIntervalSince(started),
                                         errorKind: errorKind, httpStatus: httpStatus)
        track(errorKind == nil ? .modelDownloadFinished : .modelDownloadFailed, record)
    }

    // MARK: - Lifecycle

    @MainActor
    private func enteredBackground() {
        let app = UIApplication.shared
        var task = UIBackgroundTaskIdentifier.invalid
        task = app.beginBackgroundTask(withName: "pal.analytics.flush") {
            app.endBackgroundTask(task)
            task = .invalid
        }
        queue.async { [weak self] in
            self?.save()
            self?.nextAttempt = .distantPast
            self?.send(force: true) {
                DispatchQueue.main.async {
                    if task != .invalid { app.endBackgroundTask(task) }
                }
            }
        }
    }

    // MARK: - Delivery (on `queue`)

    private func send(force: Bool, completion: @escaping () -> Void) {
        let cutoff = Int64((Date().timeIntervalSince1970 - maxAge) * 1000)
        let before = events.count
        events.removeAll { $0.occurred_at_ms < cutoff }
        if events.count != before { saveSoon() }
        guard !sending, !events.isEmpty, !device.isEmpty, force || Date() >= nextAttempt else {
            return completion()
        }
        let batch = Array(events.prefix(batchSize))
        struct Body: Encodable {
            let device: [String: PALAnalyticsValue]
            let events: [PALAnalyticsEvent]
        }
        guard let body = try? JSONEncoder().encode(Body(device: device, events: batch)) else {
            return completion()
        }
        sending = true
        var request = URLRequest(url: Self.ingest, timeoutInterval: 15)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = body
        URLSession.shared.dataTask(with: request) { [weak self] _, response, error in
            let status = (response as? HTTPURLResponse)?.statusCode ?? 0
            guard let self else { return completion() }
            self.queue.async {
                self.sending = false
                let ids = Set(batch.map(\.event_id))
                if error == nil, (200..<300).contains(status) {
                    self.events.removeAll { ids.contains($0.event_id) }
                    self.failures = 0
                    self.nextAttempt = .distantPast
                    self.saveSoon()
                    if !self.events.isEmpty { return self.send(force: force, completion: completion) }
                } else if status == 400 || status == 413 || status == 422 {
                    self.events.removeAll { ids.contains($0.event_id) }  // malformed for good
                    self.saveSoon()
                    print("[analytics] server refused a batch (\(status)), dropped \(ids.count)")
                } else {
                    self.backoff()
                }
                completion()
            }
        }.resume()
    }

    private func backoff() {
        failures += 1
        nextAttempt = Date().addingTimeInterval(min(300, pow(2, Double(min(failures, 8)))))
    }

    // MARK: - Persistence (on `queue`)

    private func load() {
        guard let data = try? Data(contentsOf: fileURL),
              let saved = try? JSONDecoder().decode([PALAnalyticsEvent].self, from: data) else { return }
        events = saved + events
        sequence = max(sequence, saved.map(\.sequence).max() ?? 0)
    }

    /// At most one write a second: a burst of records costs one file write,
    /// and going to the background writes at once.
    private func saveSoon() {
        guard !saveScheduled else { return }
        saveScheduled = true
        queue.asyncAfter(deadline: .now() + 1) { [weak self] in
            guard let self else { return }
            self.saveScheduled = false
            self.save()
        }
    }

    private func save() {
        guard let data = try? JSONEncoder().encode(events) else { return }
        try? data.write(to: fileURL, options: [.atomic, .completeFileProtectionUntilFirstUserAuthentication])
    }
}
