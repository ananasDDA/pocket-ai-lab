//
//  ModelDownloader.swift
//  local_ai_test
//
//  Downloads MLX models from Hugging Face using background URLSession.
//  Downloads continue even when the app is backgrounded or killed.
//

import Foundation
import Observation
import UIKit

// MARK: - Download state

enum ModelDownloadState: Equatable {
    case idle
    case fetchingManifest
    case downloading(progress: Double, downloaded: Int64, total: Int64)
    case compiling(progress: Double)
    case installed
    case error(String)

    var isActive: Bool {
        switch self {
        case .fetchingManifest, .downloading, .compiling: return true
        default: return false
        }
    }
}

// MARK: - HF API types

private struct HFModelInfo: Decodable {
    /// Commit the manifest describes. Downloads are pinned to it so a repo
    /// update mid-download cannot mix files from two revisions.
    let sha: String?
    let siblings: [HFFile]
}

private struct HFFile: Decodable {
    let rfilename: String
    let size: Int64?

    enum CodingKeys: String, CodingKey {
        case rfilename
        case size
    }
}

// MARK: - Persistent metadata

/// Encoded into URLSessionDownloadTask.taskDescription
private struct TaskMeta: Codable {
    let modelId: String
    let filename: String
    let fileSize: Int64
}

/// Persistent plan for a multi-file model download.
private struct DownloadPlan: Codable {
    let modelId: String
    let repo: String
    /// Git revision the file URLs were built from. Nil in plans written by
    /// builds that predated revision pinning.
    var revision: String?
    var files: [FilePlan]
    var completedFilenames: Set<String>
    var totalBytes: Int64

    var completedBytes: Int64 {
        files.filter { completedFilenames.contains($0.filename) }
            .reduce(0) { $0 + $1.size }
    }

    var isComplete: Bool {
        completedFilenames.count == files.count
    }
}

private struct FilePlan: Codable {
    let filename: String
    let size: Int64
    let downloadURL: String
    let destinationRelativePath: String
}

// MARK: - Downloader

@Observable
final class ModelDownloader: NSObject {

    static let shared = ModelDownloader()

    // Only `states` needs to be observed by SwiftUI.
    private(set) var states: [String: ModelDownloadState] = [:]

    /// Called by AppDelegate when background session events finish.
    var backgroundCompletionHandler: (() -> Void)?

    // Everything below is excluded from observation tracking.
    @ObservationIgnored
    private var backgroundSession: URLSession!

    @ObservationIgnored
    private var plans: [String: DownloadPlan] = [:]

    @ObservationIgnored
    private let lock = NSLock()

    /// Tracks bytes written per active download task (keyed by taskIdentifier).
    @ObservationIgnored
    private var taskBytesWritten: [Int: Int64] = [:]

    /// Maps taskIdentifier → modelId for fast lookup without calling getAllTasks.
    @ObservationIgnored
    private var taskModelMap: [Int: String] = [:]

    @ObservationIgnored
    private let modelsRoot: URL = {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("LocalAIModels", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }()

    @ObservationIgnored
    private var plansDirectory: URL {
        let dir = modelsRoot.appendingPathComponent(".plans", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private override init() {
        super.init()

        let config = URLSessionConfiguration.background(
            withIdentifier: "com.localaitest.modeldownloader.bg"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        backgroundSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        restoreInstalledStates()
        Task { await restoreActiveDownloads() }
    }

    // MARK: - Thread-safe plan access

    private func withPlan<T>(for modelId: String, _ body: (inout DownloadPlan?) -> T) -> T {
        lock.withLock {
            var plan = plans[modelId]
            let result = body(&plan)
            plans[modelId] = plan
            return result
        }
    }

    private func readPlan(for modelId: String) -> DownloadPlan? {
        lock.withLock { plans[modelId] }
    }

    private func setPlan(_ plan: DownloadPlan?, for modelId: String) {
        lock.withLock { plans[modelId] = plan }
    }

    // MARK: - Public API

    func state(for model: AIModel) -> ModelDownloadState {
        states[model.id] ?? .idle
    }

    func modelDirectory(for model: AIModel) -> URL {
        modelsRoot.appendingPathComponent(
            model.id.replacingOccurrences(of: "/", with: "_"),
            isDirectory: true
        )
    }

    func download(_ model: AIModel) {
        guard !state(for: model).isActive else { return }
        Task { await startDownload(model) }
    }

    func cancel(_ model: AIModel) {
        let modelId = model.id
        removePlan(for: modelId)

        backgroundSession.getAllTasks { tasks in
            for task in tasks {
                if let meta = self.decodeMeta(from: task), meta.modelId == modelId {
                    task.cancel()
                }
            }
        }

        Task { @MainActor in
            self.states[modelId] = .idle
        }
    }

    func delete(_ model: AIModel) {
        let modelId = model.id
        removePlan(for: modelId)

        // Cancel all active tasks first
        backgroundSession.getAllTasks { [weak self] tasks in
            guard let self else { return }
            for task in tasks {
                if let meta = self.decodeMeta(from: task), meta.modelId == modelId {
                    task.cancel()
                }
            }

            // Delete files after tasks are cancelled
            let dir = self.modelDirectory(for: model)
            try? FileManager.default.removeItem(at: dir)

            Task { @MainActor in
                self.states[modelId] = .idle
                InstalledModelsStore.shared.markUninstalled(model)
            }
        }
    }

    // MARK: - Download pipeline

    private func startDownload(_ model: AIModel) async {
        let modelId = model.id

        await MainActor.run { states[modelId] = .fetchingManifest }

        if model.source == .imported {
            // Persist the descriptor before any network work: a crash between
            // here and completion must not lose the only copy of it.
            InstalledModelsStore.shared.register(model)
        }

        do {
            let manifest = try await fetchManifest(repo: model.huggingFaceRepo)
            // Prefer the revision the import flow pinned; otherwise pin to
            // whatever the manifest we just read describes.
            let revision = model.revision ?? manifest.sha ?? "main"
            let files = manifest.siblings
            let filtered = filterFiles(files, for: model)
            let totalBytes = filtered.compactMap(\.size).reduce(0, +)

            let dir = modelDirectory(for: model)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            var filePlans: [FilePlan] = []
            var alreadyCompleted: Set<String> = []

            for file in filtered {
                let fileSize = file.size ?? 0
                let dest = dir.appendingPathComponent(file.rfilename)
                let encodedName = file.rfilename
                    .addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? file.rfilename
                let urlString = "https://huggingface.co/\(model.huggingFaceRepo)"
                    + "/resolve/\(revision)/\(encodedName)"
                let relativePath = model.id.replacingOccurrences(of: "/", with: "_")
                    + "/" + file.rfilename

                if let existing = try? dest.resourceValues(forKeys: [.fileSizeKey]).fileSize,
                   Int64(existing) == fileSize, fileSize > 0 {
                    alreadyCompleted.insert(file.rfilename)
                }

                filePlans.append(FilePlan(
                    filename: file.rfilename,
                    size: fileSize,
                    downloadURL: urlString,
                    destinationRelativePath: relativePath
                ))
            }

            let plan = DownloadPlan(
                modelId: modelId,
                repo: model.huggingFaceRepo,
                revision: revision,
                files: filePlans,
                completedFilenames: alreadyCompleted,
                totalBytes: totalBytes
            )

            if plan.isComplete {
                await MainActor.run {
                    states[modelId] = .installed
                    InstalledModelsStore.shared.markInstalled(model)
                }
                return
            }

            setPlan(plan, for: modelId)
            savePlan(plan)

            let progress = totalBytes > 0 ? Double(plan.completedBytes) / Double(totalBytes) : 0
            await MainActor.run {
                states[modelId] = .downloading(
                    progress: progress, downloaded: plan.completedBytes, total: totalBytes
                )
            }

            for file in filePlans where !alreadyCompleted.contains(file.filename) {
                guard let url = URL(string: file.downloadURL) else { continue }

                let dest = modelsRoot.appendingPathComponent(file.destinationRelativePath)
                try FileManager.default.createDirectory(
                    at: dest.deletingLastPathComponent(),
                    withIntermediateDirectories: true
                )

                let task = backgroundSession.downloadTask(with: url)
                let meta = TaskMeta(modelId: modelId, filename: file.filename, fileSize: file.size)
                task.taskDescription = encodeMeta(meta)
                let taskId = task.taskIdentifier

                lock.withLock { taskModelMap[taskId] = modelId }

                task.resume()
            }

        } catch {
            await MainActor.run {
                states[modelId] = .error(error.localizedDescription)
            }
        }
    }

    // MARK: - HF Manifest

    private func fetchManifest(repo: String) async throws -> HFModelInfo {
        guard let url = URL(string: "https://huggingface.co/api/models/\(repo)?blobs=true") else {
            throw URLError(.badURL)
        }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else {
            throw URLError(.badServerResponse)
        }
        return try JSONDecoder().decode(HFModelInfo.self, from: data)
    }

    private func filterFiles(_ files: [HFFile], for model: AIModel) -> [HFFile] {
        let skipExact: Set<String> = [".gitattributes", "README.md", "LICENSE"]
        let base = files.filter { !skipExact.contains($0.rfilename) }

        switch model.fileLayout {
        case .huggingFaceMLX:
            // MLX repos: safetensors + tokenizer + config. Skip any GGUF or mlpackage.
            return base.filter { f in
                !f.rfilename.hasSuffix(".gguf") &&
                !f.rfilename.contains(".mlpackage")
            }

        case .singleGGUF:
            // Pick the preferred .gguf file + any tokenizer / chat-template json.
            var result: [HFFile] = []
            if let preferred = model.preferredGGUFFilename,
               let chosen = files.first(where: { $0.rfilename == preferred }) {
                result.append(chosen)
            } else if let firstGGUF = base.first(where: {
                $0.rfilename.hasSuffix(".gguf") && !$0.rfilename.hasPrefix("mmproj")
            }) {
                result.append(firstGGUF)
            }
            result += base.filter { f in
                f.rfilename.hasSuffix("tokenizer.json") ||
                f.rfilename.hasSuffix("tokenizer_config.json") ||
                f.rfilename == "chat_template.json"
            }
            return result

        case .ggufWithMmproj:
            var result: [HFFile] = []
            if let preferred = model.preferredGGUFFilename,
               let chosen = files.first(where: { $0.rfilename == preferred }) {
                result.append(chosen)
            }
            if let mmprojName = model.mmprojFilename,
               let chosen = files.first(where: { $0.rfilename == mmprojName }) {
                result.append(chosen)
            } else if let anyMmproj = base.first(where: { $0.rfilename.hasPrefix("mmproj") }) {
                result.append(anyMmproj)
            }
            result += base.filter { f in
                f.rfilename.hasSuffix("tokenizer.json") ||
                f.rfilename.hasSuffix("tokenizer_config.json")
            }
            return result

        case .coreMLPackage:
            // .mlpackage is a directory; its children land as nested rfilenames.
            // Keep them plus tokenizer/config so swift-transformers can load.
            return base.filter { f in
                f.rfilename.contains(".mlpackage") ||
                f.rfilename.hasSuffix("tokenizer.json") ||
                f.rfilename.hasSuffix("tokenizer_config.json") ||
                f.rfilename.hasSuffix("config.json") ||
                f.rfilename.hasSuffix("special_tokens_map.json")
            }

        case .whisperGGML:
            // The whisper.cpp repo hosts EVERY model size (tiny…large-v3,
            // 10+ GB total) — a bare prefix filter would download them all.
            // The catalog entry must pin the exact file.
            if let preferred = model.preferredGGUFFilename {
                return base.filter { $0.rfilename == preferred }
            }
            return base.filter { f in
                f.rfilename.hasPrefix("ggml-") && f.rfilename.hasSuffix(".bin")
                    && !f.rfilename.contains("/")
            }
        }
    }

    // MARK: - Plan persistence

    private func planURL(for modelId: String) -> URL {
        let safe = modelId.replacingOccurrences(of: "/", with: "_")
        return plansDirectory.appendingPathComponent(safe + ".json")
    }

    private func savePlan(_ plan: DownloadPlan) {
        lock.withLock {
            if let data = try? JSONEncoder().encode(plan) {
                try? data.write(to: planURL(for: plan.modelId))
            }
        }
    }

    private func loadPlan(for modelId: String) -> DownloadPlan? {
        lock.withLock {
            guard let data = try? Data(contentsOf: planURL(for: modelId)) else { return nil }
            return try? JSONDecoder().decode(DownloadPlan.self, from: data)
        }
    }

    private func removePlan(for modelId: String) {
        lock.withLock { plans[modelId] = nil }
        try? FileManager.default.removeItem(at: planURL(for: modelId))
    }

    // MARK: - Completion marker

    private func completionMarkerURL(for model: AIModel) -> URL {
        modelDirectory(for: model).appendingPathComponent(".complete")
    }

    private func writeCompletionMarker(for model: AIModel) {
        let url = completionMarkerURL(for: model)
        try? Data().write(to: url)
    }

    private func hasCompletionMarker(for model: AIModel) -> Bool {
        FileManager.default.fileExists(atPath: completionMarkerURL(for: model).path)
    }

    private func removeCompletionMarker(for model: AIModel) {
        try? FileManager.default.removeItem(at: completionMarkerURL(for: model))
    }

    // MARK: - TaskMeta encoding

    private func encodeMeta(_ meta: TaskMeta) -> String {
        (try? String(data: JSONEncoder().encode(meta), encoding: .utf8)) ?? ""
    }

    private func decodeMeta(from task: URLSessionTask) -> TaskMeta? {
        guard let desc = task.taskDescription, let data = desc.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(TaskMeta.self, from: data)
    }

    // MARK: - Restore state on launch

    /// Single lookup path for a model id. The catalog knows nothing about
    /// user-imported models, so the persisted registry is the second stop.
    private func resolveModel(id: String) -> AIModel? {
        ModelCatalog.model(id: id) ?? InstalledModelsStore.shared.knownModel(id: id)
    }

    private func restoreInstalledStates() {
        // Union of catalog and registry: curated models can appear on disk
        // without ever having been registered (older builds), and imported
        // models exist only in the registry.
        var candidates: [String: AIModel] = [:]
        for model in ModelCatalog.all { candidates[model.id] = model }
        for model in InstalledModelsStore.shared.knownModels { candidates[model.id] = model }

        for model in candidates.values {
            if hasCompletionMarker(for: model) {
                states[model.id] = .installed
                InstalledModelsStore.shared.markInstalled(model)
                // Clean up orphaned plan files
                removePlan(for: model.id)
            } else if InstalledModelsStore.shared.isInstalled(model) {
                // Registry claims it is installed but the marker is gone
                // (user deleted the container, restored from backup, …).
                InstalledModelsStore.shared.markUninstalled(model)
            }
        }
    }

    private func restoreActiveDownloads() async {
        let tasks = await backgroundSession.allTasks
        var activeModelIds: Set<String> = []

        for task in tasks where task.state == .running || task.state == .suspended {
            if let meta = decodeMeta(from: task) {
                activeModelIds.insert(meta.modelId)
            }
        }

        for modelId in activeModelIds {
            if let plan = loadPlan(for: modelId) {
                setPlan(plan, for: modelId)
                let downloaded = plan.completedBytes
                await MainActor.run {
                    states[modelId] = .downloading(
                        progress: plan.totalBytes > 0 ? Double(downloaded) / Double(plan.totalBytes) : 0,
                        downloaded: downloaded,
                        total: plan.totalBytes
                    )
                }
            }
        }

        // Clean up orphaned plans (tasks finished while app was killed)
        if let planFiles = try? FileManager.default.contentsOfDirectory(
            at: plansDirectory, includingPropertiesForKeys: nil
        ) {
            for file in planFiles where file.pathExtension == "json" {
                if let data = try? Data(contentsOf: file),
                   let plan = try? JSONDecoder().decode(DownloadPlan.self, from: data),
                   !activeModelIds.contains(plan.modelId) {
                    // Plan exists but no active tasks — check if complete
                    if plan.isComplete,
                       let model = resolveModel(id: plan.modelId) {
                        await MainActor.run {
                            states[plan.modelId] = .installed
                            InstalledModelsStore.shared.markInstalled(model)
                        }
                    }
                    try? FileManager.default.removeItem(at: file)
                    setPlan(nil, for: plan.modelId)
                }
            }
        }
    }

    // MARK: - Model completion check

    private func markFileCompleted(modelId: String, filename: String) {
        let updatedPlan: DownloadPlan? = withPlan(for: modelId) { plan in
            plan?.completedFilenames.insert(filename)
            return plan
        }

        guard let plan = updatedPlan else { return }
        savePlan(plan)

        if plan.isComplete {
            removePlan(for: modelId)
            if let model = resolveModel(id: modelId) {
                writeCompletionMarker(for: model)
                Task { @MainActor in
                    if model.requiresCompilation {
                        await self.compileIfNeeded(model: model)
                    } else {
                        self.states[modelId] = .installed
                        InstalledModelsStore.shared.markInstalled(model)
                    }
                }
            }
        }
    }

    @MainActor
    private func compileIfNeeded(model: AIModel) async {
        let dir = modelDirectory(for: model)
        let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)) ?? []
        let mlpackages = items.filter { $0.pathExtension == "mlpackage" }
        guard let mlpackage = mlpackages.first else {
            // Nothing to compile (maybe the .mlpackage lives nested elsewhere).
            // Fall through to installed so the engine can locate it at load time.
            states[model.id] = .installed
            InstalledModelsStore.shared.markInstalled(model)
            return
        }

        states[model.id] = .compiling(progress: 0.0)
        do {
            _ = try await CoreMLCompiler.compileIfNeeded(
                mlpackageURL: mlpackage,
                cacheDirectory: dir,
                estimatedDurationSeconds: 90
            ) { [weak self] p in
                Task { @MainActor [weak self] in
                    self?.states[model.id] = .compiling(progress: p)
                }
                _ = self
            }
            states[model.id] = .installed
            InstalledModelsStore.shared.markInstalled(model)
        } catch {
            states[model.id] = .error("Compile failed: \(error.localizedDescription)")
        }
    }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloader: URLSessionDownloadDelegate {

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        // Clean up per-task tracking
        lock.withLock {
            taskBytesWritten.removeValue(forKey: downloadTask.taskIdentifier)
            taskModelMap.removeValue(forKey: downloadTask.taskIdentifier)
        }

        guard let meta = decodeMeta(from: downloadTask) else { return }

        // Validate the HTTP response BEFORE persisting. Gated repos
        // (Llama, Gemma) answer 401/403 with an HTML page — saving that
        // as the model file "installs" garbage that fails at load time
        // with a cryptic error.
        if let http = downloadTask.response as? HTTPURLResponse, http.statusCode != 200 {
            let modelId = meta.modelId
            removePlan(for: modelId)
            Task { @MainActor in
                let hint = (http.statusCode == 401 || http.statusCode == 403)
                    ? " This repo is gated — accept the license on huggingface.co and use an authenticated mirror."
                    : ""
                self.states[modelId] = .error(
                    "Download failed: HTTP \(http.statusCode) for \(meta.filename).\(hint)"
                )
            }
            return
        }

        guard let plan = readPlan(for: meta.modelId),
              let filePlan = plan.files.first(where: { $0.filename == meta.filename })
        else { return }

        let destination = modelsRoot.appendingPathComponent(filePlan.destinationRelativePath)

        do {
            try FileManager.default.createDirectory(
                at: destination.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            // Remove existing file if present, then move temp → final.
            // Must complete before this method returns — iOS deletes temp file after.
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.moveItem(at: location, to: destination)

            markFileCompleted(modelId: meta.modelId, filename: meta.filename)

        } catch {
            Task { @MainActor in
                self.states[meta.modelId] = .error(
                    "Failed to save \(meta.filename): \(error.localizedDescription)"
                )
            }
        }
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        guard let meta = decodeMeta(from: downloadTask),
              let plan = readPlan(for: meta.modelId)
        else { return }

        let activeBytes: Int64 = lock.withLock {
            taskBytesWritten[downloadTask.taskIdentifier] = totalBytesWritten
            taskModelMap[downloadTask.taskIdentifier] = meta.modelId

            var total: Int64 = 0
            for (taskId, modelId) in taskModelMap where modelId == meta.modelId {
                total += taskBytesWritten[taskId] ?? 0
            }
            return total
        }

        let downloaded = plan.completedBytes + activeBytes
        let total = plan.totalBytes
        let progress = total > 0 ? min(Double(downloaded) / Double(total), 1.0) : 0

        Task { @MainActor in
            self.states[meta.modelId] = .downloading(
                progress: progress,
                downloaded: downloaded,
                total: total
            )
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        guard let meta = decodeMeta(from: task), let error else { return }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled { return }

        let modelId = meta.modelId
        removePlan(for: modelId)

        backgroundSession.getAllTasks { tasks in
            for t in tasks {
                if let m = self.decodeMeta(from: t), m.modelId == modelId {
                    t.cancel()
                }
            }
        }

        Task { @MainActor in
            self.states[modelId] = .error(error.localizedDescription)
        }
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            self.backgroundCompletionHandler?()
            self.backgroundCompletionHandler = nil
        }
    }
}
