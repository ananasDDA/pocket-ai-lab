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

// MARK: - Errors

nonisolated enum ModelDownloadError: Error, LocalizedError, Equatable {
    /// The catalog pins a `.mlpackage` / `.mlmodelc` the repo no longer
    /// offers. Downloading "something else that looks similar" is worse than
    /// not downloading: the pin exists because the alternatives are a
    /// different quantization or an uncompiled twin, so the fallback would
    /// silently install a model whose size, RAM figure and compile
    /// requirement no longer match the catalog entry the user agreed to.
    case preferredPackageMissing(package: String, repo: String)

    var errorDescription: String? {
        switch self {
        case .preferredPackageMissing(let package, let repo):
            return "\(repo) no longer publishes \(package). "
                + "This model needs an app or catalog update before it can be downloaded."
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

/// The manifest entry shape is identical to the one the import flow already
/// models, and `HFSibling` is `nonisolated` with a memberwise init, which
/// lets `filterFiles(_:for:)` be exercised directly from tests.
typealias HFFile = HFSibling

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

    /// In-flight "import from Files" copies, keyed by model id. They are not
    /// URLSession tasks, so `cancel(_:)` has to reach them separately.
    /// Main-actor only: every access is from `installLocalFile`, `cancel`, or
    /// `delete`. Each entry carries the `id` it was stored with so the
    /// `installLocalFile` cleanup can tell "my own task" apart from a newer
    /// import for the same model id that started while it was unwinding.
    @ObservationIgnored
    private var localCopyTasks: [String: (id: UUID, task: Task<Void, any Error>)] = [:]

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
            withIdentifier: "self.pro-ai-lab.modeldownloader.bg"
        )
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        backgroundSession = URLSession(configuration: config, delegate: self, delegateQueue: nil)

        restoreInstalledStates()
        Task { await restoreActiveDownloads() }
    }

    // MARK: - Thread-safe plan access
    //
    // PLAN LIFECYCLE INVARIANT. A model has a plan for exactly as long as its
    // download is live, so "the plan is gone" means "this download is over".
    // Every terminal transition — `.installed`, `.error`, and the `.idle` of
    // cancel/delete — therefore calls `removePlan(for:)` BEFORE publishing
    // that state, and the two routines below are the only places that publish
    // one. Progress callbacks reach the main actor asynchronously and are
    // free to land after a terminal state, so they check `readPlan` on the
    // main actor and drop themselves when it is nil; without the ordering
    // above, a late tick would put a finished, cancelled or failed download
    // back into `.downloading` and `download()` would then refuse to restart
    // it (`state(for:).isActive`), stranding the model.

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

    // MARK: - Terminal transitions

    /// The ONE way a model becomes installed. Order matters and is the whole
    /// point of routing every caller through here:
    ///
    /// 1. the plan goes first, closing the download (see the invariant above);
    /// 2. the completion marker is written — that file, not the in-memory
    ///    state, is what `restoreInstalledStates` reads on the next launch, so
    ///    a path that skips it un-installs the model at the next app start;
    /// 3. a `requiresCompilation` entry is compiled before it is called
    ///    installed, because the engine would otherwise be handed a bare
    ///    `.mlpackage`.
    private func completeInstall(of model: AIModel) {
        removePlan(for: model.id)
        writeCompletionMarker(for: model)
        Task { @MainActor in
            if model.requiresCompilation {
                await self.compileIfNeeded(model: model)
            } else {
                self.states[model.id] = .installed
                InstalledModelsStore.shared.markInstalled(model)
            }
            PALAnalytics.shared.downloadEnded(model, errorKind: self.states[model.id] == .installed ? nil : "coreml_compile")
        }
    }

    /// The ONE way a download fails. Plan first, then the sibling tasks that
    /// are now pointless, then the state. `errorKind` is a short machine kind
    /// for the measurement record; the message stays on the device.
    private func fail(modelId: String, message: String, errorKind: String, httpStatus: Int? = nil) {
        if let model = resolveModel(id: modelId) {
            PALAnalytics.shared.downloadEnded(model, errorKind: errorKind, httpStatus: httpStatus)
        }
        removePlan(for: modelId)
        cancelTasks(for: modelId)
        Task { @MainActor in
            self.states[modelId] = .error(message)
        }
    }

    /// Cancels every background task belonging to a model. Cancellation comes
    /// back as `NSURLErrorCancelled` in `didCompleteWithError`, which returns
    /// without touching the state — so this never fights the caller.
    private func cancelTasks(for modelId: String) {
        backgroundSession.getAllTasks { [weak self] tasks in
            guard let self else { return }
            for task in tasks where self.decodeMeta(from: task)?.modelId == modelId {
                task.cancel()
            }
        }
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
        localCopyTasks[modelId]?.task.cancel()
        cancelTasks(for: modelId)

        Task { @MainActor in
            self.states[modelId] = .idle
        }
    }

    func delete(_ model: AIModel) {
        let modelId = model.id
        removePlan(for: modelId)
        // A local "import from Files" copy is not a URLSession task, so the
        // getAllTasks cancellation below never reaches it. Without this, a
        // Remove tapped mid-copy deletes the model directory out from under
        // the detached copy task, which then finishes "successfully" into a
        // directory with no weights and re-marks the model installed.
        localCopyTasks[modelId]?.task.cancel()

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
                // A model imported from the Files app has no repo to fetch
                // again, so removing its files removes the entry too —
                // leaving it behind would show an Install button that could
                // never succeed.
                if model.source == .imported && model.huggingFaceRepo.isEmpty {
                    InstalledModelsStore.shared.unregister(model)
                } else {
                    InstalledModelsStore.shared.markUninstalled(model)
                }
            }
        }
    }

    // MARK: - Local file import

    /// Installs a model from a file the user picked in the Files app.
    ///
    /// The copy runs on a detached task: GGUF weights are gigabytes and the
    /// main actor has to stay responsive. Progress is published through
    /// `states` exactly like a download, so model cards and the import sheet
    /// need no special case for it.
    ///
    /// The caller owns the security-scoped access to `source` and must hold
    /// it for the whole call.
    func installLocalFile(
        _ model: AIModel,
        from source: URL,
        filename: String,
        totalBytes: Int64
    ) async throws {
        let modelId = model.id

        // Persist the descriptor first: a crash mid-copy must not leave a
        // multi-gigabyte file on disk that nothing in the registry points at.
        InstalledModelsStore.shared.register(model)
        removeCompletionMarker(for: model)
        states[modelId] = .downloading(progress: 0, downloaded: 0, total: totalBytes)

        let destination = modelDirectory(for: model).appendingPathComponent(filename)

        let copyTask = Task.detached(priority: .userInitiated) {
            try LocalGGUFFile.copy(
                from: source,
                to: destination,
                totalBytes: totalBytes
            ) { fraction in
                Task { @MainActor in
                    // Progress hops to the main actor asynchronously, so a
                    // late one must not overwrite the terminal state this
                    // method sets once the copy returns.
                    guard case .downloading = ModelDownloader.shared.state(for: model) else {
                        return
                    }
                    ModelDownloader.shared.states[modelId] = .downloading(
                        progress: fraction,
                        downloaded: Int64(Double(totalBytes) * fraction),
                        total: totalBytes
                    )
                }
            }
        }
        // Tagged with its own id so the cleanup below only clears the slot if
        // it still holds *this* task — a second import for the same model id
        // started while this one was unwinding must not lose its own entry.
        let copyTaskId = UUID()
        localCopyTasks[modelId] = (id: copyTaskId, task: copyTask)
        defer {
            if localCopyTasks[modelId]?.id == copyTaskId {
                localCopyTasks[modelId] = nil
            }
        }

        do {
            try await copyTask.value
        } catch is CancellationError {
            // Cancelled from the model card. The partial file is already gone
            // and there is no repo to resume from, so the entry goes with it.
            states[modelId] = .idle
            InstalledModelsStore.shared.unregister(model)
            throw CancellationError()
        } catch {
            // Same reasoning as the cancellation path above: leaving the
            // descriptor registered here would strand a phantom entry with
            // an empty huggingFaceRepo whose Install/Retry can never succeed,
            // and would make a later import of the same file wrongly ask
            // "already imported - replace?".
            states[modelId] = .error(error.localizedDescription)
            InstalledModelsStore.shared.unregister(model)
            throw error
        }

        writeCompletionMarker(for: model)
        states[modelId] = .installed
        InstalledModelsStore.shared.markInstalled(model)
    }

    // MARK: - Download pipeline

    private func startDownload(_ model: AIModel) async {
        let modelId = model.id
        PALAnalytics.shared.downloadStarted(model)

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
            let filtered = try Self.filterFiles(files, for: model)
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
                // Every file was already on disk at full size (a re-tapped
                // Install after a completed-but-unmarked download). This must
                // go through the same routine as a fresh completion: writing
                // `.installed` on its own leaves no completion marker, so
                // `restoreInstalledStates` un-installs the model on the next
                // launch, and it would skip the Core ML compile a
                // `requiresCompilation` entry still needs.
                completeInstall(of: model)
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
            // `fail` and not a bare state assignment: some files may already
            // have been resumed before the throw, and the plan must be gone
            // before the terminal state is published (see `fail`).
            fail(modelId: modelId, message: error.localizedDescription,
                 errorKind: PALErrorClassifier.kind(of: error, phase: .download))
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

    /// Repo-root files swift-transformers needs next to the weights. Matched
    /// by suffix so that a repo keeping them in a subfolder still works.
    nonisolated private static let supportFileSuffixes: [String] = [
        "tokenizer.json", "tokenizer_config.json", "config.json", "special_tokens_map.json",
    ]

    /// The two directory shapes a Core ML repo can offer, in the order we
    /// prefer them. `.mlmodelc` is ALREADY COMPILED, so a repo that publishes
    /// one lets the app skip `MLModel.compileModel` entirely — Apple's own
    /// recommendation, and the reason a catalog entry pointing at an
    /// `.mlmodelc` sets `requiresCompilation: false`.
    nonisolated static let coreMLContainerExtensions = [".mlmodelc", ".mlpackage"]

    /// Path of the `.mlpackage` / `.mlmodelc` directory `rfilename` lives
    /// inside, or nil when the file sits outside every container (repo-root
    /// tokenizer/config).
    nonisolated static func mlpackagePath(of rfilename: String) -> String? {
        var walked: [String] = []
        for component in rfilename.split(separator: "/") {
            walked.append(String(component))
            if coreMLContainerExtensions.contains(where: { component.hasSuffix($0) }) {
                return walked.joined(separator: "/")
            }
        }
        return nil
    }

    /// Every distinct Core ML container directory in a repo listing.
    nonisolated static func mlpackagePaths(in files: [HFFile]) -> [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for file in files {
            guard let path = mlpackagePath(of: file.rfilename), seen.insert(path).inserted else { continue }
            ordered.append(path)
        }
        return ordered
    }

    /// The container with the smallest summed file size. Missing sizes count
    /// as 0; ties break on path so the choice is deterministic.
    ///
    /// This is only the fallback for an unpinned repo. Repos that ship BOTH a
    /// `.mlpackage` and the `.mlmodelc` compiled from it (they are the same
    /// model twice) must pin `preferredPackage`, because "smallest" would pick
    /// the `.mlpackage` and pay a compile the repo already did for us.
    nonisolated static func smallestMLPackage(in files: [HFFile]) -> String? {
        var totals: [String: Int64] = [:]
        for file in files {
            guard let path = mlpackagePath(of: file.rfilename) else { continue }
            totals[path, default: 0] += file.size ?? 0
        }
        return totals.min {
            $0.value == $1.value ? $0.key < $1.key : $0.value < $1.value
        }?.key
    }

    /// Not private: `DownloaderFilterTests` drives this directly. Pure over
    /// its inputs, so it needs no downloader state.
    nonisolated static func filterFiles(_ files: [HFFile], for model: AIModel) throws -> [HFFile] {
        let skipExact: Set<String> = [".gitattributes", "README.md", "LICENSE"]
        let base = files.filter { !skipExact.contains($0.rfilename) }

        switch model.fileLayout {
        case .huggingFaceMLX:
            // MLX repos: safetensors + tokenizer + config. Skip any GGUF or mlpackage.
            return base.filter { f in
                !f.rfilename.hasSuffix(".gguf") &&
                Self.mlpackagePath(of: f.rfilename) == nil
            }

        case .singleGGUF:
            // Pick the preferred .gguf file + any tokenizer / chat-template json.
            var result: [HFFile] = []
            if let preferred = model.preferredGGUFFilename,
               let chosen = files.first(where: { $0.rfilename == preferred }) {
                result.append(chosen)
            } else if let firstGGUF = base.first(where: {
                GGUFNaming.isWeights($0.rfilename)
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
            } else if let anyMmproj = base.first(where: { GGUFNaming.isProjector($0.rfilename) }) {
                result.append(anyMmproj)
            }
            result += base.filter { f in
                f.rfilename.hasSuffix("tokenizer.json") ||
                f.rfilename.hasSuffix("tokenizer_config.json")
            }
            return result

        case .coreMLPackage:
            // `.mlpackage` and `.mlmodelc` are both directories; their children
            // land as nested rfilenames. Keep the children of exactly ONE of
            // them, plus the repo-root tokenizer/config so swift-transformers
            // can load.
            //
            // Repos routinely ship several containers side by side: two
            // quantizations of the same model, or a `.mlpackage` next to the
            // `.mlmodelc` compiled from it. Taking them all multiplies the
            // download for no benefit, so exactly one is chosen — pinned by
            // `preferredPackage` where the catalog cares which.
            let isSupportFile: (HFFile) -> Bool = { f in
                Self.supportFileSuffixes.contains { f.rfilename.hasSuffix($0) }
            }
            let packages = Self.mlpackagePaths(in: base)
            // A repo-root file whose name matches a support suffix but which
            // lives inside a container is handled below; `mlpackagePath` is
            // what separates the two.

            var chosen: String?
            if let preferred = model.preferredPackage {
                chosen = packages.first { $0 == preferred || $0.hasSuffix("/" + preferred) }
                if chosen == nil {
                    // Fail loudly instead of falling back to the smallest
                    // package. The pin is what makes the entry's numbers true:
                    // a `.mlmodelc` pin exists precisely because the smallest
                    // container is the uncompiled `.mlpackage`, and taking it
                    // would install a model that needs an on-device compile
                    // and twice the disk the entry reserved.
                    print("[ModelDownloader] preferredPackage '\(preferred)' not in \(model.huggingFaceRepo); failing the download")
                    throw ModelDownloadError.preferredPackageMissing(
                        package: preferred, repo: model.huggingFaceRepo
                    )
                }
            }
            if chosen == nil, packages.count > 1 {
                chosen = Self.smallestMLPackage(in: base)
                if let chosen {
                    print("[ModelDownloader] \(model.huggingFaceRepo) ships \(packages.count) .mlpackage variants; downloading the smallest: \(chosen)")
                }
            }

            guard let chosen else {
                // Single-container (or container-less) repo: take everything
                // that belongs to a container, plus the support files.
                return base.filter { isSupportFile($0) || Self.mlpackagePath(of: $0.rfilename) != nil }
            }
            return base.filter { f in
                if f.rfilename.hasPrefix(chosen + "/") || f.rfilename == chosen { return true }
                // Tokenizer/config living inside a container we are not taking
                // would land in an unused directory — drop it.
                return isSupportFile(f) && Self.mlpackagePath(of: f.rfilename) == nil
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
    /// The descriptor for a model id, for code that only has the id: the
    /// download delegate, which runs long after the caller handed us an
    /// `AIModel`.
    ///
    /// `ModelCatalog.builtIn` is the last resort and it is NOT redundant.
    /// `ModelCatalog.all` serves the catalog the app last fetched from the
    /// public repo, which on a device that ran an older build is OLDER than
    /// the build itself. A model this build added and started downloading is
    /// then unresolvable here — and an unresolvable model never gets its
    /// completion marker or its `.installed` state, so a download that
    /// transferred every byte sits at 100% forever and the model can never be
    /// used. Falling back to the compiled-in catalog makes the downloader able
    /// to finish anything it was able to start.
    private func resolveModel(id: String) -> AIModel? {
        ModelCatalog.model(id: id)
            ?? InstalledModelsStore.shared.knownModel(id: id)
            ?? ModelCatalog.builtIn.first { $0.id == id }
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
        // Every model the session still knows about, in ANY task state. A
        // transfer that finished while the app was dead comes back
        // `.completed` with its `didFinishDownloadingTo` still queued for
        // delivery, so it is not "active" — but its plan is not orphaned
        // either, and deleting it here would race the delivery and throw the
        // finished file away.
        var knownModelIds: Set<String> = []

        for task in tasks {
            guard let meta = decodeMeta(from: task) else { continue }
            knownModelIds.insert(meta.modelId)
            if task.state == .running || task.state == .suspended {
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
                   !knownModelIds.contains(plan.modelId) {
                    // Plan exists but no active tasks — check if complete.
                    // Same routine as the live path: a plan that finished
                    // while the app was dead needs its completion marker and,
                    // for a `requiresCompilation` entry, its compile. Marking
                    // it `.installed` here and only deleting the plan file
                    // used to leave a model that un-installed itself on the
                    // very next launch, because no marker was ever written.
                    if plan.isComplete, let model = resolveModel(id: plan.modelId) {
                        completeInstall(of: model)   // removes the plan too
                    } else {
                        try? FileManager.default.removeItem(at: file)
                        setPlan(nil, for: plan.modelId)
                    }
                }
            }
        }
    }

    // MARK: - Model completion check

    private func markFileCompleted(modelId: String, filename: String) {
        // A download that finished while the app was not running is delivered
        // as soon as the background session is recreated, which is before the
        // async `restoreActiveDownloads` has put the plan back in memory (and
        // that only rehydrates plans with a still-RUNNING task, so a model
        // whose last file completed while the app was dead has none at all).
        // The persisted plan is the source of truth in that window; without
        // this fallback the completion is dropped and the model never gets its
        // marker. Read outside `withPlan` — `loadPlan` takes the same
        // non-recursive lock.
        let persisted = readPlan(for: modelId) == nil ? loadPlan(for: modelId) : nil

        let updatedPlan: DownloadPlan? = withPlan(for: modelId) { plan in
            if plan == nil { plan = persisted }
            plan?.completedFilenames.insert(filename)
            return plan
        }

        guard let plan = updatedPlan else { return }
        savePlan(plan)

        if plan.isComplete, let model = resolveModel(id: modelId) {
            completeInstall(of: model)
        } else if plan.isComplete {
            // Unresolvable model id: drop the plan anyway so a stale one does
            // not resurrect the download on the next launch.
            removePlan(for: modelId)
        }
    }

    /// Recursive on-disk size of a directory, 0 when it cannot be read.
    nonisolated static func directorySize(of url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey], options: [.skipsHiddenFiles]
        ) else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
        }
        return total
    }

    @MainActor
    private func compileIfNeeded(model: AIModel) async {
        let dir = modelDirectory(for: model)
        // Same selection rule the engine uses at load time, so the package we
        // compile is the package we later run.
        guard let mlpackage = CoreMLCompiler.locatePackage(in: dir) else {
            // Nothing to compile (maybe the .mlpackage lives nested elsewhere).
            // Fall through to installed so the engine can locate it at load time.
            states[model.id] = .installed
            InstalledModelsStore.shared.markInstalled(model)
            return
        }

        states[model.id] = .compiling(progress: 0.0)
        // The progress is an estimate driven by elapsed time; scale it with
        // the package size so a 4 GB model does not sit at 100% for most of
        // its compile. Roughly 90 s base plus two minutes per GB on device.
        let packageBytes = Self.directorySize(of: mlpackage)
        let estimatedSeconds = 90 + 120 * Double(packageBytes) / 1_073_741_824

        // Progress arrives off the main actor and is published by hopping ONTO
        // it, so a tick posted moments before the compile returns runs AFTER
        // the terminal state assigned below and pins the model at `.compiling`
        // forever. That is what made a Core ML compile which finishes in
        // seconds look like a 60-minute hang: the build-test harness polled
        // `.compiling(1.0)` until its budget ran out and then deleted a model
        // that had compiled perfectly well. `CoreMLCompiler` now drains its
        // ticker before returning; this latch closes the remaining window, and
        // costs nothing to keep.
        let latch = CompileStateLatch()
        do {
            _ = try await CoreMLCompiler.compileIfNeeded(
                mlpackageURL: mlpackage,
                cacheDirectory: dir,
                estimatedDurationSeconds: estimatedSeconds
            ) { [weak self] p in
                Task { @MainActor [weak self] in
                    guard let self, !latch.isClosed else { return }
                    self.states[model.id] = .compiling(progress: p)
                }
            }
            latch.close()
            states[model.id] = .installed
            InstalledModelsStore.shared.markInstalled(model)
        } catch {
            latch.close()
            states[model.id] = .error("Compile failed: \(error.localizedDescription)")
        }
    }
}

/// One-shot latch that stops a late compile-progress callback from
/// overwriting the terminal download state. Main-actor isolated, so the
/// `close()` that precedes the terminal assignment is ordered ahead of every
/// progress hop still queued behind it.
@MainActor
private final class CompileStateLatch {
    private(set) var isClosed = false
    func close() { isClosed = true }
}

// MARK: - URLSessionDownloadDelegate

extension ModelDownloader {
    /// 200 for a transfer that ran start to finish, 206 for one the session
    /// resumed with a Range request after an interruption. Anything else is
    /// an error page, not a model file.
    nonisolated static func isSuccessfulDownloadStatus(_ statusCode: Int) -> Bool {
        statusCode == 200 || statusCode == 206
    }
}

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
        //
        // 206 is a success: a background transfer that lost the network
        // resumes by itself with a Range request, and the response it hands
        // over at the end is that resume's, not the first request's. The
        // file at `location` is complete either way. Treating 206 as a
        // failure threw away finished multi-gigabyte downloads whenever
        // Wi-Fi blinked during an overnight sweep.
        if let http = downloadTask.response as? HTTPURLResponse,
           !Self.isSuccessfulDownloadStatus(http.statusCode) {
            let hint = (http.statusCode == 401 || http.statusCode == 403)
                ? " This repo is gated — accept the license on huggingface.co and use an authenticated mirror."
                : ""
            fail(
                modelId: meta.modelId,
                message: "Download failed: HTTP \(http.statusCode) for \(meta.filename).\(hint)",
                errorKind: "http",
                httpStatus: http.statusCode
            )
            return
        }

        // `loadPlan` fallback: a background transfer that finished while the
        // app was not running is delivered right after the session is
        // recreated, and the in-memory plan does not exist yet (see
        // `markFileCompleted`). Without it the finished file is dropped on the
        // floor — iOS deletes the temp file the moment this returns — and the
        // download restarts from zero. Rehydrating it here also means the rest
        // of this delegate sees the plan it expects. A cancelled or failed
        // download cannot come back this way: `removePlan` deletes the plan
        // FILE as well, so `loadPlan` finds nothing.
        let restored = readPlan(for: meta.modelId) ?? loadPlan(for: meta.modelId)
        guard let plan = restored,
              let filePlan = plan.files.first(where: { $0.filename == meta.filename })
        else { return }
        setPlan(plan, for: meta.modelId)

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
            fail(
                modelId: meta.modelId,
                message: "Failed to save \(meta.filename): \(error.localizedDescription)",
                errorKind: "file_save"
            )
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
            // The hop is asynchronous, so this can run AFTER the download
            // reached a terminal state: `didCompleteWithError` failing one
            // file of a multi-file model while others are still writing, or a
            // cancel/delete tapped mid-transfer. Re-checking the plan on the
            // main actor is what stops a stale tick from reinstating
            // `.downloading` over `.error`/`.idle` — a state the UI never
            // leaves again, because `download()` refuses to restart anything
            // whose state `isActive`. Safe by the plan-lifecycle invariant:
            // every terminal transition removes the plan first.
            guard self.readPlan(for: meta.modelId) != nil else { return }
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
        // Unconditional and first: this is the only callback every task is
        // guaranteed to reach, cancelled and failed ones included. Leaving a
        // dead task's byte count in `taskBytesWritten` makes the NEXT attempt
        // for the same model count it again — `didWriteData` sums every entry
        // whose `taskModelMap` id matches — and the restarted download opens
        // at an inflated percentage, or pinned at 100%. Removing entries
        // already cleared by `didFinishDownloadingTo` is a no-op.
        lock.withLock {
            taskBytesWritten.removeValue(forKey: task.taskIdentifier)
            taskModelMap.removeValue(forKey: task.taskIdentifier)
        }

        guard let meta = decodeMeta(from: task), let error else { return }

        let nsError = error as NSError
        if nsError.domain == NSURLErrorDomain && nsError.code == NSURLErrorCancelled { return }

        fail(modelId: meta.modelId, message: error.localizedDescription,
             errorKind: PALErrorClassifier.kind(of: error, phase: .download))
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        Task { @MainActor in
            self.backgroundCompletionHandler?()
            self.backgroundCompletionHandler = nil
        }
    }
}
