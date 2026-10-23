//
//  BenchmarkSuiteListView.swift
//  local_ai_test
//
//  "Task suites": the built-in one, plus whatever the user imported. Pushed
//  from the Model Benchmark section, so it owns its own import sheet — it is
//  a full screen of its own, not a row a `List` can recycle.
//

import SwiftUI

struct BenchmarkSuiteListView: View {

    @State private var store = CustomBenchmarkSuiteStore.shared
    @State private var showImporter = false
    @State private var pendingDeletion: StoredBenchmarkSuite?

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Built-in")
                        .font(.subheadline).bold()
                    Text("\(RemoteBenchmarkSuite.shared.suite.tasks.count) tasks · "
                         + "suite v\(RemoteBenchmarkSuite.shared.suite.suiteVersion)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 2)
            } footer: {
                Text("Ships with the app and refreshes from the project repo.")
            }

            Section {
                if store.suites.isEmpty {
                    Text("No imported suites yet.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(store.suites) { suite in
                        SuiteRow(suite: suite)
                            .swipeActions {
                                Button(role: .destructive) {
                                    pendingDeletion = suite
                                } label: {
                                    Label("Delete", systemImage: "trash")
                                }
                            }
                    }
                }
            } header: {
                Text("Imported")
            } footer: {
                Text("Each imported suite scores on its own scale. A run graded by one "
                     + "of these is never compared against a built-in run, or against "
                     + "another suite.")
            }

            Section {
                Button {
                    showImporter = true
                } label: {
                    Label("Import a suite", systemImage: "square.and.arrow.down")
                }
            }
        }
        .navigationTitle("Task suites")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showImporter) {
            ImportBenchmarkSuiteSheet()
        }
        .alert(
            "Delete this suite?",
            isPresented: Binding(
                get: { pendingDeletion != nil },
                set: { if !$0 { pendingDeletion = nil } }
            ),
            presenting: pendingDeletion
        ) { suite in
            Button("Cancel", role: .cancel) { pendingDeletion = nil }
            Button("Delete", role: .destructive) {
                store.delete(suite)
                pendingDeletion = nil
            }
        } message: { suite in
            Text("\(suite.name) and its \(suite.taskCount) tasks are removed from this "
                 + "device. Reports it already produced are kept.")
        }
    }
}

private struct SuiteRow: View {
    let suite: StoredBenchmarkSuite

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(suite.name)
                .font(.subheadline).bold()
                .lineLimit(1)
            Text("\(suite.taskCount) task\(suite.taskCount == 1 ? "" : "s") · "
                 + suite.format.displayName)
                .font(.caption)
                .foregroundStyle(.secondary)
            Text(suite.source)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .truncationMode(.middle)
        }
        .padding(.vertical, 2)
    }
}
