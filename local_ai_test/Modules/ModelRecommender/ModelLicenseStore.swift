//
//  ModelLicenseStore.swift
//  local_ai_test
//
//  Records which model licenses the user has explicitly accepted. Some
//  model families (Llama, Gemma) are distributed under licenses that
//  require acceptance before the weights are downloaded; the catalog marks
//  those entries with `licenseName`/`licenseURL` and the install button
//  routes through an acceptance prompt exactly once per license.
//

import Foundation

enum ModelLicenseStore {

    private static let key = "acceptedModelLicenses"

    /// Whether this model needs an acceptance prompt before downloading.
    static func requiresAcceptance(_ model: AIModel) -> Bool {
        guard let name = model.licenseName else { return false }
        return !accepted().contains(name)
    }

    static func recordAcceptance(of model: AIModel) {
        guard let name = model.licenseName else { return }
        var names = accepted()
        names.insert(name)
        UserDefaults.standard.set(Array(names).sorted(), forKey: key)
    }

    private static func accepted() -> Set<String> {
        Set(UserDefaults.standard.stringArray(forKey: key) ?? [])
    }
}
