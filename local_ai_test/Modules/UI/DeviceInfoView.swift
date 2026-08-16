//
//  DeviceInfoView.swift
//  local_ai_test
//

import SwiftUI

struct DeviceInfoView: View {
    let device: DeviceInfo

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Device")
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.bottom, 8)

            VStack(spacing: 1) {
                row(icon: "iphone", label: "Model", value: device.marketingName)
                row(icon: "barcode", label: "Identifier", value: device.identifier)
                row(icon: "cpu", label: "Chip", value: device.chip)
                row(icon: "memorychip", label: "RAM", value: device.totalRAMFormatted)
                row(icon: "internaldrive", label: "Free space", value: device.freeDiskFormatted)
                row(icon: "gear", label: "iOS", value: device.iOSVersion)
                appleIntelligenceRow
            }
            .clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }

    private func row(icon: String, label: String, value: String) -> some View {
        HStack {
            Label(label, systemImage: icon)
                .foregroundStyle(.primary)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 11)
        .background(.regularMaterial)
    }

    private var appleIntelligenceRow: some View {
        HStack {
            Label("Apple Intelligence", systemImage: "apple.intelligence")
                .foregroundStyle(.primary)
            Spacer()
            appleIntelligenceBadge
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 11)
        .background(.regularMaterial)
    }

    @ViewBuilder
    private var appleIntelligenceBadge: some View {
        switch device.appleIntelligence {
        case .available:
            Label("Enabled", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .font(.subheadline)
        case .notEnabled:
            Label("Disabled", systemImage: "exclamationmark.circle.fill")
                .foregroundStyle(.orange)
                .font(.subheadline)
        case .notSupported:
            Label("Not supported", systemImage: "xmark.circle.fill")
                .foregroundStyle(.secondary)
                .font(.subheadline)
        case .modelNotReady:
            Label("Downloading...", systemImage: "arrow.down.circle.fill")
                .foregroundStyle(.blue)
                .font(.subheadline)
        case .unknown:
            Label("Unknown", systemImage: "questionmark.circle.fill")
                .foregroundStyle(.secondary)
                .font(.subheadline)
        }
    }
}
