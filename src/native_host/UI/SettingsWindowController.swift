import Foundation
import AppKit
import SwiftUI

public final class SettingsWindowController: NSWindowController {
    public static let shared = SettingsWindowController()
    
    private convenience init() {
        let window = NSWindow(
            contentRect: NSRect(x: 150, y: 150, width: 500, height: 480),
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "EyeMouse Settings"
        window.center()
        window.contentViewController = NSHostingController(rootView: SettingsView())
        self.init(window: window)
    }
}

public struct SettingsView: View {
    @State private var selectedShortcutName: String
    @State private var selectedCameraID: String
    
    private let shortcutPresets: [ShortcutConfig] = [
        .defaultShortcut,
        .controlOptionM,
        .rightOptionHold
    ]
    
    public init() {
        _selectedShortcutName = State(initialValue: ShortcutConfig.load().name)
        _selectedCameraID = State(initialValue: CameraManager.shared.selectedDevice?.id ?? "")
    }
    
    public var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            // 1. Activation Shortcut Section
            GroupBox(label: Label("Activation Shortcut", systemImage: "keyboard.fill")) {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Eye control is active ONLY while this shortcut is held. Releasing any key immediately deactivates tracking.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Picker("Shortcut:", selection: $selectedShortcutName) {
                        ForEach(shortcutPresets, id: \.name) { preset in
                            Text("\(preset.name) (\(preset.displayString))").tag(preset.name)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: selectedShortcutName) { _, newName in
                        if let match = shortcutPresets.first(where: { $0.name == newName }) {
                            GlobalEventTap.shared.updateConfig(match)
                            MenuBarController.shared.updateShortcutDisplay()
                        }
                    }
                }
                .padding(8)
            }
            
            // 2. Camera Selection Section
            GroupBox(label: Label("Video Capture Device", systemImage: "camera.fill")) {
                VStack(alignment: .leading, spacing: 8) {
                    Picker("Camera:", selection: $selectedCameraID) {
                        ForEach(CameraManager.shared.availableDevices) { dev in
                            Text("\(dev.localizedName) (\(dev.type.rawValue))").tag(dev.id)
                        }
                    }
                    .pickerStyle(MenuPickerStyle())
                    .onChange(of: selectedCameraID) { _, newID in
                        CameraManager.shared.selectDevice(byID: newID)
                    }
                    
                    Text("Default prefers the built-in webcam. 640x480 BGRA is captured for low latency.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
                .padding(8)
            }
            
            // 3. macOS Permissions Checklist
            GroupBox(label: Label("macOS System Permissions", systemImage: "lock.shield.fill")) {
                VStack(alignment: .leading, spacing: 10) {
                    // Camera
                    HStack {
                        Image(systemName: PermissionsManager.shared.isCameraAuthorized ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                            .foregroundColor(PermissionsManager.shared.isCameraAuthorized ? .green : .red)
                        Text("Camera:")
                            .font(.subheadline)
                        Spacer()
                        Text(PermissionsManager.shared.isCameraAuthorized ? "Granted" : "Required")
                            .font(.caption)
                            .foregroundColor(PermissionsManager.shared.isCameraAuthorized ? .green : .red)
                        Button("Settings") {
                            PermissionsManager.shared.openSystemSettings(for: .camera)
                        }
                        .font(.caption)
                    }
                    
                    // Accessibility (Active Key Filtering)
                    HStack {
                        Image(systemName: PermissionsManager.shared.isAccessibilityTrusted ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                            .foregroundColor(PermissionsManager.shared.isAccessibilityTrusted ? .green : .orange)
                        Text("Accessibility (Key Suppression):")
                            .font(.subheadline)
                        Spacer()
                        Text(PermissionsManager.shared.isAccessibilityTrusted ? "Trusted" : "Recommended")
                            .font(.caption)
                            .foregroundColor(PermissionsManager.shared.isAccessibilityTrusted ? .green : .orange)
                        Button("Settings") {
                            PermissionsManager.shared.openSystemSettings(for: .accessibility)
                        }
                        .font(.caption)
                    }
                    
                    // Input Monitoring (Global Shortcut Monitoring)
                    HStack {
                        Image(systemName: PermissionsManager.shared.isInputMonitoringGranted ? "checkmark.circle.fill" : "info.circle.fill")
                            .foregroundColor(PermissionsManager.shared.isInputMonitoringGranted ? .green : .blue)
                        Text("Input Monitoring (Listen Tap):")
                            .font(.subheadline)
                        Spacer()
                        Text(PermissionsManager.shared.isInputMonitoringGranted ? "Granted" : "Optional")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Button("Settings") {
                            PermissionsManager.shared.openSystemSettings(for: .inputMonitoring)
                        }
                        .font(.caption)
                    }
                }
                .padding(8)
            }
            
            Spacer()
        }
        .padding(16)
        .frame(minWidth: 460, minHeight: 440)
    }
}
