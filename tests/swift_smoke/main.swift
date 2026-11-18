import Foundation
import AppKit
import AVFoundation

print("=== Swift & AppKit Toolchain Smoke Test ===")
print("Process Name:", ProcessInfo.processInfo.processName)
print("Operating System Version:", ProcessInfo.processInfo.operatingSystemVersionString)

#if arch(arm64)
print("Target Architecture: arm64 (Apple Silicon) - VERIFIED")
#else
print("Target Architecture: NON-ARM64")
#endif

// Verify AppKit NSApplication initialization as an accessory (menu bar) app
let app = NSApplication.shared
app.setActivationPolicy(.accessory)
print("AppKit NSApplication initialized with .accessory activation policy.")

// Verify Display discovery
let screens = NSScreen.screens
print("Discovered Displays: \(screens.count)")
for (index, screen) in screens.enumerated() {
    let frame = screen.frame
    print("  Display [\(index)]: \(Int(frame.width))x\(Int(frame.height)) at origin (\(Int(frame.origin.x)), \(Int(frame.origin.y)))")
}

// Verify AVFoundation discovery capability without triggering a user permission prompt
let discoverySession = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera, .external],
    mediaType: .video,
    position: .unspecified
)
print("Discovered Video Input Devices: \(discoverySession.devices.count)")
for device in discoverySession.devices {
    print("  Camera: \(device.localizedName) (ID: \(device.uniqueID))")
}

print("SWIFT_APPKIT_SMOKE_TEST_SUCCESS")
exit(0)
