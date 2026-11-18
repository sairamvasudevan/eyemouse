import Foundation
import AVFoundation
import CoreGraphics
import AppKit

public final class PermissionsManager {
    public static let shared = PermissionsManager()
    
    private init() {}
    
    // MARK: - Camera Authorization
    
    public var isCameraAuthorized: Bool {
        return AVCaptureDevice.authorizationStatus(for: .video) == .authorized
    }
    
    public var cameraStatus: AVAuthorizationStatus {
        return AVCaptureDevice.authorizationStatus(for: .video)
    }
    
    public func requestCameraAccess(completion: @escaping (Bool) -> Void) {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async {
                    completion(granted)
                }
            }
        case .denied, .restricted:
            completion(false)
        @unknown default:
            completion(false)
        }
    }
    
    // MARK: - Accessibility Authorization (for active modifying event tap)
    
    public var isAccessibilityTrusted: Bool {
        return AXIsProcessTrusted()
    }
    
    public func promptForAccessibility() {
        let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
        _ = AXIsProcessTrustedWithOptions(options)
    }
    
    // MARK: - Input Monitoring Authorization (for passive listen-only event tap)
    
    public var isInputMonitoringGranted: Bool {
        return CGPreflightListenEventAccess()
    }
    
    public func requestInputMonitoring() {
        _ = CGRequestListenEventAccess()
    }
    
    // MARK: - System Settings Navigation
    
    public func openSystemSettings(for pane: SettingsPane) {
        if let url = URL(string: pane.urlString) {
            NSWorkspace.shared.open(url)
        }
    }
    
    public enum SettingsPane {
        case camera
        case accessibility
        case inputMonitoring
        
        var urlString: String {
            switch self {
            case .camera:
                return "x-apple.systempreferences:com.apple.preference.security?Privacy_Camera"
            case .accessibility:
                return "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility"
            case .inputMonitoring:
                return "x-apple.systempreferences:com.apple.preference.security?Privacy_ListenEvent"
            }
        }
    }
}

