import Foundation
import AppKit

public final class AppDelegate: NSObject, NSApplicationDelegate {
    
    public func applicationDidFinishLaunching(_ notification: Notification) {
        print("=== EyeMouse Application Launching ===")
        print("Version: \(BundleConstants.appVersion) (Target: Apple Silicon macOS)")
        
        // Run as menu-bar accessory utility
        NSApp.setActivationPolicy(.accessory)
        
        // 1. Initialize Frame Pipeline
        _ = FramePipelineCoordinator.shared
        
        // 2. Setup Menu Bar Item & UI
        MenuBarController.shared.setupMenuBar()
        
        // 2b. Check for Existing Active Calibration Profile
        DiagnosticsStore.shared.loadActiveProfile()
        
        // 3. Check and Prompt Permissions
        checkPermissionsOnInit()
        
        // 4. Hook SafetyStateManager to Camera and Engine lifecycle
        setupSafetyStateCoordination()
        
        // 5. Start Global Keyboard Event Tap
        let tapStarted = GlobalEventTap.shared.start()
        if !tapStarted {
            print("[AppDelegate] Warning: Event tap did not start immediately. Checking permissions...")
        }
        
        print("=== EyeMouse Initialized Successfully ===")
    }
    
    public func applicationWillTerminate(_ notification: Notification) {
        print("[AppDelegate] Application terminating. Releasing all resources...")
        GlobalEventTap.shared.stop()
        CameraCaptureService.shared.stopCapture()
        EngineProcessManager.shared.stopEngine()
        SafetyStateManager.shared.forceInactive(reason: "Application terminated")
        print("[AppDelegate] Teardown complete.")
    }
    
    private func checkPermissionsOnInit() {
        if !PermissionsManager.shared.isCameraAuthorized {
            PermissionsManager.shared.requestCameraAccess { granted in
                print("[AppDelegate] Camera authorization result: \(granted)")
                if !granted {
                    SafetyStateManager.shared.reportError("Camera permission denied")
                }
            }
        }
        
        if !PermissionsManager.shared.isAccessibilityTrusted && !PermissionsManager.shared.isInputMonitoringGranted {
            print("[AppDelegate] Accessibility or Input Monitoring required for global shortcut monitoring.")
            PermissionsManager.shared.promptForAccessibility()
        }
    }
    
    private func setupSafetyStateCoordination() {
        SafetyStateManager.shared.onStateChanged = { state in
            print("[SafetyCoordination] State transitioned to: \(state.rawValue)")
            switch state {
            case .starting:
                // 1. Ensure engine process is running
                DispatchQueue.global(qos: .userInitiated).async {
                    if !EngineProcessManager.shared.isRunning {
                        let started = EngineProcessManager.shared.startEngine()
                        if !started {
                            SafetyStateManager.shared.reportError("Engine startup failed")
                            return
                        }
                    }
                    
                    // 2. Start Camera capture session
                    CameraCaptureService.shared.startCapture()
                }
                
            case .stopping:
                // Synchronously close frame gate, asynchronously stop camera
                CameraCaptureService.shared.stopCapture()
                
            case .inactive:
                break
                
            case .error:
                CameraCaptureService.shared.stopCapture()
                break
                
            case .active:
                break
            }
        }
    }
}

