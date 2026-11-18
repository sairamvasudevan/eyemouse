import Foundation
import CoreGraphics

func assertTest(_ condition: Bool, _ message: String) {
    if !condition {
        print("FAIL: \(message)")
        exit(1)
    }
    print("PASS: \(message)")
}

@main
struct SafetyStateTestRunner {
    static func main() {
        print("=== Running M3 Safety State Manager & Shortcut Unit Tests ===")

        // 1. Initial State
        let mgr = SafetyStateManager.shared
        assertTest(mgr.currentState == .inactive, "Initial state must be INACTIVE")
        assertTest(!mgr.isKeyHeld, "Initial isKeyHeld must be false")
        assertTest(!mgr.canProcessFrames, "Initial canProcessFrames must be false")

        // 2. Normal Activation Flow
        mgr.handleShortcutPressed()
        assertTest(mgr.currentState == .starting, "State after press must be STARTING")
        assertTest(mgr.isKeyHeld, "isKeyHeld must be true")
        assertTest(mgr.canProcessFrames, "canProcessFrames is true during STARTING to allow camera frames")

        mgr.markEngineAndCameraReady()
        assertTest(mgr.currentState == .active, "State after camera/engine ready must be ACTIVE")
        assertTest(mgr.canProcessFrames, "canProcessFrames must be true when ACTIVE")

        // 3. Synchronous Key Release
        mgr.handleShortcutReleased()
        assertTest(!mgr.isKeyHeld, "isKeyHeld must immediately be false")
        assertTest(!mgr.canProcessFrames, "canProcessFrames must synchronously be false upon key release")
        assertTest(mgr.currentState == .stopping, "State must transition to STOPPING")

        mgr.markTeardownComplete()
        assertTest(mgr.currentState == .inactive, "State after teardown must be INACTIVE")
        assertTest(!mgr.canProcessFrames, "canProcessFrames must remain false")

        // 4. Startup Cancellation (User releases key while camera/engine is still booting)
        mgr.handleShortcutPressed()
        assertTest(mgr.currentState == .starting, "State must be STARTING")

        // Release key immediately before ready signal
        mgr.handleShortcutReleased()
        assertTest(mgr.currentState == .stopping, "State must transition to STOPPING before ready")
        assertTest(!mgr.canProcessFrames, "canProcessFrames must be false")

        // Now delayed ready signal arrives: must NOT enter ACTIVE!
        mgr.markEngineAndCameraReady()
        assertTest(mgr.currentState == .stopping, "Ready signal after key release must NOT activate; state remains STOPPING")
        assertTest(!mgr.canProcessFrames, "canProcessFrames must remain false")

        mgr.markTeardownComplete()
        assertTest(mgr.currentState == .inactive, "State must return to INACTIVE")

        // 5. Sleep and Screen Lock Recovery
        mgr.handleShortcutPressed()
        mgr.markEngineAndCameraReady()
        assertTest(mgr.currentState == .active, "State is ACTIVE")

        mgr.forceInactive(reason: "System Sleep")
        assertTest(mgr.currentState == .inactive, "System sleep forces INACTIVE immediately")
        assertTest(!mgr.canProcessFrames, "canProcessFrames must be false after sleep")
        assertTest(mgr.errorMessage == "System Sleep", "Error/Inactive reason recorded")

        // 6. Camera Disconnect / Engine Error
        mgr.handleShortcutPressed()
        mgr.markEngineAndCameraReady()
        assertTest(mgr.currentState == .active, "State is ACTIVE")

        mgr.reportError("Camera disconnected")
        assertTest(mgr.currentState == .error, "Reported error must set state to ERROR")
        assertTest(!mgr.canProcessFrames, "canProcessFrames must be false in ERROR state")
        assertTest(mgr.errorMessage == "Camera disconnected", "Error message recorded")

        // Reset to inactive
        mgr.forceInactive()
        assertTest(mgr.currentState == .inactive, "Reset to INACTIVE")

        // 7. Shortcut Configuration Matching Tests
        let defaultConfig = ShortcutConfig.defaultShortcut
        assertTest(defaultConfig.displayString == "⌃⌥E", "Default display string must be ⌃⌥E")

        // Valid chord: Key Code 14 (E) + Control + Option
        let match1 = defaultConfig.matches(keyCode: KeyCodeTable.kVK_ANSI_E, flags: [.maskControl, .maskAlternate])
        assertTest(match1, "Shortcut matches key E with Control+Option")

        // Partial chord: Only Control (Option missing) -> MUST NOT MATCH
        let match2 = defaultConfig.matches(keyCode: KeyCodeTable.kVK_ANSI_E, flags: [.maskControl])
        assertTest(!match2, "Shortcut fails when Option modifier is missing")

        // Wrong key: Key Code 46 (M) with Control + Option -> MUST NOT MATCH
        let match3 = defaultConfig.matches(keyCode: KeyCodeTable.kVK_ANSI_M, flags: [.maskControl, .maskAlternate])
        assertTest(!match3, "Shortcut fails when key code is M instead of E")

        // Extra modifier: Command + Control + Option + E -> MUST NOT MATCH
        let match4 = defaultConfig.matches(keyCode: KeyCodeTable.kVK_ANSI_E, flags: [.maskControl, .maskAlternate, .maskCommand])
        assertTest(!match4, "Shortcut fails when extra Command modifier is present")

        print("=== ALL M3 SAFETY STATE & SHORTCUT UNIT TESTS PASSED! ===")
    }
}
