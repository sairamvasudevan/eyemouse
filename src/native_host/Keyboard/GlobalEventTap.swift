import Foundation
import CoreGraphics
import AppKit

public final class GlobalEventTap {
    public static let shared = GlobalEventTap()
    
    private var machPort: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var isTapActive = false
    
    private var currentConfig: ShortcutConfig
    private var isTriggerHeld = false
    
    private init() {
        self.currentConfig = ShortcutConfig.load()
        setupSleepWakeObservers()
    }
    
    public func updateConfig(_ config: ShortcutConfig) {
        self.currentConfig = config
        config.save()
    }
    
    public func start() -> Bool {
        guard !isTapActive else { return true }
        
        let mask = CGEventMask(
            (1 << CGEventType.keyDown.rawValue) |
            (1 << CGEventType.keyUp.rawValue) |
            (1 << CGEventType.flagsChanged.rawValue)
        )
        
        // Pass unmanaged self as userInfo
        let observer = Unmanaged.passUnretained(self).toOpaque()
        
        // Try active modifying tap first (.defaultTap) to suppress trigger keystrokes
        var tapPort = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: mask,
            callback: eventTapCallback,
            userInfo: observer
        )
        
        // If active tap fails (e.g. Accessibility not granted), try listen-only tap (.listenOnly)
        if tapPort == nil {
            print("[EventTap] Active tap unavailable. Attempting listen-only tap...")
            tapPort = CGEvent.tapCreate(
                tap: .cgSessionEventTap,
                place: .headInsertEventTap,
                options: .listenOnly,
                eventsOfInterest: mask,
                callback: eventTapCallback,
                userInfo: observer
            )
        }
        
        guard let validPort = tapPort else {
            print("[EventTap] Failed to create CGEventTap. Permission missing.")
            SafetyStateManager.shared.forceInactive(reason: "EventTap failed: check Accessibility / Input Monitoring")
            return false
        }
        
        self.machPort = validPort
        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, validPort, 0)
        self.runLoopSource = source
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: validPort, enable: true)
        self.isTapActive = true
        print("[EventTap] Global event tap installed and enabled.")
        return true
    }
    
    public func stop() {
        guard isTapActive else { return }
        if let port = machPort {
            CGEvent.tapEnable(tap: port, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        machPort = nil
        runLoopSource = nil
        isTapActive = false
        isTriggerHeld = false
        SafetyStateManager.shared.handleShortcutReleased()
        print("[EventTap] Event tap stopped.")
    }
    
    private func setupSleepWakeObservers() {
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.willSleepNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            print("[EventTap] System going to sleep. Forcing inactive.")
            self?.isTriggerHeld = false
            SafetyStateManager.shared.forceInactive(reason: "System sleep")
        }
        
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didWakeNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            print("[EventTap] System woke from sleep. Re-enabling tap.")
            if let port = self?.machPort {
                CGEvent.tapEnable(tap: port, enable: true)
            }
        }
        
        NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.screensDidSleepNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            print("[EventTap] Screen locked or display slept. Forcing inactive.")
            self?.isTriggerHeld = false
            SafetyStateManager.shared.forceInactive(reason: "Screen locked")
        }
    }
    
    // Internal callback handler
    fileprivate func handleEvent(proxy: CGEventTapProxy, type: CGEventType, event: CGEvent) -> Unmanaged<CGEvent>? {
        // 1. Handle OS disable notifications
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            print("[EventTap] Tap disabled by OS (\(type.rawValue)). Re-enabling...")
            if let port = machPort {
                CGEvent.tapEnable(tap: port, enable: true)
            }
            if isTriggerHeld {
                isTriggerHeld = false
                SafetyStateManager.shared.handleShortcutReleased()
            }
            return Unmanaged.passRetained(event)
        }
        
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
        let flags = event.flags
        
        // 2. Evaluate key match
        switch type {
        case .keyDown:
            if currentConfig.matches(keyCode: keyCode, flags: flags) {
                if !isTriggerHeld {
                    isTriggerHeld = true
                    // Lightweight, minimal state transition call
                    SafetyStateManager.shared.handleShortcutPressed()
                }
                // Suppress trigger key from reaching active app
                return nil
            } else if isTriggerHeld {
                // If another unrelated key was pressed while shortcut was held,
                // check if modifiers still match
                if !currentConfig.matches(keyCode: currentConfig.requiredKeyCode ?? 0, flags: flags) {
                    isTriggerHeld = false
                    SafetyStateManager.shared.handleShortcutReleased()
                }
            }
            
        case .keyUp:
            if let targetKey = currentConfig.requiredKeyCode, keyCode == targetKey {
                if isTriggerHeld {
                    isTriggerHeld = false
                    // Synchronously cut the gate immediately
                    SafetyStateManager.shared.handleShortcutReleased()
                }
                // Suppress trigger key-up
                return nil
            }
            
        case .flagsChanged:
            // Check modifier-only shortcuts or modifier release
            if currentConfig.requiredKeyCode == nil {
                // Modifier-only trigger
                let relevant = currentConfig.requiredModifiers
                let active = flags.contains(relevant)
                if active && !isTriggerHeld {
                    isTriggerHeld = true
                    SafetyStateManager.shared.handleShortcutPressed()
                } else if !active && isTriggerHeld {
                    isTriggerHeld = false
                    SafetyStateManager.shared.handleShortcutReleased()
                }
            } else {
                // Key + Modifier trigger: if modifier was released while key held, drop immediately!
                if isTriggerHeld {
                    let relevant = currentConfig.requiredModifiers
                    if !flags.contains(relevant) {
                        isTriggerHeld = false
                        SafetyStateManager.shared.handleShortcutReleased()
                    }
                }
            }
            
        default:
            break
        }
        
        return Unmanaged.passRetained(event)
    }
}

// C-convention callback for CGEventTapCreate
private func eventTapCallback(
    proxy: CGEventTapProxy,
    type: CGEventType,
    event: CGEvent,
    refcon: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    guard let refcon = refcon else {
        return Unmanaged.passRetained(event)
    }
    let tap = Unmanaged<GlobalEventTap>.fromOpaque(refcon).takeUnretainedValue()
    return tap.handleEvent(proxy: proxy, type: type, event: event)
}
