import Foundation
import os.lock

public enum SafetyState: String, Codable {
    case inactive = "INACTIVE"
    case starting = "STARTING"
    case active = "ACTIVE"
    case stopping = "STOPPING"
    case error = "ERROR"
    
    public var displayTitle: String {
        switch self {
        case .inactive: return "Idle (Hold shortcut to activate)"
        case .starting: return "Starting engine & camera..."
        case .active: return "Active [Tracking Live]"
        case .stopping: return "Stopping..."
        case .error: return "Error"
        }
    }
}

public final class SafetyStateManager {
    public static let shared = SafetyStateManager()
    
    private var lock = os_unfair_lock()
    private var _state: SafetyState = .inactive
    private var _isKeyHeld: Bool = false
    private var _errorMessage: String? = nil
    
    public var onStateChanged: ((SafetyState) -> Void)?
    
    private init() {}
    
    public var currentState: SafetyState {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _state
    }
    
    public var isKeyHeld: Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _isKeyHeld
    }
    
    public var errorMessage: String? {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return _errorMessage
    }
    
    /// Gate check for frame processing:
    /// Allowed ONLY if state is STARTING or ACTIVE AND key is physically held.
    public var canProcessFrames: Bool {
        os_unfair_lock_lock(&lock)
        defer { os_unfair_lock_unlock(&lock) }
        return (_state == .active || _state == .starting) && _isKeyHeld
    }
    
    /// Called immediately and synchronously from the CGEventTap callback on key press
    public func handleShortcutPressed() {
        os_unfair_lock_lock(&lock)
        _isKeyHeld = true
        if _state == .inactive || _state == .stopping {
            _state = .starting
            _errorMessage = nil
            let newState = _state
            os_unfair_lock_unlock(&lock)
            notifyStateChanged(newState)
            return
        }
        os_unfair_lock_unlock(&lock)
    }
    
    /// Called immediately and synchronously from the CGEventTap callback on ANY key release
    public func handleShortcutReleased() {
        os_unfair_lock_lock(&lock)
        _isKeyHeld = false
        // Immediate synchronous closure of frame gate
        if _state == .starting || _state == .active {
            _state = .stopping
            let newState = _state
            os_unfair_lock_unlock(&lock)
            notifyStateChanged(newState)
            return
        }
        os_unfair_lock_unlock(&lock)
    }
    
    /// Transition to active once camera and engine are confirmed ready
    public func markEngineAndCameraReady() {
        os_unfair_lock_lock(&lock)
        // If key was released while starting up, immediately abort
        if !_isKeyHeld {
            _state = .stopping
            let newState = _state
            os_unfair_lock_unlock(&lock)
            notifyStateChanged(newState)
            return
        }
        if _state == .starting {
            _state = .active
            let newState = _state
            os_unfair_lock_unlock(&lock)
            notifyStateChanged(newState)
            return
        }
        os_unfair_lock_unlock(&lock)
    }
    
    /// Transition to inactive once camera session stops
    public func markTeardownComplete() {
        os_unfair_lock_lock(&lock)
        if _state == .stopping {
            _state = .inactive
            let newState = _state
            os_unfair_lock_unlock(&lock)
            notifyStateChanged(newState)
            return
        }
        os_unfair_lock_unlock(&lock)
    }
    
    /// Force inactive on system sleep, screen lock, or unexpected tap disable
    public func forceInactive(reason: String? = nil) {
        os_unfair_lock_lock(&lock)
        _isKeyHeld = false
        _state = .inactive
        if let r = reason {
            _errorMessage = r
        }
        let newState = _state
        os_unfair_lock_unlock(&lock)
        notifyStateChanged(newState)
    }
    
    /// Set error state on engine crash, socket disconnect, or camera failure
    public func reportError(_ message: String) {
        os_unfair_lock_lock(&lock)
        _isKeyHeld = false
        _state = .error
        _errorMessage = message
        let newState = _state
        os_unfair_lock_unlock(&lock)
        notifyStateChanged(newState)
    }
    
    private func notifyStateChanged(_ newState: SafetyState) {
        DispatchQueue.main.async { [weak self] in
            self?.onStateChanged?(newState)
        }
    }
}

