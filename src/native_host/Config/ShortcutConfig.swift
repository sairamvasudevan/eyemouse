import Foundation
import CoreGraphics

public struct ShortcutConfig: Codable, Equatable {
    public var requiredKeyCode: UInt16?
    public var requiredModifiersRaw: UInt64
    public var name: String
    
    public init(requiredKeyCode: UInt16?, requiredModifiers: CGEventFlags, name: String = "Custom") {
        self.requiredKeyCode = requiredKeyCode
        // Strip off device-dependent flags from modifiers mask
        let mask = CGEventFlags([.maskControl, .maskAlternate, .maskShift, .maskCommand])
        self.requiredModifiersRaw = (requiredModifiers.rawValue & mask.rawValue)
        self.name = name
    }
    
    public var requiredModifiers: CGEventFlags {
        return CGEventFlags(rawValue: requiredModifiersRaw)
    }
    
    // Preset: Control + Option + E (Recommended Default)
    public static let defaultShortcut = ShortcutConfig(
        requiredKeyCode: KeyCodeTable.kVK_ANSI_E,
        requiredModifiers: [.maskControl, .maskAlternate],
        name: "Control + Option + E"
    )
    
    // Preset: Control + Option + M
    public static let controlOptionM = ShortcutConfig(
        requiredKeyCode: KeyCodeTable.kVK_ANSI_M,
        requiredModifiers: [.maskControl, .maskAlternate],
        name: "Control + Option + M"
    )
    
    // Preset: Right Option (Modifier-only hold)
    public static let rightOptionHold = ShortcutConfig(
        requiredKeyCode: KeyCodeTable.kVK_RightOption,
        requiredModifiers: [.maskAlternate],
        name: "Right Option"
    )
    
    public var displayString: String {
        var str = ""
        if requiredModifiers.contains(.maskControl) { str += "⌃" }
        if requiredModifiers.contains(.maskAlternate) { str += "⌥" }
        if requiredModifiers.contains(.maskShift) { str += "⇧" }
        if requiredModifiers.contains(.maskCommand) { str += "⌘" }
        if let kc = requiredKeyCode {
            str += KeyCodeTable.name(for: kc)
        }
        return str.isEmpty ? "None" : str
    }
    
    public func matches(keyCode: UInt16, flags: CGEventFlags) -> Bool {
        let relevantMask = CGEventFlags([.maskControl, .maskAlternate, .maskShift, .maskCommand])
        let actualModifiers = flags.rawValue & relevantMask.rawValue
        let expectedModifiers = requiredModifiersRaw & relevantMask.rawValue
        
        // Modifiers must match
        guard actualModifiers == expectedModifiers else {
            return false
        }
        
        // Key code must match if specified
        if let targetKey = requiredKeyCode {
            return keyCode == targetKey
        }
        return true
    }
    
    // Load from UserDefaults or return default
    public static func load() -> ShortcutConfig {
        guard let data = UserDefaults.standard.data(forKey: BundleConstants.keyShortcutConfig),
              let config = try? JSONDecoder().decode(ShortcutConfig.self, from: data) else {
            return .defaultShortcut
        }
        return config
    }
    
    public func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: BundleConstants.keyShortcutConfig)
        }
    }
}

