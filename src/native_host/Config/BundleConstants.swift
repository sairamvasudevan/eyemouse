import Foundation

public enum BundleConstants {
    public static let bundleIdentifier = "com.eyemouse.EyeMouse"
    public static let appName = "EyeMouse"
    public static let appVersion = "0.3.0"
    public static var projectRoot: String { FileManager.default.currentDirectoryPath }
    
    // UserDefaults keys
    public static let keySelectedCameraID = "com.eyemouse.selectedCameraUniqueID"
    public static let keyShortcutConfig = "com.eyemouse.shortcutConfiguration"
    
    // Protocol Constants matching M2
    public static let protocolVersion: UInt32 = 1
    public static let globalHeaderSize = 64
    public static let slotHeaderSize = 64
    public static let numSlots: UInt32 = 4
    public static let slotSize: UInt32 = 10 * 1024 * 1024 // 10 MiB (supports up to 1080p BGRA uncompressed)
    
    // macOS POSIX Shared Memory Limit
    public static let maxShmNameLength = 31
    
    public static func makeSocketPath(forPID pid: Int32 = getpid()) -> String {
        return "/tmp/eyemouse_ctrl_\(pid).sock"
    }
    
    public static func makeShmName(forPID pid: Int32 = getpid()) -> String {
        let ts = Int(Date().timeIntervalSince1970) % 100000
        let name = "/emt_\(pid)_\(ts)"
        // Enforce macOS PSHNAMLEN <= 31
        if name.count > maxShmNameLength {
            return String(name.prefix(maxShmNameLength))
        }
        return name
    }
}
