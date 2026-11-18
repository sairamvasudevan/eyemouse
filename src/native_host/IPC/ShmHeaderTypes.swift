import Foundation

public enum ShmHeaderTypes {
    public static let magicGlobal: [UInt8] = [0x45, 0x4D, 0x53, 0x48] // "EMSH"
    public static let magicSlot: [UInt8]   = [0x45, 0x59, 0x45, 0x4D] // "EYEM"
    
    public static let slotStateFree: UInt32    = 0
    public static let slotStateWriting: UInt32 = 1
    public static let slotStateReady: UInt32   = 2
    public static let slotStateReading: UInt32 = 3
    
    public static let pixelFormatUnknown: UInt32 = 0
    public static let pixelFormatBGRA: UInt32    = 1
    public static let pixelFormatRGBA: UInt32    = 2
    public static let pixelFormatBGR: UInt32     = 3
    public static let pixelFormatRGB: UInt32     = 4
    public static let pixelFormatNV12: UInt32    = 5
}

