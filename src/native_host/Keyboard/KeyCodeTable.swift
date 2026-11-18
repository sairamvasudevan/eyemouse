import Foundation
import CoreGraphics

public enum KeyCodeTable {
    // Virtual Key Codes (from Carbon HIToolbox Events.h)
    public static let kVK_ANSI_A: UInt16 = 0x00
    public static let kVK_ANSI_S: UInt16 = 0x01
    public static let kVK_ANSI_D: UInt16 = 0x02
    public static let kVK_ANSI_F: UInt16 = 0x03
    public static let kVK_ANSI_H: UInt16 = 0x04
    public static let kVK_ANSI_G: UInt16 = 0x05
    public static let kVK_ANSI_Z: UInt16 = 0x06
    public static let kVK_ANSI_X: UInt16 = 0x07
    public static let kVK_ANSI_C: UInt16 = 0x08
    public static let kVK_ANSI_V: UInt16 = 0x09
    public static let kVK_ANSI_B: UInt16 = 0x0B
    public static let kVK_ANSI_Q: UInt16 = 0x0C
    public static let kVK_ANSI_W: UInt16 = 0x0D
    public static let kVK_ANSI_E: UInt16 = 0x0E
    public static let kVK_ANSI_R: UInt16 = 0x0F
    public static let kVK_ANSI_Y: UInt16 = 0x10
    public static let kVK_ANSI_T: UInt16 = 0x11
    public static let kVK_ANSI_1: UInt16 = 0x12
    public static let kVK_ANSI_2: UInt16 = 0x13
    public static let kVK_ANSI_3: UInt16 = 0x14
    public static let kVK_ANSI_4: UInt16 = 0x15
    public static let kVK_ANSI_6: UInt16 = 0x16
    public static let kVK_ANSI_5: UInt16 = 0x17
    public static let kVK_ANSI_Equal: UInt16 = 0x18
    public static let kVK_ANSI_9: UInt16 = 0x19
    public static let kVK_ANSI_7: UInt16 = 0x1A
    public static let kVK_ANSI_Minus: UInt16 = 0x1B
    public static let kVK_ANSI_8: UInt16 = 0x1C
    public static let kVK_ANSI_0: UInt16 = 0x1D
    public static let kVK_ANSI_RightBracket: UInt16 = 0x1E
    public static let kVK_ANSI_O: UInt16 = 0x1F
    public static let kVK_ANSI_U: UInt16 = 0x20
    public static let kVK_ANSI_LeftBracket: UInt16 = 0x21
    public static let kVK_ANSI_I: UInt16 = 0x22
    public static let kVK_ANSI_P: UInt16 = 0x23
    public static let kVK_ANSI_L: UInt16 = 0x25
    public static let kVK_ANSI_J: UInt16 = 0x26
    public static let kVK_ANSI_Quote: UInt16 = 0x27
    public static let kVK_ANSI_K: UInt16 = 0x28
    public static let kVK_ANSI_Semicolon: UInt16 = 0x29
    public static let kVK_ANSI_Backslash: UInt16 = 0x2A
    public static let kVK_ANSI_Comma: UInt16 = 0x2B
    public static let kVK_ANSI_Slash: UInt16 = 0x2C
    public static let kVK_ANSI_N: UInt16 = 0x2D
    public static let kVK_ANSI_M: UInt16 = 0x2E
    public static let kVK_ANSI_Period: UInt16 = 0x2F
    public static let kVK_ANSI_Grave: UInt16 = 0x32
    
    public static let kVK_Space: UInt16 = 0x31
    public static let kVK_Tab: UInt16 = 0x30
    public static let kVK_Return: UInt16 = 0x24
    public static let kVK_Escape: UInt16 = 0x35
    public static let kVK_Command: UInt16 = 0x37
    public static let kVK_Shift: UInt16 = 0x38
    public static let kVK_CapsLock: UInt16 = 0x39
    public static let kVK_Option: UInt16 = 0x3A
    public static let kVK_Control: UInt16 = 0x3B
    public static let kVK_RightShift: UInt16 = 0x3C
    public static let kVK_RightOption: UInt16 = 0x3D
    public static let kVK_RightControl: UInt16 = 0x3E
    public static let kVK_Function: UInt16 = 0x3F
    
    public static func name(for keyCode: UInt16) -> String {
        switch keyCode {
        case kVK_ANSI_A: return "A"
        case kVK_ANSI_B: return "B"
        case kVK_ANSI_C: return "C"
        case kVK_ANSI_D: return "D"
        case kVK_ANSI_E: return "E"
        case kVK_ANSI_F: return "F"
        case kVK_ANSI_G: return "G"
        case kVK_ANSI_H: return "H"
        case kVK_ANSI_I: return "I"
        case kVK_ANSI_J: return "J"
        case kVK_ANSI_K: return "K"
        case kVK_ANSI_L: return "L"
        case kVK_ANSI_M: return "M"
        case kVK_ANSI_N: return "N"
        case kVK_ANSI_O: return "O"
        case kVK_ANSI_P: return "P"
        case kVK_ANSI_Q: return "Q"
        case kVK_ANSI_R: return "R"
        case kVK_ANSI_S: return "S"
        case kVK_ANSI_T: return "T"
        case kVK_ANSI_U: return "U"
        case kVK_ANSI_V: return "V"
        case kVK_ANSI_W: return "W"
        case kVK_ANSI_X: return "X"
        case kVK_ANSI_Y: return "Y"
        case kVK_ANSI_Z: return "Z"
        case kVK_Space: return "Space"
        case kVK_Return: return "Return"
        case kVK_Escape: return "Esc"
        case kVK_RightOption: return "Right Option"
        case kVK_Function: return "Fn"
        default: return "Key(\(keyCode))"
        }
    }
}

