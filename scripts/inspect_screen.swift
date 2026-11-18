import Foundation
import AppKit

if let screen = NSScreen.main {
    print("Main Screen Information:")
    print("  Frame (Points): width=\(screen.frame.width), height=\(screen.frame.height), origin=(\(screen.frame.origin.x), \(screen.frame.origin.y))")
    print("  Visible Frame (Points): width=\(screen.visibleFrame.width), height=\(screen.visibleFrame.height), origin=(\(screen.visibleFrame.origin.x), \(screen.visibleFrame.origin.y))")
    print("  Backing Scale Factor: \(screen.backingScaleFactor)")
    let pixelWidth = screen.frame.width * screen.backingScaleFactor
    let pixelHeight = screen.frame.height * screen.backingScaleFactor
    print("  Physical Retina Resolution: \(pixelWidth) x \(pixelHeight)")
    let diagonalPoints = sqrt(pow(screen.frame.width, 2) + pow(screen.frame.height, 2))
    let diagonalPixels = sqrt(pow(pixelWidth, 2) + pow(pixelHeight, 2))
    print("  Diagonal (Points): \(String(format: "%.1f", diagonalPoints)) pt")
    print("  Diagonal (Pixels): \(String(format: "%.1f", diagonalPixels)) px")
} else {
    print("No main screen detected.")
}

