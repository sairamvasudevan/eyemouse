import Foundation
import AppKit

public struct CalibrationGeometry: Codable {
    public let widthPoints: Double
    public let heightPoints: Double
    public let backingScale: Double
    public let viewingDistanceCm: Double
    public let diagonalInches: Double
    
    public init(
        widthPoints: Double,
        heightPoints: Double,
        backingScale: Double,
        viewingDistanceCm: Double = 55.0,
        diagonalInches: Double = 13.6
    ) {
        self.widthPoints = widthPoints
        self.heightPoints = heightPoints
        self.backingScale = backingScale
        self.viewingDistanceCm = viewingDistanceCm
        self.diagonalInches = diagonalInches
    }
    
    public static var current: CalibrationGeometry {
        if let screen = NSScreen.main {
            return CalibrationGeometry(
                widthPoints: Double(screen.frame.width),
                heightPoints: Double(screen.frame.height),
                backingScale: Double(screen.backingScaleFactor),
                viewingDistanceCm: 55.0,
                diagonalInches: 13.6
            )
        }
        return CalibrationGeometry(
            widthPoints: 1470.0,
            heightPoints: 956.0,
            backingScale: 2.0
        )
    }
    
    public var widthPixels: Double { widthPoints * backingScale }
    public var heightPixels: Double { heightPoints * backingScale }
    
    public var diagonalPoints: Double {
        hypot(widthPoints, heightPoints)
    }
    
    public var diagonalPixels: Double {
        hypot(widthPixels, heightPixels)
    }
    
    public var ppi: Double {
        diagonalPixels / diagonalInches
    }
    
    public var pixelsPerCm: Double {
        ppi / 2.54
    }
    
    public func normalizedToPoints(u: Double, v: Double) -> CGPoint {
        let cu = min(max(u, 0.0), 1.0)
        let cv = min(max(v, 0.0), 1.0)
        return CGPoint(x: cu * widthPoints, y: cv * heightPoints)
    }
    
    public func pointsToNormalized(x: Double, y: Double) -> (u: Double, v: Double) {
        let u = min(max(x / widthPoints, 0.0), 1.0)
        let v = min(max(y / heightPoints, 0.0), 1.0)
        return (u, v)
    }
    
    public func pointsToCocoa(point: CGPoint) -> CGPoint {
        return CGPoint(x: point.x, y: CGFloat(heightPoints) - point.y)
    }
    
    public func cocoaToPoints(point: CGPoint) -> CGPoint {
        return CGPoint(x: point.x, y: CGFloat(heightPoints) - point.y)
    }
    
    public func pointsToPixels(point: CGPoint) -> CGPoint {
        return CGPoint(x: point.x * CGFloat(backingScale), y: point.y * CGFloat(backingScale))
    }
    
    public func pixelErrorToVisualDegrees(errorPixels: Double) -> Double {
        guard pixelsPerCm > 0, viewingDistanceCm > 0 else { return 0.0 }
        let errorCm = errorPixels / pixelsPerCm
        let angleRad = 2.0 * atan2(errorCm / 2.0, viewingDistanceCm)
        return angleRad * (180.0 / .pi)
    }
}

