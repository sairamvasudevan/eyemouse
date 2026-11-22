import Foundation
import CoreGraphics

public struct CalibrationTarget: Identifiable, Codable {
    public let id: Int
    public let name: String
    public let normX: Double
    public let normY: Double
    public let isCenter: Bool
    public let isCorner: Bool
    public let isEdge: Bool
    
    public init(
        id: Int,
        name: String,
        normX: Double,
        normY: Double,
        isCenter: Bool = false,
        isCorner: Bool = false,
        isEdge: Bool = false
    ) {
        self.id = id
        self.name = name
        self.normX = normX
        self.normY = normY
        self.isCenter = isCenter
        self.isCorner = isCorner
        self.isEdge = isEdge
    }
    
    public func screenPoint(screen: CalibrationGeometry) -> CGPoint {
        return screen.normalizedToPoints(u: normX, v: normY)
    }
    
    public static func defaultThirteenPointGrid(margin: Double = 0.08) -> [CalibrationTarget] {
        let m = margin
        let cx = 0.5
        let cy = 0.5
        let left = m
        let right = 1.0 - m
        let top = m
        let bottom = 1.0 - m
        
        let qLeft = 0.25
        let qRight = 0.75
        let qTop = 0.25
        let qBottom = 0.75
        
        return [
            CalibrationTarget(id: 0, name: "Center", normX: cx, normY: cy, isCenter: true),
            CalibrationTarget(id: 1, name: "Top-Left", normX: left, normY: top, isCorner: true),
            CalibrationTarget(id: 2, name: "Top-Right", normX: right, normY: top, isCorner: true),
            CalibrationTarget(id: 3, name: "Bottom-Left", normX: left, normY: bottom, isCorner: true),
            CalibrationTarget(id: 4, name: "Bottom-Right", normX: right, normY: bottom, isCorner: true),
            CalibrationTarget(id: 5, name: "Top-Center", normX: cx, normY: top, isEdge: true),
            CalibrationTarget(id: 6, name: "Bottom-Center", normX: cx, normY: bottom, isEdge: true),
            CalibrationTarget(id: 7, name: "Left-Center", normX: left, normY: cy, isEdge: true),
            CalibrationTarget(id: 8, name: "Right-Center", normX: right, normY: cy, isEdge: true),
            CalibrationTarget(id: 9, name: "Inner-Top-Left", normX: qLeft, normY: qTop),
            CalibrationTarget(id: 10, name: "Inner-Top-Right", normX: qRight, normY: qTop),
            CalibrationTarget(id: 11, name: "Inner-Bottom-Left", normX: qLeft, normY: qBottom),
            CalibrationTarget(id: 12, name: "Inner-Bottom-Right", normX: qRight, normY: qBottom),
        ]
    }
    
    public static func defaultNinePointValidationGrid() -> [CalibrationTarget] {
        return [
            CalibrationTarget(id: 0, name: "Val-Top-Mid-Left", normX: 0.20, normY: 0.12),
            CalibrationTarget(id: 1, name: "Val-Top-Mid-Right", normX: 0.80, normY: 0.12),
            CalibrationTarget(id: 2, name: "Val-Bottom-Mid-Left", normX: 0.20, normY: 0.88),
            CalibrationTarget(id: 3, name: "Val-Bottom-Mid-Right", normX: 0.80, normY: 0.88),
            CalibrationTarget(id: 4, name: "Val-Mid-Left-Inner", normX: 0.35, normY: 0.50),
            CalibrationTarget(id: 5, name: "Val-Mid-Right-Inner", normX: 0.65, normY: 0.50),
            CalibrationTarget(id: 6, name: "Val-Mid-Top-Inner", normX: 0.50, normY: 0.30),
            CalibrationTarget(id: 7, name: "Val-Mid-Bottom-Inner", normX: 0.50, normY: 0.70),
            CalibrationTarget(id: 8, name: "Val-Center-Offset", normX: 0.40, normY: 0.40, isCenter: true),
        ]
    }
    
    public static func defaultThirtyTwoPointValidationGrid() -> [CalibrationTarget] {
        return [
            // Top Peripheral Zone (4 targets)
            CalibrationTarget(id: 0, name: "Val32-Top-Outer-Left", normX: 0.14, normY: 0.05, isCorner: true),
            CalibrationTarget(id: 1, name: "Val32-Top-Mid-Left", normX: 0.32, normY: 0.05, isEdge: true),
            CalibrationTarget(id: 2, name: "Val32-Top-Mid-Right", normX: 0.68, normY: 0.05, isEdge: true),
            CalibrationTarget(id: 3, name: "Val32-Top-Outer-Right", normX: 0.86, normY: 0.05, isCorner: true),
            
            // Upper Flanks (2 targets)
            CalibrationTarget(id: 4, name: "Val32-Upper-Edge-Left", normX: 0.06, normY: 0.20, isEdge: true),
            CalibrationTarget(id: 5, name: "Val32-Upper-Edge-Right", normX: 0.94, normY: 0.20, isEdge: true),

            // Upper-Mid Transition (4 targets)
            CalibrationTarget(id: 6, name: "Val32-UpMid-Left", normX: 0.18, normY: 0.32),
            CalibrationTarget(id: 7, name: "Val32-UpMid-CenterLeft", normX: 0.38, normY: 0.22, isCenter: true),
            CalibrationTarget(id: 8, name: "Val32-UpMid-CenterRight", normX: 0.62, normY: 0.22, isCenter: true),
            CalibrationTarget(id: 9, name: "Val32-UpMid-Right", normX: 0.82, normY: 0.32),

            // Mid Flanks (2 targets)
            CalibrationTarget(id: 10, name: "Val32-Mid-Edge-Left", normX: 0.06, normY: 0.38, isEdge: true),
            CalibrationTarget(id: 11, name: "Val32-Mid-Edge-Right", normX: 0.94, normY: 0.38, isEdge: true),

            // Central Primary Workspace (8 targets)
            CalibrationTarget(id: 12, name: "Val32-Center-NW", normX: 0.32, normY: 0.36, isCenter: true),
            CalibrationTarget(id: 13, name: "Val32-Center-NE", normX: 0.68, normY: 0.36, isCenter: true),
            CalibrationTarget(id: 14, name: "Val32-Center-West", normX: 0.42, normY: 0.50, isCenter: true),
            CalibrationTarget(id: 15, name: "Val32-Center-East", normX: 0.58, normY: 0.50, isCenter: true),
            CalibrationTarget(id: 16, name: "Val32-Center-SW", normX: 0.32, normY: 0.64, isCenter: true),
            CalibrationTarget(id: 17, name: "Val32-Center-SE", normX: 0.68, normY: 0.64, isCenter: true),
            CalibrationTarget(id: 18, name: "Val32-Center-North", normX: 0.50, normY: 0.42, isCenter: true),
            CalibrationTarget(id: 19, name: "Val32-Center-South", normX: 0.50, normY: 0.58, isCenter: true),

            // Lower-Mid Transition (4 targets)
            CalibrationTarget(id: 20, name: "Val32-LoMid-Left", normX: 0.18, normY: 0.68),
            CalibrationTarget(id: 21, name: "Val32-LoMid-CenterLeft", normX: 0.38, normY: 0.78, isCenter: true),
            CalibrationTarget(id: 22, name: "Val32-LoMid-CenterRight", normX: 0.62, normY: 0.78, isCenter: true),
            CalibrationTarget(id: 23, name: "Val32-LoMid-Right", normX: 0.82, normY: 0.68),

            // Low Flanks (2 targets)
            CalibrationTarget(id: 24, name: "Val32-Low-Edge-Left", normX: 0.06, normY: 0.62, isEdge: true),
            CalibrationTarget(id: 25, name: "Val32-Low-Edge-Right", normX: 0.94, normY: 0.62, isEdge: true),

            // Bottom Peripheral Zone (6 targets)
            CalibrationTarget(id: 26, name: "Val32-Bot-Outer-Left", normX: 0.14, normY: 0.95, isCorner: true),
            CalibrationTarget(id: 27, name: "Val32-Bot-Mid-Left", normX: 0.32, normY: 0.95, isEdge: true),
            CalibrationTarget(id: 28, name: "Val32-Bot-Mid-Right", normX: 0.68, normY: 0.95, isEdge: true),
            CalibrationTarget(id: 29, name: "Val32-Bot-Outer-Right", normX: 0.86, normY: 0.95, isCorner: true),
            CalibrationTarget(id: 30, name: "Val32-Bot-Edge-Left", normX: 0.06, normY: 0.80, isEdge: true),
            CalibrationTarget(id: 31, name: "Val32-Bot-Edge-Right", normX: 0.94, normY: 0.80, isEdge: true),
        ]
    }
    
    public static func defaultOnePointGrid() -> [CalibrationTarget] {
        return [
            CalibrationTarget(id: 0, name: "Screen Center", normX: 0.5, normY: 0.5, isCenter: true)
        ]
    }
    
    public static func defaultFivePointGrid(margin: Double = 0.15) -> [CalibrationTarget] {
        let m = margin
        return [
            CalibrationTarget(id: 0, name: "Screen Center", normX: 0.5, normY: 0.5, isCenter: true),
            CalibrationTarget(id: 1, name: "Top-Left", normX: m, normY: m, isCorner: true),
            CalibrationTarget(id: 2, name: "Top-Right", normX: 1.0 - m, normY: m, isCorner: true),
            CalibrationTarget(id: 3, name: "Bottom-Left", normX: m, normY: 1.0 - m, isCorner: true),
            CalibrationTarget(id: 4, name: "Bottom-Right", normX: 1.0 - m, normY: 1.0 - m, isCorner: true),
        ]
    }
    
    public static func defaultNinePointGrid(margin: Double = 0.12) -> [CalibrationTarget] {
        let m = margin
        let cx = 0.5
        let cy = 0.5
        return [
            CalibrationTarget(id: 0, name: "Screen Center", normX: cx, normY: cy, isCenter: true),
            CalibrationTarget(id: 1, name: "Top-Left", normX: m, normY: m, isCorner: true),
            CalibrationTarget(id: 2, name: "Top-Center", normX: cx, normY: m, isEdge: true),
            CalibrationTarget(id: 3, name: "Top-Right", normX: 1.0 - m, normY: m, isCorner: true),
            CalibrationTarget(id: 4, name: "Right-Center", normX: 1.0 - m, normY: cy, isEdge: true),
            CalibrationTarget(id: 5, name: "Bottom-Right", normX: 1.0 - m, normY: 1.0 - m, isCorner: true),
            CalibrationTarget(id: 6, name: "Bottom-Center", normX: cx, normY: 1.0 - m, isEdge: true),
            CalibrationTarget(id: 7, name: "Bottom-Left", normX: m, normY: 1.0 - m, isCorner: true),
            CalibrationTarget(id: 8, name: "Left-Center", normX: m, normY: cy, isEdge: true),
        ]
    }
}

