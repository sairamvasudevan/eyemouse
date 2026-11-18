import Foundation
import CoreGraphics

public struct ValidationMetrics: Codable {
    public let modelName: String
    public let totalValidationTargets: Int
    public let meanPixelError: Double
    public let medianPixelError: Double
    public let p95PixelError: Double
    public let maxPixelError: Double
    public let minPixelError: Double
    public let meanNormError: Double
    public let medianNormError: Double
    public let meanPixelErrorX: Double
    public let meanPixelErrorY: Double
    public let centerMeanPixelError: Double
    public let cornerMeanPixelError: Double
    public let edgeMeanPixelError: Double
    public let meanVisualDegrees: Double
    public let medianVisualDegrees: Double
    public let p95VisualDegrees: Double
    public let medianErrorPctDiagonal: Double
    public let p95ErrorPctDiagonal: Double
    public let qualityGrade: String
    public let isValidForUse: Bool
    public let problematicRegions: [String]
    public let targetResults: [TargetResultSummary]?
    
    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case totalValidationTargets = "total_validation_targets"
        case meanPixelError = "mean_pixel_error"
        case medianPixelError = "median_pixel_error"
        case p95PixelError = "p95_pixel_error"
        case maxPixelError = "max_pixel_error"
        case minPixelError = "min_pixel_error"
        case meanNormError = "mean_norm_error"
        case medianNormError = "median_norm_error"
        case meanPixelErrorX = "mean_pixel_error_x"
        case meanPixelErrorY = "mean_pixel_error_y"
        case centerMeanPixelError = "center_mean_pixel_error"
        case cornerMeanPixelError = "corner_mean_pixel_error"
        case edgeMeanPixelError = "edge_mean_pixel_error"
        case meanVisualDegrees = "mean_visual_degrees"
        case medianVisualDegrees = "median_visual_degrees"
        case p95VisualDegrees = "p95_visual_degrees"
        case medianErrorPctDiagonal = "median_error_pct_diagonal"
        case p95ErrorPctDiagonal = "p95_error_pct_diagonal"
        case qualityGrade = "quality_grade"
        case isValidForUse = "is_valid_for_use"
        case problematicRegions = "problematic_regions"
        case targetResults = "target_results"
    }
}

public struct TargetResultSummary: Codable {
    public let targetIndex: Int
    public let targetName: String
    public let groundTruthNorm: [Double]
    public let predictedNorm: [Double]
    public let errorPixels: Double
    public let errorPixelsX: Double
    public let errorPixelsY: Double
    public let errorVisualDegrees: Double
    
    enum CodingKeys: String, CodingKey {
        case targetIndex = "target_index"
        case targetName = "target_name"
        case groundTruthNorm = "ground_truth_norm"
        case predictedNorm = "predicted_norm"
        case errorPixels = "error_pixels"
        case errorPixelsX = "error_pixels_x"
        case errorPixelsY = "error_pixels_y"
        case errorVisualDegrees = "error_visual_degrees"
    }
}

public struct ModelDictionary: Codable {
    public let modelName: String
    public let alpha: Double
    public let weights: [[Double]]
    
    enum CodingKeys: String, CodingKey {
        case modelName = "model_name"
        case alpha
        case weights
    }
}

public struct CalibrationProfile: Codable {
    public let version: Int
    public let createdAtIso: String
    public let cameraUniqueID: String
    public let cameraName: String
    public let cameraResolution: String
    public let pixelFormat: String
    public let screenWidthPoints: Double
    public let screenHeightPoints: Double
    public let backingScaleFactor: Double
    public let modelName: String
    public let modelDict: ModelDictionary
    public let validationMetrics: ValidationMetrics
    public let qualityGrade: String
    public let isValid: Bool
    
    enum CodingKeys: String, CodingKey {
        case version
        case createdAtIso = "created_at_iso"
        case cameraUniqueID = "camera_unique_id"
        case cameraName = "camera_name"
        case cameraResolution = "camera_resolution"
        case pixelFormat = "pixel_format"
        case screenWidthPoints = "screen_width_points"
        case screenHeightPoints = "screen_height_points"
        case backingScaleFactor = "backing_scale_factor"
        case modelName = "model_name"
        case modelDict = "model_dict"
        case validationMetrics = "validation_metrics"
        case qualityGrade = "quality_grade"
        case isValid = "is_valid"
    }
    
    public static var defaultProfileURL: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".eyemouse/profiles/active_profile.json")
    }
    
    public static func loadDefault() -> CalibrationProfile? {
        let url = defaultProfileURL
        guard FileManager.default.fileExists(atPath: url.path) else { return nil }
        do {
            let data = try Data(contentsOf: url)
            let decoder = JSONDecoder()
            return try decoder.decode(CalibrationProfile.self, from: data)
        } catch {
            print("[CalibrationProfile] Failed to decode profile from \(url.path): \(error)")
            return nil
        }
    }
    
    public func save(to url: URL = defaultProfileURL) throws {
        let dir = url.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        let data = try encoder.encode(self)
        try data.write(to: url)
    }
    
    public func isCompatible(currentCameraID: String, screen: CalibrationGeometry) -> (Bool, String?) {
        guard isValid else { return (false, "Profile is marked invalid") }
        
        if !cameraUniqueID.isEmpty && !currentCameraID.isEmpty && cameraUniqueID != currentCameraID {
            return (false, "Camera mismatch: profile=\(cameraUniqueID), current=\(currentCameraID)")
        }
        
        if abs(screenWidthPoints - screen.widthPoints) > 10.0 {
            return (false, "Screen width changed: \(screenWidthPoints) vs \(screen.widthPoints)")
        }
        if abs(screenHeightPoints - screen.heightPoints) > 10.0 {
            return (false, "Screen height changed: \(screenHeightPoints) vs \(screen.heightPoints)")
        }
        if abs(backingScaleFactor - screen.backingScale) > 0.1 {
            return (false, "Display scale factor changed: \(backingScaleFactor) vs \(screen.backingScale)")
        }
        
        return (true, nil)
    }
}

