import Foundation
import Combine

public final class DiagnosticsStore: ObservableObject {
    public static let shared = DiagnosticsStore()
    
    @Published public var isCameraActive: Bool = false
    @Published public var faceDetected: Bool = false
    @Published public var confidence: Double = 0.0
    
    // Normalized Iris UV in local eye coordinate frame
    @Published public var leftIrisX: Double = 0.0
    @Published public var leftIrisY: Double = 0.0
    @Published public var rightIrisX: Double = 0.0
    @Published public var rightIrisY: Double = 0.0
    
    // Blink dynamics
    @Published public var blinkPhase: String = "OPEN"
    @Published public var leftBlinkScore: Double = 0.0
    @Published public var rightBlinkScore: Double = 0.0
    @Published public var blinkDurationMs: Double = 0.0
    @Published public var isSustainedBlink: Bool = false
    
    // 3D Head Pose
    @Published public var headYaw: Double = 0.0
    @Published public var headPitch: Double = 0.0
    @Published public var headRoll: Double = 0.0
    
    // Gaze State (Calibrated vs Optical Proxy)
    @Published public var isCalibrated: Bool = false
    @Published public var activeProfileName: String = "None"
    @Published public var activeQualityGrade: String = "Uncalibrated"
    @Published public var activeMedianErrorPixels: Double = 0.0
    @Published public var rawGazeX: Double = 0.0
    @Published public var rawGazeY: Double = 0.0
    @Published public var gazeX: Double = 0.0
    @Published public var gazeY: Double = 0.0
    @Published public var screenPoint: CGPoint = .zero
    @Published public var isGazeValid: Bool = false
    @Published public var showLiveGazePreview: Bool = false
    
    // Telemetry & Performance
    @Published public var cameraFps: Double = 0.0
    @Published public var engineFps: Double = 0.0
    @Published public var engineLatencyMs: Double = 0.0
    @Published public var p95LatencyMs: Double = 0.0
    @Published public var publishedFrames: Int = 0
    @Published public var droppedFrames: Int = 0
    
    // Engine process
    @Published public var enginePID: Int32? = nil
    @Published public var engineStatus: String = "Not Running"
    
    private var recentLatencies: [Double] = []
    private var fpsCounter = 0
    private var lastFpsTimestamp: TimeInterval = Date().timeIntervalSince1970
    
    private init() {}
    
    public func updateEngineTelemetry(data: [String: Any]) {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            
            if let face = data["face_detected"] as? Bool {
                self.faceDetected = face
            }
            if let conf = data["confidence"] as? Double {
                self.confidence = conf
            }
            if let lat = data["processing_time_ms"] as? Double {
                self.engineLatencyMs = lat
                self.recordLatency(lat)
            }
            
            // Gaze estimate
            if let gaze = data["gaze"] as? [String: Any] {
                self.rawGazeX = gaze["raw_x"] as? Double ?? 0.0
                self.rawGazeY = gaze["raw_y"] as? Double ?? 0.0
                self.gazeX = gaze["filtered_x"] as? Double ?? 0.0
                self.gazeY = gaze["filtered_y"] as? Double ?? 0.0
                self.isGazeValid = gaze["is_valid"] as? Bool ?? false
                if let cal = gaze["is_calibrated"] as? Bool {
                    self.isCalibrated = cal
                }
                
                // Convert normalized gaze to logical screen coordinates
                if self.isCalibrated && self.isGazeValid {
                    let geom = CalibrationGeometry.current
                    self.screenPoint = geom.normalizedToPoints(u: self.gazeX, v: self.gazeY)
                }
            } else {
                self.isGazeValid = false
            }
            
            // Raw Eye Features
            if let eye = data["eye_features"] as? [String: Any] {
                self.leftIrisX = eye["left_x"] as? Double ?? self.leftIrisX
                self.leftIrisY = eye["left_y"] as? Double ?? self.leftIrisY
                self.rightIrisX = eye["right_x"] as? Double ?? self.rightIrisX
                self.rightIrisY = eye["right_y"] as? Double ?? self.rightIrisY
            }
            
            // Blink event
            if let blink = data["blink"] as? [String: Any] {
                self.blinkPhase = (blink["phase"] as? String)?.uppercased() ?? "OPEN"
                self.leftBlinkScore = blink["left"] as? Double ?? 0.0
                self.rightBlinkScore = blink["right"] as? Double ?? 0.0
                self.blinkDurationMs = blink["duration_ms"] as? Double ?? 0.0
                self.isSustainedBlink = blink["is_sustained"] as? Bool ?? false
            }
            
            // Head pose
            if let pose = data["head_pose"] as? [String: Any] {
                self.headYaw = pose["yaw"] as? Double ?? 0.0
                self.headPitch = pose["pitch"] as? Double ?? 0.0
                self.headRoll = pose["roll"] as? Double ?? 0.0
            }
            
            // FPS computation
            self.fpsCounter += 1
            let now = Date().timeIntervalSince1970
            let dt = now - self.lastFpsTimestamp
            if dt >= 1.0 {
                self.engineFps = Double(self.fpsCounter) / dt
                self.fpsCounter = 0
                self.lastFpsTimestamp = now
            }
        }
    }
    
    private func recordLatency(_ lat: Double) {
        recentLatencies.append(lat)
        if recentLatencies.count > 100 {
            recentLatencies.removeFirst()
        }
        let sorted = recentLatencies.sorted()
        let p95Idx = Int(Double(sorted.count) * 0.95)
        if p95Idx < sorted.count {
            p95LatencyMs = sorted[p95Idx]
        }
    }
    
    public func loadActiveProfile() {
        if let profile = CalibrationProfile.loadDefault() {
            let geom = CalibrationGeometry.current
            let (compat, reason) = profile.isCompatible(currentCameraID: CameraManager.shared.selectedDevice?.id ?? "", screen: geom)
            if compat {
                self.isCalibrated = true
                self.activeProfileName = profile.modelName
                self.activeQualityGrade = profile.qualityGrade
                self.activeMedianErrorPixels = profile.validationMetrics.medianPixelError
                print("[DiagnosticsStore] Loaded active calibration profile: \(profile.modelName), Quality: \(profile.qualityGrade), Median error: \(profile.validationMetrics.medianPixelError)px")
            } else {
                print("[DiagnosticsStore] Active profile incompatible: \(reason ?? "unknown")")
                self.isCalibrated = false
                self.activeQualityGrade = "Incompatible"
            }
        } else {
            self.isCalibrated = false
            self.activeQualityGrade = "Uncalibrated"
        }
    }
    
    public func resetTrackingState() {
        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.faceDetected = false
            self.confidence = 0.0
            self.isGazeValid = false
            self.blinkPhase = "OPEN"
            self.isCameraActive = false
        }
    }
}

