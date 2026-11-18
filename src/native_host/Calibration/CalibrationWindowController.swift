import Foundation
import AppKit
import SwiftUI

public final class CalibrationWindowController: ObservableObject {
    public static let shared = CalibrationWindowController()
    
    private var window: NSWindow?
    
    @Published public var state: SessionState = .ready
    @Published public var currentTargetIndex: Int = 0
    @Published public var totalTargets: Int = 13
    @Published public var currentTarget: CalibrationTarget?
    @Published public var fixationProgress: Double = 0.0
    @Published public var statusMessage: String = "Keep your head steady and fixate on each target."
    @Published public var validationReport: ValidationMetrics?
    @Published public var selectedModelName: String = ""
    
    public enum SessionState {
        case ready
        case calibrating
        case validating
        case processing
        case completed
        case cancelled
    }
    
    private var calibTargets: [CalibrationTarget] = []
    private var valTargets: [CalibrationTarget] = []
    private var calibAggregates: [[String: Any]] = []
    private var valAggregates: [[String: Any]] = []
    
    private var currentSamples: [[Double]] = []
    private var targetStartTime: TimeInterval = 0
    private var timer: Timer?
    
    private init() {}
    
    private func resetSession() {
        self.state = .ready
        self.currentTargetIndex = 0
        self.currentTarget = nil
        self.fixationProgress = 0.0
        self.statusMessage = "Keep your head steady and fixate on each target."
        self.validationReport = nil
        self.calibAggregates.clear()
        self.valAggregates.clear()
    }
    
    public func show() {
        guard window == nil else {
            window?.makeKeyAndOrderFront(nil)
            return
        }
        guard let screen = NSScreen.main else { return }
        
        resetSession()
        
        let win = NSWindow(
            contentRect: screen.frame,
            styleMask: [.borderless],
            backing: .buffered,
            defer: false,
            screen: screen
        )
        win.isOpaque = true
        win.backgroundColor = NSColor(white: 0.05, alpha: 0.98)
        win.level = .floating
        win.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        
        let view = CalibrationView(controller: self)
        win.contentView = NSHostingView(rootView: view)
        win.makeKeyAndOrderFront(nil)
        self.window = win
        
        // Start engine and camera session
        DispatchQueue.global(qos: .userInitiated).async {
            if !EngineProcessManager.shared.isRunning {
                _ = EngineProcessManager.shared.startEngine()
            }
            CameraCaptureService.shared.startCapture()
        }
    }
    
    public func close() {
        timer?.invalidate()
        timer = nil
        window?.orderOut(nil)
        window = nil
        self.state = .ready
        
        // If not actively controlled by hold-key, stop camera
        if !SafetyStateManager.shared.canProcessFrames {
            CameraCaptureService.shared.stopCapture()
        }
    }
    
    public func startCalibration() {
        self.calibTargets = CalibrationTarget.defaultThirteenPointGrid()
        self.valTargets = CalibrationTarget.defaultNinePointValidationGrid()
        self.calibAggregates.clear()
        self.valAggregates.clear()
        
        self.state = .calibrating
        self.totalTargets = calibTargets.count
        self.currentTargetIndex = 0
        presentTarget(calibTargets[0])
    }
    
    private func presentTarget(_ target: CalibrationTarget) {
        self.currentTarget = target
        self.currentSamples.clear()
        self.targetStartTime = Date().timeIntervalSince1970
        self.fixationProgress = 0.0
        
        timer?.invalidate()
        let interval = 0.033 // ~30 Hz update
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.tickTarget()
        }
    }
    
    private func tickTarget() {
        guard let target = currentTarget else { return }
        let now = Date().timeIntervalSince1970
        let elapsed = now - targetStartTime
        let settling = 0.4
        let fixation = 1.0
        let total = settling + fixation
        
        self.fixationProgress = min(max((elapsed - settling) / fixation, 0.0), 1.0)
        
        // Sample collection during fixation window
        if elapsed >= settling && elapsed <= total {
            let diag = DiagnosticsStore.shared
            if diag.faceDetected && diag.blinkPhase == "OPEN" && diag.confidence >= 0.60 {
                // Feature vector: [left_x, left_y, right_x, right_y, yaw, pitch, roll]
                let sample: [Double] = [
                    diag.leftIrisX, diag.leftIrisY,
                    diag.rightIrisX, diag.rightIrisY,
                    diag.headYaw, diag.headPitch, diag.headRoll
                ]
                currentSamples.append(sample)
            }
        }
        
        // Target complete
        if elapsed >= total {
            timer?.invalidate()
            timer = nil
            finishCurrentTarget(target)
        }
    }
    
    private func finishCurrentTarget(_ target: CalibrationTarget) {
        // Robust median aggregation
        let validSamples = currentSamples.count
        var medianVec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        if validSamples >= 8 {
            for dim in 0..<7 {
                let vals = currentSamples.map { $0[dim] }.sorted()
                medianVec[dim] = vals[vals.count / 2]
            }
        } else {
            // Insufficient samples fallback to latest store values
            let diag = DiagnosticsStore.shared
            medianVec = [
                diag.leftIrisX, diag.leftIrisY,
                diag.rightIrisX, diag.rightIrisY,
                diag.headYaw, diag.headPitch, diag.headRoll
            ]
        }
        
        let agg: [String: Any] = [
            "target_index": target.id,
            "target_name": target.name,
            "target_norm_x": target.normX,
            "target_norm_y": target.normY,
            "feature_median": medianVec,
            "total_samples": currentSamples.count,
            "valid_samples": validSamples,
            "is_valid": true
        ]
        
        if state == .calibrating {
            calibAggregates.append(agg)
            let nextIdx = currentTargetIndex + 1
            if nextIdx < calibTargets.count {
                currentTargetIndex = nextIdx
                presentTarget(calibTargets[nextIdx])
            } else {
                // Switch to validation phase
                state = .validating
                totalTargets = valTargets.count
                currentTargetIndex = 0
                statusMessage = "Calibration complete. Now collecting 9 validation points..."
                presentTarget(valTargets[0])
            }
        } else if state == .validating {
            valAggregates.append(agg)
            let nextIdx = currentTargetIndex + 1
            if nextIdx < valTargets.count {
                currentTargetIndex = nextIdx
                presentTarget(valTargets[nextIdx])
            } else {
                // Finished both phases -> run training
                state = .processing
                statusMessage = "Training and cross-validating models..."
                runModelTrainingAndValidation()
            }
        }
    }
    
    private func runModelTrainingAndValidation() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self = self else { return }
            let geom = CalibrationGeometry.current
            let cameraID = CameraManager.shared.selectedDevice?.id ?? "MacBook Air Camera"
            
            let sessionPayload: [String: Any] = [
                "screen": [
                    "width_points": geom.widthPoints,
                    "height_points": geom.heightPoints,
                    "backing_scale": geom.backingScale
                ],
                "calibration_targets": self.calibAggregates,
                "validation_targets": self.valAggregates
            ]
            
            let tmpSessionFile = "/tmp/eyemouse_calibration_session.json"
            let tmpReportFile = "/tmp/eyemouse_validation_report.json"
            
            do {
                let data = try JSONSerialization.data(withJSONObject: sessionPayload, options: .prettyPrinted)
                try data.write(to: URL(fileURLWithPath: tmpSessionFile))
                
                // Execute Python training CLI script
                let pythonPath = BundleConstants.projectRoot + "/.venv/bin/python"
                let scriptPath = BundleConstants.projectRoot + "/scripts/train_and_validate_calibration.py"
                
                let process = Process()
                process.executableURL = URL(fileURLWithPath: pythonPath)
                var env = ProcessInfo.processInfo.environment
                env["PYTHONPATH"] = BundleConstants.projectRoot
                process.environment = env
                process.arguments = [
                    scriptPath,
                    "--input", tmpSessionFile,
                    "--output-report", tmpReportFile,
                    "--camera-id", cameraID
                ]
                
                try process.run()
                process.waitUntilExit()
                
                if process.terminationStatus == 0 {
                    let reportData = try Data(contentsOf: URL(fileURLWithPath: tmpReportFile))
                    if let scorecard = try JSONSerialization.jsonObject(with: reportData) as? [String: Any],
                       let reportDict = scorecard["report"] as? [String: Any] {
                        
                        let reportJSON = try JSONSerialization.data(withJSONObject: reportDict)
                        let parsedReport = try JSONDecoder().decode(ValidationMetrics.self, from: reportJSON)
                        
                        DispatchQueue.main.async {
                            self.validationReport = parsedReport
                            self.selectedModelName = parsedReport.modelName
                            self.state = .completed
                            
                            // Reload active profile in store
                            DiagnosticsStore.shared.loadActiveProfile()
                        }
                    }
                } else {
                    DispatchQueue.main.async {
                        self.statusMessage = "Model training failed (exit code \(process.terminationStatus))."
                        self.state = .ready
                    }
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusMessage = "Training error: \(error.localizedDescription)"
                    self.state = .ready
                }
            }
        }
    }
}

fileprivate extension Array {
    mutating func clear() {
        self.removeAll()
    }
}

struct CalibrationView: View {
    @ObservedObject var controller: CalibrationWindowController
    
    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.edgesIgnoringSafeArea(.all)
                
                switch controller.state {
                case .ready:
                    ReadyIntroView(controller: controller)
                    
                case .calibrating, .validating:
                    TargetPresentationView(controller: controller, screenBounds: geo.size)
                    
                case .processing:
                    VStack(spacing: 20) {
                        ProgressView()
                            .scaleEffect(1.5)
                        Text(controller.statusMessage)
                            .font(.system(size: 16, weight: .medium, design: .monospaced))
                            .foregroundColor(.white)
                    }
                    
                case .completed:
                    if let report = controller.validationReport {
                        ValidationScorecardView(report: report, controller: controller)
                    }
                    
                case .cancelled:
                    VStack(spacing: 16) {
                        Text("Calibration Cancelled")
                            .font(.headline)
                            .foregroundColor(.white)
                        Button("Close") { controller.close() }
                    }
                }
                
                // Top control bar
                VStack {
                    HStack {
                        Button(action: { controller.close() }) {
                            Text("ESC or Cancel")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundColor(.white.opacity(0.8))
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color.white.opacity(0.15))
                                .cornerRadius(6)
                        }
                        .buttonStyle(PlainButtonStyle())
                        
                        Spacer()
                        
                        if controller.state == .calibrating || controller.state == .validating {
                            let phaseName = controller.state == .calibrating ? "CALIBRATING" : "VALIDATING"
                            Text("\(phaseName): Target \(controller.currentTargetIndex + 1) of \(controller.totalTargets)")
                                .font(.system(size: 13, weight: .bold, design: .monospaced))
                                .foregroundColor(.cyan)
                        }
                    }
                    .padding(20)
                    Spacer()
                }
            }
        }
    }
}

struct ReadyIntroView: View {
    @ObservedObject var controller: CalibrationWindowController
    
    var body: some View {
        VStack(spacing: 24) {
            Text("EyeMouse Gaze Calibration")
                .font(.system(size: 28, weight: .bold))
                .foregroundColor(.white)
            
            VStack(alignment: .leading, spacing: 12) {
                InstructionRow(icon: "1.circle.fill", text: "Sit comfortably ~55 cm from the screen with normal head posture.")
                InstructionRow(icon: "2.circle.fill", text: "13 calibration points and 9 validation points will appear sequentially.")
                InstructionRow(icon: "3.circle.fill", text: "Focus your eyes directly on each pulsing dot until it completes.")
                InstructionRow(icon: "4.circle.fill", text: "Independent held-out targets will objectively validate accuracy.")
                InstructionRow(icon: "shield.fill", text: "No OS mouse movement or cursor injection will occur.")
            }
            .frame(maxWidth: 550)
            .padding(24)
            .background(Color.white.opacity(0.06))
            .cornerRadius(12)
            
            Button(action: { controller.startCalibration() }) {
                Text("Start Calibration")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundColor(.black)
                    .frame(width: 220, height: 44)
                    .background(Color.cyan)
                    .cornerRadius(8)
            }
            .buttonStyle(PlainButtonStyle())
        }
    }
}

struct InstructionRow: View {
    let icon: String
    let text: String
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: icon)
                .font(.system(size: 18))
                .foregroundColor(.cyan)
                .frame(width: 24)
            Text(text)
                .font(.system(size: 14))
                .foregroundColor(.white.opacity(0.9))
        }
    }
}

struct TargetPresentationView: View {
    @ObservedObject var controller: CalibrationWindowController
    let screenBounds: CGSize
    
    var body: some View {
        if let target = controller.currentTarget {
            let px = target.normX * Double(screenBounds.width)
            let py = target.normY * Double(screenBounds.height)
            let isVal = controller.state == .validating
            let color = isVal ? Color.orange : Color.cyan
            
            ZStack {
                // Shrinking outer countdown ring
                Circle()
                    .stroke(color.opacity(0.4), lineWidth: 3)
                    .frame(width: 60 * (1.0 - 0.5 * controller.fixationProgress),
                           height: 60 * (1.0 - 0.5 * controller.fixationProgress))
                
                // Steady center target
                Circle()
                    .fill(color)
                    .frame(width: 14, height: 14)
                    .shadow(color: color, radius: 8)
                
                // Inner white pip
                Circle()
                    .fill(Color.white)
                    .frame(width: 4, height: 4)
            }
            .position(x: CGFloat(px), y: CGFloat(py))
            .animation(.easeInOut(duration: 0.1), value: controller.fixationProgress)
        }
    }
}

struct ValidationScorecardView: View {
    let report: ValidationMetrics
    @ObservedObject var controller: CalibrationWindowController
    
    var body: some View {
        VStack(spacing: 24) {
            HStack(spacing: 12) {
                Image(systemName: report.isValidForUse ? "checkmark.seal.fill" : "exclamationmark.triangle.fill")
                    .font(.system(size: 32))
                    .foregroundColor(report.isValidForUse ? .green : .orange)
                
                VStack(alignment: .leading) {
                    Text("Calibration Quality: \(report.qualityGrade)")
                        .font(.system(size: 24, weight: .bold))
                        .foregroundColor(.white)
                    Text("Validated on 9 independent held-out screen positions")
                        .font(.system(size: 13))
                        .foregroundColor(.white.opacity(0.7))
                }
            }
            
            // Metrics grid
            VStack(spacing: 12) {
                MetricScoreRow(title: "Median Pixel Error:", value: "\(String(format: "%.1f", report.medianPixelError)) px (\(String(format: "%.2f", report.medianErrorPctDiagonal))% diag)")
                MetricScoreRow(title: "95th Percentile (P95) Error:", value: "\(String(format: "%.1f", report.p95PixelError)) px (\(String(format: "%.2f", report.p95ErrorPctDiagonal))% diag)")
                MetricScoreRow(title: "Mean Visual Degree Error:", value: "\(String(format: "%.2f", report.meanVisualDegrees))° visual angle")
                MetricScoreRow(title: "Center vs Corner Accuracy:", value: "Center: \(String(format: "%.1f", report.centerMeanPixelError)) px | Corner: \(String(format: "%.1f", report.cornerMeanPixelError)) px")
                MetricScoreRow(title: "Selected Model Architecture:", value: report.modelName)
            }
            .frame(maxWidth: 500)
            .padding(20)
            .background(Color.white.opacity(0.08))
            .cornerRadius(12)
            
            HStack(spacing: 20) {
                Button("Recalibrate") {
                    controller.startCalibration()
                }
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.white)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.15))
                .cornerRadius(8)
                
                Button("Save & Enable Live Preview") {
                    LiveGazePreviewController.shared.show()
                    controller.close()
                }
                .font(.system(size: 14, weight: .bold))
                .foregroundColor(.black)
                .padding(.horizontal, 24)
                .padding(.vertical, 10)
                .background(Color.cyan)
                .cornerRadius(8)
            }
            .buttonStyle(PlainButtonStyle())
        }
    }
}

struct MetricScoreRow: View {
    let title: String
    let value: String
    
    var body: some View {
        HStack {
            Text(title)
                .font(.system(size: 13))
                .foregroundColor(.white.opacity(0.8))
            Spacer()
            Text(value)
                .font(.system(size: 13, weight: .semibold, design: .monospaced))
                .foregroundColor(.cyan)
        }
    }
}
