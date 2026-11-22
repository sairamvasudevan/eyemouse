import Foundation
import AppKit
import SwiftUI

public final class CalibrationFullscreenWindow: NSWindow {
    public override var canBecomeKey: Bool { true }
    public override var canBecomeMain: Bool { true }
}

public final class CalibrationWindowController: ObservableObject {
    public static let shared = CalibrationWindowController()
    
    private var window: NSWindow?
    private var localKeyMonitor: Any?
    
    @Published public var state: SessionState = .ready
    @Published public var isInteractiveMode: Bool = true
    @Published public var currentTargetIndex: Int = 0
    @Published public var totalTargets: Int = 5
    @Published public var currentTarget: CalibrationTarget?
    @Published public var fixationProgress: Double = 0.0
    @Published public var isConfirmedFeedback: Bool = false
    @Published public var statusMessage: String = "Look directly at the dot, then press SPACEBAR to capture your eye position."
    @Published public var validationReport: ValidationMetrics?
    @Published public var selectedModelName: String = "adaptive_online_ridge"
    @Published public var pointsAdaptedCount: Int = 0
    
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
        self.isConfirmedFeedback = false
        self.statusMessage = "Look directly at the dot, then press SPACEBAR to capture your eye position."
        self.validationReport = nil
        self.pointsAdaptedCount = 0
        self.calibAggregates.removeAll()
        self.valAggregates.removeAll()
        self.currentSamples.removeAll()
    }
    
    public func show() {
        guard window == nil else {
            window?.makeKeyAndOrderFront(nil)
            return
        }
        guard let screen = NSScreen.main else { return }
        
        resetSession()
        
        let win = CalibrationFullscreenWindow(
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
        
        // Install key monitor for Spacebar, Enter, Escape, and number shortcuts
        installKeyMonitor()
        
        // Start engine and camera session
        DispatchQueue.global(qos: .userInitiated).async {
            if !EngineProcessManager.shared.isRunning {
                _ = EngineProcessManager.shared.startEngine()
            }
            CameraCaptureService.shared.startCapture()
        }
    }
    
    public func close() {
        removeKeyMonitor()
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
    
    private func installKeyMonitor() {
        removeKeyMonitor()
        localKeyMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self = self else { return event }
            
            // Escape (53): Cancel / Close
            if event.keyCode == 53 {
                self.close()
                return nil
            }
            
            // Spacebar (49) or Return (36)
            if event.keyCode == 49 || event.keyCode == 36 {
                if self.state == .calibrating || self.state == .validating {
                    self.confirmCurrentPoint()
                    return nil
                } else if self.state == .ready {
                    self.startAdaptiveCalibration(pointCount: 5)
                    return nil
                }
            }
            
            // Shortcuts while ready:
            if self.state == .ready {
                if event.characters == "1" || event.characters?.uppercased() == "C" {
                    self.startQuickOnePoint()
                    return nil
                } else if event.characters == "5" {
                    self.startAdaptiveCalibration(pointCount: 5)
                    return nil
                } else if event.characters == "9" {
                    self.startAdaptiveCalibration(pointCount: 9)
                    return nil
                }
            }
            
            return event
        }
    }
    
    private func removeKeyMonitor() {
        if let monitor = localKeyMonitor {
            NSEvent.removeMonitor(monitor)
            localKeyMonitor = nil
        }
    }
    
    // MARK: - Calibration Launchers
    
    public func startQuickOnePoint() {
        self.isInteractiveMode = true
        self.calibTargets = CalibrationTarget.defaultOnePointGrid()
        self.valTargets = []
        self.totalTargets = calibTargets.count
        self.currentTargetIndex = 0
        self.state = .calibrating
        self.statusMessage = "Look directly at the center dot, then press SPACEBAR to capture your eye position."
        presentTarget(calibTargets[0])
    }
    
    public func startAdaptiveCalibration(pointCount: Int = 5) {
        self.isInteractiveMode = true
        if pointCount == 1 {
            startQuickOnePoint()
            return
        } else if pointCount == 9 {
            self.calibTargets = CalibrationTarget.defaultNinePointGrid()
        } else {
            self.calibTargets = CalibrationTarget.defaultFivePointGrid()
        }
        self.valTargets = []
        self.totalTargets = calibTargets.count
        self.currentTargetIndex = 0
        self.state = .calibrating
        self.statusMessage = "Look directly at the dot, then press SPACEBAR to capture your eye position."
        presentTarget(calibTargets[0])
    }
    
    public func startBatchCalibration() {
        self.isInteractiveMode = false
        self.calibTargets = CalibrationTarget.defaultThirteenPointGrid()
        self.valTargets = CalibrationTarget.defaultNinePointValidationGrid()
        self.calibAggregates.removeAll()
        self.valAggregates.removeAll()
        
        self.state = .calibrating
        self.totalTargets = calibTargets.count
        self.currentTargetIndex = 0
        self.statusMessage = "Fixate on each target until the countdown ring shrinks."
        presentTarget(calibTargets[0])
    }
    
    public func resetAdaptationToBasePrior() {
        if let client = EngineProcessManager.shared.getControlClient() {
            _ = client.sendResetAdaptation()
        }
        DiagnosticsStore.shared.loadActiveProfile()
        self.statusMessage = "Gaze model reset to base physiological prior."
    }
    
    // MARK: - Target Presentation & Sampling
    
    private func presentTarget(_ target: CalibrationTarget) {
        self.currentTarget = target
        self.currentSamples.removeAll()
        self.targetStartTime = Date().timeIntervalSince1970
        self.fixationProgress = 0.0
        self.isConfirmedFeedback = false
        
        timer?.invalidate()
        let interval = 0.033 // ~30 Hz sampling
        timer = Timer.scheduledTimer(withTimeInterval: interval, repeats: true) { [weak self] _ in
            self?.tickSampling()
        }
    }
    
    private func tickSampling() {
        guard let target = currentTarget else { return }
        let now = Date().timeIntervalSince1970
        let elapsed = now - targetStartTime
        
        let diag = DiagnosticsStore.shared
        if diag.faceDetected && diag.blinkPhase == "OPEN" && diag.confidence >= 0.50 {
            let sample: [Double] = [
                diag.leftIrisX, diag.leftIrisY,
                diag.rightIrisX, diag.rightIrisY,
                diag.headYaw, diag.headPitch, diag.headRoll
            ]
            currentSamples.append(sample)
            // Keep rolling buffer of recent ~20 samples (~0.6 sec)
            if currentSamples.count > 25 {
                currentSamples.removeFirst()
            }
        }
        
        if !isInteractiveMode {
            // Automatic countdown timer mode
            let settling = 0.4
            let fixation = 1.0
            let total = settling + fixation
            self.fixationProgress = min(max((elapsed - settling) / fixation, 0.0), 1.0)
            if elapsed >= total {
                timer?.invalidate()
                timer = nil
                finishBatchCurrentTarget(target)
            }
        }
    }
    
    // MARK: - Interactive Keypress Confirmation
    
    public func confirmCurrentPoint() {
        guard let target = currentTarget, (state == .calibrating || state == .validating) else { return }
        
        // Calculate median feature vector from recent samples
        var medianVec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if currentSamples.count >= 3 {
            for dim in 0..<7 {
                let vals = currentSamples.map { $0[dim] }.sorted()
                medianVec[dim] = vals[vals.count / 2]
            }
        } else {
            let diag = DiagnosticsStore.shared
            medianVec = [
                diag.leftIrisX, diag.leftIrisY,
                diag.rightIrisX, diag.rightIrisY,
                diag.headYaw, diag.headPitch, diag.headRoll
            ]
        }
        
        // Send online adaptation point to Python vision engine via IPC
        if let client = EngineProcessManager.shared.getControlClient() {
            let ok = client.sendAdaptPoint(
                features: medianVec,
                targetNormX: target.normX,
                targetNormY: target.normY,
                weight: 1.0
            )
            print("[Calibration] Adapted point \(target.name) at (\(target.normX), \(target.normY)): \(ok)")
        }
        
        // Visual confirmation pulse
        self.isConfirmedFeedback = true
        self.pointsAdaptedCount += 1
        
        // Short pause to show green confirmation ring, then advance to next target
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) { [weak self] in
            guard let self = self else { return }
            self.isConfirmedFeedback = false
            
            let nextIdx = self.currentTargetIndex + 1
            if nextIdx < self.calibTargets.count {
                self.currentTargetIndex = nextIdx
                self.presentTarget(self.calibTargets[nextIdx])
            } else {
                // All targets confirmed!
                self.timer?.invalidate()
                self.timer = nil
                self.state = .completed
                self.currentTarget = nil
                self.statusMessage = "Self-adaptation complete! Gaze model adapted online."
                DiagnosticsStore.shared.loadActiveProfile()
            }
        }
    }
    
    // MARK: - Batch Mode Completion
    
    private func finishBatchCurrentTarget(_ target: CalibrationTarget) {
        let validSamples = currentSamples.count
        var medianVec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if validSamples >= 5 {
            for dim in 0..<7 {
                let vals = currentSamples.map { $0[dim] }.sorted()
                medianVec[dim] = vals[vals.count / 2]
            }
        } else {
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
                state = .validating
                totalTargets = valTargets.count
                currentTargetIndex = 0
                statusMessage = "Now collecting 9 validation points..."
                presentTarget(valTargets[0])
            }
        } else if state == .validating {
            valAggregates.append(agg)
            let nextIdx = currentTargetIndex + 1
            if nextIdx < valTargets.count {
                currentTargetIndex = nextIdx
                presentTarget(valTargets[nextIdx])
            } else {
                state = .processing
                statusMessage = "Training and cross-validating models..."
                runBatchTrainingAndValidation()
            }
        }
    }
    
    private func runBatchTrainingAndValidation() {
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

// MARK: - SwiftUI Views

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
                    AdaptiveCompletionView(controller: controller)
                    
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
            
            VStack(alignment: .leading, spacing: 14) {
                InstructionRow(icon: "dot.circle.viewfinder", text: "Look directly at each target point shown on screen.")
                InstructionRow(icon: "keyboard", text: "Press SPACEBAR when looking at the dot to capture your eye position.")
                InstructionRow(icon: "brain.head.profile", text: "Self-adapting model learns online and updates instantaneously.")
                InstructionRow(icon: "shield.fill", text: "No mouse movement or cursor injection will occur.")
            }
            .frame(maxWidth: 580)
            .padding(24)
            .background(Color.white.opacity(0.06))
            .cornerRadius(12)
            
            VStack(spacing: 12) {
                Button(action: { controller.startAdaptiveCalibration(pointCount: 5) }) {
                    HStack {
                        Image(systemName: "play.circle.fill")
                        Text("Start 5-Point Adaptive Calibration (Press Space / 5)")
                    }
                    .font(.system(size: 15, weight: .bold))
                    .foregroundColor(.black)
                    .frame(width: 440, height: 44)
                    .background(Color.cyan)
                    .cornerRadius(8)
                }
                .buttonStyle(PlainButtonStyle())
                
                HStack(spacing: 12) {
                    Button(action: { controller.startQuickOnePoint() }) {
                        Text("Quick 1-Point Recalibrate (Press 1)")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.white)
                            .frame(width: 214, height: 38)
                            .background(Color.white.opacity(0.12))
                            .cornerRadius(8)
                    }
                    .buttonStyle(PlainButtonStyle())
                    
                    Button(action: { controller.startAdaptiveCalibration(pointCount: 9) }) {
                        Text("9-Point Grid (Press 9)")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundColor(.white)
                            .frame(width: 214, height: 38)
                            .background(Color.white.opacity(0.12))
                            .cornerRadius(8)
                    }
                    .buttonStyle(PlainButtonStyle())
                }
                
                Button(action: { controller.resetAdaptationToBasePrior() }) {
                    Text("Reset Model to Physiological Base Prior")
                        .font(.system(size: 12))
                        .foregroundColor(.white.opacity(0.6))
                        .underline()
                }
                .buttonStyle(PlainButtonStyle())
                .padding(.top, 4)
            }
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
            let isConfirmed = controller.isConfirmedFeedback
            let dotColor = isConfirmed ? Color.green : Color.cyan
            
            ZStack {
                // Interactive guidance prompt banner
                VStack {
                    Spacer()
                    VStack(spacing: 8) {
                        HStack(spacing: 8) {
                            Image(systemName: "keyboard.fill")
                                .foregroundColor(.cyan)
                            Text("Look directly at the dot, then press SPACEBAR to capture")
                                .font(.system(size: 16, weight: .semibold))
                                .foregroundColor(.white)
                        }
                        Text("Target \(controller.currentTargetIndex + 1) of \(controller.totalTargets): \(target.name)")
                            .font(.system(size: 13, design: .monospaced))
                            .foregroundColor(.white.opacity(0.7))
                    }
                    .padding(.horizontal, 24)
                    .padding(.vertical, 14)
                    .background(Color.black.opacity(0.75))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.white.opacity(0.2), lineWidth: 1)
                    )
                    .padding(.bottom, 60)
                }
                
                // Target reticle
                ZStack {
                    if isConfirmed {
                        // Expanding green capture pulse
                        Circle()
                            .stroke(Color.green, lineWidth: 4)
                            .frame(width: 70, height: 70)
                            .scaleEffect(1.3)
                            .opacity(0.8)
                    }
                    
                    // Outer subtle halo
                    Circle()
                        .stroke(dotColor.opacity(0.4), lineWidth: 3)
                        .frame(width: 44, height: 44)
                    
                    // Crosshair guides
                    Rectangle()
                        .fill(dotColor.opacity(0.5))
                        .frame(width: 1, height: 32)
                    Rectangle()
                        .fill(dotColor.opacity(0.5))
                        .frame(width: 32, height: 1)
                    
                    // Solid central dot
                    Circle()
                        .fill(dotColor)
                        .frame(width: 16, height: 16)
                        .shadow(color: dotColor, radius: isConfirmed ? 16 : 8)
                    
                    // Center white pip
                    Circle()
                        .fill(Color.white)
                        .frame(width: 4, height: 4)
                }
                .position(x: CGFloat(px), y: CGFloat(py))
                .animation(.spring(response: 0.25, dampingFraction: 0.6), value: isConfirmed)
            }
        }
    }
}

struct AdaptiveCompletionView: View {
    @ObservedObject var controller: CalibrationWindowController
    
    var body: some View {
        VStack(spacing: 24) {
            Image(systemName: "checkmark.seal.fill")
                .font(.system(size: 48))
                .foregroundColor(.green)
            
            VStack(spacing: 8) {
                Text("Gaze Calibration Complete!")
                    .font(.system(size: 26, weight: .bold))
                    .foregroundColor(.white)
                Text("Self-adapting model has updated online and saved to your active profile.")
                    .font(.system(size: 14))
                    .foregroundColor(.white.opacity(0.8))
            }
            
            VStack(alignment: .leading, spacing: 10) {
                MetricScoreRow(title: "Architecture:", value: "Adaptive Online Ridge (Bayesian Prior)")
                MetricScoreRow(title: "Points Captured:", value: "\(controller.pointsAdaptedCount) targets")
                MetricScoreRow(title: "Adaptation Status:", value: "Active & Saved")
            }
            .frame(maxWidth: 460)
            .padding(20)
            .background(Color.white.opacity(0.08))
            .cornerRadius(12)
            
            HStack(spacing: 16) {
                Button("Recalibrate") {
                    controller.startAdaptiveCalibration(pointCount: 5)
                }
                .font(.system(size: 14, weight: .medium))
                .foregroundColor(.white)
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.15))
                .cornerRadius(8)
                
                Button("Enable Diagnostic Preview & Done") {
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
