import Foundation
import AVFoundation
import CoreVideo

@main
struct LivePipelineTestRunner {
    static func main() {
        print("=== Running M3 Real Live Camera & Vision Engine Pipeline Test ===")
        
        // 1. Verify Camera Authorization
        let authStatus = AVCaptureDevice.authorizationStatus(for: .video)
        print("Camera Authorization Status: \(authStatus.rawValue) (3 = Authorized)")
        guard authStatus == .authorized else {
            print("ERROR: Camera is not authorized for testing.")
            exit(1)
        }
        
        // 2. Discover and Select Physical Camera
        CameraManager.shared.refreshDevices()
        guard let camera = CameraManager.shared.selectedDevice else {
            print("ERROR: No camera device found.")
            exit(1)
        }
        print("Selected Camera: \(camera.localizedName) (\(camera.type.rawValue), ID: \(camera.id))")
        
        // 3. Start Vision Engine
        print("Starting Vision Engine process...")
        let engineStarted = EngineProcessManager.shared.startEngine()
        guard engineStarted else {
            print("ERROR: EngineProcessManager failed to start engine.")
            exit(1)
        }
        print("Vision Engine connected. Socket: \(EngineProcessManager.shared.socketPath), SHM: \(EngineProcessManager.shared.shmName)")
        
        // 4. Initialize Frame Pipeline
        _ = FramePipelineCoordinator.shared
        
        // 5. Simulate Shortcut Hold Activation
        SafetyStateManager.shared.handleShortcutPressed()
        SafetyStateManager.shared.markEngineAndCameraReady()
        guard SafetyStateManager.shared.canProcessFrames else {
            print("ERROR: Safety gate is not open after activation.")
            exit(1)
        }
        print("Safety Gate OPEN. Starting real AVCaptureSession...")
        
        // 6. Start Camera Capture Session
        CameraCaptureService.shared.startCapture()
        
        // 7. Collect 45 real camera frames and track performance
        let targetFrames = 45
        var receivedAcks = 0
        var totalEngineLatencyMs = 0.0
        var latencies: [Double] = []
        var detectedFaces = 0
        let startTime = Date().timeIntervalSince1970
        
        let client = EngineProcessManager.shared.getControlClient()
        client?.onTelemetryReceived = { data in
            receivedAcks += 1
            let lat = data["processing_time_ms"] as? Double ?? 0.0
            let face = data["face_detected"] as? Bool ?? false
            latencies.append(lat)
            totalEngineLatencyMs += lat
            if face { detectedFaces += 1 }
            
            if receivedAcks <= 5 || receivedAcks % 10 == 0 || receivedAcks == targetFrames {
                let gaze = data["gaze"] as? [String: Any]
                let gx = gaze?["filtered_x"] as? Double ?? 0.0
                let gy = gaze?["filtered_y"] as? Double ?? 0.0
                let pose = data["head_pose"] as? [String: Any]
                let yaw = pose?["yaw"] as? Double ?? 0.0
                print("Live Frame [\(receivedAcks)/\(targetFrames)] -> Face: \(face), Latency: \(String(format: "%.2f", lat))ms, Gaze: (\(String(format: "%+.3f", gx)), \(String(format: "%+.3f", gy))), HeadYaw: \(String(format: "%+.1f°", yaw))")
            }
        }
        
        // Wait loop for frames with timeout
        let maxWaitSeconds = 8.0
        let waitStart = Date().timeIntervalSince1970
        while receivedAcks < targetFrames && (Date().timeIntervalSince1970 - waitStart < maxWaitSeconds) {
            RunLoop.current.run(until: Date(timeIntervalSinceNow: 0.05))
        }
        
        let elapsed = Date().timeIntervalSince1970 - startTime
        print("\n--- Live Test Completed in \(String(format: "%.2f", elapsed))s ---")
        print("Frames Processed by Engine: \(receivedAcks)")
        
        guard receivedAcks >= 15 else {
            print("ERROR: Did not receive sufficient frames from real camera (\(receivedAcks) < 15)")
            EngineProcessManager.shared.stopEngine()
            CameraCaptureService.shared.stopCapture()
            exit(1)
        }
        
        // 8. Test Immediate Gate Cutoff (Shortcut Release)
        print("\nTesting Immediate Shortcut Release...")
        let publishedBeforeRelease = DiagnosticsStore.shared.publishedFrames
        SafetyStateManager.shared.handleShortcutReleased()
        assert(!SafetyStateManager.shared.canProcessFrames, "Gate must be closed immediately")
        
        // Sleep 200ms while camera is still capturing to verify no frames pass through closed gate
        let preCount = DiagnosticsStore.shared.publishedFrames
        Thread.sleep(forTimeInterval: 0.20)
        let postCount = DiagnosticsStore.shared.publishedFrames
        let leakedFrames = postCount - preCount
        print("Frames published after release: \(leakedFrames) (MUST BE 0)")
        guard leakedFrames == 0 else {
            print("ERROR: Safety gate leaked frames after shortcut release!")
            exit(1)
        }
        print("PASS: Zero frames leaked after shortcut release.")
        
        // 9. Stop Camera and Engine
        CameraCaptureService.shared.stopCapture()
        EngineProcessManager.shared.stopEngine()
        
        // 10. Compute Final Live Metrics
        let meanLat = totalEngineLatencyMs / Double(receivedAcks)
        let sortedLat = latencies.sorted()
        let p50 = sortedLat[Int(Double(sortedLat.count) * 0.50)]
        let p95 = sortedLat[Int(Double(sortedLat.count) * 0.95)]
        let fps = Double(receivedAcks) / elapsed
        
        print("\n=== Live Real-Camera Pipeline Benchmark Summary ===")
        print("Camera:            \(camera.localizedName)")
        print("Resolution:        640x480 BGRA")
        print("Live Capture Rate: \(String(format: "%.1f", fps)) FPS")
        print("Mean Latency:      \(String(format: "%.2f", meanLat)) ms")
        print("Median (p50):      \(String(format: "%.2f", p50)) ms")
        print("95th %-ile (p95):  \(String(format: "%.2f", p95)) ms")
        print("Min Latency:       \(String(format: "%.2f", sortedLat.first ?? 0)) ms")
        print("Max Latency:       \(String(format: "%.2f", sortedLat.last ?? 0)) ms")
        print("Face Detections:   \(detectedFaces)/\(receivedAcks)")
        print("Dropped Frames:    \(DiagnosticsStore.shared.droppedFrames)")
        
        // Save results to JSON
        let results: [String: Any] = [
            "camera_name": camera.localizedName,
            "camera_type": camera.type.rawValue,
            "resolution": "640x480 BGRA",
            "total_frames": receivedAcks,
            "elapsed_seconds": elapsed,
            "live_fps": fps,
            "mean_latency_ms": meanLat,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "min_latency_ms": sortedLat.first ?? 0,
            "max_latency_ms": sortedLat.last ?? 0,
            "detected_faces": detectedFaces,
            "dropped_frames": DiagnosticsStore.shared.droppedFrames,
            "gate_leak_frames": leakedFrames
        ]
        
        if let json = try? JSONSerialization.data(withJSONObject: results, options: .prettyPrinted) {
            try? json.write(to: URL(fileURLWithPath: "build/m3_live_test_results.json"))
            print("Results written to build/m3_live_test_results.json")
        }
        
        print("\n=== M3 REAL LIVE WEBCAM PIPELINE TEST PASSED! ===")
    }
}

