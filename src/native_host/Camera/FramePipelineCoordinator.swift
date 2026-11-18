import Foundation
import CoreVideo
import AVFoundation

public final class FramePipelineCoordinator {
    public static let shared = FramePipelineCoordinator()
    
    private var sequenceNumber: UInt64 = 0
    private var frameCounter = 0
    private var lastFpsTimestamp: TimeInterval = Date().timeIntervalSince1970
    
    private init() {
        setupPipeline()
    }
    
    private func setupPipeline() {
        CameraCaptureService.shared.onFrameCaptured = { [weak self] pixelBuffer, pts in
            self?.processCapturedFrame(pixelBuffer: pixelBuffer, pts: pts)
        }
    }
    
    public func processCapturedFrame(pixelBuffer: CVPixelBuffer, pts: CMTime) {
        // Double check safety gate
        guard SafetyStateManager.shared.canProcessFrames else {
            return
        }
        
        guard let producer = EngineProcessManager.shared.getShmProducer(),
              let client = EngineProcessManager.shared.getControlClient() else {
            return
        }
        
        sequenceNumber += 1
        let seq = sequenceNumber
        let tsNs = UInt64(clock_gettime_nsec_np(CLOCK_UPTIME_RAW))
        
        // Write frame to shared memory slot respecting bytesPerRow
        guard let slotIdx = producer.writeFrame(
            pixelBuffer: pixelBuffer,
            sequenceNum: seq,
            timestampNs: tsNs
        ) else {
            // Backpressure: drop frame safely rather than buffering
            DispatchQueue.main.async {
                DiagnosticsStore.shared.droppedFrames += 1
            }
            return
        }
        
        // Notify Python engine via control socket
        let sent = client.sendFrameNotification(slot: slotIdx, sequence: seq, timestampNs: tsNs)
        if sent {
            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                DiagnosticsStore.shared.publishedFrames += 1
                
                // Track Camera Capture FPS
                self.frameCounter += 1
                let now = Date().timeIntervalSince1970
                let dt = now - self.lastFpsTimestamp
                if dt >= 1.0 {
                    DiagnosticsStore.shared.cameraFps = Double(self.frameCounter) / dt
                    self.frameCounter = 0
                    self.lastFpsTimestamp = now
                }
            }
        }
    }
}

