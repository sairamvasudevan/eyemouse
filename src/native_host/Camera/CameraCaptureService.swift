import Foundation
import AVFoundation
import CoreVideo

public final class CameraCaptureService: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    public static let shared = CameraCaptureService()
    
    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.eyemouse.camera.session")
    private let frameQueue = DispatchQueue(label: "com.eyemouse.camera.output")
    
    private var activeDeviceInput: AVCaptureDeviceInput?
    private var isSessionConfigured = false
    
    public var onFrameCaptured: ((CVPixelBuffer, CMTime) -> Void)?
    
    private override init() {
        super.init()
    }
    
    public func startCapture() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            
            // Verify permission
            guard PermissionsManager.shared.isCameraAuthorized else {
                print("[CameraCapture] Camera not authorized.")
                SafetyStateManager.shared.reportError("Camera permission denied")
                return
            }
            
            self.configureSessionIfNeeded()
            
            if !self.session.isRunning {
                self.session.startRunning()
                print("[CameraCapture] AVCaptureSession started.")
                DispatchQueue.main.async {
                    DiagnosticsStore.shared.isCameraActive = true
                }
            }
        }
    }
    
    public func stopCapture() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
                print("[CameraCapture] AVCaptureSession stopped.")
                DispatchQueue.main.async {
                    DiagnosticsStore.shared.isCameraActive = false
                }
            }
            SafetyStateManager.shared.markTeardownComplete()
        }
    }
    
    private func configureSessionIfNeeded() {
        guard let device = CameraManager.shared.selectedDevice?.avDevice else {
            print("[CameraCapture] No camera device selected.")
            SafetyStateManager.shared.reportError("No camera device selected")
            return
        }
        
        // If device already attached, return
        if let currentInput = activeDeviceInput, currentInput.device.uniqueID == device.uniqueID, isSessionConfigured {
            return
        }
        
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        
        // Remove old input if any
        if let oldInput = activeDeviceInput {
            session.removeInput(oldInput)
            activeDeviceInput = nil
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
                self.activeDeviceInput = input
            } else {
                print("[CameraCapture] Cannot add device input.")
                SafetyStateManager.shared.reportError("Cannot configure camera input")
                return
            }
        } catch {
            print("[CameraCapture] Failed to create AVCaptureDeviceInput: \(error)")
            SafetyStateManager.shared.reportError("Camera input error: \(error.localizedDescription)")
            return
        }
        
        if !session.outputs.contains(videoOutput) {
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)
            ]
            videoOutput.setSampleBufferDelegate(self, queue: frameQueue)
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            } else {
                print("[CameraCapture] Cannot add video output.")
                SafetyStateManager.shared.reportError("Cannot configure camera output")
                return
            }
        }
        
        // Apply optimal preset after inputs and outputs are attached
        if session.canSetSessionPreset(.vga640x480) {
            session.sessionPreset = .vga640x480
        } else if session.canSetSessionPreset(.hd1280x720) {
            session.sessionPreset = .hd1280x720
        }
        
        // Lock frame duration to 30 FPS if supported by active format
        do {
            try device.lockForConfiguration()
            let targetFPS = 30.0
            let frameDuration = CMTime(value: 1, timescale: Int32(targetFPS))
            var supported = false
            for range in device.activeFormat.videoSupportedFrameRateRanges {
                if range.minFrameRate <= targetFPS && targetFPS <= range.maxFrameRate {
                    supported = true
                    break
                }
            }
            if supported {
                device.activeVideoMinFrameDuration = frameDuration
                device.activeVideoMaxFrameDuration = frameDuration
                print("[CameraCapture] Configured device frame rate to \(targetFPS) FPS.")
            }
            device.unlockForConfiguration()
        } catch {
            print("[CameraCapture] Warning: Could not lock device for frame duration: \(error)")
        }
        
        self.isSessionConfigured = true
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
    public func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Strict safety gate check: drop frame synchronously if control is not active
        guard SafetyStateManager.shared.canProcessFrames else {
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }
        
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        onFrameCaptured?(pixelBuffer, pts)
    }
}
