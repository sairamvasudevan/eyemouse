import Foundation
import AVFoundation
import CoreMedia

class FPSDetector: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var frameCount = 0
    var startTime: CFAbsoluteTime = 0
    let targetFrames = 60
    let semaphore = DispatchSemaphore(value: 0)
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if frameCount == 0 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        frameCount += 1
        let now = CFAbsoluteTimeGetCurrent()
        let elapsed = now - startTime
        if frameCount % 15 == 0 {
            let currentFPS = Double(frameCount) / elapsed
            print("Frame \(frameCount): \(String(format: "%.2f", currentFPS)) FPS")
        }
        if frameCount >= targetFrames {
            semaphore.signal()
        }
    }
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified
)

guard let dev = discovery.devices.first else {
    print("No built-in camera found.")
    exit(1)
}

print("Testing Camera: \(dev.localizedName)")
let session = AVCaptureSession()
session.beginConfiguration()

guard let input = try? AVCaptureDeviceInput(device: dev), session.canAddInput(input) else {
    print("Cannot add input")
    exit(1)
}
session.addInput(input)

let output = AVCaptureVideoDataOutput()
output.alwaysDiscardsLateVideoFrames = true
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
let queue = DispatchQueue(label: "test.fps.queue")
let detector = FPSDetector()
output.setSampleBufferDelegate(detector, queue: queue)

guard session.canAddOutput(output) else {
    print("Cannot add output")
    exit(1)
}
session.addOutput(output)

if session.canSetSessionPreset(.vga640x480) {
    session.sessionPreset = .vga640x480
}

session.commitConfiguration()

// Test 1: Default configuration without locking frame duration
print("\n--- Test 1: Default Auto-Exposure (unlocked frame rate) ---")
session.startRunning()
detector.semaphore.wait()
session.stopRunning()
let elapsed1 = CFAbsoluteTimeGetCurrent() - detector.startTime
let fps1 = Double(detector.frameCount) / elapsed1
print("Result 1: \(detector.frameCount) frames in \(String(format: "%.2f", elapsed1))s = \(String(format: "%.2f", fps1)) FPS")

// Test 2: Locked frame duration to 30 FPS
print("\n--- Test 2: Locking min & max frame duration to 30 FPS ---")
do {
    try dev.lockForConfiguration()
    let frameDuration = CMTime(value: 1, timescale: 30)
    dev.activeVideoMinFrameDuration = frameDuration
    dev.activeVideoMaxFrameDuration = frameDuration
    dev.unlockForConfiguration()
    print("Successfully set activeVideoMinFrameDuration and activeVideoMaxFrameDuration to 1/30s")
} catch {
    print("Failed to lock device: \(error)")
}

let detector2 = FPSDetector()
output.setSampleBufferDelegate(detector2, queue: queue)
session.startRunning()
detector2.semaphore.wait()
session.stopRunning()
let elapsed2 = CFAbsoluteTimeGetCurrent() - detector2.startTime
let fps2 = Double(detector2.frameCount) / elapsed2
print("Result 2: \(detector2.frameCount) frames in \(String(format: "%.2f", elapsed2))s = \(String(format: "%.2f", fps2)) FPS")

