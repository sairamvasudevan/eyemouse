import Foundation
import AVFoundation
import CoreMedia
import CoreVideo

class ResolutionBenchmarkService: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var frameCount = 0
    var startTime: CFAbsoluteTime = 0
    var lastFrameTime: CFAbsoluteTime = 0
    var frameIntervals: [Double] = []
    var targetFrames = 60
    var deliveredWidth = 0
    var deliveredHeight = 0
    var deliveredStride = 0
    var semaphore = DispatchSemaphore(value: 0)
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let now = CFAbsoluteTimeGetCurrent()
        if frameCount == 0 {
            startTime = now
            lastFrameTime = now
            if let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                deliveredWidth = CVPixelBufferGetWidth(imageBuffer)
                deliveredHeight = CVPixelBufferGetHeight(imageBuffer)
                deliveredStride = CVPixelBufferGetBytesPerRow(imageBuffer)
            }
        } else {
            frameIntervals.append(now - lastFrameTime)
            lastFrameTime = now
        }
        
        frameCount += 1
        if frameCount >= targetFrames {
            semaphore.signal()
        }
    }
}

func benchmarkPreset(device: AVCaptureDevice, preset: AVCaptureSession.Preset, name: String) -> [String: Any]? {
    print("\n--- Testing Preset: \(name) ---")
    let session = AVCaptureSession()
    session.beginConfiguration()
    
    guard let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
        print("Cannot add input for \(name)")
        return nil
    }
    session.addInput(input)
    
    let output = AVCaptureVideoDataOutput()
    output.alwaysDiscardsLateVideoFrames = true
    output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: Int(kCVPixelFormatType_32BGRA)]
    let bench = ResolutionBenchmarkService()
    let queue = DispatchQueue(label: "bench.resolution.queue")
    output.setSampleBufferDelegate(bench, queue: queue)
    
    guard session.canAddOutput(output) else {
        print("Cannot add output for \(name)")
        return nil
    }
    session.addOutput(output)
    
    if session.canSetSessionPreset(preset) {
        session.sessionPreset = preset
    } else {
        print("Preset \(name) not supported on this device.")
        return nil
    }
    
    // Lock frame duration to 30 FPS if supported
    do {
        try device.lockForConfiguration()
        let targetDuration = CMTime(value: 1, timescale: 30)
        device.activeVideoMinFrameDuration = targetDuration
        device.activeVideoMaxFrameDuration = targetDuration
        device.unlockForConfiguration()
    } catch {
        print("Could not lock frame rate for \(name): \(error)")
    }
    
    session.commitConfiguration()
    
    session.startRunning()
    _ = bench.semaphore.wait(timeout: .now() + 4.0)
    session.stopRunning()
    
    let elapsed = CFAbsoluteTimeGetCurrent() - bench.startTime
    let fps = Double(bench.frameCount) / elapsed
    let meanInterval = bench.frameIntervals.isEmpty ? 0 : (bench.frameIntervals.reduce(0, +) / Double(bench.frameIntervals.count))
    let jitter = bench.frameIntervals.isEmpty ? 0 : sqrt(bench.frameIntervals.reduce(0) { $0 + pow($1 - meanInterval, 2) } / Double(bench.frameIntervals.count))
    
    print("  Resolution: \(bench.deliveredWidth)x\(bench.deliveredHeight) (Stride: \(bench.deliveredStride))")
    print("  Delivered Frames: \(bench.frameCount) in \(String(format: "%.2f", elapsed))s = \(String(format: "%.2f", fps)) FPS")
    print("  Mean Frame Interval: \(String(format: "%.2f", meanInterval * 1000)) ms, Jitter: \(String(format: "%.2f", jitter * 1000)) ms")
    
    return [
        "preset": name,
        "width": bench.deliveredWidth,
        "height": bench.deliveredHeight,
        "stride": bench.deliveredStride,
        "fps": fps,
        "mean_interval_ms": meanInterval * 1000,
        "jitter_ms": jitter * 1000,
        "delivered_frames": bench.frameCount,
        "elapsed_seconds": elapsed,
    ]
}

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified
)

guard let camera = discovery.devices.first else {
    print("No built-in camera found.")
    exit(1)
}

print("=== Benchmarking Camera Presets on \(camera.localizedName) ===")
var results: [[String: Any]] = []

if let r1 = benchmarkPreset(device: camera, preset: .vga640x480, name: "640x480") {
    results.append(r1)
}
if let r2 = benchmarkPreset(device: camera, preset: .hd1280x720, name: "1280x720") {
    results.append(r2)
}
if let r3 = benchmarkPreset(device: camera, preset: .hd1920x1080, name: "1920x1080") {
    results.append(r3)
}

let json = try! JSONSerialization.data(withJSONObject: results, options: .prettyPrinted)
let outURL = URL(fileURLWithPath: "build/m4_1_camera_benchmarks.json")
try! json.write(to: outURL)
print("\nResults written to \(outURL.path)")

