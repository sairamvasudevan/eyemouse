import Foundation
import AVFoundation
import CoreMedia
import CoreVideo

class FrameCollector: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let warmupFrames = 30 // ~1 sec warmup
    let captureFrames = 45 // 1.5 sec fixation capture
    var totalFramesSeen = 0
    var capturedBuffers: [Data] = []
    let sema = DispatchSemaphore(value: 0)
    var width: Int = 0
    var height: Int = 0
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        totalFramesSeen += 1
        if totalFramesSeen <= warmupFrames {
            return
        }
        
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        let w = CVPixelBufferGetWidth(pixelBuffer)
        let h = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }
        
        width = w
        height = h
        
        // Copy contiguous rows if stride == w * 4, else copy row by row
        if bytesPerRow == w * 4 {
            let data = Data(bytes: base, count: w * h * 4)
            capturedBuffers.append(data)
        } else {
            var data = Data(count: w * h * 4)
            data.withUnsafeMutableBytes { (dst: UnsafeMutableRawBufferPointer) in
                for row in 0..<h {
                    let srcRow = base.advanced(by: row * bytesPerRow)
                    let dstRow = dst.baseAddress!.advanced(by: row * w * 4)
                    dstRow.copyMemory(from: srcRow, byteCount: w * 4)
                }
            }
            capturedBuffers.append(data)
        }
        
        if capturedBuffers.count >= captureFrames {
            sema.signal()
        }
    }
}

let collector = FrameCollector()
let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified
)

guard let dev = discovery.devices.first else {
    print("No camera found")
    exit(1)
}

print("Starting camera capture on \(dev.localizedName)...")
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
let queue = DispatchQueue(label: "jitter.frame.queue")
output.setSampleBufferDelegate(collector, queue: queue)
guard session.canAddOutput(output) else {
    print("Cannot add output")
    exit(1)
}
session.addOutput(output)
session.sessionPreset = .high
session.commitConfiguration()

session.startRunning()
print("Camera warming up (1s) and capturing 45 frames...")
_ = collector.sema.wait(timeout: .now() + 5.0)
session.stopRunning()

print("Captured \(collector.capturedBuffers.count) frames of size \(collector.width)x\(collector.height)")

guard collector.capturedBuffers.count == collector.captureFrames else {
    print("Failed to capture expected frames: count=\(collector.capturedBuffers.count)")
    exit(1)
}

let outputPath = "/tmp/eyemouse_jitter_frames.raw"
let fileHandle = FileManager.default.createFile(atPath: outputPath, contents: nil, attributes: nil)
guard let handle = FileHandle(forWritingAtPath: outputPath) else {
    print("Cannot open output file")
    exit(1)
}

for buf in collector.capturedBuffers {
    handle.write(buf)
}
handle.closeFile()
print("Wrote \(collector.capturedBuffers.count) raw frames to \(outputPath)")

