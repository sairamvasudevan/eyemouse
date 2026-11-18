import Foundation
import AVFoundation

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
    mediaType: .video,
    position: .unspecified
)

print("Discovered \(discovery.devices.count) cameras:")
for dev in discovery.devices {
    print("\nDevice: \(dev.localizedName) [\(dev.uniqueID)]")
    print("  Position: \(dev.position.rawValue)")
    print("  Is Suspended: \(dev.isSuspended)")
    print("  Active format: \(dev.activeFormat.formatDescription)")
    print("  Active min frame duration: \(dev.activeVideoMinFrameDuration.seconds)s (\(1.0 / dev.activeVideoMinFrameDuration.seconds) fps)")
    print("  Active max frame duration: \(dev.activeVideoMaxFrameDuration.seconds)s (\(1.0 / dev.activeVideoMaxFrameDuration.seconds) fps)")
    
    for (idx, format) in dev.formats.enumerated() {
        let desc = format.formatDescription
        let dims = CMVideoFormatDescriptionGetDimensions(desc)
        let ranges = format.videoSupportedFrameRateRanges.map { "\($0.minFrameRate)-\($0.maxFrameRate) fps" }.joined(separator: ", ")
        if dims.width <= 1280 && dims.height <= 720 {
            print("  Format [\(idx)]: \(dims.width)x\(dims.height), Ranges: [\(ranges)]")
        }
    }
}

