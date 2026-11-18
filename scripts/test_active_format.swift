import Foundation
import AVFoundation

let discovery = AVCaptureDevice.DiscoverySession(
    deviceTypes: [.builtInWideAngleCamera],
    mediaType: .video,
    position: .unspecified
)

guard let dev = discovery.devices.first else {
    print("No camera found.")
    exit(1)
}

print("Device: \(dev.localizedName)")
for (idx, format) in dev.formats.enumerated() {
    let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
    let formatType = CMFormatDescriptionGetMediaSubType(format.formatDescription)
    print("Format [\(idx)]: \(dims.width)x\(dims.height), subType=\(formatType)")
}

// Test setting activeFormat to 640x480
do {
    try dev.lockForConfiguration()
    for format in dev.formats {
        let dims = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        if dims.width == 640 && dims.height == 480 {
            dev.activeFormat = format
            print("Set activeFormat to 640x480: \(dev.activeFormat)")
            break
        }
    }
    dev.unlockForConfiguration()
} catch {
    print("Failed to lock device: \(error)")
}

