import Foundation
import AVFoundation

public enum CameraType: String, Codable {
    case builtIn = "Built-in"
    case continuity = "Continuity"
    case virtual = "Virtual"
    case external = "External"
}

public struct CameraDevice: Identifiable, Equatable {
    public let id: String // uniqueID
    public let localizedName: String
    public let type: CameraType
    public let avDevice: AVCaptureDevice
    
    public init(device: AVCaptureDevice) {
        self.id = device.uniqueID
        self.localizedName = device.localizedName
        self.avDevice = device
        
        let nameLower = device.localizedName.lowercased()
        if nameLower.contains("virtual") || nameLower.contains("sharing") || nameLower.contains("obs") {
            self.type = .virtual
        } else if device.deviceType == .continuityCamera || nameLower.contains("iphone") || nameLower.contains("ipad") {
            self.type = .continuity
        } else if device.deviceType == .builtInWideAngleCamera || nameLower.contains("facetime") || nameLower.contains("macbook") {
            self.type = .builtIn
        } else {
            self.type = .external
        }
    }
}

