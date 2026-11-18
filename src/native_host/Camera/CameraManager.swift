import Foundation
import AVFoundation

public final class CameraManager: NSObject {
    public static let shared = CameraManager()
    
    public private(set) var availableDevices: [CameraDevice] = []
    public private(set) var selectedDevice: CameraDevice?
    
    public var onDevicesChanged: (([CameraDevice]) -> Void)?
    public var onSelectedDeviceChanged: ((CameraDevice?) -> Void)?
    
    private override init() {
        super.init()
        refreshDevices()
        setupNotifications()
    }
    
    public func refreshDevices() {
        var deviceTypes: [AVCaptureDevice.DeviceType] = [
            .builtInWideAngleCamera,
            .external,
        ]
        if #available(macOS 13.0, *) {
            deviceTypes.append(.continuityCamera)
        }
        
        let discovery = AVCaptureDevice.DiscoverySession(
            deviceTypes: deviceTypes,
            mediaType: .video,
            position: .unspecified
        )
        
        self.availableDevices = discovery.devices.map { CameraDevice(device: $0) }
        
        // Restore selected device or select default
        let savedID = UserDefaults.standard.string(forKey: BundleConstants.keySelectedCameraID)
        if let savedID = savedID, let match = availableDevices.first(where: { $0.id == savedID }) {
            self.selectedDevice = match
        } else {
            // Prefer physical built-in camera over virtual or continuity
            let preferred = availableDevices.first(where: { $0.type == .builtIn })
                ?? availableDevices.first(where: { $0.type == .external })
                ?? availableDevices.first(where: { $0.type == .continuity })
                ?? availableDevices.first
            self.selectedDevice = preferred
            if let defID = preferred?.id {
                UserDefaults.standard.set(defID, forKey: BundleConstants.keySelectedCameraID)
            }
        }
        
        onDevicesChanged?(availableDevices)
        onSelectedDeviceChanged?(selectedDevice)
    }
    
    public func selectDevice(byID id: String) {
        guard let device = availableDevices.first(where: { $0.id == id }) else { return }
        self.selectedDevice = device
        UserDefaults.standard.set(device.id, forKey: BundleConstants.keySelectedCameraID)
        onSelectedDeviceChanged?(device)
        print("[CameraManager] Selected camera: \(device.localizedName) (\(device.type.rawValue))")
    }
    
    private func setupNotifications() {
        NotificationCenter.default.addObserver(
            forName: AVCaptureDevice.wasConnectedNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            print("[CameraManager] Camera connected. Refreshing devices...")
            self?.refreshDevices()
        }
        
        NotificationCenter.default.addObserver(
            forName: AVCaptureDevice.wasDisconnectedNotification,
            object: nil,
            queue: .main
        ) { [weak self] notification in
            guard let disconnected = notification.object as? AVCaptureDevice else { return }
            print("[CameraManager] Camera disconnected: \(disconnected.localizedName)")
            if self?.selectedDevice?.id == disconnected.uniqueID {
                print("[CameraManager] Active selected camera disconnected! Forcing inactive.")
                SafetyStateManager.shared.forceInactive(reason: "Camera disconnected")
            }
            self?.refreshDevices()
        }
    }
}

