import Foundation
import Darwin

public final class ControlClient {
    public let socketPath: String
    private var clientSock: Int32 = -1
    private var isConnected = false
    private let readQueue = DispatchQueue(label: "com.eyemouse.control.read")
    
    public var onTelemetryReceived: (([String: Any]) -> Void)?
    
    public init(socketPath: String) {
        self.socketPath = socketPath
    }
    
    deinit {
        close()
    }
    
    public func connect(timeoutSeconds: Double = 5.0) -> Bool {
        let fd = socket(AF_UNIX, SOCK_STREAM, 0)
        guard fd >= 0 else {
            perror("[ControlClient] socket() failed")
            return false
        }
        
        var addr = sockaddr_un()
        addr.sun_family = sa_family_t(AF_UNIX)
        let pathCapacity = MemoryLayout.size(ofValue: addr.sun_path)
        
        guard socketPath.utf8.count < pathCapacity else {
            print("[ControlClient] Socket path too long: \(socketPath)")
            Darwin.close(fd)
            return false
        }
        
        _ = withUnsafeMutablePointer(to: &addr.sun_path.0) { ptr in
            socketPath.withCString { cstr in
                strncpy(ptr, cstr, pathCapacity - 1)
            }
        }
        
        let addrLen = socklen_t(MemoryLayout<sockaddr_un>.size)
        let startTime = Date().timeIntervalSince1970
        
        while true {
            let res = withUnsafePointer(to: &addr) { ptr in
                ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
                    Darwin.connect(fd, sa, addrLen)
                }
            }
            if res == 0 {
                self.clientSock = fd
                self.isConnected = true
                print("[ControlClient] Connected to Unix socket at \(socketPath)")
                startReading()
                return true
            }
            
            if Date().timeIntervalSince1970 - startTime > timeoutSeconds {
                print("[ControlClient] Connection timed out for \(socketPath)")
                Darwin.close(fd)
                return false
            }
            usleep(50000) // 50ms
        }
    }
    
    public func performHandshake(shmName: String) -> Bool {
        guard isConnected else { return false }
        
        let msg: [String: Any] = [
            "cmd": "HANDSHAKE",
            "protocol_version": BundleConstants.protocolVersion,
            "shm_name": shmName
        ]
        
        guard sendMessage(msg) else { return false }
        return true
    }
    
    public func sendFrameNotification(slot: Int, sequence: UInt64, timestampNs: UInt64) -> Bool {
        guard isConnected else { return false }
        let msg: [String: Any] = [
            "cmd": "FRAME",
            "slot": slot,
            "seq": sequence,
            "timestamp_ns": timestampNs
        ]
        return sendMessage(msg)
    }
    
    public func sendAdaptPoint(features: [Double], targetNormX: Double, targetNormY: Double, weight: Double = 1.0) -> Bool {
        guard isConnected else { return false }
        let msg: [String: Any] = [
            "cmd": "ADAPT_POINT",
            "features": features,
            "target_norm_x": targetNormX,
            "target_norm_y": targetNormY,
            "weight": weight
        ]
        return sendMessage(msg)
    }
    
    public func sendResetAdaptation() -> Bool {
        guard isConnected else { return false }
        return sendMessage(["cmd": "RESET_ADAPTATION"])
    }
    
    public func sendLoadCalibration(profilePath: String) -> Bool {
        guard isConnected else { return false }
        let msg: [String: Any] = [
            "cmd": "LOAD_CALIBRATION",
            "profile_path": profilePath
        ]
        return sendMessage(msg)
    }
    
    public func sendShutdown() {
        guard isConnected else { return }
        _ = sendMessage(["cmd": "SHUTDOWN"])
    }
    
    public func close() {
        guard isConnected, clientSock >= 0 else { return }
        isConnected = false
        Darwin.close(clientSock)
        clientSock = -1
        print("[ControlClient] Closed control socket.")
    }
    
    private func sendMessage(_ dict: [String: Any]) -> Bool {
        guard clientSock >= 0,
              let jsonData = try? JSONSerialization.data(withJSONObject: dict),
              var jsonString = String(data: jsonData, encoding: .utf8) else {
            return false
        }
        jsonString += "\n"
        
        return jsonString.withCString { cstr in
            let len = strlen(cstr)
            var bytesWritten = 0
            while bytesWritten < len {
                let n = write(clientSock, cstr + bytesWritten, len - bytesWritten)
                if n <= 0 {
                    return false
                }
                bytesWritten += n
            }
            return true
        }
    }
    
    private func startReading() {
        readQueue.async { [weak self] in
            guard let self = self else { return }
            var buffer = Data()
            let chunk = UnsafeMutablePointer<UInt8>.allocate(capacity: 4096)
            defer { chunk.deallocate() }
            
            while self.isConnected && self.clientSock >= 0 {
                let bytesRead = read(self.clientSock, chunk, 4096)
                if bytesRead <= 0 {
                    // Socket closed or error
                    if self.isConnected {
                        print("[ControlClient] Socket EOF or read error.")
                        SafetyStateManager.shared.reportError("Vision engine connection closed")
                    }
                    break
                }
                
                buffer.append(chunk, count: bytesRead)
                
                // Process newline-delimited messages
                while let newlineIdx = buffer.firstIndex(of: UInt8(ascii: "\n")) {
                    let lineData = buffer.subdata(in: 0..<newlineIdx)
                    buffer.removeSubrange(0...newlineIdx)
                    
                    if let json = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] {
                        self.handleIncomingMessage(json)
                    }
                }
            }
        }
    }
    
    private func handleIncomingMessage(_ dict: [String: Any]) {
        let status = dict["status"] as? String ?? ""
        if status == "READY" {
            print("[ControlClient] Vision engine confirmed READY.")
            SafetyStateManager.shared.markEngineAndCameraReady()
        } else if status == "ACK" {
            DiagnosticsStore.shared.updateEngineTelemetry(data: dict)
            onTelemetryReceived?(dict)
        } else if status == "POINT_ADAPTED" {
            let px = dict["pred_x"] as? Double ?? 0.0
            let py = dict["pred_y"] as? Double ?? 0.0
            print("[ControlClient] Engine adapted point, predicted: (\(px), \(py))")
            DiagnosticsStore.shared.loadActiveProfile()
        } else if status == "ADAPTATION_RESET" {
            print("[ControlClient] Engine adaptation reset to base prior.")
            DiagnosticsStore.shared.loadActiveProfile()
        } else if status == "CALIBRATION_LOADED" {
            print("[ControlClient] Engine loaded calibration profile.")
            DiagnosticsStore.shared.loadActiveProfile()
        } else if status == "ERROR" || status == "ADAPTATION_ERROR" || status == "CALIBRATION_ERROR" {
            let msg = (dict["message"] as? String) ?? (dict["error"] as? String) ?? "Unknown engine error"
            print("[ControlClient] Engine error: \(msg)")
        }
    }
}
