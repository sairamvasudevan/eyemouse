import Foundation

public final class EngineProcessManager {
    public static let shared = EngineProcessManager()
    
    private var process: Process?
    private var controlClient: ControlClient?
    private var shmProducer: ShmProducer?
    
    public private(set) var socketPath: String = ""
    public private(set) var shmName: String = ""
    public private(set) var isRunning: Bool = false
    
    private init() {}
    
    public func startEngine() -> Bool {
        guard !isRunning else { return true }
        
        let pid = getpid()
        self.socketPath = BundleConstants.makeSocketPath(forPID: pid)
        self.shmName = BundleConstants.makeShmName(forPID: pid)
        
        // Clean up any stale socket
        if FileManager.default.fileExists(atPath: socketPath) {
            try? FileManager.default.removeItem(atPath: socketPath)
        }
        
        // 1. Initialize Shared Memory Producer
        guard let producer = ShmProducer(shmName: shmName) else {
            print("[EngineManager] Failed to create shared memory: \(shmName)")
            SafetyStateManager.shared.reportError("Shared memory creation failed")
            return false
        }
        self.shmProducer = producer
        
        // 2. Resolve Engine Executable Path
        guard let engineLaunchInfo = resolveEngineLaunchInfo() else {
            print("[EngineManager] No valid vision engine executable found.")
            SafetyStateManager.shared.reportError("Vision engine binary not found")
            producer.close()
            return false
        }
        
        print("[EngineManager] Launching vision engine: \(engineLaunchInfo.executableURL.path)")
        
        // 3. Launch Engine Process
        let proc = Process()
        proc.executableURL = engineLaunchInfo.executableURL
        var args = engineLaunchInfo.arguments
        args.append(contentsOf: ["--socket", socketPath])
        proc.arguments = args
        
        var env = ProcessInfo.processInfo.environment
        env["MPLCONFIGDIR"] = "/tmp"
        env["MPLBACKEND"] = "Agg"
        proc.environment = env
        
        // Create log file for diagnostic stderr/stdout
        let logDir = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Logs/EyeMouse")
        try? FileManager.default.createDirectory(at: logDir, withIntermediateDirectories: true)
        let logFile = logDir.appendingPathComponent("engine.log")
        if !FileManager.default.fileExists(atPath: logFile.path) {
            FileManager.default.createFile(atPath: logFile.path, contents: nil)
        }
        if let fileHandle = try? FileHandle(forWritingTo: logFile) {
            fileHandle.seekToEndOfFile()
            proc.standardOutput = fileHandle
            proc.standardError = fileHandle
        }
        
        proc.terminationHandler = { [weak self] p in
            print("[EngineManager] Engine process terminated with code: \(p.terminationStatus)")
            DispatchQueue.main.async {
                self?.isRunning = false
                DiagnosticsStore.shared.engineStatus = "Terminated (Exit: \(p.terminationStatus))"
                DiagnosticsStore.shared.enginePID = nil
                if SafetyStateManager.shared.currentState == .active || SafetyStateManager.shared.currentState == .starting {
                    SafetyStateManager.shared.reportError("Vision engine terminated unexpectedly")
                }
            }
        }
        
        do {
            try proc.run()
            self.process = proc
            self.isRunning = true
            DispatchQueue.main.async {
                DiagnosticsStore.shared.enginePID = proc.processIdentifier
                DiagnosticsStore.shared.engineStatus = "Running (PID: \(proc.processIdentifier))"
            }
        } catch {
            print("[EngineManager] Failed to run engine process: \(error)")
            SafetyStateManager.shared.reportError("Failed to launch vision engine: \(error.localizedDescription)")
            producer.close()
            return false
        }
        
        // 4. Wait for Socket to Appear
        let startWait = Date().timeIntervalSince1970
        var socketReady = false
        while Date().timeIntervalSince1970 - startWait < 25.0 {
            if FileManager.default.fileExists(atPath: socketPath) {
                socketReady = true
                break
            }
            usleep(50000) // 50ms
        }
        
        guard socketReady else {
            print("[EngineManager] Vision engine did not create socket within 25 seconds.")
            SafetyStateManager.shared.reportError("Vision engine socket timeout")
            stopEngine()
            return false
        }
        
        // 5. Connect Control Client and Perform Handshake
        let client = ControlClient(socketPath: socketPath)
        guard client.connect(timeoutSeconds: 3.0) else {
            print("[EngineManager] Failed to connect to vision engine socket.")
            SafetyStateManager.shared.reportError("Engine socket connection failed")
            stopEngine()
            return false
        }
        self.controlClient = client
        
        guard client.performHandshake(shmName: shmName) else {
            print("[EngineManager] Engine handshake failed.")
            SafetyStateManager.shared.reportError("Engine handshake failed")
            stopEngine()
            return false
        }
        
        print("[EngineManager] Engine started and connected successfully.")
        return true
    }
    
    public func stopEngine() {
        guard isRunning || process != nil else { return }
        print("[EngineManager] Stopping vision engine...")
        
        if let client = controlClient {
            client.sendShutdown()
            // Wait briefly for graceful stop
            usleep(100000) // 100ms
            client.close()
            controlClient = nil
        }
        
        if let proc = process, proc.isRunning {
            proc.terminate()
            // Wait up to 1 second
            let start = Date().timeIntervalSince1970
            while proc.isRunning && Date().timeIntervalSince1970 - start < 1.0 {
                usleep(50000)
            }
            if proc.isRunning {
                kill(proc.processIdentifier, SIGKILL)
            }
        }
        process = nil
        isRunning = false
        
        if let producer = shmProducer {
            producer.close()
            shmProducer = nil
        }
        
        if FileManager.default.fileExists(atPath: socketPath) {
            try? FileManager.default.removeItem(atPath: socketPath)
        }
        
        DispatchQueue.main.async {
            DiagnosticsStore.shared.engineStatus = "Stopped"
            DiagnosticsStore.shared.enginePID = nil
            DiagnosticsStore.shared.resetTrackingState()
        }
    }
    
    public func getShmProducer() -> ShmProducer? {
        return shmProducer
    }
    
    public func getControlClient() -> ControlClient? {
        return controlClient
    }
    
    private struct LaunchInfo {
        let executableURL: URL
        let arguments: [String]
    }
    
    private func resolveEngineLaunchInfo() -> LaunchInfo? {
        let fm = FileManager.default
        let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)
        
        // 1. App Bundle Contents/Resources/engine/eyemouse_engine
        let resourceEngine = Bundle.main.bundleURL.appendingPathComponent("Contents/Resources/engine/eyemouse_engine")
        if fm.isExecutableFile(atPath: resourceEngine.path) {
            return LaunchInfo(executableURL: resourceEngine, arguments: [])
        }
        
        // 2. App Bundle Contents/MacOS/eyemouse_engine
        let bundleEngine = Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/eyemouse_engine")
        if fm.isExecutableFile(atPath: bundleEngine.path) {
            return LaunchInfo(executableURL: bundleEngine, arguments: [])
        }
        
        // 3. Relative dist_engine/eyemouse_engine/eyemouse_engine (Development Frozen build)
        let distEngine = currentDir.appendingPathComponent("dist_engine/eyemouse_engine/eyemouse_engine")
        if fm.isExecutableFile(atPath: distEngine.path) {
            return LaunchInfo(executableURL: distEngine, arguments: [])
        }
        
        // 4. Fallback to .venv Python interpreter
        let venvPython = currentDir.appendingPathComponent(".venv/bin/python")
        if fm.isExecutableFile(atPath: venvPython.path) {
            print("[EngineManager] Using development fallback .venv interpreter")
            return LaunchInfo(executableURL: venvPython, arguments: ["-m", "src.vision_engine.main"])
        }
        
        return nil
    }
}
