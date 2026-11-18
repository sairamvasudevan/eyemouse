import Foundation
import Darwin

// POSIX shm_open C bridge for macOS ARM64 variadic compatibility
@_silgen_name("eyemouse_shm_open")
func eyemouse_shm_open(_ name: UnsafePointer<CChar>, _ oflag: CInt, _ mode: CInt) -> CInt

// Protocol constants
let MAGIC_GLOBAL: [UInt8] = [0x45, 0x4D, 0x53, 0x48] // "EMSH"
let MAGIC_SLOT: [UInt8]   = [0x45, 0x59, 0x45, 0x4D] // "EYEM"
let PROTOCOL_VERSION: UInt32 = 1
let GLOBAL_HEADER_SIZE = 64
let SLOT_HEADER_SIZE = 64
let NUM_SLOTS: UInt32 = 4
let SLOT_SIZE: UInt32 = 4 * 1024 * 1024 // 4 MB

let SLOT_STATE_FREE: UInt32    = 0
let SLOT_STATE_WRITING: UInt32 = 1
let SLOT_STATE_READY: UInt32   = 2
let SLOT_STATE_READING: UInt32 = 3

let PIXEL_FORMAT_BGRA: UInt32  = 1

// Parse command line arguments
guard CommandLine.arguments.count >= 3 else {
    print("Usage: swift_ipc_producer <socket_path> <shm_name>")
    exit(1)
}
let socketPath = CommandLine.arguments[1]
var rawShmName = CommandLine.arguments[2]
if !rawShmName.hasPrefix("/") {
    rawShmName = "/" + rawShmName
}
let shmName = rawShmName

print("=== Swift POSIX Shared-Memory Test Producer ===")
print("Socket Path: \(socketPath)")
print("SHM Name:    \(shmName)")

// 1. Create and map POSIX Shared Memory
shm_unlink(shmName) // Clean up any stale segment
let totalShmSize = Int(GLOBAL_HEADER_SIZE) + Int(NUM_SLOTS * SLOT_SIZE)
let shmFd = eyemouse_shm_open(shmName, O_CREAT | O_RDWR, 0o666)
guard shmFd >= 0 else {
    perror("eyemouse_shm_open failed")
    exit(1)
}

if ftruncate(shmFd, off_t(totalShmSize)) != 0 {
    perror("ftruncate failed")
    close(shmFd)
    exit(1)
}

let mmapPtr = mmap(nil, totalShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
guard mmapPtr != MAP_FAILED, let basePtr = mmapPtr else {
    perror("mmap failed")
    close(shmFd)
    exit(1)
}
close(shmFd)

// 2. Initialize Global Header (64 bytes)
memset(basePtr, 0, totalShmSize)
let globalHeader = basePtr.bindMemory(to: UInt8.self, capacity: GLOBAL_HEADER_SIZE)
// Magic
for i in 0..<4 { globalHeader[i] = MAGIC_GLOBAL[i] }
// Version
basePtr.advanced(by: 4).bindMemory(to: UInt32.self, capacity: 1).pointee = PROTOCOL_VERSION.littleEndian
// Num slots
basePtr.advanced(by: 8).bindMemory(to: UInt32.self, capacity: 1).pointee = NUM_SLOTS.littleEndian
// Slot size
basePtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = SLOT_SIZE.littleEndian
// Producer PID
basePtr.advanced(by: 16).bindMemory(to: Int32.self, capacity: 1).pointee = getpid().littleEndian
// Consumer PID
basePtr.advanced(by: 20).bindMemory(to: Int32.self, capacity: 1).pointee = 0
// Header size
basePtr.advanced(by: 24).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(GLOBAL_HEADER_SIZE).littleEndian
// Slot header size
basePtr.advanced(by: 28).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(SLOT_HEADER_SIZE).littleEndian

print("Shared memory allocated and initialized: \(totalShmSize) bytes across \(NUM_SLOTS) slots.")

// 3. Connect to Unix Domain Socket
let clientSock = socket(AF_UNIX, SOCK_STREAM, 0)
guard clientSock >= 0 else {
    perror("socket failed")
    exit(1)
}

var serverAddr = sockaddr_un()
serverAddr.sun_family = sa_family_t(AF_UNIX)
let maxPathLen = MemoryLayout.size(ofValue: serverAddr.sun_path)
withUnsafeMutablePointer(to: &serverAddr.sun_path.0) { ptr in
    _ = socketPath.withCString { cstr in
        strncpy(ptr, cstr, maxPathLen - 1)
    }
}

let connectResult = withUnsafePointer(to: &serverAddr) { ptr in
    ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sa in
        connect(clientSock, sa, socklen_t(MemoryLayout<sockaddr_un>.size))
    }
}
guard connectResult == 0 else {
    perror("connect failed")
    exit(1)
}
print("Connected to Python Vision Engine control socket.")

// Helper: send line-delimited JSON
func sendMessage(_ dict: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: dict),
          var str = String(data: data, encoding: .utf8) else {
        fatalError("JSON encoding failed")
    }
    str += "\n"
    _ = str.withCString { cstr in
        write(clientSock, cstr, strlen(cstr))
    }
}

// Helper: read line-delimited JSON
func readResponse() -> [String: Any]? {
    var lineData = Data()
    while true {
        var byte: UInt8 = 0
        let n = read(clientSock, &byte, 1)
        if n <= 0 { return nil }
        if byte == UInt8(ascii: "\n") { break }
        lineData.append(byte)
    }
    guard let obj = try? JSONSerialization.jsonObject(with: lineData) as? [String: Any] else {
        return nil
    }
    return obj
}

// 4. Send HANDSHAKE
sendMessage([
    "cmd": "HANDSHAKE",
    "version": 1,
    "shm_name": shmName,
    "num_slots": NUM_SLOTS,
    "slot_size": SLOT_SIZE
])

guard let readyMsg = readResponse(), readyMsg["status"] as? String == "READY" else {
    print("Handshake failed! Received: \(String(describing: readResponse()))")
    exit(1)
}
print("Handshake SUCCESS: Vision Engine is READY.")

// Helper function to draw a synthetic face in a BGRA buffer
func drawSyntheticFace(into buffer: UnsafeMutablePointer<UInt8>, width: Int, height: Int) {
    // Fill background with light gray (B=200, G=205, R=210, A=255)
    for y in 0..<height {
        for x in 0..<width {
            let offset = (y * width + x) * 4
            buffer[offset + 0] = 200 // B
            buffer[offset + 1] = 205 // G
            buffer[offset + 2] = 210 // R
            buffer[offset + 3] = 255 // A
        }
    }
    let cx = width / 2
    let cy = height / 2

    // Draw face oval (B=180, G=195, R=235)
    for y in (cy - 120)...(cy + 120) {
        for x in (cx - 90)...(cx + 90) {
            let dx = Double(x - cx) / 90.0
            let dy = Double(y - cy) / 120.0
            if (dx * dx + dy * dy) <= 1.0 {
                let offset = (y * width + x) * 4
                buffer[offset + 0] = 180
                buffer[offset + 1] = 195
                buffer[offset + 2] = 235
                buffer[offset + 3] = 255
            }
        }
    }

    // Helper for circle
    func drawCircle(atX ox: Int, atY oy: Int, radius: Int, b: UInt8, g: UInt8, r: UInt8) {
        let rSq = radius * radius
        for y in (oy - radius)...(oy + radius) {
            for x in (ox - radius)...(ox + radius) {
                let dSq = (x - ox) * (x - ox) + (y - oy) * (y - oy)
                if dSq <= rSq && x >= 0 && x < width && y >= 0 && y < height {
                    let offset = (y * width + x) * 4
                    buffer[offset + 0] = b
                    buffer[offset + 1] = g
                    buffer[offset + 2] = r
                    buffer[offset + 3] = 255
                }
            }
        }
    }

    // Left eye (subject left = image right cx + 38)
    drawCircle(atX: cx + 38, atY: cy - 15, radius: 14, b: 255, g: 255, r: 255)
    drawCircle(atX: cx + 38, atY: cy - 15, radius: 5, b: 40, g: 60, r: 90) // iris
    // Right eye (subject right = image left cx - 38)
    drawCircle(atX: cx - 38, atY: cy - 15, radius: 14, b: 255, g: 255, r: 255)
    drawCircle(atX: cx - 38, atY: cy - 15, radius: 5, b: 40, g: 60, r: 90) // iris
}

// 5. Publish Test Frames Across Slots
let frameWidth = 640
let frameHeight = 480
let frameStride = frameWidth * 4
let payloadSize = frameStride * frameHeight
var numFramesToSend = 30
if CommandLine.arguments.count >= 4, let n = Int(CommandLine.arguments[3]), n > 0 {
    numFramesToSend = n
}

var totalEngineMs = 0.0
let startTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

for frameIdx in 1...numFramesToSend {
    let slotIdx = (frameIdx - 1) % Int(NUM_SLOTS)
    let slotOffset = Int(GLOBAL_HEADER_SIZE) + (slotIdx * Int(SLOT_SIZE))
    let slotPtr = basePtr.advanced(by: slotOffset)

    // Mark slot WRITING
    slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = SLOT_STATE_WRITING.littleEndian

    // Write frame pixel payload
    let dataPtr = slotPtr.advanced(by: SLOT_HEADER_SIZE).bindMemory(to: UInt8.self, capacity: payloadSize)
    drawSyntheticFace(into: dataPtr, width: frameWidth, height: frameHeight)

    // Write Slot Header
    for i in 0..<4 { slotPtr.bindMemory(to: UInt8.self, capacity: 4)[i] = MAGIC_SLOT[i] }
    slotPtr.advanced(by: 4).bindMemory(to: UInt32.self, capacity: 1).pointee = PROTOCOL_VERSION.littleEndian
    slotPtr.advanced(by: 8).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(slotIdx).littleEndian
    // Slot state -> READY
    slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = SLOT_STATE_READY.littleEndian
    // Sequence num
    slotPtr.advanced(by: 16).bindMemory(to: UInt64.self, capacity: 1).pointee = UInt64(frameIdx).littleEndian
    // Timestamp nanoseconds (monotonic simulated 30fps: ~33.3ms interval)
    let tsNs = UInt64(1_000_000_000) + UInt64(frameIdx) * 33_333_333
    slotPtr.advanced(by: 24).bindMemory(to: UInt64.self, capacity: 1).pointee = tsNs.littleEndian
    // Dimensions & format
    slotPtr.advanced(by: 32).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(frameWidth).littleEndian
    slotPtr.advanced(by: 36).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(frameHeight).littleEndian
    slotPtr.advanced(by: 40).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(frameStride).littleEndian
    slotPtr.advanced(by: 44).bindMemory(to: UInt32.self, capacity: 1).pointee = PIXEL_FORMAT_BGRA.littleEndian
    slotPtr.advanced(by: 48).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(payloadSize).littleEndian

    // Notify Vision Engine via control socket
    sendMessage([
        "cmd": "FRAME",
        "slot": slotIdx,
        "seq": frameIdx,
        "timestamp_ns": tsNs
    ])

    // Wait for ACK
    guard let ack = readResponse() else {
        print("ERROR: Failed to receive ACK for frame \(frameIdx)")
        exit(1)
    }

    let status = ack["status"] as? String ?? ""
    let faceDetected = ack["face_detected"] as? Bool ?? false
    let procTime = ack["processing_time_ms"] as? Double ?? 0.0
    totalEngineMs += procTime
    if frameIdx <= 5 || frameIdx % 10 == 0 || frameIdx == numFramesToSend {
        print("Frame [\(frameIdx)/\(numFramesToSend)] -> Slot \(slotIdx) ACK: face=\(faceDetected), engine=\(procTime)ms")
    }

    guard status == "ACK" && faceDetected else {
        print("ERROR: Frame \(frameIdx) processing failed or face not detected")
        exit(1)
    }

    // Verify that the consumer released the slot to FREE
    let finalSlotState = slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee.littleEndian
    guard finalSlotState == SLOT_STATE_FREE else {
        print("ERROR: Expected slot \(slotIdx) state to be FREE (0), got \(finalSlotState)")
        exit(1)
    }
}

let endTime = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
let elapsedSec = Double(endTime - startTime) / 1_000_000_000.0
let effectiveFps = Double(numFramesToSend) / elapsedSec
let avgEngineMs = totalEngineMs / Double(numFramesToSend)
print("=== Benchmark Summary ===")
print("Frames: \(numFramesToSend) | Elapsed: \(String(format: "%.3f", elapsedSec))s | FPS: \(String(format: "%.1f", effectiveFps)) | Avg engine time: \(String(format: "%.2f", avgEngineMs))ms")

// 6. Graceful SHUTDOWN
sendMessage(["cmd": "SHUTDOWN"])
_ = readResponse()

close(clientSock)
munmap(basePtr, totalShmSize)
shm_unlink(shmName)

print("SWIFT_IPC_PRODUCER_SUCCESS")
exit(0)
