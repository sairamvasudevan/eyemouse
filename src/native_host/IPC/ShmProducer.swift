import Foundation
import CoreVideo
import Darwin

@_silgen_name("eyemouse_shm_open")
func eyemouse_shm_open(_ name: UnsafePointer<CChar>, _ oflag: CInt, _ mode: CInt) -> CInt

public final class ShmProducer {
    public let shmName: String
    public let numSlots: UInt32
    public let slotSize: UInt32
    public let totalShmSize: Int
    
    private var basePtr: UnsafeMutableRawPointer?
    private var isMapped = false
    
    public init?(shmName: String, numSlots: UInt32 = BundleConstants.numSlots, slotSize: UInt32 = BundleConstants.slotSize) {
        var cleanName = shmName
        if !cleanName.hasPrefix("/") {
            cleanName = "/" + cleanName
        }
        self.shmName = cleanName
        self.numSlots = numSlots
        self.slotSize = slotSize
        self.totalShmSize = BundleConstants.globalHeaderSize + Int(numSlots * slotSize)
        
        // Clean up preexisting
        shm_unlink(self.shmName)
        
        let fd = eyemouse_shm_open(self.shmName, O_CREAT | O_RDWR, 0o666)
        guard fd >= 0 else {
            perror("[ShmProducer] eyemouse_shm_open failed")
            return nil
        }
        defer { Darwin.close(fd) }
        
        if ftruncate(fd, off_t(totalShmSize)) != 0 {
            perror("[ShmProducer] ftruncate failed")
            shm_unlink(self.shmName)
            return nil
        }
        
        let ptr = mmap(nil, totalShmSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)
        guard ptr != MAP_FAILED, let validBase = ptr else {
            perror("[ShmProducer] mmap failed")
            shm_unlink(self.shmName)
            return nil
        }
        self.basePtr = validBase
        self.isMapped = true
        
        initializeGlobalHeader()
        print("[ShmProducer] Shared memory mapped: \(self.shmName) (\(totalShmSize) bytes, \(numSlots) slots)")
    }
    
    deinit {
        close()
    }
    
    public func close() {
        guard isMapped, let ptr = basePtr else { return }
        munmap(ptr, totalShmSize)
        shm_unlink(shmName)
        basePtr = nil
        isMapped = false
        print("[ShmProducer] Shared memory unmapped and unlinked: \(shmName)")
    }
    
    private func initializeGlobalHeader() {
        guard let base = basePtr else { return }
        memset(base, 0, totalShmSize)
        
        // Magic
        let magicPtr = base.bindMemory(to: UInt8.self, capacity: 4)
        for i in 0..<4 { magicPtr[i] = ShmHeaderTypes.magicGlobal[i] }
        
        // Version (offset 4)
        base.advanced(by: 4).bindMemory(to: UInt32.self, capacity: 1).pointee = BundleConstants.protocolVersion.littleEndian
        // Num slots (offset 8)
        base.advanced(by: 8).bindMemory(to: UInt32.self, capacity: 1).pointee = numSlots.littleEndian
        // Slot size (offset 12)
        base.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = slotSize.littleEndian
        // Producer PID (offset 16)
        base.advanced(by: 16).bindMemory(to: Int32.self, capacity: 1).pointee = getpid().littleEndian
        // Consumer PID (offset 20)
        base.advanced(by: 20).bindMemory(to: Int32.self, capacity: 1).pointee = 0
        // Header size (offset 24)
        base.advanced(by: 24).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(BundleConstants.globalHeaderSize).littleEndian
        // Slot header size (offset 28)
        base.advanced(by: 28).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(BundleConstants.slotHeaderSize).littleEndian
    }
    
    /// Write raw CVPixelBuffer into an available slot. Returns slot index on success.
    public func writeFrame(
        pixelBuffer: CVPixelBuffer,
        sequenceNum: UInt64,
        timestampNs: UInt64
    ) -> Int? {
        guard let base = basePtr else { return nil }
        
        // Find an available FREE slot
        var targetSlot: Int? = nil
        for s in 0..<Int(numSlots) {
            let offset = BundleConstants.globalHeaderSize + (s * Int(slotSize))
            let state = base.advanced(by: offset + 12).bindMemory(to: UInt32.self, capacity: 1).pointee.littleEndian
            if state == ShmHeaderTypes.slotStateFree {
                targetSlot = s
                break
            }
        }
        
        // If no slot is FREE, reclaim the oldest READY slot (stale-frame drop) to avoid blocking
        if targetSlot == nil {
            for s in 0..<Int(numSlots) {
                let offset = BundleConstants.globalHeaderSize + (s * Int(slotSize))
                let state = base.advanced(by: offset + 12).bindMemory(to: UInt32.self, capacity: 1).pointee.littleEndian
                if state == ShmHeaderTypes.slotStateReady {
                    targetSlot = s
                    break
                }
            }
        }
        
        guard let slotIdx = targetSlot else {
            // All slots are currently being written or read by consumer
            return nil
        }
        
        let slotOffset = BundleConstants.globalHeaderSize + (slotIdx * Int(slotSize))
        let slotPtr = base.advanced(by: slotOffset)
        
        // 1. Mark slot state WRITING (offset 12)
        slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = ShmHeaderTypes.slotStateWriting.littleEndian
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let srcPtr = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = ShmHeaderTypes.slotStateFree.littleEndian
            return nil
        }
        
        let width = UInt32(CVPixelBufferGetWidth(pixelBuffer))
        let height = UInt32(CVPixelBufferGetHeight(pixelBuffer))
        let bytesPerRow = UInt32(CVPixelBufferGetBytesPerRow(pixelBuffer))
        let payloadSize = bytesPerRow * height
        
        guard payloadSize <= slotSize - UInt32(BundleConstants.slotHeaderSize) else {
            print("[ShmProducer] Payload size \(payloadSize) exceeds slot payload capacity.")
            slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = ShmHeaderTypes.slotStateFree.littleEndian
            return nil
        }
        
        // 2. Copy image pixel data into slot payload
        let destDataPtr = slotPtr.advanced(by: BundleConstants.slotHeaderSize)
        memcpy(destDataPtr, srcPtr, Int(payloadSize))
        
        // 3. Write Slot Header (64 bytes)
        // Magic (0..3)
        for i in 0..<4 { slotPtr.bindMemory(to: UInt8.self, capacity: 4)[i] = ShmHeaderTypes.magicSlot[i] }
        // Version (4..7)
        slotPtr.advanced(by: 4).bindMemory(to: UInt32.self, capacity: 1).pointee = BundleConstants.protocolVersion.littleEndian
        // Slot index (8..11)
        slotPtr.advanced(by: 8).bindMemory(to: UInt32.self, capacity: 1).pointee = UInt32(slotIdx).littleEndian
        // Sequence num (16..23)
        slotPtr.advanced(by: 16).bindMemory(to: UInt64.self, capacity: 1).pointee = sequenceNum.littleEndian
        // Timestamp ns (24..31)
        slotPtr.advanced(by: 24).bindMemory(to: UInt64.self, capacity: 1).pointee = timestampNs.littleEndian
        // Width (32..35)
        slotPtr.advanced(by: 32).bindMemory(to: UInt32.self, capacity: 1).pointee = width.littleEndian
        // Height (36..39)
        slotPtr.advanced(by: 36).bindMemory(to: UInt32.self, capacity: 1).pointee = height.littleEndian
        // Stride / bytesPerRow (40..43)
        slotPtr.advanced(by: 40).bindMemory(to: UInt32.self, capacity: 1).pointee = bytesPerRow.littleEndian
        // Pixel format BGRA (44..47)
        slotPtr.advanced(by: 44).bindMemory(to: UInt32.self, capacity: 1).pointee = ShmHeaderTypes.pixelFormatBGRA.littleEndian
        // Payload size (48..51)
        slotPtr.advanced(by: 48).bindMemory(to: UInt32.self, capacity: 1).pointee = payloadSize.littleEndian
        
        // 4. Finally mark slot state READY (offset 12)
        slotPtr.advanced(by: 12).bindMemory(to: UInt32.self, capacity: 1).pointee = ShmHeaderTypes.slotStateReady.littleEndian
        
        return slotIdx
    }
}

