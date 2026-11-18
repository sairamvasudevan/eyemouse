"""Zero-copy POSIX shared-memory consumer for video frame slots."""

from __future__ import annotations

import os
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Optional, Tuple
import numpy as np

from .protocol import (
    GLOBAL_HEADER_SIZE,
    SLOT_HEADER_SIZE,
    SLOT_STATE_FREE,
    SLOT_STATE_READY,
    SLOT_STATE_READING,
    ShmGlobalMetadata,
    ShmSlotMetadata,
    unpack_global_header,
    unpack_slot_header,
)
from ..types import PixelFormat


class ShmConsumer:
    """Manages POSIX shared memory mapping and zero-copy frame buffer extraction."""

    def __init__(self, shm_name: str) -> None:
        # Strip leading slash if present for cross-platform compatibility
        self.raw_name = shm_name
        clean_name = shm_name[1:] if shm_name.startswith("/") else shm_name
        self.shm: SharedMemory = SharedMemory(name=clean_name, create=False)
        self.buf = self.shm.buf

        # Unpack global header
        self.global_meta: ShmGlobalMetadata = unpack_global_header(bytes(self.buf[:GLOBAL_HEADER_SIZE]))
        self.num_slots = self.global_meta.num_slots
        self.slot_size = self.global_meta.slot_size

        expected_total_size = GLOBAL_HEADER_SIZE + (self.num_slots * self.slot_size)
        if len(self.buf) < expected_total_size:
            raise ValueError(
                f"Shared memory buffer size mismatch: {len(self.buf)} < {expected_total_size}"
            )

    def close(self) -> None:
        """Detach from shared memory."""
        self.shm.close()

    def get_slot_offset(self, slot_index: int) -> int:
        """Calculate byte offset of a slot within the shared memory buffer."""
        if not (0 <= slot_index < self.num_slots):
            raise IndexError(f"Slot index out of bounds: {slot_index} (num_slots: {self.num_slots})")
        return GLOBAL_HEADER_SIZE + (slot_index * self.slot_size)

    def read_slot_metadata(self, slot_index: int) -> ShmSlotMetadata:
        """Read and decode the slot header."""
        offset = self.get_slot_offset(slot_index)
        header_bytes = bytes(self.buf[offset : offset + SLOT_HEADER_SIZE])
        return unpack_slot_header(header_bytes)

    def acquire_slot_for_reading(self, slot_index: int) -> Tuple[ShmSlotMetadata, np.ndarray]:
        """
        Atomically transition slot state to READING and obtain a zero-copy numpy array of the frame.
        
        Returns:
            (ShmSlotMetadata, numpy array view of raw pixel data)
        """
        offset = self.get_slot_offset(slot_index)
        meta = self.read_slot_metadata(slot_index)

        if meta.slot_state != SLOT_STATE_READY:
            # Slot is not ready for reading (might have been overwritten or is free)
            raise ValueError(f"Slot {slot_index} is not in READY state (state: {meta.slot_state})")

        # Mark slot state as READING (offset 12 in slot header is uint32 slot_state)
        state_offset = offset + 12
        struct.pack_into("<I", self.buf, state_offset, SLOT_STATE_READING)

        data_start = offset + SLOT_HEADER_SIZE
        data_end = data_start + meta.payload_size
        if data_end > offset + self.slot_size:
            raise ValueError(f"Payload size {meta.payload_size} exceeds slot size {self.slot_size}")

        # Zero-copy view directly into shared memory buffer
        raw_array = np.ndarray(
            shape=(meta.payload_size,),
            dtype=np.uint8,
            buffer=self.buf,
            offset=data_start,
        )

        return meta, raw_array

    def release_slot(self, slot_index: int) -> None:
        """Mark slot state as FREE, allowing producer to reclaim and write new frames."""
        offset = self.get_slot_offset(slot_index)
        state_offset = offset + 12
        struct.pack_into("<I", self.buf, state_offset, SLOT_STATE_FREE)

