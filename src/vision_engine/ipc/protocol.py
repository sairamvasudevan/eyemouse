"""POSIX Shared Memory and Unix Domain Socket protocol definitions."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

from ..types import PixelFormat


MAGIC_GLOBAL = b"EMSH"  # EyeMouse Shared Header
MAGIC_SLOT = b"EYEM"    # EyeMouse Slot Header
PROTOCOL_VERSION = 1

GLOBAL_HEADER_SIZE = 64
SLOT_HEADER_SIZE = 64
DEFAULT_NUM_SLOTS = 4
DEFAULT_SLOT_SIZE = 4 * 1024 * 1024  # 4 MiB per slot (supports up to 1080p BGRA)

# Slot state machine
SLOT_STATE_FREE = 0       # Available for producer to write
SLOT_STATE_WRITING = 1    # Producer is currently writing pixel data
SLOT_STATE_READY = 2      # Frame is published and ready for consumer
SLOT_STATE_READING = 3    # Consumer is actively reading/processing frame

# Struct formats: Little-endian (<)
# Global header (64 bytes):
#   char magic[4]
#   uint32 version
#   uint32 num_slots
#   uint32 slot_size_bytes
#   int32  producer_pid
#   int32  consumer_pid
#   uint32 header_size_bytes
#   uint32 slot_header_size_bytes
#   char   reserved[32]
GLOBAL_HEADER_FORMAT = "<4sIIIiiII32s"

# Slot header (64 bytes):
#   char   magic[4]
#   uint32 version
#   uint32 slot_index
#   uint32 slot_state
#   uint64 sequence_num
#   uint64 timestamp_ns
#   uint32 width
#   uint32 height
#   uint32 stride
#   uint32 pixel_format
#   uint32 payload_size
#   char   reserved[12]
SLOT_HEADER_FORMAT = "<4sIIIQQIIIII12s"

assert struct.calcsize(GLOBAL_HEADER_FORMAT) == GLOBAL_HEADER_SIZE
assert struct.calcsize(SLOT_HEADER_FORMAT) == SLOT_HEADER_SIZE


@dataclass(slots=True)
class ShmGlobalMetadata:
    version: int
    num_slots: int
    slot_size: int
    producer_pid: int
    consumer_pid: int


@dataclass(slots=True)
class ShmSlotMetadata:
    slot_index: int
    slot_state: int
    sequence_num: int
    timestamp_ns: int
    width: int
    height: int
    stride: int
    pixel_format: PixelFormat
    payload_size: int


def pack_global_header(
    num_slots: int = DEFAULT_NUM_SLOTS,
    slot_size: int = DEFAULT_SLOT_SIZE,
    producer_pid: int = 0,
    consumer_pid: int = 0,
) -> bytes:
    """Pack the 64-byte global shared memory header."""
    return struct.pack(
        GLOBAL_HEADER_FORMAT,
        MAGIC_GLOBAL,
        PROTOCOL_VERSION,
        num_slots,
        slot_size,
        producer_pid,
        consumer_pid,
        GLOBAL_HEADER_SIZE,
        SLOT_HEADER_SIZE,
        b"\x00" * 32,
    )


def unpack_global_header(data: bytes) -> ShmGlobalMetadata:
    """Unpack and validate the 64-byte global shared memory header."""
    if len(data) < GLOBAL_HEADER_SIZE:
        raise ValueError(f"Global header buffer too short: {len(data)} < {GLOBAL_HEADER_SIZE}")
    magic, ver, n_slots, slot_sz, prod_pid, cons_pid, h_sz, s_sz, _ = struct.unpack(
        GLOBAL_HEADER_FORMAT, data[:GLOBAL_HEADER_SIZE]
    )
    if magic != MAGIC_GLOBAL:
        raise ValueError(f"Invalid global header magic: {magic!r}")
    if ver != PROTOCOL_VERSION:
        raise ValueError(f"Unsupported protocol version: {ver} (expected {PROTOCOL_VERSION})")
    return ShmGlobalMetadata(
        version=ver,
        num_slots=n_slots,
        slot_size=slot_sz,
        producer_pid=prod_pid,
        consumer_pid=cons_pid,
    )


def pack_slot_header(
    slot_index: int,
    slot_state: int,
    sequence_num: int,
    timestamp_ns: int,
    width: int,
    height: int,
    stride: int,
    pixel_format: PixelFormat,
    payload_size: int,
) -> bytes:
    """Pack the 64-byte slot header."""
    return struct.pack(
        SLOT_HEADER_FORMAT,
        MAGIC_SLOT,
        PROTOCOL_VERSION,
        slot_index,
        slot_state,
        sequence_num,
        timestamp_ns,
        width,
        height,
        stride,
        int(pixel_format),
        payload_size,
        b"\x00" * 12,
    )


def unpack_slot_header(data: bytes) -> ShmSlotMetadata:
    """Unpack and validate the 64-byte slot header."""
    if len(data) < SLOT_HEADER_SIZE:
        raise ValueError(f"Slot header buffer too short: {len(data)} < {SLOT_HEADER_SIZE}")
    (
        magic,
        ver,
        idx,
        state,
        seq,
        ts,
        w,
        h,
        stride,
        fmt,
        size,
        _,
    ) = struct.unpack(SLOT_HEADER_FORMAT, data[:SLOT_HEADER_SIZE])
    if magic != MAGIC_SLOT:
        raise ValueError(f"Invalid slot header magic: {magic!r}")
    return ShmSlotMetadata(
        slot_index=idx,
        slot_state=state,
        sequence_num=seq,
        timestamp_ns=ts,
        width=w,
        height=h,
        stride=stride,
        pixel_format=PixelFormat(fmt) if fmt in PixelFormat._value2member_map_ else PixelFormat.UNKNOWN,
        payload_size=size,
    )

