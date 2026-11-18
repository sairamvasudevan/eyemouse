"""Unit tests for POSIX shared memory protocol, slot state machine, and control plane."""

from __future__ import annotations

import os
import socket
import struct
import tempfile
import threading
import time
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import pytest

from src.vision_engine.types import PixelFormat
from src.vision_engine.ipc.protocol import (
    MAGIC_GLOBAL,
    MAGIC_SLOT,
    PROTOCOL_VERSION,
    GLOBAL_HEADER_SIZE,
    SLOT_HEADER_SIZE,
    SLOT_STATE_FREE,
    SLOT_STATE_WRITING,
    SLOT_STATE_READY,
    SLOT_STATE_READING,
    pack_global_header,
    unpack_global_header,
    pack_slot_header,
    unpack_slot_header,
)
from src.vision_engine.ipc.shm_consumer import ShmConsumer
from src.vision_engine.ipc.control_server import ControlServer


def test_global_header_pack_unpack():
    packed = pack_global_header(
        num_slots=4,
        slot_size=1024 * 1024,
        producer_pid=1234,
        consumer_pid=5678,
    )
    assert len(packed) == GLOBAL_HEADER_SIZE
    meta = unpack_global_header(packed)
    assert meta.version == PROTOCOL_VERSION
    assert meta.num_slots == 4
    assert meta.slot_size == 1024 * 1024
    assert meta.producer_pid == 1234
    assert meta.consumer_pid == 5678


def test_global_header_validation_errors():
    # Truncated
    with pytest.raises(ValueError, match="too short"):
        unpack_global_header(b"SHORT")

    # Invalid magic
    packed = pack_global_header()
    corrupt_magic = b"XXXX" + packed[4:]
    with pytest.raises(ValueError, match="Invalid global header magic"):
        unpack_global_header(corrupt_magic)

    # Invalid version
    corrupt_ver = packed[:4] + struct.pack("<I", 999) + packed[8:]
    with pytest.raises(ValueError, match="Unsupported protocol version"):
        unpack_global_header(corrupt_ver)


def test_slot_header_pack_unpack():
    packed = pack_slot_header(
        slot_index=2,
        slot_state=SLOT_STATE_READY,
        sequence_num=42,
        timestamp_ns=1_000_000_000,
        width=1280,
        height=720,
        stride=5120,
        pixel_format=PixelFormat.BGRA,
        payload_size=1280 * 720 * 4,
    )
    assert len(packed) == SLOT_HEADER_SIZE
    meta = unpack_slot_header(packed)
    assert meta.slot_index == 2
    assert meta.slot_state == SLOT_STATE_READY
    assert meta.sequence_num == 42
    assert meta.timestamp_ns == 1_000_000_000
    assert meta.width == 1280
    assert meta.height == 720
    assert meta.stride == 5120
    assert meta.pixel_format == PixelFormat.BGRA
    assert meta.payload_size == 1280 * 720 * 4


def test_slot_header_validation_errors():
    with pytest.raises(ValueError, match="too short"):
        unpack_slot_header(b"SHORT")

    packed = pack_slot_header(
        slot_index=0,
        slot_state=SLOT_STATE_FREE,
        sequence_num=1,
        timestamp_ns=100,
        width=640,
        height=480,
        stride=2560,
        pixel_format=PixelFormat.BGRA,
        payload_size=1000,
    )
    corrupt_magic = b"BAD!" + packed[4:]
    with pytest.raises(ValueError, match="Invalid slot header magic"):
        unpack_slot_header(corrupt_magic)


def test_shm_consumer_slot_lifecycle():
    num_slots = 2
    slot_size = 1024  # Small slot for unit test
    total_size = GLOBAL_HEADER_SIZE + (num_slots * slot_size)
    # Note: macOS PSHNAMLEN is 31 characters max including leading slash.
    shm_name = f"emt_{os.getpid()}_{int(time.time()) % 100000}"

    shm = SharedMemory(name=shm_name, create=True, size=total_size)
    try:
        # Write global header
        global_bytes = pack_global_header(num_slots=num_slots, slot_size=slot_size)
        shm.buf[:GLOBAL_HEADER_SIZE] = global_bytes

        # Open consumer
        consumer = ShmConsumer(f"/{shm_name}")
        assert consumer.num_slots == 2
        assert consumer.slot_size == 1024

        # Slot 0 in FREE state: consumer cannot acquire
        slot0_offset = consumer.get_slot_offset(0)
        slot0_header = pack_slot_header(
            slot_index=0,
            slot_state=SLOT_STATE_FREE,
            sequence_num=1,
            timestamp_ns=100,
            width=10,
            height=10,
            stride=40,
            pixel_format=PixelFormat.BGRA,
            payload_size=400,
        )
        shm.buf[slot0_offset : slot0_offset + SLOT_HEADER_SIZE] = slot0_header

        with pytest.raises(ValueError, match="not in READY state"):
            consumer.acquire_slot_for_reading(0)

        # Transition slot 0 to READY with test pattern
        slot0_ready = pack_slot_header(
            slot_index=0,
            slot_state=SLOT_STATE_READY,
            sequence_num=100,
            timestamp_ns=500_000,
            width=10,
            height=10,
            stride=40,
            pixel_format=PixelFormat.BGRA,
            payload_size=400,
        )
        shm.buf[slot0_offset : slot0_offset + SLOT_HEADER_SIZE] = slot0_ready
        # Write payload
        payload_data = np.full(400, fill_value=42, dtype=np.uint8)
        data_start = slot0_offset + SLOT_HEADER_SIZE
        shm.buf[data_start : data_start + 400] = payload_data.tobytes()

        # Acquire slot 0
        meta, raw_view = consumer.acquire_slot_for_reading(0)
        assert meta.sequence_num == 100
        assert meta.payload_size == 400
        assert len(raw_view) == 400
        assert raw_view[0] == 42
        assert raw_view[-1] == 42

        # Verify state in buffer is now READING
        cur_meta = consumer.read_slot_metadata(0)
        assert cur_meta.slot_state == SLOT_STATE_READING

        # Release slot 0
        consumer.release_slot(0)
        cur_meta = consumer.read_slot_metadata(0)
        assert cur_meta.slot_state == SLOT_STATE_FREE

        consumer.close()
    finally:
        shm.close()
        shm.unlink()


def test_shm_consumer_stale_slot_rejection():
    """Verify that consumer safely rejects a slot if producer reclaimed/overwrote it."""
    num_slots = 2
    slot_size = 1024
    total_size = GLOBAL_HEADER_SIZE + (num_slots * slot_size)
    shm_name = f"emt_{os.getpid()}_{int(time.time()) % 100000}_stale"

    shm = SharedMemory(name=shm_name, create=True, size=total_size)
    try:
        shm.buf[:GLOBAL_HEADER_SIZE] = pack_global_header(num_slots=num_slots, slot_size=slot_size)
        consumer = ShmConsumer(f"/{shm_name}")

        slot0_offset = consumer.get_slot_offset(0)

        # Producer was writing (slot state = WRITING), not READY yet
        shm.buf[slot0_offset : slot0_offset + SLOT_HEADER_SIZE] = pack_slot_header(
            slot_index=0,
            slot_state=SLOT_STATE_WRITING,
            sequence_num=50,
            timestamp_ns=100_000,
            width=10,
            height=10,
            stride=40,
            pixel_format=PixelFormat.BGRA,
            payload_size=400,
        )

        # Consumer should reject
        with pytest.raises(ValueError, match="not in READY state"):
            consumer.acquire_slot_for_reading(0)

        consumer.close()
    finally:
        shm.close()
        shm.unlink()


def test_control_server_communication():
    with tempfile.TemporaryDirectory() as tmpdir:
        sock_path = os.path.join(tmpdir, "test_ctrl.sock")
        server = ControlServer(sock_path)
        server.start()

        client_received = []

        def client_worker():
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            # Give server a moment to start listening
            time.sleep(0.05)
            client.connect(sock_path)
            # Send handshake
            client.sendall(b'{"type":"HANDSHAKE","protocol_version":1}\n')
            # Read response
            f = client.makefile("r", encoding="utf-8")
            resp = f.readline()
            client_received.append(resp)
            client.close()

        th = threading.Thread(target=client_worker)
        th.start()

        connected = server.accept_client(timeout=2.0)
        assert connected is True

        msg = server.read_message(timeout=2.0)
        assert msg is not None
        assert msg.get("type") == "HANDSHAKE"

        server.send_message({"type": "ACK", "status": "OK"})
        th.join(timeout=2.0)

        assert len(client_received) == 1
        assert "OK" in client_received[0]

        server.close()
        assert not os.path.exists(sock_path)
