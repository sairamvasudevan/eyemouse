"""Test runner coordinating the cross-language Swift producer and Python vision engine."""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def run_cross_language_ipc_test():
    repo_root = Path(__file__).resolve().parent.parent
    swift_bin = repo_root / "build" / "swift_ipc_producer"
    assert swift_bin.is_file(), f"Swift producer binary not found: {swift_bin}. Run 'make build-swift-ipc' first."

    session_id = f"test_{uuid.uuid4().hex[:8]}"
    socket_path = f"/tmp/eyemouse_ctrl_{session_id}.sock"
    shm_name = f"eyemouse_shm_{session_id}"

    # Clean up any preexisting artifacts
    if os.path.exists(socket_path):
        os.unlink(socket_path)

    python_bin = sys.executable
    engine_cmd = [
        python_bin,
        "-m",
        "src.vision_engine.main",
        "--socket",
        socket_path,
    ]

    print(f"==> Launching Python Vision Engine (socket: {socket_path})")
    engine_proc = subprocess.Popen(
        engine_cmd,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait for the control socket to appear
    for _ in range(50):
        if os.path.exists(socket_path):
            break
        time.sleep(0.05)
    else:
        engine_proc.terminate()
        stdout, stderr = engine_proc.communicate()
        print("Engine stdout:", stdout)
        print("Engine stderr:", stderr)
        raise RuntimeError("Python Vision Engine failed to create control socket within 2.5 seconds.")

    print(f"==> Running Swift Producer: {swift_bin} {socket_path} {shm_name}")
    producer_cmd = [str(swift_bin), socket_path, shm_name]
    producer_proc = subprocess.run(
        producer_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    print("Producer Output:\n" + producer_proc.stdout)
    if producer_proc.returncode != 0:
        print("Producer Error:\n" + producer_proc.stderr)
        engine_proc.terminate()
        engine_stdout, engine_stderr = engine_proc.communicate()
        print("Engine stderr:", engine_stderr)
        raise RuntimeError(f"Swift IPC Producer exited with code {producer_proc.returncode}")

    assert "SWIFT_IPC_PRODUCER_SUCCESS" in producer_proc.stdout, "Missing success string from producer"

    # Wait for engine to exit gracefully after receiving SHUTDOWN
    engine_proc.wait(timeout=5.0)
    assert engine_proc.returncode == 0, f"Engine exited with non-zero code {engine_proc.returncode}"
    print("==> CROSS-LANGUAGE SWIFT-TO-PYTHON SHARED-MEMORY IPC TEST PASSED!")


if __name__ == "__main__":
    run_cross_language_ipc_test()

