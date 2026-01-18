#!/usr/bin/env python3
"""
Simple VisionIPC consumer test for shadow device.

Tests that frames published by camera_bridge.py can be consumed
via VisionIPC without needing full modeld or OpenCL.

Usage:
  # Terminal 1: Start camera server (in Termux)
  python3 ~/termux_camera_server.py --port 8080

  # Terminal 2: Start camera bridge (in proot)
  python3 tools/shadow/setup/camera_bridge.py --url http://<ip>:8080

  # Terminal 3: Run this consumer test (in proot)
  python3 tools/shadow/setup/test_visionipc_consumer.py
"""

import os
import sys
import time

# Ensure openpilot root is in path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPILOT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _OPENPILOT_ROOT not in sys.path:
    sys.path.insert(0, _OPENPILOT_ROOT)

try:
    from msgq.visionipc import VisionIpcClient, VisionStreamType
    from cereal import messaging
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure msgq is built: cd msgq_repo && scons -j2")
    sys.exit(1)


def test_visionipc_available():
    """Check what VisionIPC streams are available."""
    print("=== Checking VisionIPC Availability ===")

    streams = VisionIpcClient.available_streams("camerad", block=False)
    if streams:
        print(f"Available streams from 'camerad': {streams}")
        for s in streams:
            name = {0: "ROAD", 1: "DRIVER", 2: "WIDE_ROAD"}.get(s, f"UNKNOWN({s})")
            print(f"  - {name} ({s})")
        return True
    else:
        print("No streams available. Is camera_bridge.py running?")
        return False


def test_frame_reception(timeout_sec: float = 10.0):
    """Test receiving frames from VisionIPC."""
    print("\n=== Testing Frame Reception ===")

    # Create client without OpenCL context (False = no CL)
    client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, False)

    print("Connecting to VisionIPC server...")
    start = time.monotonic()
    while not client.connect(False):
        if time.monotonic() - start > timeout_sec:
            print(f"Connection timeout after {timeout_sec}s")
            return False
        time.sleep(0.1)

    print("Connected! Waiting for frames...")

    frames_received = 0
    frame_times = []
    last_frame_time = time.monotonic()

    while frames_received < 10:
        buf = client.recv()
        if buf is not None:
            now = time.monotonic()
            dt = now - last_frame_time
            last_frame_time = now
            frame_times.append(dt)
            frames_received += 1

            print(f"  Frame {frames_received}: id={buf.frame_id}, "
                  f"size={buf.width}x{buf.height}, "
                  f"stride={buf.stride}, uv_offset={buf.uv_offset}, "
                  f"dt={dt*1000:.1f}ms")

        if time.monotonic() - start > timeout_sec:
            break

    if frames_received > 0:
        avg_dt = sum(frame_times[1:]) / len(frame_times[1:]) if len(frame_times) > 1 else 0
        fps = 1.0 / avg_dt if avg_dt > 0 else 0
        print(f"\nReceived {frames_received} frames")
        print(f"Average frame interval: {avg_dt*1000:.1f}ms ({fps:.2f} FPS)")
        return True
    else:
        print("No frames received")
        return False


def test_camera_state_subscription():
    """Test subscribing to roadCameraState messages."""
    print("\n=== Testing Camera State Subscription ===")

    sm = messaging.SubMaster(['roadCameraState'])

    print("Subscribing to roadCameraState...")
    frames_received = 0
    start = time.monotonic()

    while frames_received < 5 and time.monotonic() - start < 10:
        sm.update(100)  # 100ms timeout

        if sm.updated['roadCameraState']:
            msg = sm['roadCameraState']
            frames_received += 1
            print(f"  roadCameraState: frameId={msg.frameId}")

    if frames_received > 0:
        print(f"Received {frames_received} camera state messages")
        return True
    else:
        print("No camera state messages received")
        return False


def main():
    print("VisionIPC Consumer Test")
    print("=" * 50)
    print()

    # Test 1: Check availability
    available = test_visionipc_available()

    if not available:
        print("\nNo VisionIPC server running.")
        print("Start camera_bridge.py first:")
        print("  python3 tools/shadow/setup/camera_bridge.py --url http://<ip>:8080")
        return 1

    # Test 2: Frame reception
    frames_ok = test_frame_reception()

    # Test 3: Camera state messages
    state_ok = test_camera_state_subscription()

    print("\n" + "=" * 50)
    print("Results:")
    print(f"  VisionIPC available: {'PASS' if available else 'FAIL'}")
    print(f"  Frame reception: {'PASS' if frames_ok else 'FAIL'}")
    print(f"  Camera state msgs: {'PASS' if state_ok else 'FAIL'}")

    return 0 if (available and frames_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
