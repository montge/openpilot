#!/usr/bin/env python3
"""
Frame Streamer for Shadow Device

Captures frames from VisionIPC (published by camera_bridge.py) and streams
them to a remote inference server over ZeroMQ.

Usage:
  # On shadow device (after camera_bridge.py is running):
  python frame_streamer.py --server tcp://192.168.1.100:5555

Architecture:
  camera_bridge.py → VisionIPC → frame_streamer.py → ZMQ → inference_server.py

The frame streamer:
1. Subscribes to VisionIPC frames from camerad (camera_bridge)
2. Encodes NV12 frames to JPEG for bandwidth efficiency
3. Publishes frames over ZMQ to the inference server
"""

import argparse
import os
import struct
import sys
import time
from typing import Optional

# Ensure openpilot root is in path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPILOT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _OPENPILOT_ROOT not in sys.path:
    sys.path.insert(0, _OPENPILOT_ROOT)

try:
    import zmq
    import cv2
    import numpy as np
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyzmq opencv-python numpy")
    sys.exit(1)

# VisionIPC imports
VIPC_AVAILABLE = False
try:
    from msgq.visionipc import VisionIpcClient, VisionStreamType
    VIPC_AVAILABLE = True
except ImportError:
    pass


def nv12_to_bgr(nv12_data: bytes, width: int, height: int) -> Optional[np.ndarray]:
    """Convert NV12 YUV data to BGR for JPEG encoding."""
    # NV12 layout: Y plane (width*height) + UV plane (width*height/2)
    expected_size = width * height + width * height // 2
    if len(nv12_data) != expected_size:
        print(f"NV12 size mismatch: got {len(nv12_data)}, expected {expected_size}")
        return None

    # Reshape NV12 data
    y_size = width * height
    y_plane = np.frombuffer(nv12_data[:y_size], dtype=np.uint8).reshape((height, width))
    uv_plane = np.frombuffer(nv12_data[y_size:], dtype=np.uint8).reshape((height // 2, width))

    # Create YUV image in OpenCV format (Y, UV planes stacked)
    yuv = np.vstack([y_plane, uv_plane])

    # Convert NV12 to BGR
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    return bgr


class FrameStreamer:
    """
    Streams VisionIPC frames to remote inference server via ZeroMQ.
    """

    def __init__(self, server_url: str, jpeg_quality: int = 80,
                 target_fps: float = 20.0):
        self.server_url = server_url
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.running = False
        self.frame_count = 0
        self.bytes_sent = 0

        # ZeroMQ context and socket
        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.PUB)
        self.socket.connect(server_url)
        # Give socket time to connect
        time.sleep(0.5)

        # VisionIPC client
        self.vipc_client = None
        if VIPC_AVAILABLE:
            print("Connecting to VisionIPC...")
            self.vipc_client = VisionIpcClient(
                "camerad",
                VisionStreamType.VISION_STREAM_ROAD,
                False  # conflate - drop old frames
            )
            # Wait for connection
            for i in range(50):  # 5 second timeout
                if self.vipc_client.connect(False):
                    print("VisionIPC connected!")
                    break
                time.sleep(0.1)
            else:
                print("WARNING: VisionIPC connection timeout")
        else:
            print("ERROR: VisionIPC not available - cannot stream frames")

        print(f"Frame streamer initialized:")
        print(f"  Server: {server_url}")
        print(f"  JPEG quality: {jpeg_quality}")
        print(f"  Target FPS: {target_fps}")

    def _encode_frame(self, bgr: np.ndarray) -> bytes:
        """Encode BGR frame to JPEG."""
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        success, jpeg_data = cv2.imencode('.jpg', bgr, encode_params)
        if success:
            return jpeg_data.tobytes()
        return b''

    def _send_frame(self, jpeg_data: bytes, frame_id: int, width: int, height: int):
        """Send frame over ZMQ."""
        # Message format: [topic, metadata, jpeg_data]
        # Metadata: frame_id (Q), width (I), height (I)
        metadata = struct.pack("QII", frame_id, width, height)
        self.socket.send_multipart([
            b"frame",
            metadata,
            jpeg_data
        ])
        self.frame_count += 1
        self.bytes_sent += len(jpeg_data)

    def run(self):
        """Main streaming loop."""
        if self.vipc_client is None or not self.vipc_client.is_connected():
            print("ERROR: VisionIPC not connected")
            return

        self.running = True
        start_time = time.monotonic()
        last_frame_time = 0

        print("\nStreaming frames to server...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                # Receive frame from VisionIPC
                buf = self.vipc_client.recv()
                if buf is None:
                    time.sleep(0.01)
                    continue

                # Rate limiting
                now = time.monotonic()
                if now - last_frame_time < self.frame_interval:
                    continue
                last_frame_time = now

                # Get frame dimensions
                width = buf.width
                height = buf.height
                frame_id = buf.frame_id

                # Convert NV12 to BGR
                nv12_data = bytes(buf.data)
                bgr = nv12_to_bgr(nv12_data, width, height)
                if bgr is None:
                    continue

                # Encode to JPEG
                jpeg_data = self._encode_frame(bgr)
                if not jpeg_data:
                    continue

                # Send to server
                self._send_frame(jpeg_data, frame_id, width, height)

                # Stats every 100 frames
                if self.frame_count % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    fps = self.frame_count / elapsed
                    bandwidth = self.bytes_sent / elapsed / 1024 / 1024  # MB/s
                    print(f"Frames: {self.frame_count}, FPS: {fps:.1f}, "
                          f"Bandwidth: {bandwidth:.2f} MB/s, "
                          f"Avg JPEG: {self.bytes_sent // self.frame_count // 1024}KB")

        except KeyboardInterrupt:
            print("\nStopping streamer...")
        finally:
            self.running = False

        elapsed = time.monotonic() - start_time
        print(f"\nStreamer stopped:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Total data: {self.bytes_sent / 1024 / 1024:.1f} MB")
        print(f"  Average FPS: {self.frame_count / elapsed:.1f}")
        print(f"  Average bandwidth: {self.bytes_sent / elapsed / 1024 / 1024:.2f} MB/s")

    def close(self):
        """Clean up resources."""
        self.running = False
        self.socket.close()
        self.zmq_ctx.term()


def test_vipc():
    """Test VisionIPC connection without streaming."""
    print("Testing VisionIPC connection...")
    print(f"VisionIPC available: {VIPC_AVAILABLE}")

    if not VIPC_AVAILABLE:
        print("ERROR: VisionIPC not available")
        return False

    client = VisionIpcClient(
        "camerad",
        VisionStreamType.VISION_STREAM_ROAD,
        False
    )

    print("Waiting for VisionIPC server (5s timeout)...")
    for i in range(50):
        if client.connect(False):
            print("Connected!")
            break
        time.sleep(0.1)
    else:
        print("Connection timeout - is camera_bridge.py running?")
        return False

    print("Receiving test frame...")
    buf = None
    for i in range(100):  # 10 second timeout
        buf = client.recv()
        if buf is not None:
            break
        time.sleep(0.1)

    if buf:
        print(f"Received frame:")
        print(f"  Frame ID: {buf.frame_id}")
        print(f"  Dimensions: {buf.width}x{buf.height}")
        print(f"  Data size: {len(buf.data)} bytes")

        # Test NV12 to BGR conversion
        bgr = nv12_to_bgr(bytes(buf.data), buf.width, buf.height)
        if bgr is not None:
            print(f"  NV12 → BGR: OK ({bgr.shape})")

            # Test JPEG encoding
            _, jpeg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print(f"  BGR → JPEG: OK ({len(jpeg)} bytes)")
        else:
            print("  NV12 → BGR: FAILED")
            return False

        return True
    else:
        print("Timeout waiting for frame")
        return False


def test_zmq(server_url: str):
    """Test ZMQ connection by sending a dummy frame."""
    print(f"Testing ZMQ connection to {server_url}...")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUB)
    socket.connect(server_url)
    time.sleep(0.5)  # Allow connection to establish

    # Create dummy frame (100x100 black image)
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    _, jpeg = cv2.imencode('.jpg', dummy, [cv2.IMWRITE_JPEG_QUALITY, 80])
    jpeg_data = jpeg.tobytes()

    # Send dummy frame
    metadata = struct.pack("QII", 0, 100, 100)
    socket.send_multipart([b"frame", metadata, jpeg_data])

    print(f"Sent test frame ({len(jpeg_data)} bytes)")
    print("Check if inference_server.py received it")

    socket.close()
    ctx.term()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Stream VisionIPC frames to remote inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Stream to server:
  python frame_streamer.py --server tcp://192.168.1.100:5555

  # Custom quality and FPS:
  python frame_streamer.py --server tcp://192.168.1.100:5555 --quality 90 --fps 15

  # Test VisionIPC connection:
  python frame_streamer.py --test-vipc

  # Test ZMQ connection:
  python frame_streamer.py --test-zmq --server tcp://192.168.1.100:5555
"""
    )
    parser.add_argument("--server", type=str,
                        help="Inference server URL (e.g., tcp://192.168.1.100:5555)")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Target FPS (default: 20)")
    parser.add_argument("--test-vipc", action="store_true",
                        help="Test VisionIPC connection")
    parser.add_argument("--test-zmq", action="store_true",
                        help="Test ZMQ connection")

    args = parser.parse_args()

    if args.test_vipc:
        success = test_vipc()
        sys.exit(0 if success else 1)
    elif args.test_zmq:
        if not args.server:
            print("ERROR: --server required for --test-zmq")
            sys.exit(1)
        success = test_zmq(args.server)
        sys.exit(0 if success else 1)
    else:
        if not args.server:
            print("ERROR: --server is required")
            parser.print_help()
            sys.exit(1)

        streamer = FrameStreamer(
            server_url=args.server,
            jpeg_quality=args.quality,
            target_fps=args.fps
        )
        try:
            streamer.run()
        finally:
            streamer.close()


if __name__ == "__main__":
    main()
