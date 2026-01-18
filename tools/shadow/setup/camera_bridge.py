#!/usr/bin/env python3
"""
Camera Bridge for Shadow Device

Captures frames from an IP Webcam HTTP stream and publishes them
via VisionIPC for openpilot consumption.

Usage:
  1. Install IP Webcam app on Android device
  2. Start streaming (note the IP:port)
  3. Run: python camera_bridge.py --url http://192.168.1.x:8080

Requirements:
  pip install numpy opencv-python av

This is a proof-of-concept for shadow device camera integration.
For production use, consider RTSP streaming for lower latency.
"""

import argparse
import os
import time
import sys
from typing import Optional

# Ensure openpilot root is in path for msgq/cereal imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPILOT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _OPENPILOT_ROOT not in sys.path:
    sys.path.insert(0, _OPENPILOT_ROOT)

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install numpy opencv-python av")
    sys.exit(1)

# Optional VisionIPC imports (only needed for full openpilot integration)
VIPC_AVAILABLE = False
try:
    import av
    from msgq.visionipc import VisionIpcServer, VisionStreamType
    from cereal import messaging
    VIPC_AVAILABLE = True
except ImportError:
    pass


class IPWebcamCapture:
    """Capture frames from IP Webcam Android app."""

    def __init__(self, base_url: str, camera_id: int = 0):
        self.base_url = base_url.rstrip('/')
        self.camera_id = camera_id
        self.frame_count = 0
        self.last_frame_time = 0

        # URL endpoints
        self.shot_url = f"{self.base_url}/shot.jpg"
        self.video_url = f"{self.base_url}/video"
        self.mjpeg_url = f"{self.base_url}/videofeed"

    def get_single_frame(self) -> Optional[np.ndarray]:
        """Fetch a single JPEG frame via HTTP."""
        import requests
        try:
            response = requests.get(self.shot_url, timeout=2)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, dtype=np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                self.frame_count += 1
                return frame
        except Exception as e:
            print(f"Error fetching frame: {e}")
        return None

    def open_mjpeg_stream(self) -> Optional[cv2.VideoCapture]:
        """Open MJPEG video stream for continuous capture."""
        # Try different URL formats
        urls_to_try = [
            self.mjpeg_url,
            f"{self.video_url}?dummy=param.mjpg",
            self.video_url,
        ]

        for url in urls_to_try:
            cap = cv2.VideoCapture(url)
            if cap.isOpened():
                print(f"Connected to stream: {url}")
                return cap
            cap.release()

        print("Could not open any video stream")
        return None

    @staticmethod
    def bgr2nv12(bgr: np.ndarray) -> bytes:
        """Convert BGR frame to NV12 YUV format using PyAV (preferred)."""
        if VIPC_AVAILABLE:
            frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
            return frame.reformat(format='nv12').to_ndarray().data.tobytes()
        else:
            # Fallback: manual conversion via OpenCV
            return IPWebcamCapture._bgr2nv12_opencv(bgr)

    @staticmethod
    def _bgr2nv12_opencv(frame: np.ndarray) -> bytes:
        """Fallback BGR to NV12 conversion using OpenCV."""
        # Convert BGR to YUV I420
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)

        height, width = frame.shape[:2]

        # I420 has Y, then U, then V planes
        # NV12 has Y, then interleaved UV
        y_size = width * height
        uv_size = y_size // 4

        y_plane = yuv[:height, :]
        u_plane = yuv[height:height + height//4, :].reshape(-1)
        v_plane = yuv[height + height//4:, :].reshape(-1)

        # Interleave U and V for NV12
        uv_interleaved = np.empty(uv_size * 2, dtype=np.uint8)
        uv_interleaved[0::2] = u_plane
        uv_interleaved[1::2] = v_plane

        # Combine Y and UV planes
        nv12 = np.concatenate([y_plane.flatten(), uv_interleaved])
        return nv12.tobytes()

    def frame_to_nv12(self, frame: np.ndarray) -> bytes:
        """Convert BGR frame to NV12 YUV format (openpilot's expected format)."""
        return self.bgr2nv12(frame)


def test_capture(url: str):
    """Test frame capture from IP Webcam."""
    print(f"Testing capture from: {url}")
    print(f"VisionIPC available: {VIPC_AVAILABLE}")

    cap = IPWebcamCapture(url)

    # Test single frame capture
    print("\n1. Testing single frame capture...")
    frame = cap.get_single_frame()
    if frame is not None:
        h, w = frame.shape[:2]
        print(f"   Success! Frame shape: {frame.shape} ({w}x{h})")

        # Test NV12 conversion
        nv12 = cap.frame_to_nv12(frame)
        expected_size = w * h + w * h // 2  # Y + UV planes
        print(f"   NV12 buffer size: {len(nv12)} bytes (expected: {expected_size})")
    else:
        print("   Failed to capture frame")
        return False

    # Test MJPEG stream
    print("\n2. Testing MJPEG stream...")
    stream = cap.open_mjpeg_stream()
    if stream is not None:
        frame_times = []
        for i in range(30):
            start = time.monotonic()
            ret, frame = stream.read()
            elapsed = time.monotonic() - start

            if ret:
                frame_times.append(elapsed)
            else:
                print(f"   Frame {i} failed")
                break

        stream.release()

        if frame_times:
            avg_time = sum(frame_times) / len(frame_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"   Captured {len(frame_times)} frames")
            print(f"   Average frame time: {avg_time*1000:.1f}ms ({fps:.1f} FPS)")
    else:
        print("   Could not open stream")
        return False

    print("\n3. Camera bridge test complete!")
    return True


class CameraBridge:
    """Camera bridge that publishes frames to VisionIPC."""

    def __init__(self, url: str, width: int = 1280, height: int = 720):
        self.url = url
        self.width = width
        self.height = height
        self.cap = IPWebcamCapture(url)
        self.frame_id = 0

        # Initialize VisionIPC if available
        self.vipc_server = None
        self.pm = None
        if VIPC_AVAILABLE:
            print("Initializing VisionIPC server...")
            self.pm = messaging.PubMaster(['roadCameraState'])
            self.vipc_server = VisionIpcServer("camerad")
            self.vipc_server.create_buffers(
                VisionStreamType.VISION_STREAM_ROAD,
                20,  # number of buffers
                width,
                height
            )
            self.vipc_server.start_listener()
            print(f"VisionIPC server started (buffers: {width}x{height})")
        else:
            print("VisionIPC not available - running in capture-only mode")

    def _send_frame(self, nv12_data: bytes):
        """Send frame to VisionIPC and publish metadata."""
        if self.vipc_server is None:
            return

        # Calculate timestamps (nanoseconds)
        eof = int(self.frame_id * 0.05 * 1e9)  # ~20 FPS timing

        # Send frame data via VisionIPC
        self.vipc_server.send(
            VisionStreamType.VISION_STREAM_ROAD,
            nv12_data,
            self.frame_id,
            eof,  # timestamp_sof
            eof   # timestamp_eof
        )

        # Publish camera state metadata via cereal
        dat = messaging.new_message('roadCameraState', valid=True)
        msg = {
            "frameId": self.frame_id,
            "transform": [1.0, 0.0, 0.0,
                          0.0, 1.0, 0.0,
                          0.0, 0.0, 1.0]
        }
        setattr(dat, 'roadCameraState', msg)
        self.pm.send('roadCameraState', dat)

    def run(self, display: bool = False):
        """Run the camera bridge loop."""
        print(f"Starting camera bridge from: {self.url}")

        stream = self.cap.open_mjpeg_stream()
        if stream is None:
            print("Failed to open stream, falling back to snapshot mode")
            use_stream = False
        else:
            use_stream = True

        start_time = time.monotonic()

        try:
            while True:
                if use_stream:
                    ret, frame = stream.read()
                    if not ret:
                        print("Stream read failed, reconnecting...")
                        stream.release()
                        time.sleep(1)
                        stream = self.cap.open_mjpeg_stream()
                        continue
                else:
                    frame = self.cap.get_single_frame()
                    if frame is None:
                        time.sleep(0.1)
                        continue

                # Resize if needed to match buffer dimensions
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))

                # Convert to NV12 for openpilot
                nv12_data = self.cap.frame_to_nv12(frame)

                # Publish to VisionIPC
                self._send_frame(nv12_data)
                self.frame_id += 1

                if display:
                    cv2.imshow("Camera Bridge", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                # Print stats every 100 frames
                if self.frame_id % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    fps = self.frame_id / elapsed
                    mode = "VisionIPC" if self.vipc_server else "capture-only"
                    print(f"Frames: {self.frame_id}, FPS: {fps:.1f}, Mode: {mode}")

        except KeyboardInterrupt:
            print("\nStopping camera bridge...")
        finally:
            if use_stream and stream:
                stream.release()
            if display:
                cv2.destroyAllWindows()

        elapsed = time.monotonic() - start_time
        print(f"Total frames: {self.frame_id}, Average FPS: {self.frame_id/elapsed:.1f}")


def run_bridge(url: str, display: bool = False, width: int = 1280, height: int = 720):
    """Run the camera bridge, publishing frames to VisionIPC."""
    bridge = CameraBridge(url, width, height)
    bridge.run(display)


def main():
    parser = argparse.ArgumentParser(description="Camera bridge for shadow device")
    parser.add_argument("--url", required=True, help="IP Webcam base URL (e.g., http://192.168.1.100:8080)")
    parser.add_argument("--test", action="store_true", help="Run capture test only")
    parser.add_argument("--display", action="store_true", help="Display frames in window")
    parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")

    args = parser.parse_args()

    if args.test:
        success = test_capture(args.url)
        sys.exit(0 if success else 1)
    else:
        run_bridge(args.url, display=args.display, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
