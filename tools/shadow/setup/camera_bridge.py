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
  pip install numpy opencv-python requests

This is a proof-of-concept for shadow device camera integration.
For production use, consider RTSP streaming for lower latency.
"""

import argparse
import time
import sys
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install numpy opencv-python")
    sys.exit(1)


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

    def frame_to_nv12(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to NV12 YUV format (openpilot's expected format)."""
        # Convert BGR to YUV
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
        return nv12


def test_capture(url: str):
    """Test frame capture from IP Webcam."""
    print(f"Testing capture from: {url}")

    cap = IPWebcamCapture(url)

    # Test single frame capture
    print("\n1. Testing single frame capture...")
    frame = cap.get_single_frame()
    if frame is not None:
        print(f"   Success! Frame shape: {frame.shape}")

        # Test NV12 conversion
        nv12 = cap.frame_to_nv12(frame)
        print(f"   NV12 buffer size: {len(nv12)} bytes")
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


def run_bridge(url: str, display: bool = False):
    """Run the camera bridge, publishing frames to VisionIPC."""
    print(f"Starting camera bridge from: {url}")

    cap = IPWebcamCapture(url)
    stream = cap.open_mjpeg_stream()

    if stream is None:
        print("Failed to open stream, falling back to snapshot mode")
        use_stream = False
    else:
        use_stream = True

    frame_count = 0
    start_time = time.monotonic()

    try:
        while True:
            if use_stream:
                ret, frame = stream.read()
                if not ret:
                    print("Stream read failed, reconnecting...")
                    stream.release()
                    time.sleep(1)
                    stream = cap.open_mjpeg_stream()
                    continue
            else:
                frame = cap.get_single_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

            frame_count += 1

            # Convert to NV12 for openpilot
            nv12_data = cap.frame_to_nv12(frame)

            # TODO: Publish to VisionIPC here
            # This requires the msgq/visionipc module to be built
            # vipc_server.send(VisionStreamType.VISION_STREAM_ROAD, nv12_data, frame_id, timestamp)

            if display:
                cv2.imshow("Camera Bridge", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Print stats every 100 frames
            if frame_count % 100 == 0:
                elapsed = time.monotonic() - start_time
                fps = frame_count / elapsed
                print(f"Frames: {frame_count}, FPS: {fps:.1f}, NV12 size: {len(nv12_data)}")

    except KeyboardInterrupt:
        print("\nStopping camera bridge...")
    finally:
        if use_stream and stream:
            stream.release()
        if display:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    print(f"Total frames: {frame_count}, Average FPS: {frame_count/elapsed:.1f}")


def main():
    parser = argparse.ArgumentParser(description="Camera bridge for shadow device")
    parser.add_argument("--url", required=True, help="IP Webcam base URL (e.g., http://192.168.1.100:8080)")
    parser.add_argument("--test", action="store_true", help="Run capture test only")
    parser.add_argument("--display", action="store_true", help="Display frames in window")

    args = parser.parse_args()

    if args.test:
        success = test_capture(args.url)
        sys.exit(0 if success else 1)
    else:
        run_bridge(args.url, display=args.display)


if __name__ == "__main__":
    main()
