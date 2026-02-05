#!/usr/bin/env python3
"""
MJPEG to ZMQ Streamer for Shadow Device

Captures frames from IP Webcam's MJPEG stream and sends them directly
to a remote inference server via ZeroMQ. This bypasses VisionIPC for
environments where it's not available.

Usage:
  1. Install IP Webcam app on Android device
  2. Start streaming (note the URL, e.g., http://192.168.1.100:8080)
  3. On shadow device:
     python mjpeg_zmq_streamer.py --camera http://localhost:8080 --server tcp://desktop:5555

Requirements:
  pip install pyzmq opencv-python-headless numpy
"""

import argparse
import struct
import sys
import time

try:
    import cv2
    import numpy as np
    import zmq
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyzmq opencv-python-headless numpy")
    sys.exit(1)


class MJPEGZMQStreamer:
    """Stream MJPEG frames to ZMQ server."""

    def __init__(self, camera_url: str, server_url: str,
                 jpeg_quality: int = 80, target_fps: float = 20.0,
                 resize_width: int = 1280):
        self.camera_url = camera_url
        self.server_url = server_url
        self.jpeg_quality = jpeg_quality
        self.target_fps = target_fps
        self.resize_width = resize_width
        self.frame_interval = 1.0 / target_fps

        # Stats
        self.frame_count = 0
        self.bytes_sent = 0
        self.start_time = None

        # ZMQ setup
        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.PUB)
        self.socket.connect(server_url)

        # Camera capture
        self.cap = None

        print(f"MJPEG → ZMQ Streamer")
        print(f"  Camera: {camera_url}")
        print(f"  Server: {server_url}")
        print(f"  Quality: {jpeg_quality}")
        print(f"  Target FPS: {target_fps}")
        print(f"  Resize width: {resize_width}")

    def _connect_camera(self) -> bool:
        """Connect to MJPEG stream."""
        # Try different URL patterns for IP Webcam
        urls_to_try = [
            f"{self.camera_url}/videofeed",
            f"{self.camera_url}/video?dummy=param.mjpg",
            f"{self.camera_url}/video",
            self.camera_url,
        ]

        for url in urls_to_try:
            print(f"  Trying: {url}")
            self.cap = cv2.VideoCapture(url)
            if self.cap.isOpened():
                print(f"  Connected!")
                return True
            self.cap.release()

        print("  Failed to connect to camera stream")
        return False

    def _encode_frame(self, frame: np.ndarray) -> tuple[bytes, int, int]:
        """Resize and encode frame to JPEG."""
        h, w = frame.shape[:2]

        # Resize if needed
        if w != self.resize_width:
            scale = self.resize_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (self.resize_width, new_h))
            h, w = frame.shape[:2]

        # Encode to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        _, jpeg = cv2.imencode('.jpg', frame, encode_params)

        return jpeg.tobytes(), w, h

    def _send_frame(self, jpeg_data: bytes, frame_id: int, width: int, height: int):
        """Send frame over ZMQ."""
        metadata = struct.pack("QII", frame_id, width, height)
        self.socket.send_multipart([b"frame", metadata, jpeg_data])
        self.frame_count += 1
        self.bytes_sent += len(jpeg_data)

    def run(self):
        """Main streaming loop."""
        print("\nConnecting to camera...")
        if not self._connect_camera():
            return

        # Allow ZMQ connection to establish
        time.sleep(1)

        self.start_time = time.monotonic()
        last_frame_time = 0
        last_stats_time = self.start_time

        print("\nStreaming... Press Ctrl+C to stop\n")

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Frame read failed, reconnecting...")
                    self.cap.release()
                    time.sleep(1)
                    if not self._connect_camera():
                        break
                    continue

                # Rate limiting
                now = time.monotonic()
                if now - last_frame_time < self.frame_interval:
                    continue
                last_frame_time = now

                # Encode and send
                jpeg_data, w, h = self._encode_frame(frame)
                self._send_frame(jpeg_data, self.frame_count, w, h)

                # Stats every 2 seconds
                if now - last_stats_time >= 2.0:
                    elapsed = now - self.start_time
                    fps = self.frame_count / elapsed
                    bandwidth = self.bytes_sent / elapsed / 1024  # KB/s
                    avg_size = self.bytes_sent / self.frame_count / 1024 if self.frame_count > 0 else 0
                    print(f"Frames: {self.frame_count:5d} | FPS: {fps:5.1f} | "
                          f"BW: {bandwidth:6.1f} KB/s | Avg: {avg_size:5.1f} KB/frame")
                    last_stats_time = now

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            if self.cap:
                self.cap.release()

        # Final stats
        elapsed = time.monotonic() - self.start_time
        print(f"\nSession complete:")
        print(f"  Total frames: {self.frame_count}")
        print(f"  Total data: {self.bytes_sent / 1024 / 1024:.1f} MB")
        print(f"  Average FPS: {self.frame_count / elapsed:.1f}")
        print(f"  Average bandwidth: {self.bytes_sent / elapsed / 1024:.1f} KB/s")

    def close(self):
        """Clean up."""
        if self.cap:
            self.cap.release()
        self.socket.close()
        self.zmq_ctx.term()


def test_camera(url: str):
    """Test camera connection and measure FPS."""
    print(f"Testing camera: {url}")

    # Try different URL patterns
    urls_to_try = [
        f"{url}/videofeed",
        f"{url}/video?dummy=param.mjpg",
        f"{url}/video",
        url,
    ]

    cap = None
    for test_url in urls_to_try:
        print(f"  Trying: {test_url}")
        cap = cv2.VideoCapture(test_url)
        if cap.isOpened():
            print(f"  Connected!")
            break
        cap.release()
        cap = None

    if cap is None:
        print("Failed to connect to camera")
        return False

    # Capture test frames
    print("\nCapturing 50 test frames...")
    frame_times = []
    for i in range(50):
        t0 = time.monotonic()
        ret, frame = cap.read()
        t1 = time.monotonic()

        if ret:
            frame_times.append(t1 - t0)
            if i == 0:
                h, w = frame.shape[:2]
                print(f"  Frame size: {w}x{h}")
        else:
            print(f"  Frame {i} failed")

    cap.release()

    if frame_times:
        avg_time = sum(frame_times) / len(frame_times)
        fps = 1.0 / avg_time if avg_time > 0 else 0
        print(f"\nResults:")
        print(f"  Captured: {len(frame_times)} frames")
        print(f"  Avg frame time: {avg_time*1000:.1f} ms")
        print(f"  Estimated FPS: {fps:.1f}")
        return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Stream MJPEG camera feed to ZMQ server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test camera connection:
  python mjpeg_zmq_streamer.py --test --camera http://localhost:8080

  # Stream to server:
  python mjpeg_zmq_streamer.py --camera http://localhost:8080 --server tcp://192.168.1.100:5555

  # Custom settings:
  python mjpeg_zmq_streamer.py --camera http://localhost:8080 --server tcp://192.168.1.100:5555 \\
      --quality 70 --fps 15 --width 1280
"""
    )
    parser.add_argument("--camera", required=True,
                        help="IP Webcam base URL (e.g., http://localhost:8080)")
    parser.add_argument("--server", type=str,
                        help="ZMQ server URL (e.g., tcp://192.168.1.100:5555)")
    parser.add_argument("--quality", type=int, default=80,
                        help="JPEG quality 1-100 (default: 80)")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="Target FPS (default: 20)")
    parser.add_argument("--width", type=int, default=1280,
                        help="Resize width (default: 1280)")
    parser.add_argument("--test", action="store_true",
                        help="Test camera connection only")

    args = parser.parse_args()

    if args.test:
        success = test_camera(args.camera)
        sys.exit(0 if success else 1)
    else:
        if not args.server:
            print("ERROR: --server is required for streaming")
            parser.print_help()
            sys.exit(1)

        streamer = MJPEGZMQStreamer(
            camera_url=args.camera,
            server_url=args.server,
            jpeg_quality=args.quality,
            target_fps=args.fps,
            resize_width=args.width
        )
        try:
            streamer.run()
        finally:
            streamer.close()


if __name__ == "__main__":
    main()
