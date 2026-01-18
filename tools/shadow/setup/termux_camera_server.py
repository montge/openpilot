#!/usr/bin/env python3
"""
Simple MJPEG Camera Server using termux-api

Captures frames using termux-camera-photo and serves them as an MJPEG stream.
This provides a camera streaming solution without requiring additional Android apps.

Usage:
  # In Termux (not proot):
  python3 termux_camera_server.py --port 8080

  # Then from proot or another device:
  python3 camera_bridge.py --url http://localhost:8080 --test

Requirements:
  - termux-api package (pkg install termux-api)
  - Termux:API app installed with camera permission

Note: Frame rate is limited by termux-api overhead (~2-5 FPS).
For higher frame rates, use IP Webcam or similar app.
"""

import argparse
import subprocess
import tempfile
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# Frame buffer for sharing between capture thread and HTTP handler
class FrameBuffer:
    def __init__(self):
        self.frame: Optional[bytes] = None
        self.lock = threading.Lock()
        self.frame_count = 0

    def set_frame(self, data: bytes):
        with self.lock:
            self.frame = data
            self.frame_count += 1

    def get_frame(self) -> Optional[bytes]:
        with self.lock:
            return self.frame

# Global frame buffer
frame_buffer = FrameBuffer()


def capture_loop(camera_id: int, temp_dir: str):
    """Continuously capture frames using termux-camera-photo."""
    frame_path = os.path.join(temp_dir, "frame.jpg")

    print(f"Starting capture loop with camera {camera_id}")

    while True:
        try:
            # Capture frame
            result = subprocess.run(
                ["termux-camera-photo", "-c", str(camera_id), frame_path],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0 and os.path.exists(frame_path):
                with open(frame_path, "rb") as f:
                    frame_data = f.read()

                if len(frame_data) > 0:
                    frame_buffer.set_frame(frame_data)

            # Small delay to avoid overwhelming the camera
            time.sleep(0.1)

        except subprocess.TimeoutExpired:
            print("Camera capture timeout")
        except Exception as e:
            print(f"Capture error: {e}")
            time.sleep(1)


class MJPEGHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming and single frame capture."""

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_GET(self):
        if self.path == "/shot.jpg":
            self.send_single_frame()
        elif self.path in ("/video", "/videofeed", "/stream"):
            self.send_mjpeg_stream()
        elif self.path == "/":
            self.send_index()
        else:
            self.send_error(404)

    def send_single_frame(self):
        """Send a single JPEG frame."""
        frame = frame_buffer.get_frame()

        if frame is None:
            self.send_error(503, "No frame available yet")
            return

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(frame)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(frame)

    def send_mjpeg_stream(self):
        """Send continuous MJPEG stream."""
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        last_frame_count = -1

        try:
            while True:
                # Wait for new frame
                while frame_buffer.frame_count == last_frame_count:
                    time.sleep(0.05)

                frame = frame_buffer.get_frame()
                last_frame_count = frame_buffer.frame_count

                if frame is None:
                    continue

                # Send frame
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(frame)}\r\n".encode())
                self.wfile.write(b"\r\n")
                self.wfile.write(frame)
                self.wfile.write(b"\r\n")

        except (BrokenPipeError, ConnectionResetError):
            pass  # Client disconnected

    def send_index(self):
        """Send simple HTML index page."""
        html = """<!DOCTYPE html>
<html>
<head><title>Termux Camera Server</title></head>
<body>
<h1>Termux Camera Server</h1>
<p>Endpoints:</p>
<ul>
<li><a href="/shot.jpg">/shot.jpg</a> - Single JPEG frame</li>
<li><a href="/video">/video</a> - MJPEG stream</li>
</ul>
<h2>Live Preview</h2>
<img src="/video" style="max-width: 100%;">
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    parser = argparse.ArgumentParser(description="MJPEG camera server using termux-api")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port (default: 8080)")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0 = back)")
    args = parser.parse_args()

    # Check if termux-camera-photo is available
    try:
        result = subprocess.run(
            ["termux-camera-photo", "-c", "0", "/dev/null"],
            capture_output=True,
            timeout=10
        )
        # Even if it fails to write to /dev/null, the command exists
    except FileNotFoundError:
        print("Error: termux-camera-photo not found")
        print("Install with: pkg install termux-api")
        print("Also install Termux:API app from F-Droid")
        return 1
    except subprocess.TimeoutExpired:
        pass  # Command exists but timed out, that's okay

    # Create temp directory for frame capture
    temp_dir = tempfile.mkdtemp(prefix="termux_cam_")

    try:
        # Start capture thread
        capture_thread = threading.Thread(
            target=capture_loop,
            args=(args.camera, temp_dir),
            daemon=True
        )
        capture_thread.start()

        # Wait for first frame
        print("Waiting for first frame...")
        for _ in range(50):  # 5 second timeout
            if frame_buffer.get_frame() is not None:
                break
            time.sleep(0.1)
        else:
            print("Warning: No frames captured yet, server starting anyway")

        # Start HTTP server
        server = HTTPServer(("0.0.0.0", args.port), MJPEGHandler)
        print(f"\n=== Termux Camera Server ===")
        print(f"Camera: {args.camera}")
        print(f"Port: {args.port}")
        print(f"\nEndpoints:")
        print(f"  http://localhost:{args.port}/shot.jpg  - Single frame")
        print(f"  http://localhost:{args.port}/video     - MJPEG stream")
        print(f"\nPress Ctrl+C to stop")

        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    exit(main())
