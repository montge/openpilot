#!/usr/bin/env python3
"""
Remote Inference Server for Shadow Device

Receives camera frames from a shadow device over ZeroMQ, feeds them to modeld
via VisionIPC, and returns inference results.

Usage:
  # On desktop/server with GPU:
  python inference_server.py --port 5555 --result-port 5556

  # Connect from shadow device:
  python frame_streamer.py --server tcp://desktop-ip:5555

Architecture:
  Shadow Device                Network                    This Server
  ─────────────                ───────                    ───────────
  camera_bridge.py
        ↓
  frame_streamer.py ──────────────────────────────────▶ inference_server.py
        │                                                      ↓
        │                                                 VisionIPC
        │                                                      ↓
        │                                                   modeld
        │                                                      ↓
        │                                                  modelV2
        │                                                      ↓
  result_receiver.py ◀────────────────────────────────────────┘
"""

import argparse
import os
import struct
import subprocess
import sys
import threading
import time
from typing import Optional

# Ensure openpilot root is in path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPILOT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _OPENPILOT_ROOT not in sys.path:
  sys.path.insert(0, _OPENPILOT_ROOT)

try:
  import zmq  # type: ignore[import-not-found]
  import cv2  # type: ignore[import-not-found]
  import numpy as np
except ImportError as e:
  print(f"Missing dependency: {e}")
  print("Install with: pip install pyzmq opencv-python numpy")
  sys.exit(1)

# Optional VisionIPC imports
VIPC_AVAILABLE = False
try:
  import av  # type: ignore[import-not-found]
  from msgq.visionipc import VisionIpcServer, VisionStreamType
  from cereal import messaging

  VIPC_AVAILABLE = True
except ImportError:
  pass


def jpeg_to_nv12(jpeg_data: bytes, target_width: int, target_height: int) -> Optional[bytes]:
  """Decode JPEG to NV12 format."""
  # Decode JPEG to BGR
  img_array = np.frombuffer(jpeg_data, dtype=np.uint8)
  bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
  if bgr is None:
    return None

  # Resize if needed
  h, w = bgr.shape[:2]
  if w != target_width or h != target_height:
    bgr = cv2.resize(bgr, (target_width, target_height))

  # Convert to NV12 using PyAV if available
  if VIPC_AVAILABLE:
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray().data.tobytes()

  # Fallback: manual conversion
  yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
  height, width = bgr.shape[:2]

  y_size = width * height
  uv_size = y_size // 4

  y_plane = yuv[:height, :].flatten()
  u_plane = yuv[height : height + height // 4, :].reshape(-1)
  v_plane = yuv[height + height // 4 :, :].reshape(-1)

  uv_interleaved = np.empty(uv_size * 2, dtype=np.uint8)
  uv_interleaved[0::2] = u_plane
  uv_interleaved[1::2] = v_plane

  return np.concatenate([y_plane, uv_interleaved]).tobytes()


class InferenceServer:
  """
  Receives frames from shadow device, publishes to VisionIPC for modeld,
  and returns inference results.
  """

  def __init__(self, listen_port: int, result_port: int, width: int = 1280, height: int = 720, model: str = "default"):
    self.listen_port = listen_port
    self.result_port = result_port
    self.width = width
    self.height = height
    self.model = model
    self.running = False
    self.frame_count = 0
    self.result_count = 0
    self.modeld_proc: Optional[subprocess.Popen[bytes]] = None

    # ZeroMQ context
    self.zmq_ctx = zmq.Context()

    # Socket to receive frames (SUB - subscribes to frame_streamer PUB)
    self.frame_socket = self.zmq_ctx.socket(zmq.SUB)
    self.frame_socket.bind(f"tcp://*:{listen_port}")
    self.frame_socket.subscribe(b"frame")
    self.frame_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout

    # Socket to send results (PUB - result_receiver SUBs to this)
    self.result_socket = self.zmq_ctx.socket(zmq.PUB)
    self.result_socket.bind(f"tcp://*:{result_port}")

    # VisionIPC server (feeds modeld)
    self.vipc_server = None
    self.pm = None
    self.sm = None

    if VIPC_AVAILABLE:
      print("Initializing VisionIPC server...")
      self.vipc_server = VisionIpcServer("camerad")
      self.vipc_server.create_buffers(
        VisionStreamType.VISION_STREAM_ROAD,
        20,  # buffer count
        width,
        height,
      )
      self.vipc_server.start_listener()

      self.pm = messaging.PubMaster(['roadCameraState'])
      self.sm = messaging.SubMaster(['modelV2'])
      print(f"VisionIPC ready ({width}x{height})")
    else:
      print("WARNING: VisionIPC not available - frames will be received but not processed")

    print("Inference server initialized:")
    print(f"  Frame port: tcp://*:{listen_port}")
    print(f"  Result port: tcp://*:{result_port}")
    print(f"  Model: {model}")

  def start_modeld(self) -> bool:
    """Start modeld process for model inference.

    Returns True if modeld was started (or is already running), False on error.
    """
    if not VIPC_AVAILABLE:
      print("WARNING: Cannot start modeld without VisionIPC")
      return False

    modeld_path = os.path.join(_OPENPILOT_ROOT, "selfdrive", "modeld", "modeld")
    if not os.path.exists(modeld_path):
      # Try Python entry point
      modeld_path = os.path.join(_OPENPILOT_ROOT, "selfdrive", "modeld", "modeld.py")
      if not os.path.exists(modeld_path):
        print("WARNING: modeld not found at selfdrive/modeld/modeld[.py]")
        print("Run 'scons -u -j$(nproc)' to build modeld first")
        return False

    env = os.environ.copy()
    env["OPENPILOT_ROOT"] = _OPENPILOT_ROOT

    if self.model != "default":
      env["MODELD_MODEL_PATH"] = self.model

    print(f"Starting modeld from {modeld_path}...")
    try:
      self.modeld_proc = subprocess.Popen(
        [sys.executable, modeld_path] if modeld_path.endswith('.py') else [modeld_path],
        env=env,
        cwd=_OPENPILOT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
      )
      # Give modeld time to initialize
      time.sleep(2.0)
      if self.modeld_proc.poll() is not None:
        stderr = self.modeld_proc.stderr.read().decode() if self.modeld_proc.stderr else ""
        print(f"WARNING: modeld exited immediately: {stderr[:500]}")
        self.modeld_proc = None
        return False
      print(f"modeld started (PID {self.modeld_proc.pid})")
      return True
    except OSError as e:
      print(f"WARNING: Failed to start modeld: {e}")
      return False

  def stop_modeld(self):
    """Stop the modeld process if running."""
    if self.modeld_proc is not None:
      print("Stopping modeld...")
      self.modeld_proc.terminate()
      try:
        self.modeld_proc.wait(timeout=5)
      except subprocess.TimeoutExpired:
        self.modeld_proc.kill()
      self.modeld_proc = None

  def _publish_frame(self, nv12_data: bytes, frame_id: int, timestamp_ns: int):
    """Publish frame to VisionIPC for modeld consumption."""
    if self.vipc_server is None or self.pm is None:
      return

    self.vipc_server.send(
      VisionStreamType.VISION_STREAM_ROAD,
      nv12_data,
      frame_id,
      timestamp_ns,  # timestamp_sof
      timestamp_ns,  # timestamp_eof
    )

    # Publish camera state metadata
    dat = messaging.new_message('roadCameraState', valid=True)
    msg = {"frameId": frame_id, "transform": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]}
    dat.roadCameraState = msg
    self.pm.send('roadCameraState', dat)

  def _send_result(self, model_v2_bytes: bytes, frame_id: int):
    """Send modelV2 result back to shadow device."""
    # Message format: [topic, frame_id (Q), modelV2 bytes]
    self.result_socket.send_multipart([b"modelV2", struct.pack("Q", frame_id), model_v2_bytes])
    self.result_count += 1

  def _result_forwarder(self):
    """Thread that subscribes to modelV2 and forwards to ZMQ."""
    if self.sm is None:
      return

    print("Result forwarder started")
    last_frame_id = 0

    while self.running:
      self.sm.update(100)  # 100ms timeout

      if self.sm.updated['modelV2']:
        # Get the raw bytes
        model_v2 = self.sm['modelV2']
        # For now, use frame_id from our counter
        # TODO: extract frame_id from modelV2 if available
        frame_id = last_frame_id

        try:
          # Serialize the message
          msg_bytes = model_v2.to_bytes()
          self._send_result(msg_bytes, frame_id)
        except Exception as e:
          print(f"Error forwarding result: {e}")

      last_frame_id = self.frame_count

    print("Result forwarder stopped")

  def run(self):
    """Main server loop - receive frames and process."""
    self.running = True
    start_time = time.monotonic()

    # Start result forwarder thread
    result_thread = threading.Thread(target=self._result_forwarder, daemon=True)
    result_thread.start()

    print("\nWaiting for frames from shadow device...")
    print("Press Ctrl+C to stop\n")

    try:
      while self.running:
        try:
          # Receive frame message: [topic, metadata, jpeg_data]
          parts = self.frame_socket.recv_multipart()
        except zmq.Again:
          # Timeout - no frame received
          continue

        if len(parts) != 3:
          print(f"Invalid message format: {len(parts)} parts")
          continue

        topic, metadata, jpeg_data = parts
        frame_id, width, height = struct.unpack("QII", metadata)

        # Decode JPEG to NV12
        nv12_data = jpeg_to_nv12(jpeg_data, self.width, self.height)
        if nv12_data is None:
          print(f"Failed to decode frame {frame_id}")
          continue

        # Publish to VisionIPC
        timestamp_ns = int(time.monotonic() * 1e9)
        self._publish_frame(nv12_data, frame_id, timestamp_ns)
        self.frame_count += 1

        # Stats every 100 frames
        if self.frame_count % 100 == 0:
          elapsed = time.monotonic() - start_time
          fps = self.frame_count / elapsed
          print(f"Frames: {self.frame_count}, Results: {self.result_count}, FPS: {fps:.1f}, JPEG size: {len(jpeg_data) / 1024:.1f}KB")

    except KeyboardInterrupt:
      print("\nStopping server...")
    finally:
      self.running = False
      result_thread.join(timeout=2)

    elapsed = time.monotonic() - start_time
    print("\nServer stopped:")
    print(f"  Total frames: {self.frame_count}")
    print(f"  Total results: {self.result_count}")
    print(f"  Average FPS: {self.frame_count / elapsed:.1f}")

  def close(self):
    """Clean up resources."""
    self.running = False
    self.frame_socket.close()
    self.result_socket.close()
    self.zmq_ctx.term()


def test_server(port: int):
  """Test server without VisionIPC - just receive and decode frames."""
  print(f"Testing inference server on port {port}")
  print("VisionIPC available:", VIPC_AVAILABLE)

  ctx = zmq.Context()
  socket = ctx.socket(zmq.SUB)
  socket.bind(f"tcp://*:{port}")
  socket.subscribe(b"frame")
  socket.setsockopt(zmq.RCVTIMEO, 5000)

  print(f"\nListening on tcp://*:{port}")
  print("Waiting for test frame (5s timeout)...")

  try:
    parts = socket.recv_multipart()
    if len(parts) == 3:
      topic, metadata, jpeg_data = parts
      frame_id, width, height = struct.unpack("QII", metadata)
      print("\nReceived frame:")
      print(f"  Frame ID: {frame_id}")
      print(f"  Dimensions: {width}x{height}")
      print(f"  JPEG size: {len(jpeg_data)} bytes")

      # Test decode
      nv12 = jpeg_to_nv12(jpeg_data, 1280, 720)
      if nv12:
        print(f"  NV12 decode: OK ({len(nv12)} bytes)")
      else:
        print("  NV12 decode: FAILED")
    else:
      print(f"Invalid message: {len(parts)} parts")
  except zmq.Again:
    print("Timeout - no frame received")
  finally:
    socket.close()
    ctx.term()


def main():
  parser = argparse.ArgumentParser(
    description="Remote inference server for shadow device",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Start server (default ports):
  python inference_server.py

  # Custom ports:
  python inference_server.py --port 5555 --result-port 5556

  # Test mode (no VisionIPC):
  python inference_server.py --test
""",
  )
  parser.add_argument("--port", type=int, default=5555, help="Port to receive frames (default: 5555)")
  parser.add_argument("--result-port", type=int, default=5556, help="Port to send results (default: 5556)")
  parser.add_argument("--width", type=int, default=1280, help="Frame width (default: 1280)")
  parser.add_argument("--height", type=int, default=720, help="Frame height (default: 720)")
  parser.add_argument("--model", type=str, default="default", help="Model variant or path (default: use standard modeld)")
  parser.add_argument("--no-modeld", action="store_true", help="Don't start modeld (useful if running modeld separately)")
  parser.add_argument("--test", action="store_true", help="Test mode - receive one frame and exit")

  args = parser.parse_args()

  if args.test:
    test_server(args.port)
  else:
    server = InferenceServer(
      listen_port=args.port,
      result_port=args.result_port,
      width=args.width,
      height=args.height,
      model=args.model,
    )
    try:
      if not args.no_modeld:
        server.start_modeld()
      server.run()
    finally:
      server.stop_modeld()
      server.close()


if __name__ == "__main__":
  main()
