#!/usr/bin/env python3
"""
Result Receiver for Shadow Device

Receives model inference results (modelV2) from the remote inference server
and optionally publishes them to local messaging or logs them.

Usage:
  # On shadow device:
  python result_receiver.py --server tcp://192.168.1.100:5556

Architecture:
  inference_server.py (desktop) → ZMQ → result_receiver.py (shadow device)

The result receiver:
1. Subscribes to modelV2 results from the inference server
2. Logs latency and result statistics
3. Optionally republishes to local messaging for other consumers
"""

import argparse
import json
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure openpilot root is in path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_OPENPILOT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _OPENPILOT_ROOT not in sys.path:
    sys.path.insert(0, _OPENPILOT_ROOT)

try:
    import zmq
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install pyzmq")
    sys.exit(1)

# Optional cereal imports for message deserialization
CEREAL_AVAILABLE = False
try:
    from cereal import messaging
    from cereal import log as capnp_log
    CEREAL_AVAILABLE = True
except ImportError:
    pass


class ResultReceiver:
    """
    Receives inference results from remote server and logs/republishes them.
    """

    def __init__(self, server_url: str, log_dir: Optional[str] = None,
                 republish: bool = False):
        self.server_url = server_url
        self.log_dir = Path(log_dir) if log_dir else None
        self.republish = republish
        self.running = False
        self.result_count = 0
        self.latencies = []

        # ZeroMQ context and socket
        self.zmq_ctx = zmq.Context()
        self.socket = self.zmq_ctx.socket(zmq.SUB)
        self.socket.connect(server_url)
        self.socket.subscribe(b"modelV2")
        self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout

        # Local publisher (optional)
        self.pm = None
        if republish and CEREAL_AVAILABLE:
            print("Initializing local publisher...")
            self.pm = messaging.PubMaster(['modelV2'])
        elif republish:
            print("WARNING: Cannot republish - cereal not available")

        # Logging setup
        self.log_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = self.log_dir / f"results_{timestamp}.jsonl"
            self.log_file = open(log_path, 'w')
            print(f"Logging to: {log_path}")

        print(f"Result receiver initialized:")
        print(f"  Server: {server_url}")
        print(f"  Logging: {'enabled' if self.log_file else 'disabled'}")
        print(f"  Republish: {'enabled' if self.pm else 'disabled'}")

    def _log_result(self, frame_id: int, result_bytes: bytes, latency_ms: float):
        """Log result to JSONL file."""
        if self.log_file is None:
            return

        entry = {
            "timestamp": time.time(),
            "frame_id": frame_id,
            "latency_ms": latency_ms,
            "result_size": len(result_bytes),
        }

        # Try to extract some fields from modelV2 if cereal available
        if CEREAL_AVAILABLE:
            try:
                # Parse the capnp message
                with capnp_log.Event.from_bytes(result_bytes) as event:
                    model_v2 = event.modelV2
                    entry["model_fields"] = {
                        "frameId": model_v2.frameId,
                        "frameIdExtra": model_v2.frameIdExtra,
                    }
            except Exception:
                pass

        self.log_file.write(json.dumps(entry) + '\n')
        self.log_file.flush()

    def _republish_result(self, result_bytes: bytes):
        """Republish result to local messaging."""
        if self.pm is None:
            return

        try:
            # Parse and republish
            with capnp_log.Event.from_bytes(result_bytes) as event:
                self.pm.send('modelV2', event)
        except Exception as e:
            print(f"Error republishing: {e}")

    def run(self):
        """Main receiver loop."""
        self.running = True
        start_time = time.monotonic()

        print("\nWaiting for results from server...")
        print("Press Ctrl+C to stop\n")

        try:
            while self.running:
                try:
                    # Receive result message: [topic, frame_id, modelV2_bytes]
                    parts = self.socket.recv_multipart()
                except zmq.Again:
                    # Timeout - no result received
                    continue

                recv_time = time.monotonic()

                if len(parts) != 3:
                    print(f"Invalid message format: {len(parts)} parts")
                    continue

                topic, frame_id_bytes, result_bytes = parts
                frame_id = struct.unpack("Q", frame_id_bytes)[0]

                # Calculate latency (approximate - based on local time)
                # For accurate latency, we'd need synchronized clocks
                latency_ms = 0  # Placeholder - real latency tracking needs frame send time

                self.result_count += 1
                self.latencies.append(latency_ms)

                # Log result
                self._log_result(frame_id, result_bytes, latency_ms)

                # Republish if enabled
                self._republish_result(result_bytes)

                # Stats every 100 results
                if self.result_count % 100 == 0:
                    elapsed = time.monotonic() - start_time
                    rps = self.result_count / elapsed
                    print(f"Results: {self.result_count}, Rate: {rps:.1f}/s, "
                          f"Size: {len(result_bytes)} bytes")

        except KeyboardInterrupt:
            print("\nStopping receiver...")
        finally:
            self.running = False

        elapsed = time.monotonic() - start_time
        print(f"\nReceiver stopped:")
        print(f"  Total results: {self.result_count}")
        print(f"  Average rate: {self.result_count / elapsed:.1f} results/s")

    def close(self):
        """Clean up resources."""
        self.running = False
        if self.log_file:
            self.log_file.close()
        self.socket.close()
        self.zmq_ctx.term()


def test_receiver(server_url: str, timeout: int = 10):
    """Test receiving a single result."""
    print(f"Testing result receiver connection to {server_url}")
    print(f"Timeout: {timeout}s")

    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)
    socket.connect(server_url)
    socket.subscribe(b"modelV2")
    socket.setsockopt(zmq.RCVTIMEO, timeout * 1000)

    print(f"\nConnected to {server_url}")
    print("Waiting for result...")

    try:
        parts = socket.recv_multipart()
        if len(parts) == 3:
            topic, frame_id_bytes, result_bytes = parts
            frame_id = struct.unpack("Q", frame_id_bytes)[0]
            print(f"\nReceived result:")
            print(f"  Frame ID: {frame_id}")
            print(f"  Size: {len(result_bytes)} bytes")

            if CEREAL_AVAILABLE:
                try:
                    with capnp_log.Event.from_bytes(result_bytes) as event:
                        model_v2 = event.modelV2
                        print(f"  modelV2.frameId: {model_v2.frameId}")
                        print("  Parsed successfully!")
                except Exception as e:
                    print(f"  Parse error: {e}")
            else:
                print("  (cereal not available for parsing)")

            return True
        else:
            print(f"Invalid message: {len(parts)} parts")
            return False
    except zmq.Again:
        print("Timeout - no result received")
        print("Is inference_server.py running and receiving frames?")
        return False
    finally:
        socket.close()
        ctx.term()


def main():
    parser = argparse.ArgumentParser(
        description="Receive inference results from remote server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Receive results:
  python result_receiver.py --server tcp://192.168.1.100:5556

  # With logging:
  python result_receiver.py --server tcp://192.168.1.100:5556 --log-dir ./results

  # With local republishing:
  python result_receiver.py --server tcp://192.168.1.100:5556 --republish

  # Test connection:
  python result_receiver.py --server tcp://192.168.1.100:5556 --test
"""
    )
    parser.add_argument("--server", type=str, required=True,
                        help="Inference server result URL (e.g., tcp://192.168.1.100:5556)")
    parser.add_argument("--log-dir", type=str,
                        help="Directory to log results (JSONL format)")
    parser.add_argument("--republish", action="store_true",
                        help="Republish results to local messaging")
    parser.add_argument("--test", action="store_true",
                        help="Test mode - receive one result and exit")
    parser.add_argument("--timeout", type=int, default=10,
                        help="Timeout for test mode in seconds (default: 10)")

    args = parser.parse_args()

    if args.test:
        success = test_receiver(args.server, args.timeout)
        sys.exit(0 if success else 1)
    else:
        receiver = ResultReceiver(
            server_url=args.server,
            log_dir=args.log_dir,
            republish=args.republish
        )
        try:
            receiver.run()
        finally:
            receiver.close()


if __name__ == "__main__":
    main()
