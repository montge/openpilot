#!/data/data/com.termux/files/usr/bin/bash
# Camera Streaming Server using termux-api
# Creates a simple HTTP server that serves camera frames
#
# Usage: ./camera_stream.sh [port] [camera_id]
# Default: port 8080, camera 0 (back)

PORT=${1:-8080}
CAMERA_ID=${2:-0}
TEMP_DIR="${TMPDIR:-/data/data/com.termux/files/usr/tmp}/camera_stream"

mkdir -p "$TEMP_DIR"

echo "=== Termux Camera Stream Server ==="
echo "Port: $PORT"
echo "Camera: $CAMERA_ID"
echo ""
echo "Endpoints:"
echo "  http://localhost:$PORT/shot.jpg  - Single JPEG frame"
echo "  http://localhost:$PORT/stream    - MJPEG stream (experimental)"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Cleanup function
cleanup() {
    echo "Stopping server..."
    rm -rf "$TEMP_DIR"
    exit 0
}
trap cleanup INT TERM

# Create a simple HTTP server using netcat and shell
# This is a proof-of-concept; for production use a proper server
serve_frame() {
    local frame_file="$TEMP_DIR/frame_$$.jpg"

    # Capture frame
    termux-camera-photo -c "$CAMERA_ID" "$frame_file" 2>/dev/null

    if [ -f "$frame_file" ]; then
        local size=$(stat -c%s "$frame_file")
        echo "HTTP/1.1 200 OK"
        echo "Content-Type: image/jpeg"
        echo "Content-Length: $size"
        echo "Cache-Control: no-cache"
        echo "Connection: close"
        echo ""
        cat "$frame_file"
        rm -f "$frame_file"
    else
        echo "HTTP/1.1 500 Internal Server Error"
        echo "Content-Type: text/plain"
        echo ""
        echo "Failed to capture frame"
    fi
}

# Check if we have the required tools
if ! command -v termux-camera-photo &> /dev/null; then
    echo "Error: termux-api not installed"
    echo "Run: pkg install termux-api"
    exit 1
fi

# Check camera permission
termux-camera-photo -c "$CAMERA_ID" "$TEMP_DIR/test.jpg" 2>/dev/null
if [ ! -f "$TEMP_DIR/test.jpg" ]; then
    echo "Error: Cannot access camera"
    echo "Make sure Termux:API app is installed and has camera permission"
    exit 1
fi
rm -f "$TEMP_DIR/test.jpg"
echo "Camera access verified"

# Start simple HTTP server
# Note: This is slow (~2-5 FPS) due to termux-api overhead
# For better performance, use IP Webcam or similar app
while true; do
    # Listen for connection and serve frame
    # Using socat for simplicity (install with: pkg install socat)
    if command -v socat &> /dev/null; then
        socat TCP-LISTEN:$PORT,reuseaddr,fork EXEC:"$0 --serve-frame $CAMERA_ID"
    else
        echo "Install socat for HTTP server: pkg install socat"
        echo ""
        echo "Alternative: Use continuous capture mode"
        echo "Running capture loop (frames saved to $TEMP_DIR)..."

        frame_num=0
        while true; do
            termux-camera-photo -c "$CAMERA_ID" "$TEMP_DIR/frame_$frame_num.jpg" 2>/dev/null
            echo "Captured frame $frame_num"
            frame_num=$((frame_num + 1))

            # Keep only last 10 frames
            if [ $frame_num -gt 10 ]; then
                rm -f "$TEMP_DIR/frame_$((frame_num - 11)).jpg"
            fi

            sleep 0.2  # ~5 FPS max with termux-api
        done
    fi
done
