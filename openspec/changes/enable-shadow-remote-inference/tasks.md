## 1. Infrastructure Setup

- [x] 1.1 Install ZeroMQ on shadow device (proot Ubuntu)
  - [x] `apt install libzmq3-dev`
  - [x] `pip install pyzmq`
- [x] 1.2 Verify ZeroMQ works between device and desktop
  - [x] Simple PUB/SUB test script
  - [x] Measure round-trip latency: <100ms for ZMQ messaging
- [x] 1.3 Document network requirements (ports, firewall)

## 2. Frame Streamer (Shadow Device)

- [x] 2.1 Create `tools/shadow/setup/frame_streamer.py`
  - [x] VisionIpcClient connection to camerad
  - [x] NV12 to JPEG encoding (OpenCV)
  - [x] ZMQ PUB socket for frame publishing
  - [x] Frame metadata (frame_id, width, height, timestamp)
- [x] 2.2 Add CLI arguments
  - [x] `--server` (remote server address)
  - [x] `--quality` (JPEG quality, default 80)
  - [x] `--fps` (target FPS, default 20)
- [x] 2.3 Test frame streaming (requires device)
  - [x] Verify frames reach server (tested with termux-camera-photo → ZMQ)
  - [x] Measure bandwidth usage: ~63KB/frame at 1280x1706, JPEG Q80
  - [x] Check for frame drops: 0% loss in 5-frame test

## 3. Inference Server (Desktop)

- [x] 3.1 Create `tools/shadow/setup/inference_server.py`
  - [x] ZMQ SUB socket for receiving frames
  - [x] JPEG to NV12 decoding
  - [x] VisionIpcServer for local modeld
  - [x] roadCameraState message publishing
- [ ] 3.2 Integrate with modeld
  - [ ] Start modeld process
  - [ ] Verify modeld receives frames
  - [ ] Subscribe to modelV2 output
- [x] 3.3 Result streaming back to device
  - [x] ZMQ PUB socket for results
  - [x] Serialize modelV2 messages
  - [x] Include frame_id for correlation
- [x] 3.4 Add CLI arguments
  - [x] `--port` (listen port, default 5555)
  - [x] `--result-port` (result port, default 5556)
  - [ ] `--model` (model variant if applicable)

## 4. Result Receiver (Shadow Device)

- [x] 4.1 Create `tools/shadow/setup/result_receiver.py`
  - [x] ZMQ SUB socket for receiving results
  - [x] Deserialize modelV2 messages
  - [x] Publish to local messaging (optional)
- [x] 4.2 Add logging capability
  - [x] Log received modelV2 to file
  - [x] Correlate with local frame_ids
- [x] 4.3 Add latency measurement
  - [x] Track frame_id → result round-trip time
  - [x] Report statistics

## 5. Integration Testing

- [ ] 5.1 End-to-end test
  - [ ] Start camera server on phone
  - [ ] Start camera_bridge.py
  - [ ] Start frame_streamer.py
  - [ ] Start inference_server.py on desktop
  - [ ] Verify modelV2 results received
- [ ] 5.2 Performance benchmarks
  - [ ] Measure end-to-end latency
  - [ ] Measure frame rate achieved
  - [ ] Document bandwidth requirements
- [ ] 5.3 Error handling tests
  - [ ] Network disconnect/reconnect
  - [ ] Server restart
  - [ ] Frame buffer overflow

## 6. Documentation

- [x] 6.1 Update `tools/shadow/setup/README.md`
  - [x] Add remote inference section
  - [x] Document server requirements
  - [x] Add quick start guide
- [ ] 6.2 Create `tools/shadow/setup/REMOTE_INFERENCE.md`
  - [ ] Detailed setup instructions
  - [ ] Network configuration
  - [ ] Troubleshooting guide
- [x] 6.3 Update openspec tasks.md
  - [x] Mark implemented items
  - [x] Document known limitations (see below)

## 7. Optional Enhancements (Future)

- [ ] 7.1 Multiple stream support (road + driver camera)
- [ ] 7.2 Compressed result format (reduce bandwidth)
- [ ] 7.3 Local caching/replay mode
- [ ] 7.4 Web dashboard for monitoring

## Dependencies

- Task 2 depends on Task 1 (ZMQ setup) ✓
- Task 3 depends on Task 1 (ZMQ setup) ✓
- Task 4 depends on Task 3 (result format) ✓
- Task 5 depends on Tasks 2, 3, 4
- Tasks 2 and 3 can be developed in parallel ✓

## Validation Criteria

1. **Frame delivery**: >95% of frames reach server
2. **Latency**: End-to-end < 500ms (frame capture → result received)
3. **Bandwidth**: < 5 MB/s at 720p @ 20fps with JPEG Q80
4. **Stability**: Run for 10+ minutes without crashes
5. **Recovery**: Auto-reconnect within 5s after network glitch

## Test Results (2026-01-19)

**Environment**: OnePlus 6 (LineageOS 22.2) → Desktop (Ubuntu)

**ZMQ Connectivity**:
- ✅ PUB/SUB messaging works between devices
- ✅ Port 5555 accessible over local network (10.0.1.x)
- ✅ 0% message loss in multi-frame tests

**Camera Streaming** (using termux-camera-photo):
- ✅ Frames successfully captured and transmitted
- Resolution: 1280x1706 (resized from 4000x3000)
- JPEG size: ~63KB per frame at Q80
- Capture rate: ~0.2 FPS (termux-camera-photo limitation)

**Known Limitations**:
1. termux-camera-photo is too slow (~4.7s/frame) for real-time streaming
2. VisionIPC not built on device (Python 3.13 + proot limitations)
3. Full frame_streamer.py requires VisionIPC from camera_bridge.py
4. For real-time streaming, use IP Webcam app instead (15-30 FPS possible)

**Recommended Path for Real-Time Testing**:
1. Install IP Webcam app on device
2. Use camera_bridge.py to consume MJPEG stream
3. Or implement direct MJPEG → ZMQ streamer (bypasses VisionIPC)

## Next Steps (Remote Setup)

**Blocker**: IP Webcam is NOT on F-Droid (proprietary). Options:
1. Get APK from Play Store on another device, install via `adb install`
2. Use scrcpy to mirror screen and manually start camera app
3. Find FOSS camera streaming app on F-Droid

**When IP Webcam is available**, run:
```bash
# Desktop: Start ZMQ receiver
python tools/shadow/setup/inference_server.py --test

# Device: Start MJPEG streamer (via SSH)
ssh -p 8022 10.0.1.62 "proot-distro login ubuntu -- bash -c '
cd ~/openpilot && source .venv/bin/activate
python3 /data/data/com.termux/files/home/mjpeg_zmq_streamer.py \
    --camera http://localhost:8080 --server tcp://10.0.1.123:5555 --fps 15'"
```

**Files created for remote testing**:
- `tools/shadow/setup/mjpeg_zmq_streamer.py` - MJPEG → ZMQ (no VisionIPC needed)
- `tools/shadow/setup/REMOTE_SETUP.md` - Remote setup guide with ADB/SSH commands

**Remote access available**:
- SSH: `ssh -p 8022 10.0.1.62`
- ADB: `adb devices` shows `34f011c6`
- Desktop IP: 10.0.1.123, Device IP: 10.0.1.62
