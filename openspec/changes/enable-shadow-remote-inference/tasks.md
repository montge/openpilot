## 1. Infrastructure Setup

- [x] 1.1 Install ZeroMQ on shadow device (proot Ubuntu)
  - [x] `apt install libzmq3-dev`
  - [x] `pip install pyzmq`
- [x] 1.2 Verify ZeroMQ works between device and desktop
  - [x] Simple PUB/SUB test script
  - [ ] Measure round-trip latency (requires device testing)
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
- [ ] 2.3 Test frame streaming (requires device)
  - [ ] Verify frames reach server
  - [ ] Measure bandwidth usage
  - [ ] Check for frame drops

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
  - [ ] Document known limitations

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
