## 1. Infrastructure Setup

- [ ] 1.1 Install ZeroMQ on shadow device (proot Ubuntu)
  - [ ] `apt install libzmq3-dev`
  - [ ] `pip install pyzmq`
- [ ] 1.2 Verify ZeroMQ works between device and desktop
  - [ ] Simple PUB/SUB test script
  - [ ] Measure round-trip latency
- [ ] 1.3 Document network requirements (ports, firewall)

## 2. Frame Streamer (Shadow Device)

- [ ] 2.1 Create `tools/shadow/setup/frame_streamer.py`
  - [ ] VisionIpcClient connection to camerad
  - [ ] NV12 to JPEG encoding (OpenCV)
  - [ ] ZMQ PUB socket for frame publishing
  - [ ] Frame metadata (frame_id, width, height, timestamp)
- [ ] 2.2 Add CLI arguments
  - [ ] `--server` (remote server address)
  - [ ] `--port` (default 5555)
  - [ ] `--quality` (JPEG quality, default 80)
  - [ ] `--fps` (target FPS, default 20)
- [ ] 2.3 Test frame streaming
  - [ ] Verify frames reach server
  - [ ] Measure bandwidth usage
  - [ ] Check for frame drops

## 3. Inference Server (Desktop)

- [ ] 3.1 Create `tools/shadow/setup/inference_server.py`
  - [ ] ZMQ SUB socket for receiving frames
  - [ ] JPEG to NV12 decoding
  - [ ] VisionIpcServer for local modeld
  - [ ] roadCameraState message publishing
- [ ] 3.2 Integrate with modeld
  - [ ] Start modeld process
  - [ ] Verify modeld receives frames
  - [ ] Subscribe to modelV2 output
- [ ] 3.3 Result streaming back to device
  - [ ] ZMQ PUB socket for results
  - [ ] Serialize modelV2 messages
  - [ ] Include frame_id for correlation
- [ ] 3.4 Add CLI arguments
  - [ ] `--port` (listen port, default 5555)
  - [ ] `--result-port` (result port, default 5556)
  - [ ] `--model` (model variant if applicable)

## 4. Result Receiver (Shadow Device)

- [ ] 4.1 Create `tools/shadow/setup/result_receiver.py`
  - [ ] ZMQ SUB socket for receiving results
  - [ ] Deserialize modelV2 messages
  - [ ] Publish to local messaging (optional)
- [ ] 4.2 Add logging capability
  - [ ] Log received modelV2 to file
  - [ ] Correlate with local frame_ids
- [ ] 4.3 Add latency measurement
  - [ ] Track frame_id → result round-trip time
  - [ ] Report statistics

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

- [ ] 6.1 Update `tools/shadow/setup/README.md`
  - [ ] Add remote inference section
  - [ ] Document server requirements
  - [ ] Add quick start guide
- [ ] 6.2 Create `tools/shadow/setup/REMOTE_INFERENCE.md`
  - [ ] Detailed setup instructions
  - [ ] Network configuration
  - [ ] Troubleshooting guide
- [ ] 6.3 Update openspec tasks.md
  - [ ] Mark remote inference as implemented
  - [ ] Document known limitations

## 7. Optional Enhancements (Future)

- [ ] 7.1 Multiple stream support (road + driver camera)
- [ ] 7.2 Compressed result format (reduce bandwidth)
- [ ] 7.3 Local caching/replay mode
- [ ] 7.4 Web dashboard for monitoring

## Dependencies

- Task 2 depends on Task 1 (ZMQ setup)
- Task 3 depends on Task 1 (ZMQ setup)
- Task 4 depends on Task 3 (result format)
- Task 5 depends on Tasks 2, 3, 4
- Tasks 2 and 3 can be developed in parallel

## Validation Criteria

1. **Frame delivery**: >95% of frames reach server
2. **Latency**: End-to-end < 500ms (frame capture → result received)
3. **Bandwidth**: < 5 MB/s at 720p @ 20fps with JPEG Q80
4. **Stability**: Run for 10+ minutes without crashes
5. **Recovery**: Auto-reconnect within 5s after network glitch
