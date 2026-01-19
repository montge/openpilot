## ADDED Requirements

### Requirement: Remote Inference Streaming
The system SHALL support streaming camera frames to a remote server for GPU-accelerated model inference.

#### Scenario: Frame streaming to inference server
- **GIVEN** the shadow device is running camera_bridge.py
- **AND** frame_streamer.py is configured with a server URL
- **WHEN** frames are captured via VisionIPC
- **THEN** frames are encoded to JPEG and sent over ZeroMQ
- **AND** the server receives frames with frame_id, width, height metadata

#### Scenario: Configurable streaming parameters
- **GIVEN** frame_streamer.py is started
- **WHEN** --quality and --fps arguments are provided
- **THEN** JPEG quality and target frame rate are respected
- **AND** bandwidth usage scales with quality setting

#### Scenario: Connection resilience
- **GIVEN** frame_streamer.py is running
- **WHEN** the network connection is interrupted
- **THEN** ZeroMQ handles reconnection automatically
- **AND** streaming resumes when connection is restored

### Requirement: Inference Server
The system SHALL provide a server component that receives frames and runs modeld inference.

#### Scenario: Frame reception and decoding
- **GIVEN** inference_server.py is running on a desktop with GPU
- **WHEN** JPEG frames are received over ZeroMQ
- **THEN** frames are decoded to NV12 format
- **AND** frames are published to VisionIPC for modeld consumption

#### Scenario: Model inference execution
- **GIVEN** inference_server.py is running with VisionIPC active
- **WHEN** frames are published to VisionIPC
- **THEN** modeld processes frames and produces modelV2 outputs
- **AND** results are captured by the server

#### Scenario: Result streaming back to device
- **GIVEN** modeld has produced modelV2 results
- **WHEN** inference_server.py receives the results
- **THEN** serialized modelV2 messages are sent back over ZeroMQ
- **AND** frame_id is included for correlation

### Requirement: Result Reception
The system SHALL receive and process inference results from the remote server.

#### Scenario: Receive modelV2 results
- **GIVEN** result_receiver.py is connected to inference_server
- **WHEN** modelV2 results are received over ZeroMQ
- **THEN** results are deserialized and logged
- **AND** statistics (rate, size) are reported

#### Scenario: Result logging
- **GIVEN** result_receiver.py is running with --log-dir specified
- **WHEN** results are received
- **THEN** results are logged to JSONL format
- **AND** frame_id and timestamp are recorded

#### Scenario: Local republishing
- **GIVEN** result_receiver.py is running with --republish flag
- **WHEN** modelV2 results are received
- **THEN** results are published to local messaging
- **AND** other openpilot components can consume them

### Requirement: End-to-End Latency
The system SHALL achieve acceptable latency for shadow mode operation.

#### Scenario: Latency under normal conditions
- **GIVEN** shadow device and server are on same local network
- **WHEN** frames are streamed and inference runs
- **THEN** end-to-end latency (frame capture to result received) is under 500ms
- **AND** latency is logged for monitoring

#### Scenario: Bandwidth efficiency
- **GIVEN** frame streaming at 720p 20fps
- **WHEN** JPEG quality is set to 80
- **THEN** bandwidth usage is under 5 MB/s
- **AND** frame quality is sufficient for model inference
