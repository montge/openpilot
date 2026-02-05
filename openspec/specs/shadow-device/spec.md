# Shadow Device

**Purpose:** Define shadow device mode for parallel testing of experimental algorithms using an OnePlus 6 alongside the production comma device, with full actuator lockout, comparison logging, and divergence analysis.

## Requirements

### Requirement: Shadow Mode Detection
The system SHALL detect when running as a shadow device and enable shadow mode.

#### Scenario: OnePlus 6 without panda detected as shadow
- **GIVEN** the device is a OnePlus 6
- **WHEN** no panda hardware is connected (USB or WiFi)
- **THEN** `is_shadow_mode()` returns `True`
- **AND** shadow mode is logged at startup

#### Scenario: Environment variable override
- **GIVEN** the environment variable `SHADOW_MODE=1` is set
- **WHEN** the system starts
- **THEN** shadow mode is enabled regardless of device type or panda status

#### Scenario: Comma device with panda is not shadow
- **GIVEN** the device is a comma 3 or comma 3X
- **WHEN** panda hardware is connected
- **THEN** `is_shadow_mode()` returns `False`
- **AND** normal operation proceeds

### Requirement: Actuator Lockout
The system SHALL prevent ALL actuator commands when in shadow mode.

#### Scenario: No steering commands in shadow mode
- **GIVEN** shadow mode is active
- **WHEN** controlsd computes a steering command
- **THEN** the command is logged but NOT published to CAN
- **AND** no steering torque is applied to the vehicle

#### Scenario: No acceleration commands in shadow mode
- **GIVEN** shadow mode is active
- **WHEN** controlsd computes acceleration/braking commands
- **THEN** the commands are logged but NOT sent to the vehicle
- **AND** no throttle or brake actuation occurs

#### Scenario: Defense in depth lockout
- **GIVEN** shadow mode is active
- **WHEN** any code attempts to write to CAN bus
- **THEN** pandad refuses the write operation
- **AND** an error is logged

#### Scenario: Startup warning
- **GIVEN** shadow mode is active
- **WHEN** the system starts
- **THEN** a prominent warning is logged: "SHADOW MODE - NO VEHICLE CONTROL"
- **AND** the UI displays a shadow mode indicator (if running)

### Requirement: Full Sensor Pipeline
The system SHALL run the complete sensor and model pipeline in shadow mode.

#### Scenario: Camera capture active
- **GIVEN** shadow mode is active
- **WHEN** the system is running
- **THEN** all cameras capture frames at normal rate
- **AND** frames are processed by the driving model

#### Scenario: Model inference active
- **GIVEN** shadow mode is active
- **WHEN** camera frames are captured
- **THEN** modeld runs inference on frames
- **AND** model outputs are computed normally

#### Scenario: Path planning active
- **GIVEN** shadow mode is active
- **WHEN** model outputs are available
- **THEN** longitudinal and lateral planners compute trajectories
- **AND** planned paths are available for logging

#### Scenario: Control computation without actuation
- **GIVEN** shadow mode is active
- **WHEN** planned paths are available
- **THEN** controlsd computes control commands
- **AND** commands are logged for comparison
- **AND** commands are NOT sent to vehicle

### Requirement: Comparison Logging
The system SHALL log control outputs for offline comparison with primary device.

#### Scenario: Log model outputs
- **GIVEN** shadow mode is active
- **WHEN** modeld produces outputs
- **THEN** model output tensors are captured to comparison log
- **AND** frame ID and timestamp are recorded

#### Scenario: Log planned trajectory
- **GIVEN** shadow mode is active
- **WHEN** planner produces trajectory
- **THEN** planned path (x, y, heading) is captured
- **AND** planned velocity profile is captured

#### Scenario: Log control commands
- **GIVEN** shadow mode is active
- **WHEN** controlsd computes commands
- **THEN** steering command (-1 to 1) is captured
- **AND** acceleration command (m/s^2) is captured
- **AND** active states (lat_active, long_active) are captured

#### Scenario: GPS timestamp for synchronization
- **GIVEN** shadow mode is active
- **WHEN** log entries are written
- **THEN** GPS time is captured with each entry
- **AND** monotonic timestamp is captured for ordering

### Requirement: Log Alignment
The system SHALL provide tools to align shadow and primary device logs.

#### Scenario: GPS-based alignment
- **GIVEN** shadow log and primary device log with GPS timestamps
- **WHEN** alignment tool is run
- **THEN** logs are aligned by GPS time
- **AND** aligned pairs are output for comparison

#### Scenario: Frame-based alignment
- **GIVEN** logs with matching frame sequences
- **WHEN** alignment uses frame matching
- **THEN** frames with similar content are paired
- **AND** time offset between devices is estimated

#### Scenario: Alignment quality validation
- **GIVEN** aligned log pairs
- **WHEN** alignment is complete
- **THEN** alignment quality metric is computed
- **AND** warnings are issued if alignment is poor (>100ms offset)

### Requirement: Divergence Analysis
The system SHALL compute divergence metrics between shadow and primary logs.

#### Scenario: Model output divergence
- **GIVEN** aligned model outputs from shadow and primary
- **WHEN** divergence is computed
- **THEN** cosine similarity between outputs is calculated
- **AND** RMSE of output differences is calculated

#### Scenario: Trajectory divergence
- **GIVEN** aligned planned trajectories
- **WHEN** divergence is computed
- **THEN** path error (lateral deviation) is calculated
- **AND** speed error (velocity difference) is calculated

#### Scenario: Control command divergence
- **GIVEN** aligned control commands
- **WHEN** divergence is computed
- **THEN** steering command difference is calculated
- **AND** acceleration command difference is calculated
- **AND** statistics (mean, max, std) are reported

#### Scenario: Generate comparison report
- **GIVEN** computed divergence metrics
- **WHEN** report generation is requested
- **THEN** markdown or HTML report is generated
- **AND** report includes summary statistics
- **AND** report includes time-series visualizations

### Requirement: OnePlus 6 Compatibility
The system SHALL support OnePlus 6 as a shadow device platform.

#### Scenario: Device detection
- **GIVEN** software running on OnePlus 6
- **WHEN** hardware detection runs
- **THEN** device is identified as OnePlus 6
- **AND** appropriate hardware parameters are loaded

#### Scenario: Camera functionality
- **GIVEN** OnePlus 6 in shadow mode
- **WHEN** camera capture is initiated
- **THEN** road-facing camera captures at expected resolution
- **AND** driver-facing camera captures (if available)

#### Scenario: Thermal management
- **GIVEN** OnePlus 6 running model inference
- **WHEN** device temperature is monitored
- **THEN** temperature is logged periodically
- **AND** warning is issued if throttling detected

### Requirement: Integration with Algorithm Harness
The system SHALL integrate shadow logs with the algorithm test harness.

#### Scenario: Import shadow logs as scenarios
- **GIVEN** shadow comparison logs
- **WHEN** imported into algorithm harness
- **THEN** logs are converted to Scenario format
- **AND** scenarios can be replayed through harness

#### Scenario: Compare algorithms against shadow baseline
- **GIVEN** shadow log scenario and new algorithm
- **WHEN** harness runs comparison
- **THEN** new algorithm outputs are compared to shadow baseline
- **AND** divergence metrics are computed

### Requirement: OnePlus 6 LineageOS Setup Scripts

The system SHALL provide automated setup scripts for configuring OnePlus 6 as a shadow device running LineageOS with Termux and proot-distro Ubuntu.

#### Scenario: Fresh device setup
- **WHEN** a user has a OnePlus 6 with LineageOS installed
- **AND** runs the Termux setup script
- **THEN** proot-distro with Ubuntu is installed and configured

#### Scenario: Ubuntu environment setup
- **WHEN** a user runs the Ubuntu setup script inside proot
- **THEN** all openpilot build dependencies are installed

#### Scenario: Openpilot clone and configure
- **WHEN** a user runs the clone script
- **THEN** openpilot is cloned with submodules and Python venv configured

### Requirement: Android Getprop Device Detection

The shadow mode detection system SHALL support Android `getprop` as a fallback method for device identification when running in proot/chroot environments where `/sys/firmware/devicetree/base/model` is inaccessible.

#### Scenario: Device detection in proot environment
- **WHEN** openpilot runs on OnePlus 6 inside proot-distro
- **AND** the device tree file is not accessible
- **THEN** the system detects OnePlus 6 via `getprop ro.product.device`
- **AND** shadow mode is automatically enabled

#### Scenario: Native device detection preserved
- **WHEN** openpilot runs natively on OnePlus 6 (not in proot)
- **THEN** the system detects OnePlus 6 via device tree first
- **AND** falls back to getprop only if device tree unavailable

### Requirement: SSH Remote Development Support

The shadow device setup SHALL include SSH server configuration for remote development access.

#### Scenario: SSH server setup
- **WHEN** a user runs the SSH setup script in Termux
- **THEN** OpenSSH server starts on port 8022
- **AND** the user can connect via `ssh user@device-ip -p 8022`

#### Scenario: SSH key authentication
- **WHEN** a user adds their public key to authorized_keys
- **THEN** passwordless SSH authentication is enabled

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
