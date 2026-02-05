## ADDED Requirements

### Requirement: DINOv2 Feature Extraction
The system SHALL provide DINOv2 feature extraction for perception enhancement.

#### Scenario: Extract features on DGX Spark
- **GIVEN** a batch of input images on DGX Spark
- **WHEN** DINOv2 feature extraction is invoked
- **THEN** dense feature maps are returned (batch, channels, H/14, W/14)
- **AND** features are robust across lighting and weather conditions

#### Scenario: DINOv2 depth estimation
- **GIVEN** a monocular camera image
- **WHEN** DINOv2 depth estimation is invoked
- **THEN** per-pixel depth map is returned
- **AND** depth accuracy exceeds current openpilot baseline on benchmarks

### Requirement: SAM 2 Video Object Tracking
The system SHALL provide SAM 2 video segmentation for object tracking.

#### Scenario: Track objects through video
- **GIVEN** initial object detections in frame 0
- **WHEN** SAM 2 tracking propagates through subsequent frames
- **THEN** object masks are maintained across frames
- **AND** objects are re-identified after brief occlusions

#### Scenario: Handle occlusion and re-appearance
- **GIVEN** a tracked object that becomes occluded
- **WHEN** the object re-appears after N frames
- **THEN** the same track ID is maintained
- **AND** track continuity is preserved in output

### Requirement: CoTracker Lane Point Tracking
The system SHALL provide CoTracker for dense lane marking tracking.

#### Scenario: Track lane marking points
- **GIVEN** initial lane marking point coordinates
- **WHEN** CoTracker processes a video sequence
- **THEN** point trajectories are returned for each frame
- **AND** visibility confidence indicates trackable vs lost points

#### Scenario: Handle challenging lane conditions
- **GIVEN** video with shadows, glare, or faded markings
- **WHEN** CoTracker tracks lane points
- **THEN** points are tracked through transient occlusions
- **AND** tracking degrades gracefully in severely degraded regions

### Requirement: Knowledge Distillation Pipeline
The system SHALL provide knowledge distillation from FAIR teachers to efficient students.

#### Scenario: Soft label distillation
- **GIVEN** a teacher model and student model
- **WHEN** distillation training is executed
- **THEN** student learns from teacher's soft probability outputs
- **AND** temperature parameter controls softness of targets

#### Scenario: Feature distillation
- **GIVEN** intermediate features from teacher and student
- **WHEN** feature distillation loss is computed
- **THEN** student features are aligned to teacher features
- **AND** projection layer handles dimension mismatch

#### Scenario: Multi-task distillation
- **GIVEN** teacher outputs for depth, segmentation, and detection
- **WHEN** multi-task distillation is executed
- **THEN** student learns all tasks simultaneously
- **AND** task weights balance learning across objectives

### Requirement: Student Model for Comma 4
The system SHALL provide distilled student models deployable on comma 4.

#### Scenario: Student model inference latency
- **GIVEN** a distilled student model
- **WHEN** inference is run on comma 4 hardware
- **THEN** latency is under 50ms per frame
- **AND** throughput supports 20 Hz operation

#### Scenario: Student model accuracy
- **GIVEN** a distilled student model
- **WHEN** evaluated on benchmark datasets
- **THEN** accuracy is within 10% of teacher model
- **AND** safety-critical metrics (detection recall) are within 5%

#### Scenario: Quantized model deployment
- **GIVEN** a quantization-aware trained student model
- **WHEN** exported to INT8 format
- **THEN** inference runs on comma 4 NPU/DSP
- **AND** accuracy degradation from quantization is under 2%

### Requirement: DoRA Integration with FAIR Teachers
The system SHALL support DoRA fine-tuning using FAIR models as teachers.

#### Scenario: DoRA adapter with DINOv2 teacher
- **GIVEN** a base openpilot model with DoRA adapters
- **WHEN** training with DINOv2 feature distillation
- **THEN** adapters learn to produce DINOv2-like features
- **AND** adapter size is under 1% of base model

#### Scenario: Adapter merging for deployment
- **GIVEN** trained DoRA adapters
- **WHEN** merge operation is executed
- **THEN** adapters are folded into base model weights
- **AND** inference runs without adapter overhead

### Requirement: Training Pipeline
The system SHALL provide training pipelines for FAIR model distillation.

#### Scenario: Training on DGX Spark
- **GIVEN** route log data and FAIR teacher models
- **WHEN** distillation training is launched
- **THEN** training utilizes DGX Spark GPU efficiently
- **AND** checkpoints are saved at configurable intervals

#### Scenario: Pseudo-label generation
- **GIVEN** unlabeled route log video
- **WHEN** FAIR teachers process the video
- **THEN** pseudo-labels (depth, segmentation, tracks) are generated
- **AND** labels are saved in training-compatible format

#### Scenario: Dataset augmentation
- **GIVEN** training images with pseudo-labels
- **WHEN** augmentation pipeline is applied
- **THEN** augmented samples include geometric and photometric transforms
- **AND** augmentations are consistent across image and labels

### Requirement: Benchmarking and Evaluation
The system SHALL provide benchmarking tools for FAIR model evaluation.

#### Scenario: Compare FAIR vs current openpilot
- **GIVEN** FAIR model and current openpilot model
- **WHEN** benchmark is run on test scenarios
- **THEN** metrics are computed for both models
- **AND** comparison report highlights improvements/regressions

#### Scenario: Distillation quality evaluation
- **GIVEN** teacher model and distilled student
- **WHEN** evaluation is run on held-out data
- **THEN** accuracy gap between teacher and student is measured
- **AND** report identifies tasks where distillation underperforms

### Requirement: Hardware Abstraction
The system SHALL support both DGX Spark and comma 4 hardware.

#### Scenario: Automatic device selection
- **GIVEN** FAIR integration code running on unknown hardware
- **WHEN** model loading is invoked
- **THEN** full model loads on DGX Spark
- **AND** distilled model loads on comma 4

#### Scenario: Graceful degradation on limited hardware
- **GIVEN** FAIR integration running on PC (no DGX Spark or comma 4)
- **WHEN** inference is requested
- **THEN** system uses available GPU or CPU
- **AND** warning indicates non-optimal performance
