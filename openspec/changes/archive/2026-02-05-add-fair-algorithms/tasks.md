## 1. DINOv2 Integration

- [x] 1.1 Create `tools/fair/` directory structure
- [x] 1.2 Add DINOv2 model loading wrapper
- [x] 1.3 Implement feature extraction pipeline
- [x] 1.4 Add depth estimation head (LinearDepthHead, DPTDepthHead, MultiScaleDepthHead)
- [ ] 1.5 Benchmark DINOv2 depth vs current openpilot depth (DEFERRED: requires model weights + GPU)
- [x] 1.6 Create DINOv2 feature visualization utilities (tools/fair/visualization/)
- [x] 1.7 Add unit tests for DINOv2 wrapper

## 2. SAM 2/3 Video Segmentation

- [x] 2.1 Add SAM 2 model loading wrapper
- [x] 2.2 Implement video object tracking interface
- [ ] 2.3 Create vehicle/pedestrian tracking benchmarks (DEFERRED: requires model weights)
- [ ] 2.4 Benchmark SAM 2 vs current object detection (DEFERRED: requires model weights)
- [ ] 2.5 Implement SAM 3 text-prompted detection (DEFERRED: SAM 3 not yet released)
- [ ] 2.6 Create occlusion handling test scenarios (DEFERRED: requires model weights)
- [x] 2.7 Add unit tests for SAM wrappers

## 3. CoTracker Lane Tracking

- [x] 3.1 Add CoTracker model loading wrapper
- [x] 3.2 Implement lane point initialization from model output (track_lane_points in CoTrackerWrapper)
- [ ] 3.3 Create lane tracking through video pipeline (DEFERRED: requires model weights)
- [ ] 3.4 Benchmark CoTracker vs current lane detection (DEFERRED: requires model weights)
- [ ] 3.5 Test on challenging conditions (shadows, glare, faded markings) (DEFERRED)
- [ ] 3.6 Generate pseudo-labels for training data augmentation (DEFERRED)
- [x] 3.7 Add unit tests for CoTracker wrapper

## 4. DETR Detection

- [x] 4.1 Add DETR model loading wrapper
- [x] 4.2 Implement detection post-processing
- [ ] 4.3 Benchmark DETR vs current detection approach (DEFERRED: requires model weights)
- [ ] 4.4 Evaluate panoptic segmentation capabilities (DEFERRED: requires model weights)
- [x] 4.5 Add unit tests for DETR wrapper

## 5. UnSAMFlow Optical Flow

- [x] 5.1 Add UnSAMFlow model loading wrapper
- [x] 5.2 Implement flow estimation pipeline
- [ ] 5.3 Benchmark flow accuracy for ego-motion estimation (DEFERRED: requires model weights)
- [ ] 5.4 Integrate with object velocity estimation (DEFERRED: requires working flow model)
- [x] 5.5 Add unit tests for UnSAMFlow wrapper

## 6. Knowledge Distillation Framework

- [x] 6.1 Create `tools/fair/distillation/` directory structure
- [x] 6.2 Implement soft label distillation loss
- [x] 6.3 Implement feature distillation loss
- [x] 6.4 Implement attention transfer loss
- [x] 6.5 Create `DistillationTrainer` class
- [x] 6.6 Add learning rate scheduling and warmup
- [x] 6.7 Implement checkpoint management
- [x] 6.8 Add unit tests for distillation components

## 7. Student Model Architecture

- [x] 7.1 Design EfficientViT-based student backbone (TinyViT, MobileViT, EfficientStudent)
- [x] 7.2 Implement multi-head student (TinyDETR, MobileDetector)
- [x] 7.3 Add quantization-friendly layers (replace_batchnorm_with_layernorm, fuse_conv_bn)
- [x] 7.4 Implement model profiling utilities (tools/fair/profiling/)
- [ ] 7.5 Target comma 4 latency constraints (<50ms inference) (DEFERRED: requires hardware)
- [x] 7.6 Add unit tests for student models

## 8. DoRA + FAIR Distillation

- [x] 8.1 Integrate DoRA adapters with distillation pipeline
- [x] 8.2 Implement combined task + distillation loss (TaskDistillationLoss)
- [ ] 8.3 Create training pipeline for DoRA + DINOv2 teacher (DEFERRED: requires model weights + GPU)
- [ ] 8.4 Create training pipeline for DoRA + SAM teacher (DEFERRED: requires model weights + GPU)
- [ ] 8.5 Benchmark DoRA distillation vs full distillation (DEFERRED: requires training runs)
- [x] 8.6 Add adapter checkpointing for incremental training (save_dora_checkpoint/load_dora_checkpoint)

## 9. Dataset Preparation

- [x] 9.1 Create route log → training data converter (RouteDataset, StreamingRouteDataset)
- [x] 9.2 Implement FAIR model pseudo-label generation (create_pseudo_labels)
- [x] 9.3 Create data augmentation pipeline
- [x] 9.4 Implement hard example mining (DifficultyTracker, HardExampleSampler)
- [ ] 9.5 Create dataset statistics and visualization (DEFERRED)

## 10. Quantization for Comma 4

- [x] 10.1 Implement quantization-aware training (QAT) (prepare_qat, calibrate)
- [x] 10.2 Add INT8 quantization support (convert_to_quantized)
- [ ] 10.3 Benchmark accuracy vs latency trade-offs (DEFERRED: requires trained models)
- [ ] 10.4 Test on comma 4 hardware (DEFERRED: hardware not available)
- [x] 10.5 Create deployment export utilities (export_onnx, export_tensorrt)

## 11. Integration Testing

- [ ] 11.1 Benchmark distilled models on algorithm test harness (DEFERRED: requires trained models)
- [ ] 11.2 Compare perception quality vs current openpilot (DEFERRED)
- [ ] 11.3 Test in simulation with diverse scenarios (DEFERRED)
- [ ] 11.4 Measure end-to-end latency impact (DEFERRED)
- [ ] 11.5 Create regression test suite (DEFERRED)

## 12. Documentation

- [x] 12.1 Add README for tools/fair/
- [ ] 12.2 Document distillation hyperparameters and tuning (DEFERRED)
- [ ] 12.3 Create training playbook for DGX Spark (DEFERRED)
- [ ] 12.4 Document comma 4 deployment process (DEFERRED)
- [ ] 12.5 Add example notebooks for model evaluation (DEFERRED)

## 13. Perception Module Integration

- [x] 13.1 Create `selfdrive/modeld/fair/` directory structure
- [x] 13.2 Implement DepthEstimator module
- [x] 13.3 Implement SegmentationModule with driving classes
- [x] 13.4 Implement LaneTracker with temporal consistency
- [x] 13.5 Add unit tests for perception modules
