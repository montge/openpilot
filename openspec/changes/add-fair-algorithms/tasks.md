## 1. DINOv2 Integration

- [ ] 1.1 Create `tools/fair/` directory structure
- [ ] 1.2 Add DINOv2 model loading wrapper
- [ ] 1.3 Implement feature extraction pipeline
- [ ] 1.4 Add depth estimation head (linear probe)
- [ ] 1.5 Benchmark DINOv2 depth vs current openpilot depth
- [ ] 1.6 Create DINOv2 feature visualization utilities
- [ ] 1.7 Add unit tests for DINOv2 wrapper

## 2. SAM 2/3 Video Segmentation

- [ ] 2.1 Add SAM 2 model loading wrapper
- [ ] 2.2 Implement video object tracking interface
- [ ] 2.3 Create vehicle/pedestrian tracking benchmarks
- [ ] 2.4 Benchmark SAM 2 vs current object detection
- [ ] 2.5 Implement SAM 3 text-prompted detection (if available)
- [ ] 2.6 Create occlusion handling test scenarios
- [ ] 2.7 Add unit tests for SAM wrappers

## 3. CoTracker Lane Tracking

- [ ] 3.1 Add CoTracker model loading wrapper
- [ ] 3.2 Implement lane point initialization from model output
- [ ] 3.3 Create lane tracking through video pipeline
- [ ] 3.4 Benchmark CoTracker vs current lane detection
- [ ] 3.5 Test on challenging conditions (shadows, glare, faded markings)
- [ ] 3.6 Generate pseudo-labels for training data augmentation
- [ ] 3.7 Add unit tests for CoTracker wrapper

## 4. DETR Detection

- [ ] 4.1 Add DETR model loading wrapper
- [ ] 4.2 Implement detection post-processing
- [ ] 4.3 Benchmark DETR vs current detection approach
- [ ] 4.4 Evaluate panoptic segmentation capabilities
- [ ] 4.5 Add unit tests for DETR wrapper

## 5. UnSAMFlow Optical Flow

- [ ] 5.1 Add UnSAMFlow model loading wrapper
- [ ] 5.2 Implement flow estimation pipeline
- [ ] 5.3 Benchmark flow accuracy for ego-motion estimation
- [ ] 5.4 Integrate with object velocity estimation
- [ ] 5.5 Add unit tests for UnSAMFlow wrapper

## 6. Knowledge Distillation Framework

- [ ] 6.1 Create `tools/fair/distillation/` directory structure
- [ ] 6.2 Implement soft label distillation loss
- [ ] 6.3 Implement feature distillation loss
- [ ] 6.4 Implement attention transfer loss
- [ ] 6.5 Create `DistillationTrainer` class
- [ ] 6.6 Add learning rate scheduling and warmup
- [ ] 6.7 Implement checkpoint management
- [ ] 6.8 Add unit tests for distillation components

## 7. Student Model Architecture

- [ ] 7.1 Design EfficientViT-based student backbone
- [ ] 7.2 Implement multi-head student (depth + segmentation + detection)
- [ ] 7.3 Add quantization-friendly layers (avoid BatchNorm issues)
- [ ] 7.4 Implement model profiling utilities
- [ ] 7.5 Target comma 4 latency constraints (<50ms inference)
- [ ] 7.6 Add unit tests for student models

## 8. DoRA + FAIR Distillation

- [ ] 8.1 Integrate DoRA adapters with distillation pipeline
- [ ] 8.2 Implement combined task + distillation loss
- [ ] 8.3 Create training pipeline for DoRA + DINOv2 teacher
- [ ] 8.4 Create training pipeline for DoRA + SAM teacher
- [ ] 8.5 Benchmark DoRA distillation vs full distillation
- [ ] 8.6 Add adapter checkpointing for incremental training

## 9. Dataset Preparation

- [ ] 9.1 Create route log → training data converter
- [ ] 9.2 Implement FAIR model pseudo-label generation
- [ ] 9.3 Create data augmentation pipeline
- [ ] 9.4 Implement hard example mining
- [ ] 9.5 Create dataset statistics and visualization

## 10. Quantization for Comma 4

- [ ] 10.1 Implement quantization-aware training (QAT)
- [ ] 10.2 Add INT8 quantization support
- [ ] 10.3 Benchmark accuracy vs latency trade-offs
- [ ] 10.4 Test on comma 4 hardware (when available)
- [ ] 10.5 Create deployment export utilities (ONNX, TensorRT)

## 11. Integration Testing

- [ ] 11.1 Benchmark distilled models on algorithm test harness
- [ ] 11.2 Compare perception quality vs current openpilot
- [ ] 11.3 Test in simulation with diverse scenarios
- [ ] 11.4 Measure end-to-end latency impact
- [ ] 11.5 Create regression test suite

## 12. Documentation

- [ ] 12.1 Add README for tools/fair/
- [ ] 12.2 Document distillation hyperparameters and tuning
- [ ] 12.3 Create training playbook for DGX Spark
- [ ] 12.4 Document comma 4 deployment process
- [ ] 12.5 Add example notebooks for model evaluation
