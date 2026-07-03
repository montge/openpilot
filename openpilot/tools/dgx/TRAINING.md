# DGX Spark Training Pipeline

Training openpilot driving models on DGX Spark using DoRA fine-tuning.

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DGX Spark Training Pipeline                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │  Route Logs  │───▶│ Data Loader  │───▶│   Batches    │                  │
│  │  (rlogs)     │    │              │    │ (frames +    │                  │
│  └──────────────┘    └──────────────┘    │  metadata)   │                  │
│                                          └──────┬───────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Teacher    │───▶│   Pseudo     │───▶│    Loss      │                  │
│  │  (TensorRT)  │    │   Labels     │    │  Computation │                  │
│  │   800+ FPS   │    │              │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────┬───────┘                  │
│                                                 │                          │
│                                                 ▼                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │   Student    │◀───│   DoRA       │◀───│  Gradients   │                  │
│  │   Model      │    │   Adapters   │    │  (backward)  │                  │
│  │  (PyTorch)   │    │  (~0.1% Δ)   │    │              │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Pipeline

**Input**: Route logs from comma device (`/data/media/0/realdata/`)

**Route structure**:
```
route_id/
├── rlog.bz2          # Full message log (Cap'n Proto)
├── qlog.bz2          # Downsampled log
├── fcamera.hevc      # Front camera (1164x874 @ 20fps)
├── ecamera.hevc      # Wide camera
├── dcamera.hevc      # Driver camera
└── qcamera.ts        # Downsampled video
```

**Extracted data per frame**:
- Camera frame (YUV420 → RGB → normalized)
- Desire vector (8-dim: lane changes, turns)
- Traffic convention (2-dim: left/right hand drive)
- GPS coordinates (for filtering/validation)
- Vehicle state (speed, steering angle, etc.)

### 2. Teacher Model (TensorRT)

Uses pre-trained comma models to generate pseudo-labels:

```python
# Teacher generates targets at 800+ FPS
teacher_vision = TensorRTEngine("driving_vision.onnx")
teacher_policy = TensorRTEngine("driving_policy.onnx")

# For each batch of frames
features = teacher_vision(frames)           # Visual features
targets = teacher_policy(features, desire)  # Path predictions
```

**Outputs (pseudo-labels)**:
- Path predictions (33 future timestamps × 2 coords)
- Lane line positions
- Road edge positions
- Lead vehicle detection

### 3. DoRA Fine-Tuning

**DoRA** (Weight-Decomposed Low-Rank Adaptation):
- Decomposes weights into magnitude and direction
- Only trains low-rank direction updates
- ~0.1% of parameters vs full fine-tuning

```python
class DoRALayer(nn.Module):
    def __init__(self, base_layer, rank=16):
        self.base_weight = base_layer.weight  # Frozen
        self.magnitude = nn.Parameter(base_layer.weight.norm(dim=1))
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        # Direction = base + low-rank update
        delta_W = self.lora_B @ self.lora_A
        direction = self.base_weight + delta_W
        direction = direction / direction.norm(dim=1, keepdim=True)

        # Apply magnitude scaling
        weight = self.magnitude.unsqueeze(1) * direction
        return F.linear(x, weight)
```

**Target layers for adaptation**:
- Final dense layers in policy network
- GRU hidden state projections
- Output heads (paths, lanes, edges)

### 4. Loss Functions

**Primary: Laplacian NLL (winner-takes-all)**
```python
def laplacian_nll_loss(pred_mean, pred_std, target):
    """
    pred_mean: (batch, num_hypotheses, horizon, 2)  # x, y positions
    pred_std: (batch, num_hypotheses, horizon, 2)   # uncertainties
    target: (batch, horizon, 2)                     # ground truth path
    """
    # Compute NLL for each hypothesis
    diff = pred_mean - target.unsqueeze(1)
    nll = torch.abs(diff) / pred_std + torch.log(2 * pred_std)
    nll = nll.sum(dim=(-1, -2))  # Sum over horizon and coords

    # Winner-takes-all: only backprop through best hypothesis
    best_idx = nll.argmin(dim=1)
    loss = nll.gather(1, best_idx.unsqueeze(1)).mean()
    return loss
```

**Auxiliary losses**:
- Lane line position loss (L1)
- Road edge loss (L1)
- Lead vehicle distance loss (Huber)

### 5. Training Loop

```python
def train_epoch(student, teacher, dataloader, optimizer, device):
    student.train()

    for batch in dataloader:
        frames = batch['frames'].to(device)        # (B, T, C, H, W)
        desire = batch['desire'].to(device)        # (B, T, 8)
        traffic = batch['traffic'].to(device)      # (B, 2)

        # Teacher generates pseudo-labels (no grad)
        with torch.no_grad():
            features = teacher.vision(frames)
            targets = teacher.policy(features, desire, traffic)

        # Student forward pass
        hidden = None
        total_loss = 0

        for t in range(frames.shape[1]):
            pred, hidden = student(
                frames[:, t],
                desire[:, t],
                traffic,
                hidden
            )
            loss = compute_loss(pred, targets[:, t])
            total_loss += loss

        # Backward pass (only DoRA params have gradients)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
```

### 6. Export Pipeline

After training, merge DoRA weights and export:

```python
def merge_dora_weights(model):
    """Merge DoRA adapters back into base weights."""
    for name, module in model.named_modules():
        if isinstance(module, DoRALayer):
            delta_W = module.lora_B @ module.lora_A
            direction = module.base_weight + delta_W
            direction = direction / direction.norm(dim=1, keepdim=True)
            merged_weight = module.magnitude.unsqueeze(1) * direction
            module.base_weight.copy_(merged_weight)

def export_to_onnx(model, path):
    """Export merged model to ONNX for comma device."""
    dummy_inputs = {
        'img': torch.randn(1, 12, 128, 256),
        'big_img': torch.randn(1, 12, 128, 256),
        'desire': torch.randn(1, 8),
        'traffic_convention': torch.randn(1, 2),
    }
    torch.onnx.export(model, dummy_inputs, path, opset_version=14)
```

## File Structure

```
tools/dgx/
├── training/
│   ├── __init__.py
│   ├── dataloader.py      # Route log → training batches
│   ├── teacher.py         # TensorRT teacher wrapper
│   ├── dora.py            # DoRA layer implementation
│   ├── losses.py          # Loss functions
│   ├── model.py           # Student model with DoRA
│   ├── train.py           # Training script
│   └── export.py          # ONNX export utilities
├── benchmark_inference.py
├── benchmark_tensorrt.py
└── README.md
```

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU Memory | 16GB | 128GB (unified) |
| CPU Cores | 8 | 20 (DGX Spark) |
| Storage | 100GB | 1TB+ |
| Training Time | ~1 hour | - |

DGX Spark advantages:
- 128GB unified memory = large batch sizes
- TensorRT teacher at 800+ FPS = fast pseudo-label generation
- Single device = no distributed training complexity

## Usage

```bash
# 1. Prepare data from route logs
python tools/dgx/training/dataloader.py --routes /path/to/routes --output /data/training

# 2. Train with DoRA
python tools/dgx/training/train.py \
    --data /data/training \
    --epochs 10 \
    --batch-size 32 \
    --dora-rank 16 \
    --lr 1e-4

# 3. Export to ONNX
python tools/dgx/training/export.py \
    --checkpoint best_model.pt \
    --output custom_driving_policy.onnx
```

## Next Steps

1. [ ] Implement route log data loader
2. [ ] Create TensorRT teacher wrapper
3. [ ] Implement DoRA layer
4. [ ] Create training loop
5. [ ] Add export/merge utilities
6. [ ] Test end-to-end pipeline
7. [ ] Benchmark training performance
