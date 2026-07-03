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
- Desire vector (8-dim per frame: lane changes, turns; the model consumes a 25-frame `desire_pulse` history, shape (25, 8))
- Traffic convention (2-dim: left/right hand drive)
- GPS coordinates (for filtering/validation)
- Vehicle state (speed, steering angle, etc.)

### 2. Teacher Model (TensorRT)

Uses the pre-trained comma supercombo model to generate pseudo-labels. Since the
July 2026 upstream restructure there is no separate vision stage: the combined
`driving_supercombo.onnx` does vision + policy in a single forward pass.

```python
# Teacher generates targets (800+ FPS measured historically on the split models)
teacher = TensorRTEngine("driving_supercombo.onnx")

# Model metadata (input_shapes, output_shapes, output_slices, model_checkpoint)
# is embedded in the ONNX metadata_props -- there is no sidecar *_metadata.pkl
from openpilot.selfdrive.modeld.get_model_metadata import make_metadata_dict
metadata = make_metadata_dict("driving_supercombo.onnx")
output_slices = metadata["output_slices"]

# For each frame: one forward pass with the 6 supercombo inputs
outputs = teacher(
    img=img,                          # (1, 12, 128, 256) uint8, road camera
    big_img=big_img,                  # (1, 12, 128, 256) uint8, wide camera
    features_buffer=features_buffer,  # (1, 24, 512) float, rolling recurrent state
    desire_pulse=desire_pulse,        # (1, 25, 8) float
    traffic_convention=traffic,       # (1, 2) float
    action_t=action_t,                # (1, 2) float
)  # -> single flat output vector (1, 2576)

# Roll the recurrent state: this frame's hidden_state (512) feeds the next frame
hidden = outputs[:, output_slices["hidden_state"]]
features_buffer = np.concatenate([features_buffer[:, 1:], hidden[:, None]], axis=1)
```

**Outputs (pseudo-labels)**: the flat `outputs` vector is decomposed with
`output_slices` (meta, desire_pred, pose, wide_from_device_euler, road_transform,
lane_lines, lane_lines_prob, road_edges, lead, lead_prob, hidden_state, plan,
desire_state, pad) and parsed with `Parser().parse_outputs()` from
`openpilot.selfdrive.modeld.parse_model_outputs`:
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
- Final dense layers in the supercombo policy head
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
        frames = batch['frames'].to(device)              # (B, T, C, H, W) road camera
        big_frames = batch['big_frames'].to(device)      # (B, T, C, H, W) wide camera
        desire_pulse = batch['desire_pulse'].to(device)  # (B, T, 25, 8)
        traffic = batch['traffic'].to(device)            # (B, 2)
        action_t = batch['action_t'].to(device)          # (B, 2)

        # Teacher generates pseudo-labels (no grad): one supercombo pass per
        # frame, rolling the previous frames' hidden_state through features_buffer
        with torch.no_grad():
            feat_buf = torch.zeros(frames.shape[0], 24, 512, device=device)
            targets = []
            for t in range(frames.shape[1]):
                out = teacher(frames[:, t], big_frames[:, t], feat_buf,
                              desire_pulse[:, t], traffic, action_t)
                targets.append(out)
                hidden = out[:, output_slices["hidden_state"]]  # (B, 512)
                feat_buf = torch.cat([feat_buf[:, 1:], hidden.unsqueeze(1)], dim=1)

        # Student forward pass (same rolling recurrent state)
        student_buf = torch.zeros(frames.shape[0], 24, 512, device=device)
        total_loss = 0

        for t in range(frames.shape[1]):
            pred, hidden = student(
                frames[:, t],
                big_frames[:, t],
                student_buf,
                desire_pulse[:, t],
                traffic,
                action_t,
            )
            student_buf = torch.cat([student_buf[:, 1:], hidden.unsqueeze(1)], dim=1)
            loss = compute_loss(pred, targets[t])
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
        'img': torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8),
        'big_img': torch.randint(0, 256, (1, 12, 128, 256), dtype=torch.uint8),
        'features_buffer': torch.randn(1, 24, 512),
        'desire_pulse': torch.randn(1, 25, 8),
        'traffic_convention': torch.randn(1, 2),
        'action_t': torch.randn(1, 2),
    }
    torch.onnx.export(model, dummy_inputs, path, opset_version=14)
```

## File Structure

```
openpilot/tools/dgx/
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
- TensorRT teacher at 800+ FPS (historical split-model measurement) = fast pseudo-label generation
- Single device = no distributed training complexity

## Usage

```bash
# 1. Prepare data from route logs
python openpilot/tools/dgx/training/dataloader.py --routes /path/to/routes --output /data/training

# 2. Train with DoRA
python openpilot/tools/dgx/training/train.py \
    --data /data/training \
    --epochs 10 \
    --batch-size 32 \
    --dora-rank 16 \
    --lr 1e-4

# 3. Export to ONNX
python openpilot/tools/dgx/training/export.py \
    --checkpoint best_model.pt \
    --output custom_driving_supercombo.onnx
```

## Next Steps

1. [ ] Implement route log data loader
2. [ ] Create TensorRT teacher wrapper
3. [ ] Implement DoRA layer
4. [ ] Create training loop
5. [ ] Add export/merge utilities
6. [ ] Test end-to-end pipeline
7. [ ] Benchmark training performance
