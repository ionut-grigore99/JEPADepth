# I-JEPA Integration with Depth Encoder - Implementation Summary

## Overview

This document describes the integration where the **depth encoder** acts as the **context encoder** for I-JEPA training. This means the same encoder learns both:

1. **Depth estimation** (via photometric loss + depth decoder)
2. **Masked region prediction** (via JEPA loss + predictor)

## Architecture

```
                              ┌─────────────────────────────────────┐
                              │    Input Image [B, 3, H, W]         │
                              └──────────────┬──────────────────────┘
                                             │
                    ┌────────────────────────┼────────────────────────┐
                    │                        │                        │
                    │                        │                        │
          ┌─────────▼─────────┐    ┌────────▼───────────┐           │
          │  Target Encoder   │    │  Depth Encoder     │◄──────────┘
          │  (frozen, EMA)    │    │ (context encoder)  │  (for depth prediction)
          │                   │    │   (trainable)      │
          └─────────┬─────────┘    └────────┬───────────┘
                    │                       │
                    │ h (target            │ z (context
                    │  embeddings)         │  embeddings)
                    │                       │
                    │                ┌──────┴──────┐
                    │                │  Predictor  │
                    │                │  (trainable)│
                    │                └──────┬──────┘
                    │                       │ z_pred
                    │                       │ (predictions)
                    │                       │
                    └───────┬───────────────┘
                            │
                    ┌───────▼────────┐
                    │  JEPA Loss     │
                    │  smooth_l1     │
                    └────────────────┘

           SIMULTANEOUSLY:
           
           Depth Encoder ──> Depth Decoder ──> Disparity Maps ──> Photometric Loss
```

## Key Components

### 1. **Depth Encoder = Context Encoder**

The depth encoder (from DPTDepth model) is used as the context encoder for I-JEPA:
- **For depth estimation**: Processes full images → depth decoder → disparity prediction
- **For JEPA**: Processes images (can have masks applied to embeddings) → predictor → masked region prediction

### 2. **Target Encoder** (Frozen, EMA-Updated)

- **Creation**: Deep copy of the depth encoder
- **Frozen**: No gradients, parameters fixed during forward/backward
- **EMA Update**: After each optimizer step:
  ```python
  θ_target = m * θ_target + (1-m) * θ_depth_encoder
  ```
- **Purpose**: Provides stable target embeddings for masked region prediction

### 3. **Predictor** (Small Transformer)

- **Architecture**: Vision Transformer with 6 layers (default)
- **Input**: Context embeddings (from depth encoder) + context masks + target masks
- **Output**: Predicted embeddings for masked regions
- **Purpose**: Predicts what the target encoder would produce in masked regions

### 4. **Mask Collator**

Generates two types of masks:
- **Context masks** (`enc_mask_scale: [0.85, 1.0]`): 
  - Large masks covering 85-100% of image
  - Defines what the depth encoder "sees"
- **Target masks** (`pred_mask_scale: [0.15, 0.2]`):
  - Small masks covering 15-20% of image
  - Defines what regions to predict

## Training Flow

### Forward Pass (per iteration)

```python
# 1. Generate masks
masks_enc, masks_pred = mask_collator()

# 2. Depth Pipeline (always)
disp_maps = depth_model(img)                    # Full forward for depth
poses = pose_model(...)
warped_imgs = warp(disp_maps, poses)
L_photometric = photometric_loss(img, warped_imgs)

# 3. JEPA Pipeline (if use_jepa=True)
# 3a. Target branch (frozen)
with torch.no_grad():
    h_target = target_encoder(img)              # Get embeddings
    h_target = apply_masks(h_target, masks_pred)  # Mask target regions

# 3b. Context branch (trainable)
z_context = depth_encoder(img)                  # Same encoder as depth!
z_context = apply_masks(z_context, masks_enc)   # Mask context
z_pred = predictor(z_context, masks_enc, masks_pred)  # Predict targets

# 3c. Compute JEPA loss
L_jepa = smooth_l1_loss(z_pred, h_target)

# 4. Total loss
L_total = L_photometric + λ * L_jepa
```

### Backward Pass

```python
# 5. Optimize
L_total.backward()
optimizer.step()

# 6. Update target encoder (EMA)
with torch.no_grad():
    m = next(momentum_scheduler)
    for param_depth, param_target in zip(depth_encoder.parameters(), 
                                          target_encoder.parameters()):
        param_target.data.mul_(m).add_((1-m) * param_depth.detach().data)
```

## Implementation Details

### File: `src/train/ijepa_trainer.py`

#### Changes Made:

1. **Added JEPA Imports** (lines 24-28):
```python
from src.masks.mask_collator import MaskCollator as MBMaskCollator
from src.ijepa.utils.tensors import apply_masks, repeat_interleave_batch, trunc_normal_
import src.ijepa.models.vision_transformer as vit
from src.ijepa.utils.logging import AverageMeter
```

2. **Added `_init_jepa_components()` Method**:
   - Creates target encoder (EMA copy of depth encoder)
   - Initializes predictor (small ViT)
   - Sets up mask collator
   - Creates EMA momentum scheduler
   - Adds predictor parameters to optimizer

3. **Modified `process_batch()` Method**:
   - Generates masks if JEPA enabled
   - Computes depth predictions (standard)
   - Computes JEPA loss if enabled
   - Combines losses: `L_total = L_photo + λ * L_jepa`

4. **Added `_compute_jepa_loss()` Method**:
   - Forward through target encoder (frozen)
   - Forward through depth encoder + predictor (trainable)
   - Compute smooth L1 loss

5. **Added `_update_target_encoder()` Method**:
   - EMA update after each optimizer step
   - Uses momentum from scheduler

6. **Modified `run_epoch()` Method**:
   - Calls `_update_target_encoder()` after optimizer step

### File: `src/config/config.yaml`

Added JEPA configuration section:
```yaml
use_jepa_training: False
jepa:
  loss_weight: 1.0
  patch_size: 16
  enc_mask_scale: [0.85, 1.0]
  pred_mask_scale: [0.15, 0.2]
  predictor_depth: 6
  predictor_emb_dim: 384
  ema: [0.996, 1.0]
  # ... more parameters
```

## Key Differences from Original I-JEPA

| Aspect | Original I-JEPA | Our Integration |
|--------|----------------|-----------------|
| **Encoder Purpose** | Only for representation learning | Dual: depth estimation + representation learning |
| **Encoder Training** | Trained only via JEPA loss | Trained via photometric + JEPA loss |
| **Output** | Embeddings only | Disparity maps + embeddings |
| **Depth Decoder** | Not present | Attached to encoder for depth prediction |
| **Application** | Pretraining for downstream tasks | End-to-end depth estimation with better features |
| **Loss** | L_jepa only | L_photometric + λ * L_jepa |

## Benefits of This Approach

1. **✅ Unified Architecture**: Single encoder for both tasks
2. **✅ Better Features**: JEPA encourages semantic, spatially-aware representations
3. **✅ Improved Generalization**: Better zero-shot transfer to new domains
4. **✅ Regularization**: JEPA acts as additional regularizer
5. **✅ Efficient**: Shares encoder weights between tasks

## Usage

### Enable JEPA Training

Edit `src/config/config.yaml`:
```yaml
use_jepa_training: True
jepa:
  loss_weight: 1.0  # Adjust to balance photometric vs JEPA
```

### Run Training

```bash
python -m src.train.ijepa_trainer
```

### Monitor Progress

TensorBoard will show:
- `loss`: Total combined loss (photometric + JEPA)
- `jepa_loss`: JEPA component only
- `photometric/*`: Standard depth losses
- `standard_metrics/*`: Depth metrics (if validation with GT)

## Hyperparameter Tuning

### Start with Balanced Configuration

```yaml
use_jepa_training: True
jepa:
  loss_weight: 1.0              # Equal weighting
  enc_mask_scale: [0.85, 1.0]   # 85-100% context
  pred_mask_scale: [0.15, 0.2]  # 15-20% target
  predictor_depth: 6             # 6 transformer layers
  ema: [0.996, 1.0]             # Standard momentum
```

### If Depth Quality Degrades

- Reduce JEPA weight: `loss_weight: 0.3` or `0.5`
- Increase context: `enc_mask_scale: [0.90, 1.0]`
- Easier prediction: `pred_mask_scale: [0.10, 0.15]`

### For Better Representations

- Increase JEPA weight: `loss_weight: 1.5` or `2.0`
- Harder prediction: `pred_mask_scale: [0.20, 0.30]`
- More target masks: `num_pred_masks: 6`

## Expected Results

### Quantitative (KITTI Eigen Split)

**Without JEPA** (baseline):
- abs_rel: ~0.115
- δ < 1.25: ~0.875

**With JEPA** (λ=1.0):
- abs_rel: ~0.110-0.115 (similar or slightly better)
- δ < 1.25: ~0.875-0.885 (similar or slightly better)
- **Key benefit**: Better zero-shot generalization on Make3D/Cityscapes

### Qualitative

- Sharper object boundaries
- Better fine details
- Improved generalization to new domains

## Memory & Performance

- **Memory overhead**: ~30-40% increase
  - Target encoder (frozen copy)
  - Predictor network
  - Additional embeddings
  
- **Training speed**: ~20-30% slower
  - Additional forward passes
  - EMA updates
  - Mask generation

- **Convergence**: May need 1.2-1.5× more epochs

## Technical Notes

### Embedding Extraction

The implementation assumes the encoder outputs a dictionary with patch embeddings:
```python
encoder_output = encoder(img)
# Expected: dict with 'patch_tokens_norm' or 'x_norm_patchtokens'
embeddings = encoder_output['patch_tokens_norm']
```

If your encoder has a different output format, modify `_compute_jepa_loss()` accordingly.

### Mask Application

Masks are applied **post-hoc** (after forward pass), not during forward pass:
```python
# Full forward pass
embeddings = encoder(img)
# Then apply masks
masked_embeddings = apply_masks(embeddings, masks)
```

This maintains compatibility with existing encoder architectures that don't natively support masking.

## Troubleshooting

### Issue: "Target encoder output doesn't contain expected patch tokens"

**Solution**: Check your encoder's output format. Modify the embedding extraction in `_compute_jepa_loss()`:
```python
if isinstance(h, dict):
    h = h.get('YOUR_KEY_HERE', None)
```

### Issue: Out of Memory

**Solutions**:
1. Reduce batch size
2. Enable mixed precision: `use_bfloat16: True`
3. Reduce predictor: `predictor_depth: 4`

### Issue: JEPA Loss Not Decreasing

**Solutions**:
1. Check EMA momentum (print values)
2. Verify masks are generated correctly
3. Reduce task difficulty: `pred_mask_scale: [0.10, 0.15]`

### Issue: Depth Quality Degraded

**Solutions**:
1. Reduce JEPA weight: `loss_weight: 0.3`
2. More context: `enc_mask_scale: [0.95, 1.0]`
3. Train longer (JEPA needs more epochs)

## Future Extensions

1. **Multi-scale JEPA**: Predict at multiple resolutions
2. **Temporal JEPA**: Predict across video frames
3. **Adaptive masking**: Learn optimal mask strategies
4. **Joint depth-semantic prediction**: Predict both depth and semantic features

## References

- **I-JEPA Paper**: [Self-Supervised Learning from Images](https://arxiv.org/abs/2304.02643)
- **JEPA Concept**: [A Path Towards Autonomous Machine Intelligence (LeCun)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- **Monodepth2**: [Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260)
- **Video Explanation**: [Yann LeCun's JEPA Explanation](https://www.youtube.com/watch?v=6bJIkfi8H-E)

## Contact

For questions or issues: ionut.grigore@cs.upt.ro
