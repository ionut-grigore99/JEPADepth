# JEPA Training Integration - Summary

## What Was Done

Successfully integrated JEPA (Joint-Embedding Predictive Architecture) style training into the self-supervised depth estimation pipeline by adapting `src/ijepa/train.py` into `src/train/ijepa_trainer.py`.

## Key Changes

### 1. **Added JEPA Imports** (`ijepa_trainer.py` lines 1-32)
```python
from src.ijepa.masks.multiblock import MaskCollator as MBMaskCollator
from src.ijepa.utils.tensors import apply_masks, repeat_interleave_batch
from src.ijepa.utils.logging import AverageMeter
import src.ijepa.models.vision_transformer as vit
from src.ijepa.utils.tensors import trunc_normal_
```

### 2. **Enhanced Trainer.__init__()** (lines 130-250)
Added JEPA components initialization:
- **Target Encoder**: EMA copy of depth encoder (frozen, no gradients)
- **Predictor Network**: Small Vision Transformer for masked prediction
- **Mask Collator**: Generates context/target masks
- **Momentum Scheduler**: For EMA updates of target encoder
- **Loss Meters**: Track JEPA loss separately

Key features:
- Conditional initialization via `use_jepa_training` config flag
- Compatible with existing depth + pose model setup
- Predictor parameters added to optimizer
- Configurable hyperparameters from config.yaml

### 3. **Updated process_batch()** (lines 335-365)
Enhanced to compute JEPA loss alongside photometric loss:
```python
if self.use_jepa:
    jepa_loss = self.compute_jepa_loss(inputs["color_aug", 0, 0])
    losses["jepa_loss"] = jepa_loss
    losses["loss"] = losses["loss"] + self.jepa_weight * jepa_loss
```

### 4. **New compute_jepa_loss()** (lines 367-420)
Implements JEPA masked prediction pipeline:
1. Generate random context and target masks
2. Forward through frozen target encoder → get target embeddings
3. Forward through trainable encoder + predictor → get predictions
4. Compute smooth L1 loss between predictions and targets

### 5. **New update_target_encoder()** (lines 422-432)
Momentum update after each optimization step:
```python
θ_target = m * θ_target + (1-m) * θ_encoder
```

### 6. **Modified run_epoch()** (lines 284-320)
Added target encoder update after optimizer step:
```python
self.model_optimizer.step()
if self.use_jepa:
    self.update_target_encoder()
```

### 7. **Updated Configuration** (`src/config/config.yaml`)
Added complete JEPA configuration section:
```yaml
use_jepa_training: false  # Enable/disable JEPA
jepa:
  loss_weight: 1.0
  patch_size: 16
  enc_mask_scale: [0.85, 1.0]
  pred_mask_scale: [0.15, 0.2]
  predictor_depth: 6
  predictor_emb_dim: 384
  ema: [0.996, 1.0]
  # ... and more
```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    IJEPA Depth Trainer                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Image ──┬──> Depth Model ──> Disp Maps ──> Warp    │
│                │         │                         │        │
│                │         │                    Photometric   │
│                │         │                        Loss      │
│                │         │                                  │
│                └──> [JEPA Pipeline] ───────> JEPA Loss     │
│                     (if use_jepa=True)                      │
│                                                             │
│  JEPA Pipeline:                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │ 1. Generate Masks (context + target)             │      │
│  │                                                   │      │
│  │ 2. Target Encoder (frozen, EMA)                  │      │
│  │    └─> Target Embeddings                         │      │
│  │                                                   │      │
│  │ 3. Context Encoder (trainable)                   │      │
│  │    └─> Context Embeddings                        │      │
│  │         └─> Predictor Network                    │      │
│  │              └─> Predicted Embeddings            │      │
│  │                                                   │      │
│  │ 4. Loss = SmoothL1(predicted, target)            │      │
│  │                                                   │      │
│  │ 5. After optimizer.step():                       │      │
│  │    θ_target ← m*θ_target + (1-m)*θ_encoder       │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  Total Loss = Photometric Loss + λ * JEPA Loss             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## How It Works

### Training Loop (per iteration)

1. **Standard Depth Pipeline** (always):
   - Input image → Depth model → Disparity maps
   - Predict poses → Warp adjacent frames
   - Compute photometric loss (reprojection + SSIM + smoothness)

2. **JEPA Pipeline** (if `use_jepa_training=True`):
   - Generate context masks (85-100% of patches)
   - Generate target masks (15-20% of patches)
   - **Target branch** (frozen):
     - Input → Target Encoder → Target Embeddings
     - Apply target masks → Normalized embeddings
   - **Context branch** (trainable):
     - Input → Depth Encoder (with context masks) → Context Embeddings
     - Context Embeddings → Predictor → Predicted Embeddings
   - Compute JEPA loss: `SmoothL1(predicted, target)`

3. **Optimization**:
   - Total loss = Photometric + λ × JEPA
   - Backward pass
   - Optimizer step
   - **EMA update**: Target encoder ← momentum × target + (1-momentum) × encoder

### Masking Strategy

**Context Masks** (what the encoder sees):
- Large regions (85-100% of image)
- Provides spatial context for prediction
- Allows encoder to build global understanding

**Target Masks** (what to predict):
- Small regions (15-20% of image)
- Randomly placed with controlled aspect ratio
- Predictor must fill in these "missing" regions

## Usage

### Enable JEPA Training

In `src/config/config.yaml`:
```yaml
use_jepa_training: true
jepa:
  loss_weight: 1.0  # Adjust to balance photometric vs JEPA loss
```

### Run Training
```bash
python -m src.train.ijepa_trainer
```

### Monitor Progress

TensorBoard will show:
- `loss`: Total combined loss
- `jepa_loss`: JEPA component only
- `photometric/*`: Standard depth metrics
- `standard_metrics/*`: Depth evaluation metrics

## Benefits

1. **Better Representations**: JEPA encourages semantic, spatially-aware features
2. **Regularization**: Additional learning signal prevents overfitting
3. **Improved Generalization**: Better zero-shot transfer to new domains
4. **Complementary Learning**: Geometric (photometric) + semantic (JEPA)

## Hyperparameter Tuning Guide

### Start Here (Recommended Defaults)
```yaml
use_jepa_training: true
jepa:
  loss_weight: 1.0
  patch_size: 16
  enc_mask_scale: [0.85, 1.0]
  pred_mask_scale: [0.15, 0.2]
  predictor_depth: 6
  predictor_emb_dim: 384
  ema: [0.996, 1.0]
```

### If Depth Quality Degrades
- **Reduce JEPA weight**: `loss_weight: 0.5` or `0.3`
- **Increase context mask**: `enc_mask_scale: [0.90, 1.0]`
- **Decrease target mask**: `pred_mask_scale: [0.10, 0.15]`

### If Training is Unstable
- **Increase EMA momentum**: `ema: [0.998, 1.0]`
- **Reduce predictor depth**: `predictor_depth: 4`
- **Enable mixed precision**: `use_bfloat16: true`

### For Better Representations
- **Increase JEPA weight**: `loss_weight: 1.5` or `2.0`
- **Harder prediction task**: `pred_mask_scale: [0.20, 0.30]`
- **More target masks**: `num_pred_masks: 6`

## Files Modified

1. **`src/train/ijepa_trainer.py`**:
   - Added JEPA imports
   - Enhanced `__init__()` with JEPA components
   - Updated `process_batch()` to compute JEPA loss
   - Added `compute_jepa_loss()` method
   - Added `update_target_encoder()` method
   - Modified `run_epoch()` for EMA updates

2. **`src/config/config.yaml`**:
   - Added `use_jepa_training` flag
   - Added complete `jepa:` configuration section

3. **New Files Created**:
   - `IJEPA_TRAINING_README.md`: Comprehensive documentation
   - `IJEPA_INTEGRATION_SUMMARY.md`: This summary

## Testing

To test the integration:

1. **Without JEPA** (baseline):
```yaml
use_jepa_training: false
```
```bash
python -m src.train.ijepa_trainer
```

2. **With JEPA**:
```yaml
use_jepa_training: true
```
```bash
python -m src.train.ijepa_trainer
```

Compare:
- Training loss convergence
- Validation metrics
- Zero-shot evaluation on Make3D/Cityscapes
- Visual quality of predictions

## Next Steps

1. **Test on small dataset**: Verify training runs without errors
2. **Ablation studies**: Compare with/without JEPA
3. **Hyperparameter search**: Find optimal `loss_weight`
4. **Evaluate generalization**: Test on zero-shot benchmarks
5. **Visualize masks**: Add mask visualization to TensorBoard

## References

- **JEPA Paper**: https://arxiv.org/abs/2301.08243
- **I-JEPA (Image)**: https://arxiv.org/abs/2304.02643
- **Original IJEPA Code**: `src/ijepa/train.py`
- **Monodepth2**: https://arxiv.org/abs/1806.01260

## Contact

For questions or issues with the JEPA integration, check:
1. This summary document
2. `IJEPA_TRAINING_README.md` for detailed documentation
3. Original JEPA code in `src/ijepa/`
