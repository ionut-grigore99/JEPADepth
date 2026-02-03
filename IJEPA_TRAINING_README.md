# JEPA-Style Training Integration for Self-Supervised Depth Estimation

## Overview

This document describes the integration of JEPA (Joint-Embedding Predictive Architecture) style training into the self-supervised depth estimation pipeline. The integration combines:

1. **Traditional photometric loss**: Warping-based self-supervised depth learning
2. **JEPA masked prediction loss**: Learning representations through masked region prediction

## Architecture

### Dual Training Objective

The integrated trainer (`ijepa_trainer.py`) optimizes two complementary objectives:

1. **Depth Estimation Loss** (Traditional):
   - Photometric reprojection loss
   - SSIM loss
   - Smoothness regularization
   - Auto-masking

2. **JEPA Representation Loss** (New):
   - Masked region prediction on the encoder
   - Target encoder (EMA-updated, frozen)
   - Predictor network (small transformer)
   - Context-to-target prediction using smooth L1 loss

### Key Components

#### 1. Target Encoder
- EMA (Exponential Moving Average) copy of the depth model's encoder
- Frozen during training (no gradients)
- Updated via momentum schedule: `θ_target = m * θ_target + (1-m) * θ_encoder`
- Provides stable target embeddings for masked prediction

#### 2. Predictor Network
- Lightweight Vision Transformer
- Takes context embeddings + masks
- Predicts target embeddings in masked regions
- Parameters:
  - `predictor_depth`: Number of transformer layers (default: 6)
  - `predictor_emb_dim`: Hidden dimension (default: 384)

#### 3. Mask Collator
- Generates context masks (85-100% of patches)
- Generates target masks (15-20% of patches)
- Configurable overlap policy
- Aspect ratio and scale randomization

## Configuration

Add the following section to `src/config/config.yaml`:

```yaml
# JEPA-style training (optional)
# -----------
use_jepa_training: false  # Set to true to enable JEPA training
jepa:
  loss_weight: 1.0  # Weight for JEPA loss relative to photometric loss
  use_bfloat16: false  # Use mixed precision training
  
  # Masking strategy
  patch_size: 16  # Patch size for masking (16x16 pixels)
  enc_mask_scale: [0.85, 1.0]  # Context mask covers 85-100% of image
  pred_mask_scale: [0.15, 0.2]  # Target mask covers 15-20% of image
  aspect_ratio: [0.75, 1.5]  # Aspect ratio range for masks
  num_enc_masks: 1  # Number of context masks per image
  num_pred_masks: 4  # Number of target masks per image
  allow_overlap: false  # Allow overlap between context and target masks
  min_keep: 10  # Minimum number of patches to keep
  
  # Predictor network
  predictor_depth: 6  # Number of transformer layers in predictor
  predictor_emb_dim: 384  # Hidden dimension of predictor
  
  # EMA (Exponential Moving Average) for target encoder
  ema: [0.996, 1.0]  # Momentum schedule [start, end]
```

## Training

To use JEPA training, simply set `use_jepa_training: true` in your config:

```bash
python -m src.train.ijepa_trainer
```

The trainer will:
1. Initialize JEPA components (target encoder, predictor, mask collator)
2. Compute both photometric and JEPA losses each iteration
3. Update the target encoder with momentum after each optimization step
4. Log JEPA loss to TensorBoard

## Loss Computation

### Total Loss
```
L_total = L_photometric + λ * L_jepa
```

Where:
- `L_photometric`: Standard self-supervised depth loss (reprojection + smoothness)
- `L_jepa`: JEPA masked prediction loss
- `λ`: JEPA loss weight (configurable via `jepa.loss_weight`)

### JEPA Loss Details
```
L_jepa = SmoothL1(z_pred, h_target)
```

Where:
- `z_pred`: Predicted embeddings from context (encoder + predictor)
- `h_target`: Target embeddings from target encoder (frozen)
- Both are masked and normalized

## Benefits

1. **Improved Feature Learning**: JEPA encourages the encoder to learn more semantic, spatially-aware representations
2. **Regularization**: Acts as an additional regularizer, potentially reducing overfitting
3. **Better Generalization**: Masked prediction can improve zero-shot transfer to new domains
4. **Complementary Objectives**: Geometric (photometric) + semantic (JEPA) learning

## Implementation Details

### Forward Pass (per iteration)

1. **Standard Depth Pipeline**:
   ```
   img → depth_model → disp_maps → warp → photometric_loss
   ```

2. **JEPA Pipeline** (if enabled):
   ```
   img → [mask generation]
        ↓
   context_encoder → predictor → z_pred ──┐
        ↓                                  │
   target_encoder (frozen) → h_target ────┤
                                           │
                                      [smooth_l1_loss]
   ```

3. **Target Encoder Update** (after optimization):
   ```
   θ_target ← m * θ_target + (1-m) * θ_encoder
   ```

### Masking Strategy

- **Context Mask**: Large, covers most of the image (85-100%)
  - Provides spatial context
  - Input to encoder

- **Target Mask**: Small, sparse regions (15-20%)
  - Regions to predict
  - Randomly sampled with controlled aspect ratio

### Memory Considerations

JEPA training adds:
- Target encoder (same size as main encoder, but frozen)
- Predictor network (small, ~10-20% of encoder size)
- Additional embeddings during forward pass

Memory overhead: ~30-40% increase compared to standard training.

## Monitoring

TensorBoard logs include:
- `loss`: Total loss (photometric + JEPA)
- `jepa_loss`: JEPA masked prediction loss only
- `photometric/*`: Standard depth metrics
- `jepa/*`: JEPA-specific metrics (if available)

## Tips for Tuning

1. **Loss Weight (`jepa.loss_weight`)**:
   - Start with 1.0 (equal weighting)
   - Reduce if depth quality degrades
   - Increase if targeting better representations

2. **Masking**:
   - Larger context masks → more spatial information
   - Larger target masks → harder prediction task
   - Adjust based on image resolution and patch size

3. **EMA Momentum**:
   - Higher momentum (closer to 1.0) → slower target updates
   - Start: 0.996, End: 1.0 is a good default
   - Adjust based on training stability

4. **Predictor Depth**:
   - Deeper predictor → more capacity but slower training
   - 6 layers is a good starting point
   - Reduce to 4 if training is too slow

## References

- JEPA: [https://arxiv.org/abs/2301.08243](https://arxiv.org/abs/2301.08243)
- I-JEPA (Image-based): [https://arxiv.org/abs/2304.02643](https://arxiv.org/abs/2304.02643)
- Monodepth2 (base depth method): [https://arxiv.org/abs/1806.01260](https://arxiv.org/abs/1806.01260)

## Future Extensions

Potential improvements:
1. Multi-scale JEPA loss (predict at multiple resolutions)
2. Temporal JEPA (predict across video frames)
3. Joint depth-semantic prediction
4. Adaptive masking strategies
