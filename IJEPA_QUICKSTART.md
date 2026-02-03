# JEPA Training Integration - Quick Reference

## What is JEPA Training?

JEPA (Joint-Embedding Predictive Architecture) is a self-supervised learning method that learns representations by predicting masked regions of an image in embedding space.

**Combined with depth estimation**, it provides:
- 🎯 **Geometric learning** (photometric loss from video)
- 🧠 **Semantic learning** (masked prediction loss from JEPA)

## Quick Start

### 1. Enable JEPA in Config

Edit `src/config/config.yaml`:
```yaml
use_jepa_training: true
```

### 2. Run Training
```bash
python -m src.train.ijepa_trainer
```

That's it! The trainer will now optimize both photometric and JEPA losses.

## Architecture at a Glance

```
Input Frame ─┬─> Depth Model ───> Disparity Maps ──> Photometric Loss
             │
             └─> JEPA Pipeline:
                 ├─> Target Encoder (frozen) ──> Target Embeddings
                 └─> Context Encoder (trainable) ──> Predictor ──> Predictions
                                                                        │
                                                    JEPA Loss ──────────┘
                                                    
Total Loss = Photometric Loss + λ × JEPA Loss
```

## Key Configuration Parameters

### Essential
```yaml
use_jepa_training: true/false  # Enable/disable JEPA
jepa:
  loss_weight: 1.0  # How much to weight JEPA vs photometric loss
```

### Masking (affects difficulty)
```yaml
jepa:
  enc_mask_scale: [0.85, 1.0]   # Context: 85-100% of image
  pred_mask_scale: [0.15, 0.2]  # Target: 15-20% of image
```

### Network Architecture
```yaml
jepa:
  predictor_depth: 6        # Transformer layers in predictor
  predictor_emb_dim: 384    # Hidden dimension
```

### Training Dynamics
```yaml
jepa:
  ema: [0.996, 1.0]         # Momentum for target encoder update
  use_bfloat16: false       # Mixed precision training
```

## Common Scenarios

### Scenario 1: Baseline (No JEPA)
```yaml
use_jepa_training: false
```
```bash
python -m src.train.ijepa_trainer
```
**Use when**: Standard depth training, comparing against baseline

---

### Scenario 2: Balanced JEPA + Depth
```yaml
use_jepa_training: true
jepa:
  loss_weight: 1.0
```
```bash
python -m src.train.ijepa_trainer
```
**Use when**: Want both geometric and semantic learning equally

---

### Scenario 3: Prioritize Depth Quality
```yaml
use_jepa_training: true
jepa:
  loss_weight: 0.3          # Lower JEPA weight
  enc_mask_scale: [0.90, 1.0]  # Easier context
```
**Use when**: Depth metrics more important than representation quality

---

### Scenario 4: Prioritize Representation Learning
```yaml
use_jepa_training: true
jepa:
  loss_weight: 2.0          # Higher JEPA weight
  pred_mask_scale: [0.20, 0.30]  # Harder prediction
  num_pred_masks: 6
```
**Use when**: Targeting zero-shot generalization, transfer learning

## Monitoring Training

### TensorBoard Metrics

**Loss Components**:
- `loss`: Total combined loss
- `jepa_loss`: JEPA masked prediction loss
- `photometric/reprojection_loss`: Standard depth loss
- `photometric/smoothness_loss`: Depth smoothness regularization

**Depth Metrics** (if validation with GT):
- `standard_metrics/abs_rel`
- `standard_metrics/rmse`
- `threshold_metrics/a1`

### What to Watch

✅ **Good Signs**:
- Both losses decreasing steadily
- Validation metrics improving
- No sudden spikes or divergence

⚠️ **Warning Signs**:
- JEPA loss much larger than photometric → reduce `loss_weight`
- Depth quality degrading → reduce `loss_weight` or adjust masks
- Training unstable → increase `ema` momentum, reduce predictor depth

## Troubleshooting

### Issue: Out of Memory
**Solutions**:
1. Reduce batch size in config.yaml
2. Use mixed precision: `use_bfloat16: true`
3. Reduce predictor size: `predictor_depth: 4`

### Issue: JEPA Loss Not Decreasing
**Solutions**:
1. Check that target encoder is updating (print momentum values)
2. Verify masks are being generated correctly
3. Reduce prediction difficulty: `pred_mask_scale: [0.10, 0.15]`

### Issue: Depth Quality Worse with JEPA
**Solutions**:
1. Reduce JEPA weight: `loss_weight: 0.3` or `0.5`
2. Increase context information: `enc_mask_scale: [0.95, 1.0]`
3. Train longer (JEPA needs more epochs to converge)

### Issue: Training Too Slow
**Solutions**:
1. Disable JEPA for faster iteration: `use_jepa_training: false`
2. Reduce predictor complexity: `predictor_depth: 4`
3. Use mixed precision: `use_bfloat16: true`

## File Structure

```
src/
├── train/
│   ├── trainer.py           # Standard depth training
│   └── ijepa_trainer.py     # JEPA-enhanced training ⭐
├── ijepa/
│   ├── train.py             # Original JEPA training (reference)
│   ├── models/
│   │   └── vision_transformer.py  # Predictor architecture
│   ├── masks/
│   │   └── multiblock.py    # Mask generation
│   └── utils/
│       └── tensors.py       # JEPA utilities
└── config/
    └── config.yaml          # Configuration (includes JEPA section)
```

## Documentation Files

- 📘 **IJEPA_TRAINING_README.md**: Complete technical documentation
- 📋 **IJEPA_INTEGRATION_SUMMARY.md**: What was changed and why
- 🚀 **IJEPA_QUICKSTART.md**: This quick reference (you are here)

## Performance Expectations

### Memory Usage
- **Baseline**: ~X GB
- **With JEPA**: ~1.3-1.4× baseline (30-40% increase)
  - Target encoder (frozen copy)
  - Predictor network
  - Additional embeddings

### Training Speed
- **Baseline**: ~Y iterations/sec
- **With JEPA**: ~0.7-0.8× baseline (20-30% slower)
  - Additional forward passes
  - EMA updates
  - Mask generation

### Convergence
- **JEPA loss**: Typically converges faster (first 5-10 epochs)
- **Photometric loss**: Similar to baseline
- **Total training time**: May need 1.2-1.5× more epochs for optimal results

## Typical Hyperparameter Values

| Parameter | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| `loss_weight` | 0.3 | 1.0 | 2.0 |
| `enc_mask_scale` | [0.90, 1.0] | [0.85, 1.0] | [0.80, 0.95] |
| `pred_mask_scale` | [0.10, 0.15] | [0.15, 0.20] | [0.20, 0.30] |
| `num_pred_masks` | 2 | 4 | 6 |
| `predictor_depth` | 4 | 6 | 12 |
| `ema` range | [0.998, 1.0] | [0.996, 1.0] | [0.994, 0.999] |

**Conservative**: Prioritize depth quality, minimal JEPA influence
**Balanced**: Equal focus on depth and representations
**Aggressive**: Maximize representation learning, accept some depth quality trade-off

## Next Steps

1. ✅ Enable JEPA in config
2. ✅ Run training with `ijepa_trainer.py`
3. 📊 Monitor TensorBoard for loss curves
4. 🔬 Evaluate on KITTI/Make3D/Cityscapes
5. 🎛️ Tune hyperparameters based on results
6. 📈 Compare with baseline (non-JEPA) training

## Additional Resources

- **JEPA Paper**: [A Path Towards Autonomous Machine Intelligence (LeCun)](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- **I-JEPA**: [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2304.02643)
- **Monodepth2**: [Digging Into Self-Supervised Monocular Depth Estimation](https://arxiv.org/abs/1806.01260)

## Support

Questions? Check:
1. This quick reference
2. `IJEPA_TRAINING_README.md` for details
3. Code comments in `src/train/ijepa_trainer.py`
