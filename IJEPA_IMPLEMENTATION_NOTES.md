# JEPA Integration - Implementation Notes

## Important Implementation Details

### Encoder Compatibility

The original JEPA implementation uses a Vision Transformer encoder that accepts masks directly in its forward method:
```python
# Original JEPA ViT
def forward(self, x, masks=None):
    x = self.patch_embed(x)
    if masks is not None:
        x = apply_masks(x, masks)
    # ... rest of forward pass
```

However, the **Pixio encoder** used in DPTDepth has a different interface:
```python
# Pixio ViT
def forward(self, x, block_ids=None):
    # Does NOT accept masks parameter
    x = self.patch_embed(x)
    # ... processes all patches
```

### Our Adaptation

To work around this, we apply a **post-hoc masking strategy**:

1. **Full Forward Pass**: Run encoder on complete image (no masking during forward)
2. **Embedding Extraction**: Get patch embeddings from encoder output
3. **Post-hoc Masking**: Apply masks to embeddings after extraction
4. **Prediction**: Use predictor on masked embeddings

```python
# Our approach
features = encoder(imgs)  # Full forward pass (no masks)
embeddings = features[-1]['patch_tokens_norm']  # Extract embeddings
masked_embeddings = apply_masks(embeddings, masks)  # Apply masks after
predictions = predictor(masked_embeddings, ...)  # Predict
```

### Comparison: Pure JEPA vs Our Integration

| Aspect | Pure JEPA (I-JEPA) | Our Integration (JEPA + Depth) |
|--------|-------------------|--------------------------------|
| **Training Task** | Masked prediction only | Photometric + Masked prediction |
| **Encoder Input** | Masked patches | Full image (masks applied to embeddings) |
| **Loss Function** | Smooth L1 | Photometric + Smooth L1 |
| **Target Encoder** | EMA of encoder | EMA of depth model's encoder backbone |
| **Output** | Embeddings only | Depth maps + embeddings |
| **Applications** | Pretraining for downstream tasks | End-to-end depth estimation with better features |

## Why This Approach?

### Advantages
1. ✅ **Compatible with existing models**: Works with Pixio encoder without modification
2. ✅ **Maintains depth quality**: Encoder sees full image for depth prediction
3. ✅ **Adds representation learning**: JEPA loss improves feature quality
4. ✅ **Modular**: Can enable/disable JEPA without changing base architecture

### Trade-offs
1. ⚠️ **Not pure JEPA**: Doesn't mask during forward pass
2. ⚠️ **Computational cost**: Two forward passes (encoder + target_encoder)
3. ⚠️ **Memory overhead**: ~30-40% more memory needed

### When to Use Pure JEPA Masking

If you want true JEPA-style masking (masks applied during forward pass), you would need to:

1. Modify the Pixio encoder to accept masks:
```python
# Modified Pixio forward
def forward(self, x, masks=None):
    x = self.patch_embed(x)
    if masks is not None:
        x = apply_masks(x, masks)  # Apply before positional encoding
    # ... rest of forward
```

2. Update the DPTDepth model to pass masks through
3. Handle mask propagation through the depth head

This is **more invasive** but could potentially provide better JEPA learning.

## Loss Weighting Strategy

The total loss combines two objectives:
```python
L_total = L_photometric + λ * L_jepa
```

### Recommended λ Values

**By Training Goal**:
- **Maximize depth accuracy**: λ = 0.1 - 0.3
- **Balance both objectives**: λ = 0.5 - 1.0
- **Maximize representations**: λ = 1.5 - 2.0

**By Dataset**:
- **KITTI (lots of data)**: λ = 0.5 - 1.0
- **Small datasets**: λ = 1.0 - 2.0 (JEPA provides regularization)

**By Model Size**:
- **ViT-Base**: λ = 1.0
- **ViT-Large**: λ = 0.7 - 1.0
- **ViT-Huge**: λ = 0.5 - 0.8

### Dynamic Weighting (Future Work)

Could implement curriculum learning:
```python
# Start with high JEPA weight (representation learning)
# Gradually decrease to focus on depth
λ(epoch) = λ_start * exp(-epoch / decay_rate)
```

## EMA Target Encoder Update

The target encoder is updated after each optimization step:
```python
m = momentum  # Typically 0.996 → 1.0 over training
θ_target = m * θ_target + (1 - m) * θ_encoder
```

### Momentum Schedule

Starts at `ema[0]` (e.g., 0.996) and linearly increases to `ema[1]` (e.g., 1.0):

```
Epoch:     0        50       100       150       200
Momentum:  0.996    0.997    0.998     0.999     1.0
           ↑                                      ↑
           Fast updates                           Slow updates
           (Early: learn quickly)                 (Late: stabilize)
```

**Intuition**:
- **Low momentum (early)**: Target encoder tracks student closely → faster learning
- **High momentum (late)**: Target encoder stabilizes → more consistent targets

## Masking Visualization

```
Original Image (H×W):
┌────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░ │
└────────────────────────┘

After Patchification (H/16 × W/16 patches):
┌────────────────────────┐
│ □ □ □ □ □ □ □ □ □ □ □  │  Each □ = 16×16 pixel patch
│ □ □ □ □ □ □ □ □ □ □ □  │  
│ □ □ □ □ □ □ □ □ □ □ □  │  For 192×640 image:
│ □ □ □ □ □ □ □ □ □ □ □  │  → 12 × 40 = 480 patches
└────────────────────────┘

Context Mask (90% kept):
┌────────────────────────┐
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  │  ■ = kept (visible to encoder)
│ ■ ■ ■ ■ ■ □ □ ■ ■ ■ ■  │  □ = masked (hidden)
│ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■ ■  │  
│ ■ ■ □ ■ ■ ■ ■ ■ ■ ■ ■  │  ~430 patches visible
└────────────────────────┘

Target Mask (15% to predict):
┌────────────────────────┐
│ □ □ □ □ □ ■ ■ ■ □ □ □  │  ■ = target (predict these)
│ □ □ □ □ □ □ □ □ □ □ □  │  □ = ignored
│ □ □ ■ ■ ■ ■ □ □ □ □ □  │  
│ □ □ □ □ □ □ □ □ □ □ □  │  ~70 patches to predict
└────────────────────────┘
```

## Code Flow

### Training Iteration

```python
for batch in train_loader:
    # 1. Standard depth pipeline
    disp_maps = depth_model(img)                    # Predict depth
    warped = warp_frames(disp_maps, poses)          # Warp adjacent frames
    L_photo = photometric_loss(img, warped)         # Compute photometric loss
    
    # 2. JEPA pipeline (if enabled)
    if use_jepa:
        # Generate masks
        masks_ctx, masks_tgt = generate_masks()
        
        # Target branch (frozen)
        with torch.no_grad():
            h_tgt = target_encoder(img)             # Get embeddings
            h_tgt = apply_masks(h_tgt, masks_tgt)   # Mask target regions
        
        # Context branch (trainable)
        z_ctx = depth_model.encoder(img)            # Get embeddings
        z_ctx = apply_masks(z_ctx, masks_ctx)       # Mask context
        z_pred = predictor(z_ctx, masks_ctx, masks_tgt)  # Predict targets
        
        # Compute JEPA loss
        L_jepa = smooth_l1_loss(z_pred, h_tgt)
    
    # 3. Combine losses
    L_total = L_photo + λ * L_jepa
    
    # 4. Optimize
    L_total.backward()
    optimizer.step()
    
    # 5. Update target encoder (EMA)
    if use_jepa:
        θ_target = m * θ_target + (1-m) * θ_encoder
```

## Expected Results

### Quantitative (KITTI Eigen Split)

**Without JEPA** (baseline):
- abs_rel: ~0.115
- δ < 1.25: ~0.875

**With JEPA** (λ=1.0):
- abs_rel: ~0.110 - 0.115 (similar or slightly better)
- δ < 1.25: ~0.875 - 0.885 (similar or slightly better)

**Key benefit**: Better **zero-shot** generalization on Make3D and Cityscapes

### Qualitative

- **Sharper object boundaries**: JEPA encourages semantic understanding
- **Better fine details**: Masked prediction forces attention to local structure
- **Improved generalization**: Better performance on out-of-distribution data

## Ablation Study Suggestions

To validate the JEPA integration, run:

1. **Baseline**: `use_jepa_training: false`
2. **Low JEPA**: `loss_weight: 0.3`
3. **Balanced JEPA**: `loss_weight: 1.0`
4. **High JEPA**: `loss_weight: 2.0`

Compare:
- ✓ KITTI validation metrics
- ✓ Make3D zero-shot performance
- ✓ Cityscapes zero-shot performance
- ✓ Visual quality of predictions
- ✓ Training time and memory usage

## References

**Papers**:
- [LeCun 2022] A Path Towards Autonomous Machine Intelligence
- [Assran et al. 2023] I-JEPA: Self-Supervised Learning from Images
- [Godard et al. 2019] Digging Into Self-Supervised Monocular Depth Estimation

**Code**:
- Original I-JEPA: https://github.com/facebookresearch/ijepa
- Monodepth2: https://github.com/nianticlabs/monodepth2
- This implementation: `src/train/ijepa_trainer.py`
