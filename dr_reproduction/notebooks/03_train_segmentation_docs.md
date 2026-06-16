# Notebook 3 — Segmentation Model Training: Code Documentation

**Notebook:** `03_train_segmentation.ipynb`  
**Paper:** *A Comprehensive Deep-Learning Framework Integrating Lesion Segmentation and Stage
Classification for Enhanced Diabetic Retinopathy Diagnosis* — Incir & Bozkurt, 2026

---

## Overview

This notebook trains the **Improved U-Net** segmentation model for each of the four lesion
types (EX, HE, MA, SE) using:

1. **Harmony Search Algorithm (HSA)** — finds optimal learning rate and dropout rate
2. **Hybrid BCE + Dice loss** — handles the severe class imbalance in lesion masks
3. **Multi-scale output fusion** — averages predictions at three decoder resolutions

### Where this fits in the pipeline

```
Notebook 1 (Segmentation preprocessing)    <- DONE
Notebook 2 (Classification preprocessing)  <- DONE
Notebook 3 (U-Net training + HSA)          <- THIS NOTEBOOK
Notebook 4 (Lesion overlay + ViT-CBAM)
Notebook 5 (Evaluation)
```

### One lesion per Colab session

Training one lesion requires 6–12 hours on a T4 GPU. Run the notebook four times,
changing `CURRENT_LESION` in Cell 2 before each run:

```
EX -> HE -> MA -> SE
```

HSA results and model checkpoints are saved to Drive after each lesion, so you can
always resume from where you left off if the session disconnects.

---

## Cell 1 — Mount Drive & Imports

```python
from google.colab import drive
drive.mount('/content/drive')
import os, gc, json, time, random
import numpy as np, matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
```

### What it does

Mounts Google Drive and imports all required libraries.  
`gc` (garbage collector) is used to release GPU memory between HSA evaluations.  
`time` is used to report how long each HSA evaluation takes.

### GPU check

The cell prints the available GPU. If it prints `GPU found: False`, training on this
notebook will be extremely slow. Go to **Runtime > Change runtime type > T4 GPU**.

---

## Cell 2 — Configuration

```python
CURRENT_LESION = 'EX'   # CHANGE THIS between runs

BATCH_SIZE  = 16        # lower to 8 if you see OOM errors
MAX_EPOCHS  = 500
ES_PATIENCE = 50        # EarlyStopping patience
RLROP_WAIT  = 20        # ReduceLROnPlateau patience

HMS         = 20        # HSA Harmony Memory Size
HSA_ITER    = 20        # HSA improvisation iterations
HMCR        = 0.95      # Harmony Memory Considering Rate
PAR         = 0.35      # Pitch Adjustment Rate
HSA_WARMUP  = 5         # Epochs per HSA evaluation
HSA_SUBSET  = 500       # Training patches per HSA evaluation

LR_LOG_MIN, LR_LOG_MAX = -6.0, -4.0   # log10([1e-6, 1e-4])
DR_MIN, DR_MAX         =  0.20, 0.45
```

### Key configuration choices

**`HSA_SUBSET = 500`**: Rather than running 40 full-scale training evaluations during
HSA, each evaluation uses only 500 randomly selected training patches. This reduces
HSA time from days to 60–90 minutes while still providing a reliable signal for which
hyperparameters are better.

**`HSA_WARMUP = 5`**: Five epochs is enough to distinguish good hyperparameters
(like `lr=1e-4`) from poor ones (like `lr=1e-6`). The relative ranking between
configurations is consistent even with short warm-ups.

**`BATCH_SIZE = 16`**: Larger than the batch size of 8 mentioned in some U-Net papers,
but appropriate here because the T4 GPU has 16 GB VRAM and the model is ~30M parameters.
If you see CUDA OOM errors, reduce to 8.

---

## Cell 3 — Load Preprocessed Patch Data

```python
def load_lesion_data(lesion):
    def _load(split):
        d    = np.load(os.path.join(data_dir, f'{split}.npz'))
        imgs = d['images'].astype(np.float32) / 255.0      # [0,255] -> [0,1]
        msks = (d['masks'] > 0).astype(np.float32)         # {0,255} -> {0,1}
        msks = msks[..., np.newaxis]                        # (N,H,W) -> (N,H,W,1)
        return imgs, msks
    ...
    perm = np.random.permutation(len(X_tr))
    X_tr, y_tr = X_tr[perm], y_tr[perm]
```

### Normalization

| Array | Source dtype | Source range | Target dtype | Target range |
|---|---|---|---|---|
| `imgs` | uint8 | [0, 255] | float32 | [0.0, 1.0] |
| `msks` | uint8 | {0, 255} | float32 | {0.0, 1.0} |

**Why `> 0` and not `> 127`?** IDRiD stores lesion pixels with value **1**, not 255.
The threshold `> 0` correctly captures these pixels. Using `> 127` would erase all
lesion annotations and produce empty masks. This was the key bug fixed during
Notebook 1 development.

### Channel dimension

Masks are stored in `.npz` without a channel dimension: shape `(N, 224, 224)`.
We add it with `[..., np.newaxis]` to get `(N, 224, 224, 1)`, which matches what
Keras expects for binary segmentation output.

### Shuffle

Patches in the `.npz` files are ordered by source image (patches from image 1,
then image 2, etc.). Without shuffling, early training batches would all come from
the same images, making the gradient noisy and biased. `np.random.permutation` gives
a random order for each training run.

### Memory requirements per lesion

| Lesion | Train patches | RAM (images + masks) |
|---|---|---|
| EX | 5276 | ~800 MB + ~265 MB ≈ 1.1 GB |
| HE | 4968 | ~750 MB |
| MA | 5616 | ~850 MB |
| SE | 1224 | ~185 MB |

All lesions fit comfortably in Colab's 12 GB RAM.

---

## Cell 4 — Improved U-Net Architecture

### Residual block

```python
def residual_block(x, filters, dropout_rate):
    skip = Conv2D(filters, 1, padding='same')(x)    # 1x1 to match channels
    skip = BatchNormalization()(skip)

    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Add()([x, skip])
    return ReLU()(x)
```

**Why residual connections?** In deep networks, gradients can vanish before reaching
the early layers during backpropagation. The skip connection provides a direct gradient
path from the loss to earlier layers, making 5-level U-Nets trainable.

**Why 1×1 conv on the skip path?** The skip connection must have the same number of
channels as the main path. If the input `x` has 3 channels (RGB) and the block outputs
64 channels, a 3→64 1×1 convolution on the skip path handles this mismatch.

**Why `use_bias=False` with BatchNormalization?** BatchNormalization already adds a
learnable bias shift (`beta` parameter). Using a separate conv bias alongside BN wastes
parameters without benefit.

### U-Net architecture

```
Input  224x224x3
  |
  e1 = residual_block(64)    224x224x64   <- skip connection to d1
  pool  -> 112x112
  e2 = residual_block(128)   112x112x128  <- skip connection to d2
  pool  -> 56x56
  e3 = residual_block(256)   56x56x256    <- skip connection to d3
  pool  -> 28x28
  e4 = residual_block(512)   28x28x512    <- skip connection to d4
  pool  -> 14x14
  b  = residual_block(1024)  14x14x1024   <- bottleneck
  |
  d4 = TransposeConv(512) + Concat(e4) + residual_block  28x28
  d3 = TransposeConv(256) + Concat(e3) + residual_block  56x56
  d2 = TransposeConv(128) + Concat(e2) + residual_block  112x112
  d1 = TransposeConv(64)  + Concat(e1) + residual_block  224x224
  |
  out_1x = Conv(1, sigmoid)(d1)  224x224  (full resolution)
  out_2x = Conv(1, sigmoid)(d2)  112x112  (half resolution)  -> Upsample 2x
  out_4x = Conv(1, sigmoid)(d3)  56x56    (quarter resolution) -> Upsample 4x
  |
  output = Average([out_1x, out_2x, out_4x])   224x224x1
```

**Total parameters:** approximately 31 million (varies slightly with dropout placement).

### Multi-scale output fusion

The multi-scale fusion (paper Section 3.2) generates predictions at three decoder
levels and averages them:

- `out_1x` — full resolution decoder output (finest detail)
- `out_2x` — half-resolution prediction upsampled with bilinear interpolation
- `out_4x` — quarter-resolution prediction upsampled

**Why average three predictions?**

The full-resolution output sees fine detail but may miss global context. The
half/quarter outputs have a wider receptive field and capture the coarse shape of
lesions better. Averaging combines the strengths of all three scales without
requiring additional training supervision signals.

**Why bilinear interpolation for upsampling?** Bilinear interpolation avoids the
checkerboard artefacts that transposed convolutions sometimes introduce in output
predictions.

---

## Cell 5 — Loss Functions

### Dice coefficient

```python
def dice_coef(y_true, y_pred, smooth=1e-6):
    yt = K.flatten(tf.cast(y_true, tf.float32))
    yp = K.flatten(y_pred)
    inter = K.sum(yt * yp)
    return (2 * inter + smooth) / (K.sum(yt) + K.sum(yp) + smooth)
```

The Dice coefficient measures the overlap between the predicted mask and the ground
truth. A value of 1.0 means perfect overlap; 0.0 means no overlap.

**Why `smooth=1e-6`?** If both the prediction and ground truth are all zeros (no
lesion in the patch), both numerator and denominator would be 0/0. The smoothing
constant avoids this while having negligible effect when lesions are present.

**Why flatten?** We compute a single Dice score across all pixels in all images in
the batch. Flattening treats the whole batch as one prediction tensor, which is the
standard approach for binary segmentation.

### Hybrid loss

```python
def hybrid_loss(y_true, y_pred):
    bce = K.mean(K.binary_crossentropy(y_true, y_pred))  # per-pixel BCE
    return bce + dice_loss(y_true, y_pred)                # + Dice
```

**Why not BCE alone?** BCE is unaware of class imbalance. If 97% of pixels are
background, the model can achieve low BCE by predicting all-zeros. Dice loss forces
the model to pay attention to the positive (lesion) pixels.

**Why not Dice alone?** Dice loss is unstable when lesions are very small (as in MA
patches). BCE provides a stable pixel-level gradient that complements the global Dice
gradient.

**Why add (not weight)?** Equal weighting (weight=1 each) works well empirically for
binary medical image segmentation and avoids the need to tune a weighting factor.

---

## Cell 6 — Harmony Search Algorithm (HSA)

### Algorithm overview

The HSA (Geem et al., 2001) is a metaheuristic inspired by musical improvisation:
musicians recall and adjust remembered harmonies to find better ones. Here, each
"harmony" is a (learning_rate, dropout_rate) pair, and "better" means higher
validation Dice after 5 warm-up epochs.

```
Phase 1 — Initialise Harmony Memory:
  For i = 1 to HMS=20:
    Generate random (lr, dropout) from search space
    Train model for HSA_WARMUP=5 epochs on 500-patch subset
    Record val Dice as the fitness of this harmony
  Sort harmonies by val Dice (best first)

Phase 2 — Improvisation:
  For iteration = 1 to HSA_ITER=20:
    Generate new_harmony using:
      - HMCR=0.95 probability: pick a value from memory
        - PAR=0.35 probability: apply pitch adjustment (small perturbation)
      - 1-HMCR=0.05 probability: generate randomly from search space
    Train model for HSA_WARMUP epochs with new_harmony
    If new_harmony val Dice > worst in memory:
      Replace worst harmony with new_harmony
      Re-sort memory
```

Total evaluations: HMS + HSA_ITER = 20 + 20 = **40 model trainings**.

### LR in log space

```python
LR_LOG_MIN, LR_LOG_MAX = -6.0, -4.0   # represents [1e-6, 1e-4]

# Random initialisation
lr = 10 ** random.uniform(LR_LOG_MIN, LR_LOG_MAX)

# Pitch adjustment (perturbation)
new_log += random.uniform(-bw, bw)   # bw = 10% of log range = 0.2
```

Learning rates span several orders of magnitude. A linear search space would spend
most iterations near the upper end (1e-4) and almost never explore near 1e-6.
Working in log10 space ensures all parts of the range are equally reachable.

### Pitch adjustment bandwidth

| Parameter | Search range | Bandwidth (10%) |
|---|---|---|
| LR (log scale) | [-6, -4] (width=2) | ±0.2 log units ≈ ×1.6 or ÷1.6 |
| Dropout (linear) | [0.20, 0.45] (width=0.25) | ±0.025 |

### GPU memory management

```python
def _eval_harmony(lr, dropout_rate, ...):
    tf.keras.backend.clear_session()   # reset Keras graph
    model = build_improved_unet(...)
    ...
    del model, history
    gc.collect()
    tf.keras.backend.clear_session()
```

Building and deleting 40 models during HSA would accumulate GPU memory without
explicit clearing. `tf.keras.backend.clear_session()` resets the Keras computational
graph and releases memory. This is called both before and after each evaluation.

---

## Cell 7 — Run HSA & Save Results

```python
hsa_json = os.path.join(MODEL_DIR, 'hsa_results.json')

if CURRENT_LESION in hsa_results:
    # Use previously saved result (skip re-running HSA)
    best_lr, best_dropout = r['lr'], r['dropout']
else:
    best_lr, best_dropout, hsa_vd = run_hsa(...)
    # Save to JSON so subsequent cells can use the result
    json.dump(hsa_results, f)
```

### Why save to JSON?

Colab sessions are not permanent. If the runtime disconnects or restarts after HSA
but before full training completes, the `hsa_results.json` file on Drive preserves
the search result. Re-running Cell 7 detects the existing file and skips the 60–90
minute HSA search automatically.

### Typical HSA output

```
Phase 1 -- Initialising harmony memory...
  [01/20] lr=2.34e-05  dr=0.317  dice=0.2341  (87s)
  [02/20] lr=8.71e-06  dr=0.402  dice=0.1823  (85s)
  ...
  Best init: lr=2.34e-05  dr=0.317  dice=0.2341

Phase 2 -- Improvisation (20 iterations)...
  iter 01/20  lr=2.11e-05  dr=0.298  dice=0.2412  best=0.2412  (87s) <-- IMPROVED
  ...
HSA result [EX]:
  Best LR      = 2.11e-05
  Best Dropout = 0.298
  Best ValDice = 0.2412
```

The HSA val Dice values during warm-up are lower than the final training val Dice
because the model only sees 500 patches for 5 epochs. The warm-up serves as a
relative ranking signal, not an absolute performance measure.

---

## Cell 8 — Full Training

```python
cbs = [
    ModelCheckpoint(monitor='val_dice_coef', save_best_only=True),
    EarlyStopping(patience=ES_PATIENCE=50, restore_best_weights=True),
    ReduceLROnPlateau(patience=RLROP_WAIT=20, factor=0.5, min_lr=1e-7)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=MAX_EPOCHS,
    ...
)
```

### Callbacks explained

**ModelCheckpoint** (`save_best_only=True`): Writes the model to Drive only when
validation Dice improves. This means the file on Drive always contains the best-ever
weights, not the most recent.

**EarlyStopping** (`patience=50`): If validation Dice does not improve for 50
consecutive epochs, training stops. `restore_best_weights=True` reloads the weights
from the epoch where the best Dice was achieved, not the final epoch. This ensures
the final model reflects peak performance.

**ReduceLROnPlateau** (`patience=20, factor=0.5`): If validation Dice does not
improve for 20 epochs, the learning rate is halved. This helps escape local plateaus.
`min_lr=1e-7` prevents the LR from dropping below a useful value.

### Resume after disconnect

```python
if os.path.exists(model_path):
    model = tf.keras.models.load_model(model_path, ...)
else:
    model = build_improved_unet(INPUT_SHAPE, best_dropout)
    model.compile(...)
```

If the Colab session disconnects mid-training, the last checkpoint saved by
`ModelCheckpoint` is on Drive. Re-running this cell loads that checkpoint and
continues training from it. History files are also merged so training curves remain
continuous across sessions.

### Expected training behaviour

| Phase | Epochs | Behaviour |
|---|---|---|
| Early | 1–30 | Loss drops rapidly; val Dice rises from 0.05 to 0.40+ |
| Mid | 30–100 | Slower improvement; val Dice plateaus around 0.6–0.8 |
| Late | 100+ | Minor gains; LR reduced by ReduceLROnPlateau |
| Stopping | Varies | EarlyStopping triggers 50 epochs after best val Dice |

### Expected training time per lesion

| Lesion | Train patches | Steps/epoch | ~Time/epoch | Early stop at ~ep | Total |
|---|---|---|---|---|---|
| EX | 5276 | 330 | 3.5 min | 150 | ~8–9 h |
| HE | 4968 | 311 | 3.3 min | 150 | ~8 h |
| MA | 5616 | 351 | 3.7 min | 150 | ~9 h |
| SE | 1224 | 77 | 0.8 min | 150 | ~2 h |

Times assume a T4 GPU and batch_size=16. SE is much faster due to fewer patches.

---

## Cell 9 — Evaluate & Plot

```python
test_loss, test_dice = best_model.evaluate(X_test, y_test, ...)
targets = {'EX': 0.8800, 'HE': 0.8686, 'MA': 0.7349, 'SE': 0.8927}
```

### Paper targets (Table 2)

| Lesion | Paper Dice | Description |
|---|---|---|
| EX | 88.00% | Hard exudates — bright yellow deposits |
| HE | 86.86% | Haemorrhages — dark blot lesions |
| MA | 73.49% | Microaneurysms — tiny dot lesions, hardest to segment |
| SE | 89.27% | Soft exudates — cotton wool spots |

MA has the lowest target because microaneurysms are small (often only a few pixels
in the 1024px patches before downscaling to 224px), making them genuinely difficult
to segment.

### What the plots show

**Loss curves:** Both train and val loss should decrease and converge. A large gap
between train and val loss indicates overfitting — increase dropout or reduce epochs.

**Dice curves:** Val Dice should rise steadily and plateau. The red dashed line shows
the paper target; reaching it confirms the implementation is correct.

**Prediction samples:** Visually compare Input / Ground truth / Prediction. You should
see the model highlight the correct lesion regions in red even for small lesions.

### Saving results

Curves and prediction samples are saved as `.png` files to `MODEL_DIR` on Drive:

```
models/segmentation/
├── hsa_results.json              <- HSA best hyperparameters (all lesions)
├── unet_EX_best.h5               <- Best EX model weights
├── unet_EX_history.json          <- Training history (loss + dice per epoch)
├── unet_EX_curves.png            <- Training curve plot
├── unet_EX_predictions.png       <- Sample test set predictions
├── unet_HE_best.h5
└── ...
```

These files are inputs to Notebook 4, which uses the trained models to generate
lesion overlays on APTOS images before ViT-CBAM training.

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Batch too large | Set `BATCH_SIZE = 8` in Cell 2 |
| `val_dice_coef` stuck at ~0.001 | Wrong mask threshold | Confirm masks have non-zero pixels: `y_train.max()` should be 1.0 |
| HSA all dice ≈ 0 | GPU not available | Check Cell 1 GPU printout; change runtime type |
| Training not improving past 0.5 | LR too low or high | Check HSA result; best LR is usually 1e-5 to 5e-5 |
| Model file not updating on Drive | Drive unmounted | Re-run Cell 1 to remount |
| Session disconnects during HSA | Free Colab 90-min limit | HSA result is NOT saved mid-run; use Colab Pro or reduce HMS/HSA_ITER |
| Session disconnects during training | Free Colab 90-min limit | ModelCheckpoint has saved best weights; re-run Cell 8 to resume |

---

## Next Step

After training all four lesion models, proceed to:

**`04_train_classification.ipynb`** — Run the trained U-Net models on APTOS images
to generate colour-coded lesion overlays, then train the **ViT-CBAM + GCN** 
classification model on the augmented dataset.
