# Notebook 2 — APTOS 2019 Classification Preprocessing: Code Documentation

**Notebook:** `02_preprocess_classification.ipynb`  
**Paper:** *A Comprehensive Deep-Learning Framework Integrating Lesion Segmentation and Stage Classification for Enhanced Diabetic Retinopathy Diagnosis* — Incir & Bozkurt, Int J Imaging Syst Tech, 2026

---

## Overview

This notebook converts the raw APTOS 2019 dataset (3662 retinal fundus images with 5-class DR
severity labels) into a class-balanced, training-ready folder structure for the ViT-CBAM
classification model used in Stage 2 of the paper.

### Where this fits in the pipeline

```
Notebook 1 (Segmentation preprocessing)    <- DONE
Notebook 2 (Classification preprocessing)  <- THIS NOTEBOOK
Notebook 3 (U-Net training + HSA)
Notebook 4 (Lesion overlay generation + ViT-CBAM training)
Notebook 5 (Evaluation)
```

The lesion overlay step — running the trained U-Net on APTOS images to annotate them with
lesion masks — happens in Notebook 4, AFTER the U-Net is trained. This notebook only prepares
the raw APTOS images (OD removal + augmentation + resize).

---

## Cell 1 — Mount Drive & Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q tqdm pandas
import os, random, zipfile
import numpy as np, cv2, pandas as pd, matplotlib.pyplot as plt
from tqdm import tqdm
```

### What it does
Mounts Google Drive and imports all required libraries. `pandas` is needed here (not in
Notebook 1) because APTOS provides CSV label files rather than organized folder structure.

---

## Cell 2 — Configuration

```python
APTOS_ZIP       = '...archive.zip'
APTOS_OUT       = '.../APTOS_classification'
FINAL_SIZE      = 224
AUGMENT_FACTORS = {0: 1, 1: 9, 2: 3, 3: 18, 4: 12}
CLASS_NAMES     = {0: 'No_DR', 1: 'Mild', 2: 'Moderate', 3: 'Severe', 4: 'Proliferative'}
```

### The augmentation factors explained

APTOS 2019 has a severe class imbalance. If you train a model on the raw counts, it learns to
predict "No DR" for most inputs simply because that is the most common class.

The paper addresses this with per-class augmentation that brings every class to approximately
the same size:

| Class | Raw count | × Factor | Approx. after |
|---|---|---|---|
| 0 No DR | 1805 | 1 | 1805 |
| 1 Mild | 370 | 9 | 3330 |
| 2 Moderate | 999 | 3 | 2997 |
| 3 Severe | 193 | 18 | 3474 |
| 4 Proliferative | 295 | 12 | 3540 |

The target is approximately 3500 images per class. Class 0 is already the largest at 1805 and
is not augmented further — it stays at 1805, making it slightly under-represented relative to
the others, which is acceptable since the model must not over-fit to the majority class.

---

## Cell 3 — Validate APTOS Zip & Load CSV Labels

```python
with zipfile.ZipFile(APTOS_ZIP, 'r') as zf:
    APTOS_NAMES = set(zf.namelist())
    with zf.open('train_1.csv') as f:
        train_df = pd.read_csv(f)
    ...

def find_prefix(names, sample_id):
    for n in names:
        if sample_id in n and n.endswith('.png'):
            return n[:n.rfind('/') + 1]
    return ''
```

### What it does

1. Reads the zip's table of contents into `APTOS_NAMES` (a Python set for O(1) lookups).
2. Opens and parses both CSV files **directly from within the zip** without extracting anything.
3. Auto-detects where the images are stored inside the zip using `find_prefix`.

### Why auto-detect the image prefix?

The APTOS archive from Kaggle often uses a nested folder structure:
`train_images/train_images/000abc.png`

Rather than hard-coding this path, `find_prefix` takes a sample `id_code` from the CSV,
searches for it among all zip entries, and returns the folder path. This makes the code work
regardless of whether the zip has one or two levels of nesting.

### How APTOS CSVs are structured

| Column | Description |
|---|---|
| `id_code` | Image filename without `.png` extension |
| `diagnosis` | DR grade: 0 (No DR), 1 (Mild), 2 (Moderate), 3 (Severe), 4 (Proliferative) |

---

## Cell 4 — OD Removal & Augmentation Functions

### `load_from_zip(zf, zip_path, grayscale=False)`
```python
with zf.open(zip_path) as f:
    buf = np.frombuffer(f.read(), np.uint8)
    return cv2.imdecode(buf, flag)
```
Same pattern as Notebook 1 — reads image bytes directly from the open zip handle and decodes
with OpenCV. Returns `None` if the path is not found (graceful handling of missing files).

---

### `detect_od_mask(img_bgr)`
```python
green   = img_bgr[:, :, 1].astype(np.float32)
blurred = cv2.GaussianBlur(green, (31, 31), 0)
_, _, _, max_loc = cv2.minMaxLoc(blurred)
radius = int(min(h, w) * 0.07)
cv2.circle(mask, max_loc, radius, 255, -1)
```

**Why this approach for APTOS (not ground-truth masks)?**

IDRiD provides GT OD masks for all 81 images, so Notebook 1 uses them directly.
APTOS has 3662 images and no OD annotations. We must detect the OD automatically.

The optic disc is **the brightest compact region in a fundus image's green channel**. Hard
exudates (EX) are also bright, but they are smaller and scattered. A 31×31 Gaussian blur
suppresses small bright spots while keeping the OD peak dominant. `cv2.minMaxLoc` then
finds the single peak location.

A circle of radius = 7% of the image's shorter side is drawn around that location. This is
a rough but effective approximation — OD size varies slightly between patients but 7% covers
most cases.

---

### `remove_od_aptos(img_bgr)`
```python
mask = detect_od_mask(img_bgr)
return cv2.inpaint(img_bgr, mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
```

Uses the same TELEA inpainting algorithm as Notebook 1 for consistency. The mask (binary
circle) tells inpaint which pixels to reconstruct from their neighbours.

---

### `augment_single(img_bgr, aug_idx)`
```python
s = aug_idx % 4
if s == 0:  return _rot(img_bgr, ANGLES[aug_idx % 8])   # rotation
elif s == 1: return cv2.flip(img_bgr, 1)                  # horizontal flip
elif s == 2: return cv2.flip(_rot(...), 1)                 # rotation + flip
else:        return img + gaussian_noise                   # noise
```

**Design choices:**
- **Deterministic by index** — `augment_single(img, 0)` always produces the same result.
  This is important for reproducibility: the same augmented dataset is created every time.
- **4 strategies cycling** — Using `aug_idx % 4` ensures that even for 18 augmentations
  (class 3), we cycle through all strategy types rather than applying the same one 18 times.
- **No random rotation angle** — angles are drawn from a fixed list `[15, 45, 90, ...]` by
  index, making the result fully deterministic.

**Why rotate and flip for classification (not just segmentation)?**

For DR classification, the model must recognise lesions at any position and orientation.
A fundus image is inherently symmetric — flipping horizontally gives a plausible image from
the other eye. Rotating by multiples of 45° generates diverse views without unrealistic poses.

---

## Cell 5 — Preprocess Train & Val Splits

```python
def preprocess_aptos_split(df, img_prefix, split, augment):
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(split_dir, str(cls)), exist_ok=True)

    with zipfile.ZipFile(APTOS_ZIP, 'r') as zf:
        for _, row in tqdm(df.iterrows(), ...):
            img = load_from_zip(zf, zip_path)
            img = remove_od_aptos(img)
            img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(f'{out_dir}/{img_id}_base.png', img)

            if augment:
                for i in range(AUGMENT_FACTORS[label] - 1):
                    aug = augment_single(img, i)
                    cv2.imwrite(f'{out_dir}/{img_id}_a{i:02d}.png', aug)
```

### What it does
For each image in the CSV:
1. Loads from zip → removes OD → resizes to 224×224 → saves as `_base.png`
2. For training only: generates `AUGMENT_FACTORS[label] - 1` additional augmented versions

The `-1` in `AUGMENT_FACTORS[label] - 1` accounts for the base image that is already saved.
So for class 1 (factor=9): 1 base + 8 augmented = 9 total images per original.

### Why save PNG files to Drive for classification?

Segmentation (Notebook 1) saved compressed `.npz` arrays because 28 patches × 81 images =
~2000 small arrays that load faster as a batch. For classification, each augmented image is
an independent training sample — the standard Keras `flow_from_directory` and PyTorch
`ImageFolder` APIs expect **PNG files organized in class folders**, which is exactly the
structure we create here.

### Output folder structure

```
APTOS_classification/
├── train/
│   ├── 0/   1805 images  (No DR, no extra augmentation)
│   ├── 1/   3330 images  (Mild × 9)
│   ├── 2/   2997 images  (Moderate × 3)
│   ├── 3/   3474 images  (Severe × 18)
│   └── 4/   3540 images  (Proliferative × 12)
└── val/
    ├── 0/   (original val images for No DR)
    ├── 1/   ...
    └── 4/
```

### Why no augmentation for val?

Validation is used to measure model performance on unseen data. Augmenting validation would
make the score inconsistent across runs (different random transformations) and would not
reflect how the model behaves on real clinical images.

---

## Cell 6 — Verify Outputs & Plot Class Distribution

```python
orig_counts = [int((train_df['diagnosis'] == c).sum()) for c in range(5)]
aug_counts  = [len(os.listdir(...train/str(c)...)) for c in range(5)]
```

### What to check

**Before chart**: Should show the steep imbalance — class 0 bar towering over classes 1 and 3.  
**After chart**: Should show bars of roughly equal height (~1800–3500). Small differences are
expected because we can't augment class 0 without making it larger than the others.

If the after chart still shows severe imbalance, check:
- `AUGMENT_FACTORS` values in Cell 2
- Whether Cell 5 ran completely (progress bar should reach 100%)
- Whether Drive has enough space for ~15k PNG files

---

## Common Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `KeyError: 'train_1.csv'` | CSV name differs in your zip | Print `zf.namelist()` to see actual names |
| `find_prefix` returns empty string | `id_code` format doesn't match filenames | Check `train_df['id_code'].iloc[0]` vs actual zip paths |
| Skipped images reported | Image file missing from zip | Normal for a small number; if >10% skip, check zip integrity |
| Drive write errors | Drive not mounted or full | Re-run Cell 1; check Drive quota |
| Class folder has wrong count | Cell 5 ran twice | Delete `APTOS_classification/train/` folder and rerun |

---

## Next Step

After this notebook completes, proceed to:

**`03_train_segmentation.ipynb`** — Build the Improved U-Net architecture, run Harmony Search
Algorithm (HSA) to find optimal learning rate and dropout, then train one model per lesion
(EX, HE, MA, SE) for up to 500 epochs using the `.npz` patches from Notebook 1.
