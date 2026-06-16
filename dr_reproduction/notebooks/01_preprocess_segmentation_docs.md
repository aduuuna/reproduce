# Notebook 1 — IDRiD Segmentation Preprocessing: Code Documentation

**Notebook:** `01_preprocess_segmentation.ipynb`  
**Paper:** *A Comprehensive Deep-Learning Framework Integrating Lesion Segmentation and Stage Classification for Enhanced Diabetic Retinopathy Diagnosis* — Incir & Bozkurt, Int J Imaging Syst Tech, 2026  

---

## Overview

This notebook converts the raw IDRiD segmentation dataset (81 high-resolution fundus images) into
training-ready patch arrays for four lesion types. Each lesion type (EX, HE, MA, SE) gets its
own set of `.npz` files: one for training, one for validation, and one for testing.

The pipeline follows Section 3.3 of the paper exactly, with one intentional improvement: the paper
mentions optic disc removal but does not specify the method. Rather than estimating the disc with
Hough circles, we use the **ground-truth OD masks** that IDRiD provides (`5. Optic Disc/`), which
gives a precise boundary with no detection errors.

---

## Cell 1 — Mount Drive & Install Dependencies

```python
from google.colab import drive
drive.mount('/content/drive')
!pip install -q tqdm
import os, random, zipfile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
print('Libraries loaded.')
```

### What it does
Mounts your Google Drive at `/content/drive` so the notebook can read your dataset zip and write
output files. It then installs `tqdm` (progress bars) and imports all libraries needed in later
cells.

### Libraries used
| Library | Purpose |
|---|---|
| `cv2` (OpenCV) | Reading TIFF/JPEG images, resizing, inpainting, rotation |
| `numpy` | Array stacking, numerical operations, saving `.npz` files |
| `matplotlib` | Final visualisation of sample patches |
| `tqdm` | Progress bars during the processing loop |
| `zipfile` | Extracting the dataset without needing unzip on the command line |

### Why mount first?
Colab notebooks run on a virtual machine that starts fresh every session. Google Drive is not
automatically available — `drive.mount()` connects the two so files in `MyDrive/` become
accessible via the path `/content/drive/MyDrive/`.

---

## Cell 2 — Configuration

```python
BASE_PATH     = '/content/drive/MyDrive/DR_PROJECT/datasets'
OUTPUT_BASE   = '/content/drive/MyDrive/DR_PROJECT/preprocessed'
IDRID_SEG_ZIP = os.path.join(BASE_PATH, 'A. Segmentation.zip')

EXTRACT_DIR = '/content/idrid_seg'
...
RESIZE_W, RESIZE_H = 4096, 2560
PATCH_SIZE  = 1024
STRIDE      = 512
FINAL_SIZE  = 224
AUGMENT_FACTOR = 3
```

### What it does
Centralises all paths and numerical parameters so they only need to be changed in one place.

### Parameters and their paper source

| Parameter | Value | Paper reference |
|---|---|---|
| `RESIZE_W, RESIZE_H` | 4096 × 2560 | Section 3.3 — images are resized to this dimension before patching |
| `PATCH_SIZE` | 1024 | Section 3.3 — sliding window size |
| `STRIDE` | 512 | Section 3.3 — "50% overlap" (stride = patch_size / 2) |
| `FINAL_SIZE` | 224 | Section 3.3 — patches are downscaled to 224 × 224 for the U-Net input |
| `AUGMENT_FACTOR` | 3 | Section 3.3 — three augmentation variants are created per training patch |

### Why `EXTRACT_DIR = '/content/idrid_seg'`?
Colab's local SSD (`/content/`) is roughly 10x faster than Google Drive for random file access.
Extracting the zip once to local storage means every subsequent read during the processing loop
hits the fast local disk rather than the slower Drive network mount.

### Lesion folder map
IDRiD stores masks in numbered subfolders. The `LESION_FOLDER_MAP` and `LESION_SUFFIX_MAP`
dictionaries translate between our short codes (EX, HE, MA, SE) and the actual file/folder names
used inside the zip.

---

## Cell 3 — Extract Zip & Locate Dataset

```python
with zipfile.ZipFile(IDRID_SEG_ZIP, 'r') as zf:
    zf.extractall(EXTRACT_DIR)

SEG_BASE = None
for root, dirs, files in os.walk(EXTRACT_DIR):
    if '1. Original Images' in dirs and '2. All Segmentation Groundtruths' in dirs:
        SEG_BASE = root
        break
```

### What it does
1. Extracts `A. Segmentation.zip` to `/content/idrid_seg/` on the local SSD.
2. Walks the resulting directory tree to find the dataset root automatically — this handles
   situations where the zip contains an extra top-level folder (e.g., `A. Segmentation/...`).
3. Builds absolute paths to all image and mask subdirectories.

### Why auto-detect the root?
Zip files are sometimes created with or without a top-level folder depending on how they were
zipped. Rather than hard-coding the path and breaking if the zip structure is slightly different,
the `os.walk` loop searches for the two known landmark folders and infers the root from their
location.

### OD directory paths
```python
TRAIN_OD_DIR = os.path.join(TRAIN_MASK_DIR, '5. Optic Disc')
TEST_OD_DIR  = os.path.join(TEST_MASK_DIR,  '5. Optic Disc')
```
IDRiD provides ground-truth optic disc masks for **all 81 images** in the subfolder
`5. Optic Disc`. These are used later in `remove_optic_disc()` to precisely erase the disc.

---

## Cell 4 — Build Image Index & 80/10/10 Split

```python
for fname in sorted(os.listdir(TRAIN_IMG_DIR)):
    entry = _parse_entry(fname, TRAIN_IMG_DIR, TRAIN_OD_DIR, TRAIN_MASK_DIR)
    if entry: all_images.append(entry)
# ... same for TEST_IMG_DIR

random.seed(RANDOM_SEED)
indices = list(range(len(all_images)))
random.shuffle(indices)

n_train = int(0.8 * n)
n_val   = int(0.1 * n)
# first 80% -> train, next 10% -> val, remainder -> test
```

### What it does
1. Scans both the official training folder (54 images) and testing folder (27 images) to build
   one unified list of all 81 images.
2. Shuffles the list with a fixed seed (`RANDOM_SEED = 42`) for reproducibility.
3. Assigns each image to train / val / test based on its rank in the shuffled list.

### Why re-split instead of using IDRiD's official train/test split?
The paper combines all 81 images and applies an 80/10/10 ratio (Section 3.3, dataset
description). Using a fresh split gives the model a validation set that IDRiD's original
split does not provide.

### Why split at the image level — not the patch level?
Each high-resolution image produces up to 28 patches. If we split patches randomly, the same
fundus image can appear in both training and test sets. The model would then be evaluated on
regions it has essentially "seen" during training, artificially inflating test accuracy.

Splitting by image first ensures that every patch from an image ends up in exactly one split.

### The `_parse_entry` helper
Extracts the numeric image ID from the filename (`IDRiD_01.jpg` → `01`) and builds all relevant
file paths for that image (RGB image, OD mask, mask root directory) into a dictionary. This
dictionary is passed around in later cells without needing to reconstruct paths again.

---

## Cell 5 — Preprocessing & Augmentation Functions

This cell defines six helper functions. None of them run immediately — they are called inside
the main loop in Cell 6.

### `load_rgb(path)`
```python
img = cv2.imread(path)
return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None
```
OpenCV loads images as BGR. All further processing and saving expects RGB, so we convert
immediately after reading.

---

### `remove_optic_disc(image, od_mask)`
```python
binary  = (od_mask > 127).astype(np.uint8) * 255
kernel  = np.ones((20, 20), np.uint8)
dilated = cv2.dilate(binary, kernel, iterations=2)
return cv2.inpaint(image, dilated, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
```

**Why remove the optic disc?**  
The optic disc is a bright, well-defined circular structure that is not a lesion. Its brightness
and texture closely resemble hard exudates (EX), which causes the segmentation model to produce
false-positive predictions in the disc region. Removing it before training prevents the model
from learning this spurious association.

**How it works:**
1. Threshold the OD mask to a clean binary image.
2. Dilate the mask by 20 pixels to cover the full disc and a small safety margin around the
   boundary (where the disc blends into the retina).
3. Use TELEA inpainting — a fast, content-aware fill algorithm that reconstructs the masked
   region from surrounding pixels, producing a natural-looking retinal background.

---

### `extract_patches(image, mask)`
```python
for y in range(0, h - PATCH_SIZE + 1, STRIDE):
    for x in range(0, w - PATCH_SIZE + 1, STRIDE):
        ip = image[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        mp = mask[y:y+PATCH_SIZE,  x:x+PATCH_SIZE]
        if mp.max() > 0:
            patches.append((ip, mp))
```

Slides a 1024 × 1024 window across the resized image with a step of 512 pixels in both
directions (50% overlap). Only patches where the corresponding mask has at least one non-zero
pixel are kept. This filters out the large background-only regions that make up most of a fundus
image, which would cause severe class imbalance (far too many "no lesion" patches).

---

### `to_final(img, msk)`
```python
img_r = cv2.resize(img, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_LINEAR)
msk_r = cv2.resize(msk, (FINAL_SIZE, FINAL_SIZE), interpolation=cv2.INTER_NEAREST)
return img_r, (msk_r > 127).astype(np.uint8) * 255
```

Downscales a 1024 × 1024 patch to 224 × 224.

Two different interpolation methods are used deliberately:
- **INTER_LINEAR** for the image: bilinear interpolation produces smooth, visually correct
  colours when downsampling.
- **INTER_NEAREST** for the mask: nearest-neighbour interpolation copies the value of the
  closest pixel without blending. This preserves the binary (0 or 255) nature of the mask.
  Using INTER_LINEAR on a mask would introduce grey pixels at boundaries (e.g., 127) that
  could corrupt the ground truth.

---

### `aug_rotate_noise(img, msk)` — Augmentation Variant 1
```python
angle = random.choice([15, 30, 45, 90, 135, 180, 270])
img   = _rot(img, angle)
noise = np.random.normal(0, 8, img.shape).astype(np.int16)
img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
```

Rotates by a random discrete angle, then adds Gaussian noise with σ = 8 (roughly 3% of the
0–255 intensity range). Rotation teaches the model that lesions can appear at any orientation.
Noise simulates variability between different retinal cameras and acquisition conditions.

---

### `aug_sharpen_flip(img, msk)` — Augmentation Variant 2
```python
blurred   = cv2.GaussianBlur(img, (0, 0), 3)
sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
return cv2.flip(sharpened, 1), cv2.flip(msk, 1)
```

Applies **unsharp masking**: blurs the image slightly, then subtracts the blurred version from
the original at a defined ratio. The result is a version with more pronounced edges — which is
useful for making subtle lesion boundaries more distinct for the model to learn.

Then flips both image and mask horizontally. Retinal images have no canonical orientation
(the patient's left or right fundus are both valid inputs), so a horizontal flip is a safe,
realistic augmentation.

---

### `aug_crop_rotate(img, msk)` — Augmentation Variant 3
```python
scale = random.uniform(0.80, 0.95)
nh, nw = int(h * scale), int(w * scale)
y0, x0 = random.randint(0, h - nh), random.randint(0, w - nw)
return img[y0:y0+nh, x0:x0+nw], msk[y0:y0+nh, x0:x0+nw]
```

Rotates by a random angle, then crops a random 80–95% sub-region. This forces the model to
handle partial lesion views and lesions near patch boundaries — increasing robustness to the
common case where a lesion is split across two adjacent patches.

Note: cropped patches are returned at variable sizes. `to_final()` in Cell 6 resizes them back
to 224 × 224 before saving.

---

## Cell 6 — Run Preprocessing

```python
def preprocess_lesion(lesion):
    ...
    for info in tqdm(all_images, ...):
        # 1. Load image, mask, OD mask
        # 2. Remove optic disc
        # 3. Resize to 4096x2560
        # 4. Extract 1024x1024 patches (mask-filtered)
        # 5. Resize patches to 224x224
        # 6. Augment training patches x3
        # 7. Accumulate into buf dict
    # Save train/val/test arrays to .npz files

for lesion in LESIONS:
    preprocess_lesion(lesion)
```

### What it does
This is the main processing loop. For each of the four lesion types (EX, HE, MA, SE) it:

1. Checks if output files already exist on Drive (skip if so — **idempotent design**).
2. Loops over all 81 images with a tqdm progress bar.
3. Applies the full pipeline from Cell 5 functions.
4. Stores patches in an in-memory buffer (`buf`) keyed by split name.
5. Writes three `.npz` files (train, val, test) to Drive for the current lesion.

### Idempotency
```python
if all(os.path.exists(os.path.join(out_dir, f'{s}.npz')) for s in ['train', 'val', 'test']):
    print(f'[{lesion}] Already preprocessed - skipping.')
    return
```
If Colab disconnects mid-run and you restart, the function skips any lesion whose three output
files already exist. You don't need to reprocess from scratch.

### Why augment at the 1024px scale, not at 224px?
Augmentation functions (rotation, crop) are applied to the raw 1024 × 1024 patch before
resizing to 224 × 224. This is intentional:
- Spatial distortions at 1024px scale are more numerically meaningful.
- The final resize smooths out any artefacts introduced by the rotation border fill.
- Applying rotation on a 224 × 224 image would produce coarser results due to fewer pixels.

### Output file format: `.npz`
NumPy's compressed archive format stores multiple named arrays in a single file. Loading
`train.npz` returns a dictionary-like object:
```python
data = np.load('EX/train.npz')
images = data['images']  # shape: (N, 224, 224, 3), dtype uint8
masks  = data['masks']   # shape: (N, 224, 224),    dtype uint8
```
This is faster to load during training than reading thousands of individual PNG files, and uses
significantly less Drive storage than uncompressed TIF files.

### Expected patch counts (approximate)
| Lesion | Why | Expected train patches |
|---|---|---|
| EX | Common, large lesions | ~1200–2000 |
| HE | Moderate coverage | ~800–1500 |
| MA | Tiny, sparse lesions | ~300–700 |
| SE | Only ~40/81 images annotated | ~400–900 |

Training patches are roughly 4× higher (×4 = 1 original + 3 augmentations).

---

## Cell 7 — Verify Outputs & Visualise Samples

```python
print(f"{'Lesion':<6} {'Train':>8} {'Val':>6} {'Test':>6}  {'Total':>7}")
for lesion in LESIONS:
    ...  # loads each .npz, prints shape[0]
```

### What it does
1. Prints a summary table of patch counts per lesion per split.
2. Loads sample patches from the training set for each lesion.
3. Plots three image/mask pairs per lesion in a grid (4 rows × 7 columns).
4. Saves the grid as `sample_patches.png` to Drive for documentation.

### How to read the visualisation
Each row corresponds to one lesion type (EX, HE, MA, SE). Column 0 shows the lesion name.
Columns 1–6 alternate between a retinal patch image and its corresponding binary mask (coloured
red). Regions where the mask is bright red indicate where the lesion was annotated.

### What to check before continuing to Notebook 2
- All four lesions have non-zero counts in all three splits.
- The ratio between train and val/test roughly matches 80/10/10 at the patch level.
- The visualised masks look plausible — red regions should correspond to visible lesion areas in
  the image.

If any lesion shows 0 patches in any split, check the folder names inside the extracted zip
against the `LESION_FOLDER_MAP` dictionary in Cell 2.

---

## Common Issues & Fixes

| Issue | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: A. Segmentation.zip` | Wrong Drive path | Update `BASE_PATH` in Cell 2 |
| `Dataset root not found in zip` | Zip has unexpected folder nesting | Print `EXTRACT_DIR` contents, adjust path in `SEG_BASE` detection |
| 0 patches for SE | Normal — SE only in ~40 images | Expected; no action needed |
| Colab RAM exceeded | Images too large to hold in memory | Reduce `RESIZE_W/H` or process one image at a time |
| Output `.npz` missing on Drive | Colab disconnected before write | Rerun from Cell 1; idempotency skips completed lesions |

---

## Next Step

After this notebook completes successfully, proceed to:

**`02_train_segmentation.ipynb`** — Build the Improved U-Net, run Harmony Search Algorithm
(HSA) hyperparameter optimisation (HMS=20, 20 iterations), and train one model per lesion for
up to 500 epochs.
