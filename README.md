# Snow Ratio Detection

Detect and measure the snow cover ratio on solar panels from fixed-camera images using **U-Net** or **YOLO-Seg** segmentation models.

---

## Installation

```bash
# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision pillow numpy
pip install ultralytics   # only needed if using YOLO
```

---

## 1. `infer.py` — Image Segmentation (U-Net or YOLO)

Run inference on a **single image** or an **entire image directory**, supporting both model architectures.

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `--model` | No | `unet` (default) or `yolo` |
| `--checkpoint` | **Yes** | Path to the `.pth` (U-Net) or `.pt` (YOLO) file |
| `--img` | * | Path to a specific image |
| `--img_dir` | * | Directory containing images (`.jpg`, `.jpeg`, `.png`) |
| `--out_dir` | No | Directory to save masks and overlays (if not specified → only prints results) |
| `--device` | No | `cuda` or `cpu` (auto-detected if omitted) |

> \* You must specify **at least one** of `--img` or `--img_dir`.

### Examples

**Infer 1 image with U-Net, save results:**
```bash
python infer.py \
  --model unet \
  --checkpoint model/best.pth \
  --img data/sample.jpg \
  --out_dir outputs/
```

**Infer entire directory with YOLO:**
```bash
python infer.py \
  --model yolo \
  --checkpoint model/yolov8s-seg.pt \
  --img_dir data/test_images/ \
  --out_dir outputs/
```

**Infer directory with U-Net, only print results (do not save files):**
```bash
python infer.py \
  --checkpoint model/best.pth \
  --img_dir data/test_images/
```

### Output

- `<stem>_mask.png` — Binary mask (0/255), same size as the original image.
- `<stem>_overlay.png` — Original image + transparent red snow area for visual inspection.
- `<img_dir_name>_grid_result.png` — Grid image combining all overlays (when inferring a directory).

---

## 2. `snow_ratio_detect.py` — Calculate Snow Ratio (U-Net only)

Calculate the **snow ratio** based on the paper's formula, using a pair of images: **reference (no snow)** and **current image**:

$$\text{snow\_ratio} = \frac{Apv\_ref - Apv\_cur}{Apv\_ref}$$

Where `Apv` is the area proportion of the visible solar panel in the image.

### Parameters

| Parameter | Required | Description |
|---|---|---|
| `--ckpt` | **Yes** | Path to the U-Net checkpoint `.pth` file |
| `--ref` | **Yes** | Reference image (no snow) |
| `--cur` | **Yes** | Current image (may contain snow) |
| `--out_dir` | No | Directory to save debug masks and overlays |
| `--device` | No | `cuda` or `cpu` (auto-detected if omitted) |

### Examples

**Calculate snow ratio, do not save files:**
```bash
python snow_ratio_detect.py \
  --ckpt model/best.pth \
  --ref data/reference.jpg \
  --cur data/current.jpg
```

**Calculate snow ratio and save debug masks:**
```bash
python snow_ratio_detect.py \
  --ckpt model/best.pth \
  --ref data/reference.jpg \
  --cur data/current.jpg \
  --out_dir outputs/debug/ \
  --device cpu
```

### Output (terminal)

```
[Result]
  Apv_ref   = 0.412500
  Apv_cur   = 0.287300
  snow_ratio= 0.303515  (30.35%)
```

If `--out_dir` is provided, the mask and overlay files for both the reference and current images will be saved in that directory.