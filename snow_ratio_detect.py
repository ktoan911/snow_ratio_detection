import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

# ── Import model ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet

# ── Hằng số (phải khớp với train_unet.py) ────────────────────────────────────
IMG_SIZE = 512
THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=MEAN, std=STD)


# ─────────────────────────────────────────────────────────────────────────────
def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    """Load UNet từ checkpoint .pth đã lưu bởi train_unet.py."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_dice = ckpt.get("val_dice", float("nan"))
    lr_tag = ckpt.get("lr_tag", "?")
    print(f"[Checkpoint] {checkpoint_path}")
    print(f"  lr_tag={lr_tag}  epoch={epoch}  val_dice={val_dice:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Giống pipeline val trong train_unet.py:
      resize 512×512 → to_tensor → normalize ImageNet
    Trả về Tensor shape (1, 3, 512, 512).
    """
    img = TF.resize(img, (IMG_SIZE, IMG_SIZE), TF.InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)  # [3, H, W] float32 [0,1]
    t = normalize(t)
    return t.unsqueeze(0)  # [1, 3, H, W]


# ─────────────────────────────────────────────────────────────────────────────
def _save_mask_and_overlay(
    img_rgb: Image.Image,
    mask_512: np.ndarray,
    out_dir: Path,
    stem: str,
    orig_size: tuple[int, int],
    alpha: float = 0.35,
) -> None:
    """
    Lưu:
      - <stem>_mask.png: mask nhị phân 0/255 theo kích thước gốc
      - <stem>_overlay.png: overlay đỏ để nhìn nhanh
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    orig_w, orig_h = orig_size

    # Resize mask về size gốc (nearest để giữ nhị phân)
    mask_img = Image.fromarray(mask_512, mode="L")
    mask_orig = mask_img.resize((orig_w, orig_h), resample=Image.NEAREST)

    mask_path = out_dir / f"{stem}_mask.png"
    mask_orig.save(mask_path)

    # Overlay: tô đỏ vùng panel (mask=255)
    base = img_rgb.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (255, 0, 0, 0))

    m = np.array(mask_orig, dtype=np.uint8)
    # alpha channel theo mask (0 hoặc alpha*255)
    a = (m > 0).astype(np.uint8) * int(255 * alpha)
    overlay_np = np.array(overlay)
    overlay_np[..., 3] = a
    overlay = Image.fromarray(overlay_np, mode="RGBA")

    out = Image.alpha_composite(base, overlay)
    overlay_path = out_dir / f"{stem}_overlay.png"
    out.save(overlay_path)

    print(f"[Saved] {mask_path}")
    print(f"[Saved] {overlay_path}")


@torch.no_grad()
def predict_one(
    model: UNet, img_path: Path, device: torch.device, out_dir: Path | None = None
) -> np.ndarray:
    """
    Predict mask panel cho 1 ảnh.
    Trả về mask_np shape (512,512) uint8 {0,255} (panel=255).
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    inp = preprocess(img).to(device)  # [1, 3, 512, 512]
    logits = model(inp)  # [1, 1, 512, 512]
    prob = torch.sigmoid(logits)
    mask = (prob > THRESHOLD).squeeze(0).squeeze(0)  # [512,512] bool

    mask_np = (mask.cpu().numpy().astype(np.uint8) * 255)

    if out_dir is not None:
        _save_mask_and_overlay(
            img_rgb=img,
            mask_512=mask_np,
            out_dir=out_dir,
            stem=img_path.stem,
            orig_size=(orig_w, orig_h),
        )

    return mask_np


def calculate_apv(mask: np.ndarray) -> float:
    """
    Apv = (#pixel panel) / (total pixel) trong mask (mask 0/255 hoặc 0/1).
    """
    mask_bin = (mask > 0).astype(np.uint8)
    n_white = int(np.sum(mask_bin))
    h, w = mask_bin.shape
    total_pixels = h * w
    return n_white / float(total_pixels)


def calculate_apv_pair(mask_ref: np.ndarray, mask_test: np.ndarray) -> float:
    """
    Snow ratio theo paper:
      Apv_ref = panel area (visible) ở ảnh reference (no-snow)
      Apv_test = panel area (visible) ở ảnh current
      snow_ratio = (Apv_ref - Apv_test) / Apv_ref  (clamp 0..1)

    Note: tránh chia 0 nếu ref lỗi.
    """
    apv_ref = calculate_apv(mask_ref)
    apv_test = calculate_apv(mask_test)

    if apv_ref <= 1e-8:
        raise ValueError("Apv_ref ~ 0. Reference mask có vẻ lỗi (không thấy panel).")

    snow_ratio = (apv_ref - apv_test) / apv_ref
    snow_ratio = float(np.clip(snow_ratio, 0.0, 1.0))
    return snow_ratio


def _pick_device(device_arg: str | None) -> torch.device:
    if device_arg is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate snow cover ratio from fixed-camera PV images using UNet panel segmentation."
    )
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--ref", type=str, required=True, help="Reference (no-snow) image path")
    parser.add_argument("--cur", type=str, required=True, help="Current image path (may contain snow)")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Optional: directory to save masks/overlays for debug",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help='Device string, e.g. "cpu" or "cuda". Default auto.',
    )

    args = parser.parse_args()

    device = _pick_device(args.device)
    print(f"[Device] {device}")

    ckpt_path = Path(args.ckpt)
    ref_path = Path(args.ref)
    cur_path = Path(args.cur)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    if not cur_path.exists():
        raise FileNotFoundError(f"Current image not found: {cur_path}")

    model = load_model(str(ckpt_path), device)

    # Predict masks (512x512)
    mask_ref = predict_one(model, ref_path, device, out_dir=out_dir)
    mask_cur = predict_one(model, cur_path, device, out_dir=out_dir)

    # Compute ratio
    apv_ref = calculate_apv(mask_ref)
    apv_cur = calculate_apv(mask_cur)
    snow_ratio = calculate_apv_pair(mask_ref, mask_cur)

    print("\n[Result]")
    print(f"  Apv_ref   = {apv_ref:.6f}")
    print(f"  Apv_cur   = {apv_cur:.6f}")
    print(f"  snow_ratio= {snow_ratio:.6f}  ({snow_ratio*100:.2f}%)")