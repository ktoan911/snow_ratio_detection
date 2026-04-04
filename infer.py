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
@torch.no_grad()
def predict_one(
    model: UNet, img_path: Path, device: torch.device, out_dir: Path | None = None
) -> float:
    """
    Predict mask cho 1 ảnh, trả về snow_ratio (0.0–1.0).

    Nếu out_dir không None → lưu:
      - <stem>_mask.png   : mask nhị phân (0/255), cùng kích thước ảnh gốc
      - <stem>_overlay.png: ảnh gốc + mask đỏ trong suốt để kiểm tra trực quan
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size  # kích thước gốc

    inp = preprocess(img).to(device)  # [1, 3, 512, 512]

    logits = model(inp)  # [1, 1, 512, 512]  raw logits
    prob = torch.sigmoid(logits)  # [0, 1]
    mask = (prob > THRESHOLD).squeeze()  # [512, 512] bool Tensor

    snow_ratio = float(mask.float().mean().item())

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)

        # Resize mask về kích thước ảnh gốc
        mask_pil = Image.fromarray(mask_np, mode="L").resize(
            (orig_w, orig_h), Image.NEAREST
        )
        save_path = out_dir / f"{img_path.stem}_mask.png"
        mask_pil.save(save_path)

        # Lưu overlay: mask đỏ trong suốt (alpha=120) chồng lên ảnh gốc
        overlay = img.copy().convert("RGBA")
        red_layer = Image.new("RGBA", (orig_w, orig_h), (255, 0, 0, 0))
        red_layer.putalpha(
            Image.fromarray(
                (np.array(mask_pil) // 255 * 120).astype(np.uint8), mode="L"
            )
        )
        overlay = Image.alpha_composite(overlay, red_layer)
        overlay_path = out_dir / f"{img_path.stem}_overlay.png"
        overlay.convert("RGB").save(overlay_path)

    return snow_ratio


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Inference U-Net snow segmentation")
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path tới file .pth (ví dụ: results/best_lr1.pth)",
    )
    ap.add_argument("--img", default=None, help="Infer 1 ảnh cụ thể")
    ap.add_argument(
        "--img_dir", default=None, help="Infer cả thư mục ảnh (.jpg / .jpeg / .png)"
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Thư mục lưu mask đầu ra (nếu không chỉ định → chỉ in ra terminal)",
    )
    ap.add_argument(
        "--device",
        default=None,
        help="'cuda' hoặc 'cpu' (tự detect nếu không chỉ định)",
    )
    args = ap.parse_args()

    if args.img is None and args.img_dir is None:
        ap.error("Cần chỉ định --img hoặc --img_dir")

    # ── Device ────────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(args.checkpoint, device)

    out_dir = Path(args.out_dir) if args.out_dir else None

    # ── Infer ─────────────────────────────────────────────────────────────────
    if args.img:
        img_path = Path(args.img)
        ratio = predict_one(model, img_path, device, out_dir)
        print(f"\n{'=' * 50}")
        print(f"Ảnh       : {img_path.name}")
        print(f"Snow ratio: {ratio * 100:.2f}%")
        if out_dir:
            print(f"Mask lưu  : {out_dir / (img_path.stem + '_mask.png')}")
        print(f"{'=' * 50}")

    else:
        img_dir = Path(args.img_dir)
        assert img_dir.exists(), f"img_dir không tồn tại: {img_dir}"

        img_paths = sorted(
            [
                p
                for p in img_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if not img_paths:
            raise SystemExit(f"Không tìm thấy ảnh trong: {img_dir}")

        print(f"\nInfer {len(img_paths)} ảnh từ: {img_dir}")
        print(f"{'─' * 60}")

        ratios = []
        for p in img_paths:
            r = predict_one(model, p, device, out_dir)
            ratios.append(r)
            print(f"  {p.name:<40}  snow={r * 100:6.2f}%")

        print(f"{'─' * 60}")
        print(f"  Trung bình snow ratio: {np.mean(ratios) * 100:.2f}%")
        print(f"  Min: {np.min(ratios) * 100:.2f}%   Max: {np.max(ratios) * 100:.2f}%")
        if out_dir:
            print(f"\nMask đã lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
