# debug_masks.py
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms


IMG_SIZE = 512
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def preprocess_like_train(img: Image.Image, mask: Image.Image, do_augment: bool, seed: int = 42):
    """
    Áp dụng preprocess giống train_unet.py của bạn:
      - convert RGB/L trước khi vào đây
      - augment: RandomResizedCrop + HFlip + Rotation + Affine
      - resize về IMG_SIZE
      - to_tensor + normalize
      - mask -> tensor binary theo threshold (0/1)
    """
    rnd = random.Random(seed)

    if do_augment:
        # 1) RandomResizedCrop (scale 0.8–1.0)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        img = TF.resized_crop(img, i, j, h, w, (IMG_SIZE, IMG_SIZE), TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask, i, j, h, w, (IMG_SIZE, IMG_SIZE), TF.InterpolationMode.NEAREST)

        # 2) Horizontal flip p=0.5
        if rnd.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 3) Rotation ±20
        angle = rnd.uniform(-20, 20)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)

        # 4) Affine translate ±10%, shear ±10
        affine_params = transforms.RandomAffine.get_params(
            degrees=(0, 0),
            translate=[0.1, 0.1],
            scale_ranges=None,
            shears=[-10, 10, -10, 10],
            img_size=[IMG_SIZE, IMG_SIZE],
        )
        img = TF.affine(img, *affine_params, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.affine(mask, *affine_params, interpolation=TF.InterpolationMode.NEAREST, fill=0)

    else:
        img = TF.resize(img, (IMG_SIZE, IMG_SIZE), TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (IMG_SIZE, IMG_SIZE), TF.InterpolationMode.NEAREST)

    # to tensor + normalize
    img_t = TF.to_tensor(img)
    img_t = transforms.Normalize(mean=MEAN, std=STD)(img_t)

    # mask -> binary {0,1} giống code bạn
    mask_arr = np.array(mask)
    mask_t = torch.from_numpy(mask_arr).float()
    if mask_t.max() > 1.0:
        mask_bin = (mask_t > 127).float().unsqueeze(0)
    else:
        mask_bin = (mask_t > 0.5).float().unsqueeze(0)

    return img_t, mask_bin, mask_arr


def save_debug_images(out_dir: Path, img: Image.Image, mask_arr: np.ndarray, mask_bin: torch.Tensor, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save mask after preprocess as PNG 0/255
    mb = (mask_bin.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(mb, mode="L").save(out_dir / f"{stem}_mask_pre.png")

    # Save raw mask (as grayscale)
    Image.fromarray(mask_arr.astype(np.uint8), mode="L").save(out_dir / f"{stem}_mask_raw.png")

    # Save input image (original)
    img.save(out_dir / f"{stem}_img_raw.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", type=str, required=True, help="Folder ảnh .jpg/.jpeg")
    ap.add_argument("--label_dir", type=str, required=True, help="Folder mask .png")
    ap.add_argument("--mode", type=str, default="train", choices=["train", "val"], help="Test preprocess train (augment) hoặc val (no augment)")
    ap.add_argument("--limit", type=int, default=0, help="Giới hạn số ảnh để test (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_debug", action="store_true", help="Lưu vài ảnh debug")
    ap.add_argument("--debug_out", type=str, default="mask_debug_out")
    ap.add_argument("--topk", type=int, default=20, help="In top K ảnh có fg_ratio thấp nhất")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    label_dir = Path(args.label_dir)
    assert img_dir.exists(), f"img_dir not found: {img_dir}"
    assert label_dir.exists(), f"label_dir not found: {label_dir}"

    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    pairs = []
    for p in img_paths:
        mp = label_dir / f"{p.stem}.png"
        if mp.exists():
            pairs.append((p, mp))

    if not pairs:
        raise SystemExit("Không tìm thấy cặp ảnh–mask nào (cần <stem>.png trong label_dir).")

    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    do_augment = args.mode == "train"

    stats = []
    zero_count = 0
    near_zero_count = 0

    # thresholds để đánh dấu "gần như toàn 0"
    NEAR_ZERO_FG = 1e-4  # 0.01% pixel là foreground

    for idx, (img_path, mask_path) in enumerate(pairs):
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        _, mask_bin, mask_arr = preprocess_like_train(img, mask, do_augment=do_augment, seed=args.seed + idx)

        fg_ratio = float(mask_bin.mean().item())
        fg_pixels = int(mask_bin.sum().item())
        total_pixels = int(mask_bin.numel())
        raw_min, raw_max = int(mask_arr.min()), int(mask_arr.max())
        raw_unique = np.unique(mask_arr)
        raw_unique_preview = raw_unique[:10].tolist()
        raw_unique_count = int(raw_unique.size)

        if fg_pixels == 0:
            zero_count += 1
        if fg_ratio <= NEAR_ZERO_FG:
            near_zero_count += 1

        stats.append(
            {
                "name": img_path.name,
                "fg_ratio": fg_ratio,
                "fg_pixels": fg_pixels,
                "total_pixels": total_pixels,
                "raw_min": raw_min,
                "raw_max": raw_max,
                "raw_unique_count": raw_unique_count,
                "raw_unique_preview": raw_unique_preview,
                "img_path": str(img_path),
                "mask_path": str(mask_path),
            }
        )

        # Save a few debug samples: first 5 problematic + first 3 normal
        if args.save_debug:
            out_dir = Path(args.debug_out)
            if fg_pixels == 0 and zero_count <= 5:
                save_debug_images(out_dir, img, mask_arr, mask_bin, stem=f"ZERO_{zero_count:02d}_{img_path.stem}")
            if fg_pixels > 0 and idx < 3:
                save_debug_images(out_dir, img, mask_arr, mask_bin, stem=f"NORMAL_{idx:02d}_{img_path.stem}")

    # summary
    fg_ratios = [s["fg_ratio"] for s in stats]
    print("\n========== MASK PREPROCESS CHECK ==========")
    print(f"Mode          : {args.mode}  (augment={'ON' if do_augment else 'OFF'})")
    print(f"Pairs checked : {len(stats)}")
    print(f"All-zero masks: {zero_count}")
    print(f"Near-zero masks (fg_ratio <= {NEAR_ZERO_FG}): {near_zero_count}")
    print(f"FG ratio min/mean/max: {np.min(fg_ratios):.6f} / {np.mean(fg_ratios):.6f} / {np.max(fg_ratios):.6f}")

    # show topk lowest fg_ratio
    stats_sorted = sorted(stats, key=lambda x: x["fg_ratio"])
    print(f"\n--- TOP {args.topk} LOWEST fg_ratio ---")
    for s in stats_sorted[: args.topk]:
        print(
            f"{s['name']:<35} fg_ratio={s['fg_ratio']:.6f} fg_pixels={s['fg_pixels']}/{s['total_pixels']} "
            f"raw[min,max]=[{s['raw_min']},{s['raw_max']}] unique_count={s['raw_unique_count']} "
            f"unique_preview={s['raw_unique_preview']}"
        )
        print(f"   mask: {s['mask_path']}")

    if args.save_debug:
        print(f"\n[DEBUG] Saved samples to: {Path(args.debug_out).resolve()}")
    print("==========================================\n")


if __name__ == "__main__":
    main()