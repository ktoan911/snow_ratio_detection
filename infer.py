import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet

IMG_SIZE = 512
THRESHOLD = 0.5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=MEAN, std=STD)


def load_model(checkpoint_path: str, device: torch.device) -> UNet:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    val_dice = ckpt.get("val_dice", float("nan"))
    lr_tag = ckpt.get("lr_tag", "?")
    print(f"[Checkpoint] UNet: {checkpoint_path}")
    print(f"  lr_tag={lr_tag}  epoch={epoch}  val_dice={val_dice:.4f}")
    return model


def preprocess(img: Image.Image) -> torch.Tensor:
    img = TF.resize(img, (IMG_SIZE, IMG_SIZE), TF.InterpolationMode.BILINEAR)
    t = TF.to_tensor(img)
    t = normalize(t)
    return t.unsqueeze(0)


@torch.no_grad()
def predict_one(
    model,
    model_type: str,
    img_path: Path,
    device: torch.device,
    out_dir: Path | None = None,
) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size

    if model_type == "unet":
        inp = preprocess(img).to(device)

        logits = model(inp)
        prob = torch.sigmoid(logits)
        mask = (prob > THRESHOLD).squeeze()

        mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode="L").resize(
            (orig_w, orig_h), Image.NEAREST
        )

    elif model_type == "yolo":
        import cv2

        results = model.predict(
            str(img_path),
            imgsz=640,
            conf=THRESHOLD,
            iou=0.3,
            device=device.type if str(device.type) != "cpu" else "cpu",
            verbose=False,
            stream=False,
        )
        r = results[0]

        pred = np.zeros((orig_h, orig_w), dtype=bool)
        if r.masks is not None:
            for seg_mask in r.masks.data.cpu().numpy():
                seg_resized = cv2.resize(
                    seg_mask.astype(np.uint8),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                pred |= seg_resized

        mask_np = (pred * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_np, mode="L")

    overlay = img.copy().convert("RGBA")
    red_layer = Image.new("RGBA", (orig_w, orig_h), (255, 0, 0, 0))
    red_layer.putalpha(
        Image.fromarray((np.array(mask_pil) // 255 * 120).astype(np.uint8), mode="L")
    )
    overlay = Image.alpha_composite(overlay, red_layer)
    overlay_rgb = overlay.convert("RGB")

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

        save_path = out_dir / f"{img_path.stem}_mask.png"
        mask_pil.save(save_path)

        overlay_path = out_dir / f"{img_path.stem}_overlay.png"
        overlay_rgb.save(overlay_path)

    return overlay_rgb


def main():
    ap = argparse.ArgumentParser(
        description="Inference snow segmentation (U-Net hoặc YOLO)"
    )
    ap.add_argument(
        "--model",
        type=str,
        choices=["unet", "yolo"],
        default="unet",
        help="Loại model sẽ dùng để infer (unet hoặc yolo)",
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path tới file .pth (ví dụ: results/best_lr1.pth hoặc yolov8s-seg.pt)",
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

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.model == "unet":
        model = load_model(args.checkpoint, device)
    elif args.model == "yolo":
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Vui lòng cài đặt ultralytics: pip install ultralytics")
        print(f"[Checkpoint] YOLO: {args.checkpoint}")
        model = YOLO(args.checkpoint)

    out_dir = Path(args.out_dir) if args.out_dir else None

    if args.img:
        img_path = Path(args.img)
        _ = predict_one(model, args.model, img_path, device, out_dir)
        print(f"\n{'=' * 50}")
        print(f"Ảnh       : {img_path.name}")
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

        overlays = []
        for p in img_paths:
            overlay_rgb = predict_one(model, args.model, p, device, out_dir)
            overlays.append(overlay_rgb)
            print(f"  {p.name:<40}")

        print(f"{'─' * 60}")

        import math

        n_images = len(overlays)
        if n_images > 0:
            cols = math.ceil(math.sqrt(n_images))
            rows = math.ceil(n_images / cols)

            grid_w, grid_h = 512, 512
            grid_img = Image.new("RGB", (cols * grid_w, rows * grid_h))

            for i, overlay in enumerate(overlays):
                overlay_resized = overlay.resize((grid_w, grid_h))
                r_idx = i // cols
                c_idx = i % cols
                grid_img.paste(overlay_resized, (c_idx * grid_w, r_idx * grid_h))

            save_grid_dir = out_dir if out_dir else Path(".")
            save_grid_dir.mkdir(parents=True, exist_ok=True)
            img_dir_name = img_dir.name if img_dir.name else "result"
            grid_path = save_grid_dir / f"{img_dir_name}_grid_result.png"
            grid_img.save(grid_path)

            print(f"\nẢnh ghép tổng hợp (grid) đã được lưu tại: {grid_path.resolve()}")

        if out_dir:
            print(f"Mask/Overlay từng ảnh đã lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
