"""
train_unet.py — Training U-Net cho binary segmentation (snow mask)

Theo paper specs:
  - Input      : resize 512×512, normalize ImageNet mean/std
  - Augment    : RandomResizedCrop, HorizontalFlip, Rotation, Affine
  - Split      : 80 / 20 (train / val)
  - Loss       : BCEWithLogitsLoss
  - Optimizer  : Adam + ReduceLROnPlateau(factor=0.5, patience=10)
  - 3 LR runs  : lr1=1e-4, lr2=3e-4, lr3=1e-3
  - Epochs     : 200, early-stopping patience=20 theo val Dice
  - Metrics    : Dice, Precision, Recall, F1, Accuracy, mIoU
  - Lưu        : best_<lr_tag>.pth (val Dice cao nhất)
  - Biểu đồ   : results/<lr_tag>_curves.png
"""

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ── Import model ─────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Đường dẫn mặc định (có thể override qua CLI --img_dir, --label_dir, --result_dir)
BASE_DIR = Path(__file__).parent
DEFAULT_IMG_DIR = BASE_DIR / "example"
DEFAULT_LABEL_DIR = BASE_DIR / "label_unet"
DEFAULT_RESULT_DIR = BASE_DIR / "results"

IMG_SIZE = 512  # resize về 512×512
THRESHOLD = 0.5  # ngưỡng sigmoid để ra mask nhị phân

# ImageNet mean/std (phổ biến, chuẩn hoá tốt cho pretrained-style features)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ═══════════════════════════════════════════════════════════════════════════
# Dataset với augmentation
# ═══════════════════════════════════════════════════════════════════════════
class SnowSegDataset(Dataset):
    """
    Đọc cặp (ảnh .jpg từ img_dir/, mask .png từ label_dir/).
    Mode 'train' → áp dụng augmentation.
    Mode 'val'   → chỉ resize + normalize.
    """

    def __init__(self, img_paths: list, label_dir: Path, mode: str = "train"):
        self.img_paths = img_paths
        self.label_dir = label_dir
        self.mode = mode
        self.img_size = IMG_SIZE

        # Normalize (áp dụng sau khi đã convert sang Tensor)
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    # ── Augmentation đồng bộ image + mask ──────────────────────────────────
    def _augment(self, img: Image.Image, mask: Image.Image):
        """Áp dụng cùng biến đổi hình học lên img và mask."""

        # 1. RandomResizedCrop (scale 0.8–1.0)
        i, j, h, w = transforms.RandomResizedCrop.get_params(
            img, scale=(0.8, 1.0), ratio=(3 / 4, 4 / 3)
        )
        img = TF.resized_crop(
            img,
            i,
            j,
            h,
            w,
            (self.img_size, self.img_size),
            TF.InterpolationMode.BILINEAR,
        )
        mask = TF.resized_crop(
            mask,
            i,
            j,
            h,
            w,
            (self.img_size, self.img_size),
            TF.InterpolationMode.NEAREST,
        )

        # 2. Horizontal flip (p=0.5)
        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 3. Rotation (±20°)
        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(
            mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0
        )

        # 4. Affine: translate ±10%, shear ±10° (không rotate thêm vì bước 3 đã rotate)
        affine_params = transforms.RandomAffine.get_params(
            degrees=(0, 0),
            translate=[0.1, 0.1],
            scale_ranges=None,
            shears=[-10, 10, -10, 10],
            img_size=[self.img_size, self.img_size],
        )
        img = TF.affine(
            img, *affine_params, interpolation=TF.InterpolationMode.BILINEAR, fill=0
        )
        mask = TF.affine(
            mask, *affine_params, interpolation=TF.InterpolationMode.NEAREST, fill=0
        )

        return img, mask

    # ────────────────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.label_dir / (img_path.stem + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        if self.mode == "train":
            img, mask = self._augment(img, mask)
        else:
            # Val: chỉ resize
            img = TF.resize(
                img, (self.img_size, self.img_size), TF.InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask, (self.img_size, self.img_size), TF.InterpolationMode.NEAREST
            )

        # Chuyển sang Tensor
        img = TF.to_tensor(img)  # [3, H, W], float32 [0,1]
        img = self.normalize(img)

        mask = torch.from_numpy(np.array(mask)).float()  # [H, W]
        mask = (mask > 127).float().unsqueeze(0)  # [1, H, W], binary {0,1}

        return img, mask, str(img_path.name)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics — micro-average (accumulate raw TP/FP/FN/TN qua cả epoch)
# ═══════════════════════════════════════════════════════════════════════════
EPS = 1e-7


def accumulate_counts(acc: dict, pred_logits: torch.Tensor, target: torch.Tensor):
    """
    Cộng dồn raw TP/FP/FN/TN vào acc cho toàn epoch.
    acc: dict {'tp', 'fp', 'fn', 'tn'}
    """
    pred = (torch.sigmoid(pred_logits) > THRESHOLD).float()
    acc["tp"] = acc.get("tp", 0.0) + (pred * target).sum().item()
    acc["fp"] = acc.get("fp", 0.0) + (pred * (1 - target)).sum().item()
    acc["fn"] = acc.get("fn", 0.0) + ((1 - pred) * target).sum().item()
    acc["tn"] = acc.get("tn", 0.0) + ((1 - pred) * (1 - target)).sum().item()


def counts_to_metrics(acc: dict) -> dict:
    """Tính tất cả metric từ TP/FP/FN/TN tích lũy (micro-average đúng chuẩn)."""
    tp, fp, fn, tn = acc["tp"], acc["fp"], acc["fn"], acc["tn"]
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    iou_pos = tp / (tp + fp + fn + EPS)
    iou_neg = tn / (tn + fp + fn + EPS)
    miou = (iou_pos + iou_neg) / 2
    return {
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "miou": miou,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Vẽ biểu đồ
# ═══════════════════════════════════════════════════════════════════════════
def plot_curves(history: dict, lr_tag: str, result_dir: Path):
    """Vẽ loss, dice (train+val), và các val metric (1 đường). Lưu vào results/."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves  |  {lr_tag}", fontsize=14)

    def _plot_both(ax, train_vals, val_vals, ylabel):
        """Vẽ 2 đường: train vs val (dùng cho loss và dice)."""
        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals, label="val", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_val(ax, val_vals, ylabel):
        """Vẽ 1 đường val-only (precision/recall/F1/mIoU)."""
        ax.plot(epochs, val_vals, color="tab:orange", label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_both(axes[0, 0], history["train_loss"], history["val_loss"], "BCE Loss")
    _plot_both(axes[0, 1], history["train_dice"], history["val_dice"], "Dice")
    _plot_val(axes[0, 2], history["val_precision"], "Precision (val)")
    _plot_val(axes[1, 0], history["val_recall"], "Recall (val)")
    _plot_val(axes[1, 1], history["val_f1"], "F1 (val)")
    _plot_val(axes[1, 2], history["val_miou"], "mIoU (val)")

    plt.tight_layout()
    save_path = result_dir / f"{lr_tag}_curves.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"  [PLOT] Lưu biểu đồ: {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Training một lần với lr cụ thể
# ═══════════════════════════════════════════════════════════════════════════
def train_one_config(
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    lr_tag: str,
    device: torch.device,
    result_dir: Path,
    max_epochs: int = 200,
    early_patience: int = 20,
):
    print(f"\n{'=' * 60}")
    print(f"  Bắt đầu run: {lr_tag}  (lr={lr})")
    print(f"{'=' * 60}")

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6, verbose=True
    )

    best_dice = -1.0
    no_improve = 0
    best_path = result_dir / f"best_{lr_tag}.pth"

    history = {
        k: []
        for k in [
            "train_loss",
            "val_loss",
            "train_dice",
            "val_dice",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_accuracy",
            "val_miou",
        ]
    }

    for epoch in range(1, max_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        t_loss, t_counts = 0.0, {}
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            accumulate_counts(t_counts, logits.detach(), masks)  # cộng dồn raw counts

        n_train = len(train_loader)
        t_loss /= n_train
        t_metrics = counts_to_metrics(t_counts)  # micro-average toàn epoch

        # ── Val ────────────────────────────────────────────────────────────
        model.eval()
        v_loss, v_counts = 0.0, {}
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                loss = criterion(logits, masks)
                v_loss += loss.item()
                accumulate_counts(v_counts, logits, masks)  # cộng dồn raw counts

        n_val = len(val_loader)
        v_loss /= n_val
        v_metrics = counts_to_metrics(v_counts)  # micro-average toàn epoch

        # Scheduler bước theo val loss
        scheduler.step(v_loss)

        # Ghi history
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_dice"].append(t_metrics["dice"])
        history["val_dice"].append(v_metrics["dice"])
        for k in ["precision", "recall", "f1", "accuracy", "miou"]:
            history[f"val_{k}"].append(v_metrics[k])

        # Log mỗi 10 epoch
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f} | "
                f"train_dice={t_metrics['dice']:.4f}  val_dice={v_metrics['dice']:.4f} | "
                f"mIoU={v_metrics['miou']:.4f}  F1={v_metrics['f1']:.4f}"
            )

        # Lưu model tốt nhất
        if v_metrics["dice"] > best_dice:
            best_dice = v_metrics["dice"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_dice": best_dice,
                    "lr_tag": lr_tag,
                },
                best_path,
            )
        else:
            no_improve += 1

        # Early stopping
        if no_improve >= early_patience:
            print(
                f"\n  [EARLY STOP] Không cải thiện sau {early_patience} epoch. Dừng tại epoch {epoch}."
            )
            break

    print(f"\n  Best val Dice ({lr_tag}): {best_dice:.4f}  → lưu tại {best_path}")

    # Vẽ biểu đồ
    plot_curves(history, lr_tag, result_dir)

    # Lưu history JSON
    hist_path = result_dir / f"{lr_tag}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_dice


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Train U-Net for snow segmentation")
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Thư mục chứa ảnh đầu vào (.jpg). Mặc định: <script_dir>/example",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="Thư mục chứa mask nhãn (.png). Mặc định: <script_dir>/label_unet",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Thư mục lưu kết quả (model, biểu đồ). Mặc định: <script_dir>/results",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Chỉ chạy 1 LR cụ thể thay vì cả 3. Ví dụ: --lr 1e-4",
    )
    args = parser.parse_args()

    # ── Xác định đường dẫn ───────────────────────────────────────────────────
    img_dir = Path(args.img_dir) if args.img_dir else DEFAULT_IMG_DIR
    label_dir = Path(args.label_dir) if args.label_dir else DEFAULT_LABEL_DIR
    result_dir = Path(args.result_dir) if args.result_dir else DEFAULT_RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"img_dir    : {img_dir}")
    print(f"label_dir  : {label_dir}")
    print(f"result_dir : {result_dir}")

    # ── Tìm tất cả ảnh .jpg trong img_dir/ có nhãn tương ứng ──────────────
    all_imgs = sorted(
        [
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg"}
            and (label_dir / (p.stem + ".png")).exists()
        ]
    )
    print(f"Tổng cặp image–mask: {len(all_imgs)}")

    # ── Split 80/20 ────────────────────────────────────────────────────────
    n_total = len(all_imgs)
    n_train = int(n_total * 0.8)

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_imgs = [all_imgs[i] for i in indices[:n_train]]
    val_imgs = [all_imgs[i] for i in indices[n_train:]]

    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    train_ds = SnowSegDataset(train_imgs, label_dir=label_dir, mode="train")
    val_ds = SnowSegDataset(val_imgs, label_dir=label_dir, mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 3 LR configs theo paper ────────────────────────────────────────────
    lr_configs = [
        (1e-4, "lr1"),
        (3e-4, "lr2"),
        (1e-3, "lr3"),
    ]

    if args.lr is not None:
        lr_configs = [(args.lr, f"lr_{args.lr:.0e}")]

    summary = {}
    for lr, lr_tag in lr_configs:
        best_dice = train_one_config(
            train_loader,
            val_loader,
            lr=lr,
            lr_tag=lr_tag,
            device=device,
            result_dir=result_dir,
            max_epochs=args.max_epochs,
            early_patience=args.early_patience,
        )
        summary[lr_tag] = {"lr": lr, "best_val_dice": best_dice}

    # ── In tóm tắt ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  TỔNG KẾT:")
    for tag, info in summary.items():
        print(
            f"    {tag} (lr={info['lr']:.0e}):  Best val Dice = {info['best_val_dice']:.4f}"
        )
    print(f"{'=' * 60}")

    # Lưu summary
    summary_path = result_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary lưu tại: {summary_path}")


if __name__ == "__main__":
    main()
