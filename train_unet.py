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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from unet_model import UNet

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

BASE_DIR = Path(__file__).parent
DEFAULT_IMG_DIR = BASE_DIR / "example"
DEFAULT_LABEL_DIR = BASE_DIR / "label_unet"
DEFAULT_RESULT_DIR = BASE_DIR / "results"

IMG_SIZE = 512
THRESHOLD = 0.5

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class SnowSegDataset(Dataset):
    def __init__(self, img_paths: list, label_dir: Path, mode: str = "train"):
        self.img_paths = img_paths
        self.label_dir = label_dir
        self.mode = mode
        self.img_size = IMG_SIZE
        self.normalize = transforms.Normalize(mean=MEAN, std=STD)

    def _augment(self, img: Image.Image, mask: Image.Image):
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

        if random.random() < 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(
            mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0
        )

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

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.label_dir / (img_path.stem + ".png")

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.mode == "train":
            img, mask = self._augment(img, mask)
        else:
            img = TF.resize(
                img, (self.img_size, self.img_size), TF.InterpolationMode.BILINEAR
            )
            mask = TF.resize(
                mask, (self.img_size, self.img_size), TF.InterpolationMode.NEAREST
            )

        img = TF.to_tensor(img)
        img = self.normalize(img)

        arr = np.array(mask)
        mask = torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)

        return img, mask, str(img_path.name)


EPS = 1e-7


def accumulate_counts(acc: dict, pred_logits: torch.Tensor, target: torch.Tensor):
    with torch.no_grad():
        pred = (torch.sigmoid(pred_logits) > THRESHOLD).float()
        acc["tp"] = acc.get("tp", 0.0) + (pred * target).sum()
        acc["fp"] = acc.get("fp", 0.0) + (pred * (1 - target)).sum()
        acc["fn"] = acc.get("fn", 0.0) + ((1 - pred) * target).sum()
        acc["tn"] = acc.get("tn", 0.0) + ((1 - pred) * (1 - target)).sum()


def counts_to_metrics(acc: dict) -> dict:
    tp = acc["tp"].item() if isinstance(acc["tp"], torch.Tensor) else acc["tp"]
    fp = acc["fp"].item() if isinstance(acc["fp"], torch.Tensor) else acc["fp"]
    fn = acc["fn"].item() if isinstance(acc["fn"], torch.Tensor) else acc["fn"]
    tn = acc["tn"].item() if isinstance(acc["tn"], torch.Tensor) else acc["tn"]

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


def plot_curves(history: dict, lr_tag: str, result_dir: Path):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves  |  {lr_tag}", fontsize=14)

    def _plot_both(ax, train_vals, val_vals, ylabel):
        ax.plot(epochs, train_vals, label="train")
        ax.plot(epochs, val_vals, label="val", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_val(ax, val_vals, ylabel):
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
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_dice = -1.0
    best_metrics = {}
    best_epoch = 0
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
        model.train()
        t_loss, t_counts = 0.0, {}
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(imgs)
                loss = criterion(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            accumulate_counts(t_counts, logits.detach(), masks)

        n_train = len(train_loader)
        t_loss /= n_train
        t_metrics = counts_to_metrics(t_counts)

        model.eval()
        v_loss, v_counts = 0.0, {}
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(imgs)
                    loss = criterion(logits, masks)

                v_loss += loss.item()
                accumulate_counts(v_counts, logits, masks)

        n_val = len(val_loader)
        v_loss /= n_val
        v_metrics = counts_to_metrics(v_counts)

        scheduler.step(v_loss)

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_dice"].append(t_metrics["dice"])
        history["val_dice"].append(v_metrics["dice"])
        for k in ["precision", "recall", "f1", "accuracy", "miou"]:
            history[f"val_{k}"].append(v_metrics[k])

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f} | "
                f"train_dice={t_metrics['dice']:.4f}  val_dice={v_metrics['dice']:.4f} | "
                f"mIoU={v_metrics['miou']:.4f}  F1={v_metrics['f1']:.4f}"
            )

        if v_metrics["dice"] > best_dice:
            best_dice = v_metrics["dice"]
            best_epoch = epoch
            best_metrics = {
                "val_loss": v_loss,
                "dice": v_metrics["dice"],
                "precision": v_metrics["precision"],
                "recall": v_metrics["recall"],
                "f1": v_metrics["f1"],
                "accuracy": v_metrics["accuracy"],
                "miou": v_metrics["miou"],
            }
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

        if no_improve >= early_patience:
            print(
                f"\n  [EARLY STOP] Không cải thiện sau {early_patience} epoch. Dừng tại epoch {epoch}."
            )
            break

    print(
        f"\n  Best val Dice ({lr_tag}): {best_dice:.4f}  @ epoch {best_epoch}  → lưu tại {best_path}"
    )

    plot_curves(history, lr_tag, result_dir)

    hist_path = result_dir / f"{lr_tag}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_epoch, best_metrics


def main():
    parser = argparse.ArgumentParser(description="Train U-Net for snow segmentation")
    parser.add_argument("--img_dir", type=str, default=None)
    parser.add_argument("--label_dir", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_patience", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    img_dir = Path(args.img_dir) if args.img_dir else DEFAULT_IMG_DIR
    label_dir = Path(args.label_dir) if args.label_dir else DEFAULT_LABEL_DIR
    result_dir = Path(args.result_dir) if args.result_dir else DEFAULT_RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"img_dir    : {img_dir}")
    print(f"label_dir  : {label_dir}")
    print(f"result_dir : {result_dir}")

    all_imgs = sorted(
        [
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg"}
            and (label_dir / (p.stem + ".png")).exists()
        ]
    )
    print(f"Tổng cặp image–mask: {len(all_imgs)}")

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
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    lr_configs = [(1e-4, "lr1"), (3e-4, "lr2"), (1e-3, "lr3")]

    if args.lr is not None:
        lr_configs = [(args.lr, f"lr_{args.lr:.0e}")]

    summary = {}
    for lr, lr_tag in lr_configs:
        best_epoch, best_metrics = train_one_config(
            train_loader,
            val_loader,
            lr=lr,
            lr_tag=lr_tag,
            device=device,
            result_dir=result_dir,
            max_epochs=args.max_epochs,
            early_patience=args.early_patience,
        )
        summary[lr_tag] = {"lr": lr, "best_epoch": best_epoch, **best_metrics}

    col_w = 10
    metrics_order = ["dice", "precision", "recall", "f1", "accuracy", "miou"]
    header_names = ["Dice", "Precision", "Recall", "F1", "Accuracy", "mIoU"]

    sep = "─" * (8 + 6 + 8 + len(metrics_order) * (col_w + 3) + 1)
    print(f"\n\n{'═' * len(sep)}")
    print("  📊  KẾT QUẢ TỔNG KẾT SAU TRAINING")
    print(f"{'═' * len(sep)}")

    row_fmt = "  {:<8}  {:<6}  {:<8}"
    for _ in metrics_order:
        row_fmt += f"  {{:>{col_w}}}"
    print(row_fmt.format("Tag", "LR", "BestEpoch", *header_names))
    print(sep)

    best_vals = {m: max(info[m] for info in summary.values()) for m in metrics_order}

    for tag, info in summary.items():
        lr_str = f"{info['lr']:.0e}"
        vals = []
        for m in metrics_order:
            v = info[m]
            cell = f"{v:.4f}"
            if abs(v - best_vals[m]) < 1e-9:
                cell = f"★{v:.4f}"
            vals.append(cell.rjust(col_w))
        print(f"  {tag:<8}  {lr_str:<6}  {info['best_epoch']:<8}  {'  '.join(vals)}")

    print(sep)
    print("  ★ = giá trị tốt nhất trong nhóm LR")
    print(f"{'═' * len(sep)}\n")

    summary_path = result_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary lưu tại: {summary_path}")


if __name__ == "__main__":
    main()
