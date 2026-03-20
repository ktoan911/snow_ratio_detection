"""
train_yolo.py — Finetune YOLOv8s-seg cho binary segmentation (snow mask)

Cấu trúc giống train_unet.py:
  - Input      : resize 640×640 (YOLO default), normalize nội bộ bởi ultralytics
  - Augment    : dùng augmentation mặc định của ultralytics (hsv, flip, mosaic…)
  - Split      : 80 / 20 (train / val) — tạo dataset YOLO tạm thời trên disk
  - 3 LR runs  : lr1=1e-4, lr2=3e-4, lr3=1e-3
  - Epochs     : 200, early-stopping patience=20 (theo val Dice tự tính)
  - Metrics    : Dice, Precision, Recall, F1, Accuracy, mIoU
  - Lưu        : best_<lr_tag>/weights/best.pt  (val Dice cao nhất)
  - Biểu đồ   : results/<lr_tag>_curves.png

Yêu cầu:
    pip install ultralytics
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ultralytics ───────────────────────────────────────────────────────────────
try:
    import yaml as _yaml
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Cần cài ultralytics: pip install ultralytics")
    raise

# ── Seed ─────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Đường dẫn mặc định ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DEFAULT_IMG_DIR = BASE_DIR / "example"
DEFAULT_LABEL_DIR = BASE_DIR / "label_unet"
DEFAULT_RESULT_DIR = BASE_DIR / "results"

IMG_SIZE = 640  # YOLO default
THRESHOLD = 0.5  # ngưỡng tạo binary mask từ polygon khi tính metrics

EPS = 1e-7


# ═══════════════════════════════════════════════════════════════════════════
# Tiện ích: chuyển binary mask PNG → YOLO polygon label (class 0)
# ═══════════════════════════════════════════════════════════════════════════
def mask_png_to_yolo_label(mask_path: Path, label_path: Path):
    """
    Đọc mask grayscale (0/255), tìm contour, xuất YOLO-seg format:
      0 x1 y1 x2 y2 … (tọa độ chuẩn hóa [0,1])
    Nếu không có contour (ảnh trắng toàn bộ hoặc đen toàn bộ) → file rỗng.
    """
    arr = np.array(Image.open(mask_path).convert("L"))
    # Hỗ trợ cả mask 0/1 (label_unet) lẫn 0/255
    binary = (arr > 0).astype(np.uint8) * 255
    h, w = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        if cnt.shape[0] < 3:
            continue
        # Epsilon-approximation để giảm điểm
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.extend([x / w, y / h])
        lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))

    with open(label_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════════════
# Tạo cấu trúc YOLO dataset tạm thời trên disk
# ═══════════════════════════════════════════════════════════════════════════
def build_yolo_dataset(
    train_imgs: list[Path],
    val_imgs: list[Path],
    label_dir: Path,
    tmp_dir: Path,
) -> Path:
    """
    Tạo cây thư mục:
      tmp_dir/
        images/train/  ← symlink hoặc copy ảnh
        images/val/
        labels/train/  ← label YOLO polygon
        labels/val/
        dataset.yaml
    Trả về đường dẫn dataset.yaml.
    """
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_out = tmp_dir / "images" / split
        lbl_out = tmp_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for p in imgs:
            # Copy ảnh
            dst_img = img_out / p.name
            shutil.copy2(p, dst_img)

            # Tạo label polygon
            mask_path = label_dir / (p.stem + ".png")
            lbl_path = lbl_out / (p.stem + ".txt")
            mask_png_to_yolo_label(mask_path, lbl_path)

    # dataset.yaml
    yaml_path = tmp_dir / "dataset.yaml"
    cfg = {
        "path": str(tmp_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["snow"],
    }
    with open(yaml_path, "w") as f:
        _yaml.dump(cfg, f, allow_unicode=True)

    return yaml_path


# ═══════════════════════════════════════════════════════════════════════════
# Metrics — tính trực tiếp từ YOLO mask predictions
# ═══════════════════════════════════════════════════════════════════════════
def compute_dice_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    """Tính Dice/Precision/Recall/F1/Accuracy/mIoU từ cặp binary mask (bool)."""
    tp = float(np.logical_and(pred_mask, gt_mask).sum())
    fp = float(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = float(np.logical_and(~pred_mask, gt_mask).sum())
    tn = float(np.logical_and(~pred_mask, ~gt_mask).sum())

    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    iou_pos = tp / (tp + fp + fn + EPS)
    iou_neg = tn / (tn + fp + fn + EPS)
    miou = (iou_pos + iou_neg) / 2

    return dict(
        dice=dice,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        miou=miou,
    )


def accumulate_metrics(
    acc: dict,
    model: "YOLO",
    img_paths: list[Path],
    label_dir: Path,
    device: str,
):
    """
    Chạy inference trên từng ảnh, so sánh mask predict vs GT mask,
    cộng dồn TP/FP/FN/TN.
    """
    tp_total = fp_total = fn_total = tn_total = 0.0

    for p in img_paths:
        # ── GT mask ──────────────────────────────────────────────────────
        mask_path = label_dir / (p.stem + ".png")
        gt = (
            np.array(
                Image.open(mask_path)
                .convert("L")
                .resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
            )
            > 0  # dùng > 0 thay vì > 127 vì mask label_unet lưu giá trị 0/1 (không phải 0/255)
        )  # bool HxW

        # ── Predict ──────────────────────────────────────────────────────
        results = model.predict(
            str(p),
            imgsz=IMG_SIZE,
            conf=0.01,  # thấp để catch prediction dù model chưa confident
            iou=0.3,
            device=device,
            verbose=False,
            stream=False,
        )
        r = results[0]

        pred = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
        if r.masks is not None:
            for seg_mask in r.masks.data.cpu().numpy():  # (H', W')
                seg_resized = cv2.resize(
                    seg_mask.astype(np.uint8),
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                pred |= seg_resized

        tp_total += float(np.logical_and(pred, gt).sum())
        fp_total += float(np.logical_and(pred, ~gt).sum())
        fn_total += float(np.logical_and(~pred, gt).sum())
        tn_total += float(np.logical_and(~pred, ~gt).sum())

    acc["tp"] = acc.get("tp", 0.0) + tp_total
    acc["fp"] = acc.get("fp", 0.0) + fp_total
    acc["fn"] = acc.get("fn", 0.0) + fn_total
    acc["tn"] = acc.get("tn", 0.0) + tn_total


def counts_to_metrics(acc: dict) -> dict:
    tp, fp, fn, tn = acc["tp"], acc["fp"], acc["fn"], acc["tn"]
    dice = (2 * tp) / (2 * tp + fp + fn + EPS)
    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)
    f1 = (2 * precision * recall) / (precision + recall + EPS)
    accuracy = (tp + tn) / (tp + tn + fp + fn + EPS)
    iou_pos = tp / (tp + fp + fn + EPS)
    iou_neg = tn / (tn + fp + fn + EPS)
    miou = (iou_pos + iou_neg) / 2
    return dict(
        dice=dice,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        miou=miou,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Vẽ biểu đồ (giống train_unet.py)
# ═══════════════════════════════════════════════════════════════════════════
def plot_curves(history: dict, lr_tag: str, result_dir: Path):
    """Vẽ loss, dice (train+val), và các val metric (1 đường). Lưu vào results/."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Training Curves  |  {lr_tag}  [YOLOv8s-seg]", fontsize=14)

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

    _plot_both(axes[0, 0], history["train_loss"], history["val_loss"], "Seg Loss")
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
# Finetune một lần với lr cụ thể  — dùng ultralytics callback
# ═══════════════════════════════════════════════════════════════════════════
def train_one_config(
    yaml_path: Path,
    train_imgs: list[Path],
    val_imgs: list[Path],
    label_dir: Path,
    lr: float,
    lr_tag: str,
    device: str,
    result_dir: Path,
    max_epochs: int = 200,
    early_patience: int = 20,
    batch_size: int = 4,
):
    print(f"\n{'=' * 60}")
    print(f"  Bắt đầu run: {lr_tag}  (lr={lr})")
    print(f"{'=' * 60}")

    run_dir = result_dir / f"best_{lr_tag}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── State được chia sẻ với callback (dùng dict để tránh closure capture issue) ──
    state = {
        "history": {
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
        },
        "best_dice": -1.0,
        "best_metrics": {},
        "best_epoch": 0,
        "no_improve": 0,
        "stop": False,
    }

    # ── Callback chạy sau mỗi epoch val ─────────────────────────────────────
    # on_val_end nhận Validator (không phải Trainer) — lấy trainer qua .trainer
    def on_val_end(validator):
        tr = getattr(validator, "trainer", None)
        if tr is None:
            return
        epoch = tr.epoch + 1  # ultralytics dùng 0-indexed

        # ── Train / Val loss từ trainer metrics ──────────────────────────
        rd = tr.metrics if hasattr(tr, "metrics") else {}
        t_loss = float(
            rd.get(
                "train/seg_loss",
                rd.get("train/loss", rd.get("metrics/seg_loss(B)", 0.0)),
            )
        )
        v_loss = float(
            rd.get("val/seg_loss", rd.get("val/loss", rd.get("val(B)/seg_loss", 0.0)))
        )

        # ── Custom metrics bằng inference ────────────────────────────────
        last_pt = run_dir / "train" / "weights" / "last.pt"
        if not last_pt.exists():
            return

        yolo_model = YOLO(str(last_pt))

        t_counts, v_counts = {}, {}
        accumulate_metrics(t_counts, yolo_model, train_imgs, label_dir, device)
        accumulate_metrics(v_counts, yolo_model, val_imgs, label_dir, device)
        t_metrics = counts_to_metrics(t_counts)
        v_metrics = counts_to_metrics(v_counts)

        # Ghi history
        h = state["history"]
        h["train_loss"].append(t_loss)
        h["val_loss"].append(v_loss)
        h["train_dice"].append(t_metrics["dice"])
        h["val_dice"].append(v_metrics["dice"])
        for k in ["precision", "recall", "f1", "accuracy", "miou"]:
            h[f"val_{k}"].append(v_metrics[k])

        # Log mỗi 10 epoch
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{max_epochs} | "
                f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f} | "
                f"train_dice={t_metrics['dice']:.4f}  val_dice={v_metrics['dice']:.4f} | "
                f"mIoU={v_metrics['miou']:.4f}  F1={v_metrics['f1']:.4f}"
            )

        # Early stopping theo val Dice
        if v_metrics["dice"] > state["best_dice"]:
            state["best_dice"] = v_metrics["dice"]
            state["best_epoch"] = epoch
            state["best_metrics"] = {
                "val_loss": v_loss,
                "dice": v_metrics["dice"],
                "precision": v_metrics["precision"],
                "recall": v_metrics["recall"],
                "f1": v_metrics["f1"],
                "accuracy": v_metrics["accuracy"],
                "miou": v_metrics["miou"],
            }
            state["no_improve"] = 0

            # Copy best weights
            candidate = run_dir / "train" / "weights" / "best.pt"
            if not candidate.exists():
                candidate = run_dir / "train" / "weights" / "last.pt"
            if candidate.exists():
                shutil.copy2(candidate, run_dir / "best_model.pt")
        else:
            state["no_improve"] += 1

        # Báo trainer dừng khi hết patience
        if state["no_improve"] >= early_patience:
            print(
                f"\n  [EARLY STOP] Không cải thiện sau {early_patience} epoch. "
                f"Dừng tại epoch {epoch}."
            )
            tr.stop = True  # ultralytics kiểm tra flag này trên Trainer

    # ── Load model và gắn callback ────────────────────────────────────────
    model = YOLO("yolov8s-seg.pt")
    model.add_callback("on_val_end", on_val_end)

    # ── Train toàn bộ epochs trong 1 lần gọi ─────────────────────────────
    model.train(
        data=str(yaml_path),
        epochs=max_epochs,
        imgsz=IMG_SIZE,
        batch=batch_size,
        lr0=lr,
        lrf=0.01,  # cosine decay từ lr0 → lr0*0.01 theo đúng schedule
        warmup_epochs=3,  # warmup chuẩn (3 epoch đầu)
        optimizer="Adam",
        device=device,
        project=str(run_dir),
        name="train",
        exist_ok=True,
        verbose=False,
        # Augmentation
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=20.0,
        translate=0.1,
        flipud=0.0,
        fliplr=0.5,
        seed=SEED,
        patience=0,  # tắt YOLO early-stopping, dùng callback tự quản
        save=True,
        save_period=-1,  # chỉ lưu best.pt + last.pt, không lưu epoch*.pt (tốn disk)
    )

    best_dice = state["best_dice"]
    best_epoch = state["best_epoch"]
    best_metrics = state["best_metrics"]
    best_pt_src = run_dir / "best_model.pt"
    history = state["history"]

    print(
        f"\n  Best val Dice ({lr_tag}): {best_dice:.4f}  "
        f"@ epoch {best_epoch}  → lưu tại {best_pt_src}"
    )

    # Vẽ biểu đồ
    plot_curves(history, lr_tag, result_dir)

    # Lưu history JSON
    hist_path = result_dir / f"{lr_tag}_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return best_epoch, best_metrics


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Finetune YOLOv8s-seg for snow segmentation"
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="Thư mục chứa ảnh .jpg. Mặc định: <script_dir>/example",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        default=None,
        help="Thư mục chứa mask .png. Mặc định: <script_dir>/label_unet",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help="Thư mục lưu kết quả. Mặc định: <script_dir>/results",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--early_patience", type=int, default=20)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Số worker DataLoader (truyền vào ultralytics workers).",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Chỉ chạy 1 LR cụ thể. Ví dụ: --lr 1e-4"
    )
    args = parser.parse_args()

    # ── Paths ─────────────────────────────────────────────────────────────
    img_dir = Path(args.img_dir) if args.img_dir else DEFAULT_IMG_DIR
    label_dir = Path(args.label_dir) if args.label_dir else DEFAULT_LABEL_DIR
    result_dir = Path(args.result_dir) if args.result_dir else DEFAULT_RESULT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Device     : {device}")
    print(f"img_dir    : {img_dir}")
    print(f"label_dir  : {label_dir}")
    print(f"result_dir : {result_dir}")

    # ── Tìm tất cả ảnh .jpg có mask tương ứng ────────────────────────────
    all_imgs = sorted(
        [
            p
            for p in img_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg"}
            and (label_dir / (p.stem + ".png")).exists()
        ]
    )
    print(f"Tổng cặp image–mask: {len(all_imgs)}")

    # ── Split 80/20 ───────────────────────────────────────────────────────
    n_total = len(all_imgs)
    n_train = int(n_total * 0.8)
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_imgs = [all_imgs[i] for i in indices[:n_train]]
    val_imgs = [all_imgs[i] for i in indices[n_train:]]
    print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

    # ── Tạo YOLO dataset tạm thời ─────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="yolo_snow_"))
    print(f"Dataset tạm: {tmp_dir}")
    try:
        yaml_path = build_yolo_dataset(train_imgs, val_imgs, label_dir, tmp_dir)

        # ── 3 LR configs theo paper ───────────────────────────────────────
        lr_configs = [
            (1e-4, "lr1"),
            (3e-4, "lr2"),
            (1e-3, "lr3"),
        ]
        if args.lr is not None:
            lr_configs = [(args.lr, f"lr_{args.lr:.0e}")]

        summary = {}
        for lr, lr_tag in lr_configs:
            best_epoch, best_metrics = train_one_config(
                yaml_path=yaml_path,
                train_imgs=train_imgs,
                val_imgs=val_imgs,
                label_dir=label_dir,
                lr=lr,
                lr_tag=lr_tag,
                device=device,
                result_dir=result_dir,
                max_epochs=args.max_epochs,
                early_patience=args.early_patience,
                batch_size=args.batch_size,
            )
            summary[lr_tag] = {"lr": lr, "best_epoch": best_epoch, **best_metrics}

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n[INFO] Đã xóa dataset tạm: {tmp_dir}")

    # ── Bảng tổng kết chi tiết (giống train_unet.py) ──────────────────────
    col_w = 10
    metrics_order = ["dice", "precision", "recall", "f1", "accuracy", "miou"]
    header_names = ["Dice", "Precision", "Recall", "F1", "Accuracy", "mIoU"]

    sep = "─" * (8 + 6 + 8 + len(metrics_order) * (col_w + 3) + 1)
    print(f"\n\n{'═' * len(sep)}")
    print("  📊  KẾT QUẢ TỔNG KẾT SAU FINETUNE  [YOLOv8s-seg]")
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

    # Lưu summary JSON
    summary_path = result_dir / "summary_yolo.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary lưu tại: {summary_path}")


if __name__ == "__main__":
    main()
