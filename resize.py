import os
from pathlib import Path
from PIL import Image
import numpy as np

def create_missing_masks(img_dir, mask_dir, mask_ext=".png"):
    img_dir = Path(img_dir)
    mask_dir = Path(mask_dir)

    mask_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    existed = 0

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        mask_path = mask_dir / (img_path.stem + mask_ext)

        # Nếu đã có mask → bỏ qua
        if mask_path.exists():
            existed += 1
            continue

        # Đọc kích thước ảnh
        img = Image.open(img_path)
        w, h = img.size

        # Tạo mask đen (toàn background)
        mask = np.zeros((h, w), dtype=np.uint8)

        # Save
        Image.fromarray(mask).save(mask_path)

        created += 1
        print(f"🆕 Created mask: {mask_path.name}")

    print("\n===== DONE =====")
    print(f"Created masks : {created}")
    print(f"Existing masks: {existed}")


if __name__ == "__main__":
    create_missing_masks(
        img_dir="/mnt/data/Python/Python/Shirokuma/snow_ratio_detection/data/raw",
        mask_dir="/mnt/data/Python/Python/Shirokuma/snow_ratio_detection/data/label_unet"
    )