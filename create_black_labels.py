"""
Script tạo ảnh nhãn đen cho các ảnh trong thư mục `example`
mà không có nhãn tương ứng trong thư mục `label_unet`.

- Input: ảnh .jpg trong thư mục example/ (ví dụ: a.jpg)
- Label tương ứng: label_unet/a.png
- Nếu nhãn không tồn tại → tạo ảnh đen cùng kích thước, lưu vào label_unet/a.png
"""

from pathlib import Path

import numpy as np
from PIL import Image

# ── Cấu hình đường dẫn ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
EXAMPLE_DIR = BASE_DIR / "example"
LABEL_DIR = BASE_DIR / "label_unet"

LABEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Lấy danh sách ảnh trong example (chỉ .jpg và .jpeg) ─────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

example_images = [
    f for f in EXAMPLE_DIR.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
]

print(f"Tổng số ảnh trong example: {len(example_images)}")

created = 0
already_exists = 0

for img_path in sorted(example_images):
    # Tên nhãn tương ứng (cùng stem, đuôi .png)
    label_path = LABEL_DIR / (img_path.stem + ".png")

    if label_path.exists():
        already_exists += 1
        continue

    # Đọc ảnh gốc để lấy kích thước
    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"  [LỖI] Không thể đọc {img_path.name}: {e}")
        continue

    # Tạo ảnh đen cùng kích thước (mode L = grayscale hoặc RGB tuỳ bạn cần)
    black_img = Image.fromarray(np.zeros((height, width), dtype=np.uint8), mode="L")
    black_img.save(label_path, format="PNG")

    print(f"  [TẠO]  {img_path.name}  →  {label_path.name}  ({width}×{height})")
    created += 1

print("\nKết quả:")
print(f"  Đã có nhãn sẵn : {already_exists}")
print(f"  Tạo nhãn đen mới: {created}")
print(f"  Tổng cộng       : {already_exists + created}")
