import os
from PIL import Image

def remove_bad_images(folder_path):
    bad_count = 0
    total_count = 0

    # duyệt toàn bộ folder (recursive)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                total_count += 1
                path = os.path.join(root, file)

                try:
                    # Bước 1: check integrity
                    img = Image.open(path)
                    img.verify()

                    # Bước 2: decode thật (quan trọng)
                    img = Image.open(path).convert("RGB")

                except Exception as e:
                    print(f"[DELETE] {path} | Error: {e}")
                    try:
                        os.remove(path)
                        bad_count += 1
                    except Exception as remove_error:
                        print(f"[FAILED DELETE] {path} | Error: {remove_error}")

    print("\n========== RESULT ==========")
    print(f"Total images checked: {total_count}")
    print(f"Bad images deleted: {bad_count}")
    print(f"Remaining good images: {total_count - bad_count}")\

remove_bad_images("/media/kateee/New Volume/Python/Python/Shirokuma/snow_ratio_detection/data/label_unet")