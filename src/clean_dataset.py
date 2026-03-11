import os
import shutil
from PIL import Image

DATA_DIR = r"c:\Users\suhas\sugarcane_pathology_detection\data"
CORRUPTED_DIR = os.path.join(DATA_DIR, "corrupted")

def is_image_corrupted(filepath):
    try:
        img = Image.open(filepath)
        img.verify()  # verify integrity
        return False
    except Exception:
        return True

def clean_dataset():
    total_scanned = 0
    corrupted_count = 0
    clean_count = 0

    for root, _, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_scanned += 1
                filepath = os.path.join(root, file)

                if is_image_corrupted(filepath):
                    corrupted_count += 1
                    # path to move corrupted file
                    relative_path = os.path.relpath(filepath, DATA_DIR)
                    corrupted_path = os.path.join(CORRUPTED_DIR, relative_path)

                    os.makedirs(os.path.dirname(corrupted_path), exist_ok=True)
                    shutil.move(filepath, corrupted_path)
                    print(f"❌ Moved corrupted: {filepath} → {corrupted_path}")
                else:
                    clean_count += 1

    print("\n📊 Scan Summary:")
    print(f"   🔍 Total images scanned: {total_scanned}")
    print(f"   ✅ Clean images: {clean_count}")
    print(f"   ❌ Corrupted images moved: {corrupted_count}")

if __name__ == "__main__":
    clean_dataset()
