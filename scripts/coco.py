"""
MS-COCO Dataset Downloader
==========================
Downloads and sets up MS-COCO dataset for all phases of the Vision-Language-Agent.

Phase 1 → Image Captioning  : COCO Images + Captions
Phase 2 → Visual Q&A        : COCO Images + VQA v2 Labels
Phase 3 → Image Generation  : COCO Captions as text prompts
Phase 4 → Agent Testing     : COCO Images as test inputs
Phase 5 → Memory (ChromaDB) : COCO Image Embeddings
"""

import os
import json
import zipfile
import urllib.request
from pathlib import Path
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data" / "coco"
IMAGES_DIR = DATA_DIR / "images"
ANN_DIR    = DATA_DIR / "annotations"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
ANN_DIR.mkdir(parents=True, exist_ok=True)

# ── MS-COCO 2017 URLs ─────────────────────────────────────────────────────────
COCO_URLS = {
    # Annotations (captions + instances + keypoints)
    "annotations": {
        "url"  : "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "dest" : DATA_DIR / "annotations_trainval2017.zip",
        "label": "Annotations (Captions + Instances)",
    },
    # VQA v2 annotations (built on COCO images)
    "vqa_questions_train": {
        "url"  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
        "dest" : DATA_DIR / "vqa_questions_train.zip",
        "label": "VQA v2 Train Questions",
    },
    "vqa_annotations_train": {
        "url"  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
        "dest" : DATA_DIR / "vqa_annotations_train.zip",
        "label": "VQA v2 Train Annotations",
    },
    "vqa_questions_val": {
        "url"  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
        "dest" : DATA_DIR / "vqa_questions_val.zip",
        "label": "VQA v2 Val Questions",
    },
    "vqa_annotations_val": {
        "url"  : "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
        "dest" : DATA_DIR / "vqa_annotations_val.zip",
        "label": "VQA v2 Val Annotations",
    },
    # Images (large — download only what you need)
    "train2017": {
        "url"  : "http://images.cocodataset.org/zips/train2017.zip",
        "dest" : IMAGES_DIR / "train2017.zip",
        "label": "Train Images (18 GB) — optional, see README",
    },
    "val2017": {
        "url"  : "http://images.cocodataset.org/zips/val2017.zip",
        "dest" : IMAGES_DIR / "val2017.zip",
        "label": "Validation Images (1 GB)",
    },
}

# ── Progress-aware downloader ─────────────────────────────────────────────────
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, dest: Path, label: str) -> None:
    if dest.exists():
        print(f"  ✅ Already downloaded: {dest.name}")
        return
    print(f"  ⬇️  Downloading {label} ...")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)
    print(f"  ✅ Saved to {dest}")

def extract_zip(zip_path: Path, extract_to: Path) -> None:
    if not zip_path.exists():
        print(f"  ⚠️  Zip not found: {zip_path}")
        return
    print(f"  📦 Extracting {zip_path.name} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)
    print(f"  ✅ Extracted.")

# ── Main setup ────────────────────────────────────────────────────────────────
def setup_dataset(download_train_images: bool = False):
    """
    Args:
        download_train_images: Set True only if you want the full 18 GB train set.
                               Val images (1 GB) are always downloaded.
    """
    print("\n🚀 MS-COCO Dataset Setup")
    print("=" * 60)

    # 1. Annotations (captions, instances, etc.)
    print("\n📄 Step 1 — Downloading Annotations ...")
    key = "annotations"
    download_file(COCO_URLS[key]["url"], COCO_URLS[key]["dest"], COCO_URLS[key]["label"])
    extract_zip(COCO_URLS[key]["dest"], DATA_DIR)

    # 2. VQA v2 data
    print("\n🧠 Step 2 — Downloading VQA v2 Data ...")
    for key in ["vqa_questions_train", "vqa_annotations_train",
                "vqa_questions_val",   "vqa_annotations_val"]:
        download_file(COCO_URLS[key]["url"], COCO_URLS[key]["dest"], COCO_URLS[key]["label"])
        extract_zip(COCO_URLS[key]["dest"], ANN_DIR)

    # 3. Val images (always)
    print("\n🖼️  Step 3 — Downloading Val Images (1 GB) ...")
    key = "val2017"
    download_file(COCO_URLS[key]["url"], COCO_URLS[key]["dest"], COCO_URLS[key]["label"])
    extract_zip(COCO_URLS[key]["dest"], IMAGES_DIR)

    # 4. Train images (optional)
    if download_train_images:
        print("\n🖼️  Step 4 — Downloading Train Images (18 GB) ...")
        key = "train2017"
        download_file(COCO_URLS[key]["url"], COCO_URLS[key]["dest"], COCO_URLS[key]["label"])
        extract_zip(COCO_URLS[key]["dest"], IMAGES_DIR)
    else:
        print("\n⏭️  Skipping train images (set download_train_images=True to include)")

    # 5. Verify
    print("\n🔍 Step 5 — Verifying Setup ...")
    verify_setup()

def verify_setup():
    checks = {
        "Captions train"   : ANN_DIR / "captions_train2017.json",
        "Captions val"     : ANN_DIR / "captions_val2017.json",
        "Instances train"  : ANN_DIR / "instances_train2017.json",
        "Instances val"    : ANN_DIR / "instances_val2017.json",
        "Val images folder": IMAGES_DIR / "val2017",
    }
    all_ok = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✅" if exists else "❌"
        print(f"  {status}  {name}: {path}")
        if not exists:
            all_ok = False

    # Count val images
    val_img_dir = IMAGES_DIR / "val2017"
    if val_img_dir.exists():
        count = len(list(val_img_dir.glob("*.jpg")))
        print(f"\n  📸 Val images found: {count:,} / 5,000 expected")

    if all_ok:
        print("\n🎉 Dataset setup complete! Ready to train.\n")
    else:
        print("\n⚠️  Some files are missing. Re-run this script.\n")

# ── Quick caption preview ──────────────────────────────────────────────────────
def preview_captions(n: int = 5):
    cap_file = ANN_DIR / "captions_val2017.json"
    if not cap_file.exists():
        print("❌ Captions file not found. Run setup_dataset() first.")
        return
    with open(cap_file) as f:
        data = json.load(f)
    print(f"\n📝 Sample Captions from COCO val2017 (showing {n}):\n")
    for ann in data["annotations"][:n]:
        img_id = ann["image_id"]
        caption = ann["caption"]
        print(f"  Image ID {img_id}: {caption}")

if __name__ == "__main__":
    # Change download_train_images=True if you want the full 18GB train set
    setup_dataset(download_train_images=False)
    preview_captions()