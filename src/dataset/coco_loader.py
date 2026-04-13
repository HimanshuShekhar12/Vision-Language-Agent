"""
COCO Dataset Loader
===================
A unified data loader for all 5 phases that use MS-COCO.

Usage
-----
from src.dataset.coco_loader import CocoLoader

loader = CocoLoader(split="val")

# Phase 1 — Captioning
for image, captions in loader.captioning_pairs():
    ...

# Phase 2 — VQA
for image, question, answer in loader.vqa_triplets():
    ...

# Phase 3 — Generation (captions as prompts)
for prompt in loader.caption_prompts():
    ...

# Phase 4 — Agent test inputs
for image, meta in loader.agent_test_inputs():
    ...

# Phase 5 — Memory (embeddings)
for image_id, image_path in loader.image_paths():
    ...
"""

import json
import random
from pathlib import Path
from typing import Generator, Tuple, List, Optional

from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent
DATA_DIR   = BASE_DIR / "data" / "coco"
IMAGES_DIR = DATA_DIR / "images"
ANN_DIR    = DATA_DIR / "annotations"


class CocoLoader:
    """
    Lightweight COCO loader — no pycocotools dependency.
    Works across all project phases.
    """

    def __init__(self, split: str = "val", max_samples: Optional[int] = None, seed: int = 42):
        """
        Args:
            split       : "train" or "val"
            max_samples : limit dataset size (useful for quick experiments)
            seed        : random seed for reproducible sampling
        """
        assert split in ("train", "val"), "split must be 'train' or 'val'"
        self.split       = split
        self.max_samples = max_samples
        self.seed        = seed
        self.img_dir     = IMAGES_DIR / f"{split}2017"

        self._captions   = self._load_json(f"captions_{split}2017.json")
        self._instances  = self._load_json(f"instances_{split}2017.json")

        # VQA (only v2 train/val match COCO images)
        self._vqa_q = self._load_vqa_json(f"v2_OpenEnded_mscoco_{split}2014_questions.json")
        self._vqa_a = self._load_vqa_json(f"v2_mscoco_{split}2014_annotations.json")

        # Build quick lookup: image_id → file_name
        self._id_to_file = {
            img["id"]: img["file_name"]
            for img in self._captions.get("images", [])
        } if self._captions else {}

    # ── Internals ────────────────────────────────────────────────────────────
    def _load_json(self, filename: str) -> Optional[dict]:
        path = ANN_DIR / filename
        if not path.exists():
            print(f"  ⚠️  Not found (run download_coco.py first): {path.name}")
            return None
        with open(path) as f:
            return json.load(f)

    def _load_vqa_json(self, filename: str) -> Optional[dict]:
        path = ANN_DIR / filename
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def _sample(self, lst: list) -> list:
        if self.max_samples is None or len(lst) <= self.max_samples:
            return lst
        rng = random.Random(self.seed)
        return rng.sample(lst, self.max_samples)

    def _load_image(self, image_id: int) -> Optional[Image.Image]:
        fname = self._id_to_file.get(image_id)
        if fname is None:
            return None
        path = self.img_dir / fname
        if not path.exists():
            return None
        return Image.open(path).convert("RGB")

    # ── Phase 1 — Image Captioning ───────────────────────────────────────────
    def captioning_pairs(self) -> Generator[Tuple[Image.Image, List[str]], None, None]:
        """
        Yields: (PIL Image, list_of_captions)
        Groups all captions per image.
        """
        if not self._captions:
            return
        # Group captions by image_id
        from collections import defaultdict
        cap_map = defaultdict(list)
        for ann in self._captions["annotations"]:
            cap_map[ann["image_id"]].append(ann["caption"])

        image_ids = self._sample(list(cap_map.keys()))
        for image_id in image_ids:
            img = self._load_image(image_id)
            if img is None:
                continue
            yield img, cap_map[image_id]

    # ── Phase 2 — Visual Q&A ─────────────────────────────────────────────────
    def vqa_triplets(self) -> Generator[Tuple[Image.Image, str, str], None, None]:
        """
        Yields: (PIL Image, question_str, answer_str)
        Uses VQA v2 annotations.
        """
        if not self._vqa_q or not self._vqa_a:
            print("⚠️  VQA data not loaded. Run download_coco.py first.")
            return
        # Build answer lookup by question_id
        ans_map = {
            ann["question_id"]: ann["multiple_choice_answer"]
            for ann in self._vqa_a["annotations"]
        }
        questions = self._sample(self._vqa_q["questions"])
        for q in questions:
            img = self._load_image(q["image_id"])
            if img is None:
                continue
            answer = ans_map.get(q["question_id"], "unknown")
            yield img, q["question"], answer

    # ── Phase 3 — Generation (captions as prompts) ───────────────────────────
    def caption_prompts(self) -> Generator[str, None, None]:
        """
        Yields: caption strings to use as Stable Diffusion prompts.
        No image loading needed — just text.
        """
        if not self._captions:
            return
        annotations = self._sample(self._captions["annotations"])
        for ann in annotations:
            yield ann["caption"]

    # ── Phase 4 — Agent test inputs ──────────────────────────────────────────
    def agent_test_inputs(self) -> Generator[Tuple[Image.Image, dict], None, None]:
        """
        Yields: (PIL Image, metadata_dict)
        metadata includes image_id, file_name, and available captions.
        """
        if not self._captions:
            return
        from collections import defaultdict
        cap_map = defaultdict(list)
        for ann in self._captions["annotations"]:
            cap_map[ann["image_id"]].append(ann["caption"])

        images = self._sample(self._captions["images"])
        for img_info in images:
            img = self._load_image(img_info["id"])
            if img is None:
                continue
            meta = {
                "image_id"  : img_info["id"],
                "file_name" : img_info["file_name"],
                "captions"  : cap_map[img_info["id"]],
                "width"     : img_info.get("width"),
                "height"    : img_info.get("height"),
            }
            yield img, meta

    # ── Phase 5 — Memory: image paths for embedding ──────────────────────────
    def image_paths(self) -> Generator[Tuple[int, Path], None, None]:
        """
        Yields: (image_id, Path)
        Use to compute embeddings and store in ChromaDB.
        """
        if not self._id_to_file:
            return
        items = self._sample(list(self._id_to_file.items()))
        for image_id, fname in items:
            path = self.img_dir / fname
            if path.exists():
                yield image_id, path

    # ── Utility ───────────────────────────────────────────────────────────────
    def __repr__(self):
        n_images = len(self._id_to_file)
        return f"CocoLoader(split='{self.split}', images={n_images:,}, max_samples={self.max_samples})"