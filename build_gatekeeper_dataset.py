# -*- coding: utf-8 -*-
"""
==============================================================================
 AgriTech: Gatekeeper Dataset Builder (build_gatekeeper_dataset.py)
==============================================================================
 Automates the creation of the Binary Gatekeeper Dataset:
   Gatekeeper_Dataset/
   ├── train/
   │   ├── Leaf/     (200 samples)
   │   └── Not_Leaf/ (200 samples)
   └── val/
       ├── Leaf/     (50 samples)
       └── Not_Leaf/ (50 samples)
==============================================================================
"""

import os
import shutil
import random
from PIL import Image, ImageEnhance, ImageFilter

random.seed(42)

OUTPUT_DIR = "Gatekeeper_Dataset"
LEAF_SOURCE_TRAIN = r"D:\Coffee_Dataset_Split\train"
LEAF_SOURCE_VAL = r"D:\Coffee_Dataset_Split\val"
NOT_LEAF_SOURCES = [
    r"D:\probe_foreign",
    r"D:\Coffee_OOD_Run\probe_foreign"
]

def collect_images_from_subdirs(root_dir):
    images = []
    if not os.path.exists(root_dir):
        return images
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
                images.append(os.path.join(dirpath, f))
    return images

def augment_non_leaf(img_path, out_path, variant_idx):
    """Generates synthetic variations of non-leaf images to enrich dataset."""
    with Image.open(img_path) as img:
        img = img.convert('RGB')
        w, h = img.size
        
        # Variations: Rotate, Zoom, Color jitter, Crop
        if variant_idx % 4 == 0:
            img = img.rotate(random.choice([90, 180, 270]))
        elif variant_idx % 4 == 1:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.6, 1.4))
        elif variant_idx % 4 == 2:
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(random.uniform(0.4, 1.6))
        else:
            box = (int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9))
            img = img.crop(box)
            
        img = img.resize((224, 224), Image.BILINEAR)
        img.save(out_path, quality=90)

def main():
    print("=" * 70)
    print("🌿 Building Gatekeeper Dataset Automatically...")
    print("=" * 70)

    train_leaf_dir = os.path.join(OUTPUT_DIR, "train", "Leaf")
    train_not_leaf_dir = os.path.join(OUTPUT_DIR, "train", "Not_Leaf")
    val_leaf_dir = os.path.join(OUTPUT_DIR, "val", "Leaf")
    val_not_leaf_dir = os.path.join(OUTPUT_DIR, "val", "Not_Leaf")

    for d in [train_leaf_dir, train_not_leaf_dir, val_leaf_dir, val_not_leaf_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Populate Leaf/ from existing coffee dataset
    print("[1/2] Sampling coffee leaves for Leaf/ class...")
    train_leaves = collect_images_from_subdirs(LEAF_SOURCE_TRAIN)
    val_leaves = collect_images_from_subdirs(LEAF_SOURCE_VAL)

    if not train_leaves and os.path.exists(r"D:\Coffee_OOD_Run\Coffee_Dataset_Split\test"):
        train_leaves = collect_images_from_subdirs(r"D:\Coffee_OOD_Run\Coffee_Dataset_Split\test")
        val_leaves = train_leaves[200:]
        train_leaves = train_leaves[:200]

    random.shuffle(train_leaves)
    random.shuffle(val_leaves)

    for i, src in enumerate(train_leaves[:200]):
        shutil.copy2(src, os.path.join(train_leaf_dir, f"leaf_train_{i:03d}.jpg"))

    for i, src in enumerate(val_leaves[:50]):
        shutil.copy2(src, os.path.join(val_leaf_dir, f"leaf_val_{i:03d}.jpg"))

    print(f"      ✓ Copied 200 train leaves and 50 val leaves.")

    # 2. Populate Not_Leaf/ from foreign/OOD images with augmentation
    print("[2/2] Generating diverse non-botanical samples for Not_Leaf/ class...")
    seed_not_leaves = []
    for s_dir in NOT_LEAF_SOURCES:
        if os.path.exists(s_dir):
            for f in os.listdir(s_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.jfif')):
                    seed_not_leaves.append(os.path.join(s_dir, f))

    seed_not_leaves = list(set(seed_not_leaves))
    print(f"      Found {len(seed_not_leaves)} seed non-leaf foreign images (logos, clocks, objects, humans).")

    if seed_not_leaves:
        # Generate 200 train samples
        for i in range(200):
            src = seed_not_leaves[i % len(seed_not_leaves)]
            out_file = os.path.join(train_not_leaf_dir, f"not_leaf_train_{i:03d}.jpg")
            augment_non_leaf(src, out_file, i)

        # Generate 50 val samples
        for i in range(50):
            src = seed_not_leaves[(i + 7) % len(seed_not_leaves)]
            out_file = os.path.join(val_not_leaf_dir, f"not_leaf_val_{i:03d}.jpg")
            augment_non_leaf(src, out_file, i + 99)

        print(f"      ✓ Generated 200 train and 50 val non-leaf samples.")

    print("\n" + "=" * 70)
    print(f"✅ Gatekeeper Dataset successfully built at: '{os.path.abspath(OUTPUT_DIR)}'")
    print("=" * 70)

if __name__ == "__main__":
    main()
