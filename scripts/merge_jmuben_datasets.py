# -*- coding: utf-8 -*-
"""
🌿 AgriTech - Phase 0: Merge JMuBEN + JMuBEN2 Datasets
==============================================================================
يدمج مجموعتي البيانات JMuBEN و JMuBEN2 في مجلد موحد JMuBEN_Combined
بالهيكل المعتمد:
  - Cerscospora
  - Leaf rust
  - Miner
  - Healthy
  - Phoma
مع توحيد الأسماء ومعالجة التكرار البايتي الصريح.
==============================================================================
"""

import os
import sys
import shutil
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Merge_Datasets]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Merge_Datasets")

NAME_MAP = {
    "cerscospora": "Cerscospora",
    "cercospora": "Cerscospora",
    "leaf rust": "Leaf rust",
    "leaf_rust": "Leaf rust",
    "rust": "Leaf rust",
    "coffee___rust": "Leaf rust",
    "miner": "Miner",
    "healthy": "Healthy",
    "coffee___healthy": "Healthy",
    "phoma": "Phoma"
}

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def merge_datasets(sources, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    target_classes = ["Cerscospora", "Leaf rust", "Miner", "Healthy", "Phoma"]
    for c in target_classes:
        os.makedirs(os.path.join(output_dir, c), exist_ok=True)

    stats = {c: {"added": 0, "skipped_duplicate": 0} for c in target_classes}
    seen_files = {c: set() for c in target_classes}

    logger.info("بدء دمج مصادر البيانات في: %s", output_dir)

    for src_path, default_cls in sources:
        if not os.path.exists(src_path):
            logger.warning("المسار غير موجود، تم تخطيه: %s", src_path)
            continue

        logger.info("معالجة المصدر: %s", src_path)
        subdirs = [d for d in os.listdir(src_path) if os.path.isdir(os.path.join(src_path, d))]

        if subdirs:
            for sd in subdirs:
                norm_c = NAME_MAP.get(sd.lower().replace("_", " ").strip())
                if not norm_c:
                    continue
                sd_full = os.path.join(src_path, sd)
                _process_folder(sd_full, norm_c, output_dir, seen_files, stats)
        else:
            norm_c = NAME_MAP.get(default_cls.lower().replace("_", " ").strip())
            if norm_c:
                _process_folder(src_path, norm_c, output_dir, seen_files, stats)

    logger.info("=" * 65)
    logger.info("📊 تقرير دمج البيانات النهائي (Phase 0 Summary):")
    logger.info("=" * 65)
    total_merged = 0
    for c in target_classes:
        count = stats[c]["added"]
        total_merged += count
        logger.info("  • الفئة [%-11s]: %d صورة (تم تخطي %d مكرر)",
                    c, count, stats[c]["skipped_duplicate"])
    logger.info("=" * 65)
    logger.info("✓ الإجمالي الكلي المدمج: %d صورة في %s", total_merged, output_dir)
    return stats, total_merged


def _process_folder(folder_path, target_class, output_dir, seen_files, stats):
    target_folder = os.path.join(output_dir, target_class)
    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext not in VALID_EXTS:
                continue

            src_file = os.path.join(root, f)
            size = os.path.getsize(src_file)
            file_key = (f, size)

            if file_key in seen_files[target_class]:
                stats[target_class]["skipped_duplicate"] += 1
                continue

            seen_files[target_class].add(file_key)
            dst_file = os.path.join(target_folder, f)

            # إذا وُجد ملف بنفس الاسم ولكن بحجم مختلف، نعيد تسميته برقم تسلسلي
            if os.path.exists(dst_file):
                base_name, file_ext = os.path.splitext(f)
                idx = 1
                while os.path.exists(dst_file):
                    dst_file = os.path.join(target_folder, f"{base_name}_dup{idx}{file_ext}")
                    idx += 1

            # استخدام hardlink إن أمكن للسرعة وتوفير المساحة، وإلا copy2
            try:
                os.link(src_file, dst_file)
            except (OSError, NotImplementedError):
                shutil.copy2(src_file, dst_file)

            stats[target_class]["added"] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JMuBEN + JMuBEN2")
    parser.add_argument("--output", default=r"D:\JMuBEN_Combined", help="Output directory")
    args = parser.parse_args()

    default_sources = [
        (r"D:\dataset\JMuBEN", "JMuBEN"),
        (r"D:\tgv3zb82nd-1\Healthy-20210326T083815Z-001\Healthy", "Healthy"),
        (r"D:\tgv3zb82nd-1\Miner-20210326T082341Z-001\Miner", "Miner"),
        (r"D:\dataset\Healthy", "Healthy"),
        (r"D:\dataset\coffee___healthy", "Healthy"),
        (r"D:\dataset\coffee___rust", "Leaf rust")
    ]

    merge_datasets(default_sources, args.output)
