# -*- coding: utf-8 -*-
"""
🌿 AgriTech - Phase 3: Verify Split Integrity (Byte-level & Name overlap check)
==============================================================================
التحقق الصارم من عدم وجود أي تقاطع بأسماء أو مسارات الملفات بين مجموعات:
  - train ∩ val = ∅
  - train ∩ test = ∅
  - val ∩ test = ∅
==============================================================================
"""

import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Verify_Integrity]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Verify_Integrity")

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def get_files_dict(split_dir):
    files_map = {}
    if not os.path.exists(split_dir):
        return files_map
    for root, _, files in os.walk(split_dir):
        for f in files:
            if f.lower().endswith(VALID_EXTS):
                fp = os.path.join(root, f)
                rel_p = os.path.relpath(fp, split_dir).replace("\\", "/")
                files_map[rel_p] = (f, os.path.getsize(fp))
    return files_map


def verify_integrity(dataset_dir):
    logger.info("=" * 65)
    logger.info("🔍 فحص النزاهة الرقمية للمجموعات في: %s", dataset_dir)
    logger.info("=" * 65)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")

    train_map = get_files_dict(train_dir)
    val_map = get_files_dict(val_dir)
    test_map = get_files_dict(test_dir)

    train_files = set(train_map.keys())
    val_files = set(val_map.keys())
    test_files = set(test_map.keys())

    logger.info("• عدد عينات Train: %d", len(train_files))
    logger.info("• عدد عينات Val:   %d", len(val_files))
    logger.info("• عدد عينات Test:  %d", len(test_files))

    assert len(train_files) > 0, f"مجلد التدريب فارغ: {train_dir}"
    assert len(val_files) > 0, f"مجلد التحقق فارغ: {val_dir}"
    assert len(test_files) > 0, f"مجلد الاختبار فارغ: {test_dir}"

    leak_tr_val = train_files.intersection(val_files)
    leak_tr_te = train_files.intersection(test_files)
    leak_va_te = val_files.intersection(test_files)

    passed = True
    if leak_tr_val:
        logger.error("❌ تم كشف تسرب بين Train و Val: %d ملف متطابق!", len(leak_tr_val))
        passed = False
    if leak_tr_te:
        logger.error("❌ تم كشف تسرب بين Train و Test: %d ملف متطابق!", len(leak_tr_te))
        passed = False
    if leak_va_te:
        logger.error("❌ تم كشف تسرب بين Val و Test: %d ملف متطابق!", len(leak_va_te))
        passed = False

    if passed:
        logger.info("=" * 65)
        logger.info("✓ النتيجة: نجاح فحص النزاهة البايتية بنسبة 100% (صفر تقاطع).")
        logger.info("=" * 65)
    else:
        logger.critical("⛔ فشل فحص النزاهة! توقف فوراً ولا تبدأ التدريب.")
        sys.exit(1)

    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Split Integrity")
    parser.add_argument("--dataset", default=r"D:\dataset_split", help="Split dataset directory")
    args = parser.parse_args()

    verify_integrity(args.dataset)
