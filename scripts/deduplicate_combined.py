# -*- coding: utf-8 -*-
"""
🌿 AgriTech - Phase 1: Perceptual Deduplication (pHash DCT + Hamming <= 4)
==============================================================================
إزالة التكرار الإدراكي الناتج عن التعزيز المسبق (Pre-Augmentation) في مجموعة البيانات
باستخدام خوارزمية pHash (DCT 8x8) ومعايرة مسافة هامينغ <= 4.
يتم الفحص لكل فئة على حدة لإنتاج JMuBEN_Dedup النظيفة تماماً.
==============================================================================
"""

import os
import sys
import json
import time
import shutil
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Deduplication]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Deduplication")


def deduplicate_class(class_name, class_input_dir, class_output_dir, max_hamming=4, max_workers=8):
    os.makedirs(class_output_dir, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = [f for f in os.listdir(class_input_dir) if f.lower().endswith(valid_exts)]
    total_imgs = len(image_files)

    logger.info("بدء معالجة فئة [%s]: إجمالي الصور = %d", class_name, total_imgs)
    t0 = time.time()

    # 1. حساب الـ pHash بالتوازي
    full_paths = [os.path.join(class_input_dir, f) for f in image_files]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hashes = list(executor.map(hl.compute_perceptual_hash, full_paths))

    t_hash = time.time()
    logger.info("تم حساب البصمات الإدراكية لـ [%s] في %.1f ثانية.", class_name, t_hash - t0)

    # 2. الفهرسة السريعة والكشف عن التكرار عبر Pigeonhole Inverted Index
    n_chunks = 6
    chunk_bits = 10
    bucket_index = [{} for _ in range(n_chunks)]

    accepted_files = []
    accepted_hashes = []
    removed_count = 0

    for idx, (f, h) in enumerate(zip(image_files, hashes)):
        # استخراج كتل الـ 10 بت
        chunks = [(h >> (i * chunk_bits)) & 0x3FF for i in range(n_chunks)]

        # البحث في المرشحين
        candidate_indices = set()
        for i, chk in enumerate(chunks):
            if chk in bucket_index[i]:
                candidate_indices.update(bucket_index[i][chk])

        is_duplicate = False
        for c_idx in candidate_indices:
            cand_h = accepted_hashes[c_idx]
            dist = bin(h ^ cand_h).count("1")
            if dist <= max_hamming:
                is_duplicate = True
                break

        if is_duplicate:
            removed_count += 1
        else:
            # قبول الصورة وإضافتها للفهرس والمجلد
            accepted_idx = len(accepted_hashes)
            accepted_hashes.append(h)
            accepted_files.append(f)
            for i, chk in enumerate(chunks):
                bucket_index[i].setdefault(chk, []).append(accepted_idx)

            src_file = os.path.join(class_input_dir, f)
            dst_file = os.path.join(class_output_dir, f)
            try:
                os.link(src_file, dst_file)
            except (OSError, NotImplementedError):
                shutil.copy2(src_file, dst_file)

    t_dedup = time.time()
    pct_removed = (removed_count / float(total_imgs)) * 100.0 if total_imgs > 0 else 0.0

    logger.info(
        "✓ اكتملت فئة [%s]: تم الإبقاء على %d صورة | حُذف %d صورة مكررة (%.1f%%) في %.1f ثانية.",
        class_name, len(accepted_files), removed_count, pct_removed, t_dedup - t_hash
    )

    return {
        "class": class_name,
        "original_count": total_imgs,
        "dedup_count": len(accepted_files),
        "removed_count": removed_count,
        "removed_percentage": round(pct_removed, 2)
    }


def run_full_deduplication(input_dir, output_dir, max_hamming=4, max_workers=8):
    os.makedirs(output_dir, exist_ok=True)
    classes = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    logger.info("=" * 65)
    logger.info("🚀 بدء عملية إزالة التكرار الإدراكي (Phase 1 Deduplication)")
    logger.info("   المصدر: %s | المخرج: %s | العتبة D <= %d", input_dir, output_dir, max_hamming)
    logger.info("=" * 65)

    summary = {}
    total_original = 0
    total_dedup = 0
    total_removed = 0

    for c in sorted(classes):
        in_c = os.path.join(input_dir, c)
        out_c = os.path.join(output_dir, c)
        res = deduplicate_class(c, in_c, out_c, max_hamming=max_hamming, max_workers=max_workers)
        summary[c] = res
        total_original += res["original_count"]
        total_dedup += res["dedup_count"]
        total_removed += res["removed_count"]

    total_pct = (total_removed / float(total_original)) * 100.0 if total_original > 0 else 0.0

    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_hamming_distance": max_hamming,
        "total_original": total_original,
        "total_deduplicated": total_dedup,
        "total_removed": total_removed,
        "total_removed_percentage": round(total_pct, 2),
        "per_class": summary
    }

    report_path = os.path.join(output_dir, "deduplication_report.json")
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report, rf, indent=2)

    logger.info("=" * 65)
    logger.info("📊 تقرير إزالة التكرار النهائي (Phase 1 Summary):")
    logger.info("=" * 65)
    for c, data in summary.items():
        logger.info(
            "  • [%-11s]: الأصل = %-5d | بعد الحذف = %-5d | المحذوف = %-5d (%.1f%%)",
            c, data["original_count"], data["dedup_count"], data["removed_count"], data["removed_percentage"]
        )
    logger.info("=" * 65)
    logger.info(
        "✓ الإجمالي: الأصل = %d | النظيف = %d | إجمالي المحذوف = %d صورة (%.1f%%)",
        total_original, total_dedup, total_removed, total_pct
    )
    logger.info("✓ تم حفظ التقرير في: %s", report_path)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perceptual Deduplication")
    parser.add_argument("--input", default=r"D:\JMuBEN_Combined", help="Input directory")
    parser.add_argument("--output", default=r"D:\JMuBEN_Dedup", help="Output directory")
    parser.add_argument("--hamming", type=int, default=4, help="Max Hamming distance")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    args = parser.parse_args()

    run_full_deduplication(args.input, args.output, max_hamming=args.hamming, max_workers=args.workers)
