# -*- coding: utf-8 -*-
"""
🌿 AgriTech - Phase 2: Dataset Splitting (70/15/15)
==============================================================================
تقسيم مجموعة البيانات المنقاة JMuBEN_Dedup إلى:
  - train: 70%
  - val:   15%
  - test:  15%
مع تثبيت البذرة SEED = 42 لضمان تكرارية التقسيم.
==============================================================================
"""

import os
import sys
import shutil
import argparse
import logging
import splitfolders

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Split_Dataset]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Split_Dataset")


def run_split(input_dir, output_dir, seed=42, ratio=(0.70, 0.15, 0.15)):
    logger.info("=" * 65)
    logger.info("📦 بدء تقسيم مجموعة البيانات (Phase 2 Splitting)")
    logger.info("   المصدر: %s | المخرج: %s | النسب: %s | Seed: %d",
                input_dir, output_dir, ratio, seed)
    logger.info("=" * 65)

    if os.path.exists(output_dir):
        logger.info("حذف مجلد التقسيم القديم إن وجد: %s", output_dir)
        shutil.rmtree(output_dir, ignore_errors=True)

    splitfolders.ratio(
        input_dir,
        output=output_dir,
        seed=seed,
        ratio=ratio
    )

    logger.info("=" * 65)
    logger.info("📊 إحصائيات الأقسام الناتجة (Phase 2 Split Summary):")
    logger.info("=" * 65)

    splits = ["train", "val", "test"]
    classes = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

    table_data = {}
    for s in splits:
        table_data[s] = {}
        s_path = os.path.join(output_dir, s)
        total_s = 0
        for c in classes:
            c_path = os.path.join(s_path, c)
            cnt = len([f for f in os.listdir(c_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(c_path) else 0
            table_data[s][c] = cnt
            total_s += cnt
        table_data[s]["total"] = total_s

    # طباعة الجدول
    header = f"{'Class':<12} | {'Train (70%)':<12} | {'Val (15%)':<12} | {'Test (15%)':<12} | {'Total':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    for c in classes:
        tr = table_data["train"].get(c, 0)
        va = table_data["val"].get(c, 0)
        te = table_data["test"].get(c, 0)
        tot = tr + va + te
        logger.info(f"{c:<12} | {tr:<12} | {va:<12} | {te:<12} | {tot:<10}")

    logger.info("-" * len(header))
    logger.info(
        f"{'TOTAL':<12} | {table_data['train']['total']:<12} | {table_data['val']['total']:<12} | {table_data['test']['total']:<12} | {sum(table_data[s]['total'] for s in splits):<10}"
    )
    logger.info("=" * len(header))
    return table_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Dedup Dataset")
    parser.add_argument("--input", default=r"D:\JMuBEN_Dedup", help="Input clean directory")
    parser.add_argument("--output", default=r"D:\dataset_split", help="Output split directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_split(args.input, args.output, seed=args.seed)
