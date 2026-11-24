# -*- coding: utf-8 -*-
"""
🌿 AgriTech: Phase 2 & 3 - Splitting & Post-Split Leakage Audit
==============================================================
1. Performs strict split using splitfolders.ratio (70/15/15, seed=42)
   Input:  D:\Coffee_Dataset_Final
   Output: D:\Coffee_Dataset_Split
2. Asserts Byte-level preflight integrity (assert_dataset_preflight)
3. Performs exhaustive Perceptual Leakage Audit (pHash DCT + Hamming <= 4)
   across Train vs Val, Train vs Test, and Val vs Test.
4. Generates detailed class distribution and statistical summary.
"""

import os
import sys
import shutil
import time
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import splitfolders

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl

INPUT_DIR = r"D:\Coffee_Dataset_Final"
OUTPUT_DIR = r"D:\Coffee_Dataset_Split"
SEED = 42
RATIO = (0.70, 0.15, 0.15)
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def perform_split():
    print("=" * 75)
    print("📦 Task 2: Strict Dataset Splitting (70% Train, 15% Val, 15% Test)...")
    print(f"   Source: {INPUT_DIR}")
    print(f"   Target: {OUTPUT_DIR}")
    print(f"   Seed:   {SEED}")
    print("=" * 75)

    if os.path.exists(OUTPUT_DIR):
        print(f"Removing previous split directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)

    t0 = time.time()
    splitfolders.ratio(
        INPUT_DIR,
        output=OUTPUT_DIR,
        seed=SEED,
        ratio=RATIO
    )
    t1 = time.time()
    print(f"✓ Splitting completed in {t1 - t0:.2f}s")


def audit_splits():
    print("\n" + "=" * 75)
    print("🔍 Task 3: Post-Split Leakage & Integrity Audit...")
    print("=" * 75)

    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")
    test_dir = os.path.join(OUTPUT_DIR, "test")

    # 1. Byte-level preflight assertion
    print("1. Running assert_dataset_preflight (Byte-level & relative paths)...")
    hl.assert_dataset_preflight(train_dir, val_dir, test_dir)
    print("   ✓ Byte-level preflight PASSED: 0 file name/path collisions between splits.")

    # 2. Extract pHashes for all files across all splits
    print("\n2. Computing perceptual hashes (pHash DCT 64-bit) for all split images...")
    
    def get_split_files(s_dir):
        files_list = []
        for c in CLASSES:
            c_dir = os.path.join(s_dir, c)
            if os.path.exists(c_dir):
                for f in os.listdir(c_dir):
                    if f.lower().endswith(VALID_EXTS):
                        files_list.append((c, f, os.path.join(c_dir, f)))
        return files_list

    train_files = get_split_files(train_dir)
    val_files = get_split_files(val_dir)
    test_files = get_split_files(test_dir)

    print(f"   Total images: Train = {len(train_files)} | Val = {len(val_files)} | Test = {len(test_files)}")

    def hash_file_worker(item):
        cls_name, fname, fp = item
        ph = hl.compute_perceptual_hash(fp)
        return cls_name, fname, ph

    with ThreadPoolExecutor(max_workers=8) as ex:
        train_hashes = list(ex.map(hash_file_worker, train_files))
        val_hashes = list(ex.map(hash_file_worker, val_files))
        test_hashes = list(ex.map(hash_file_worker, test_files))

    print("   ✓ Perceptual hashes extracted.")

    # 3. Exhaustive Cross-Split Perceptual Leakage Check (Hamming <= 4)
    print("\n3. Exhaustive Cross-Split Perceptual Leakage Check (Hamming Distance <= 4)...")
    
    # Check within same class (where leakage matters most) as well as across all classes
    def check_leakage(split1_name, split1_data, split2_name, split2_data, max_dist=4):
        leaks = []
        # Group by class for targeted inspection
        for c1, f1, h1 in split1_data:
            for c2, f2, h2 in split2_data:
                dist = bin(h1 ^ h2).count("1")
                if dist <= max_dist:
                    leaks.append({
                        "file1": f"{split1_name}/{c1}/{f1}",
                        "file2": f"{split2_name}/{c2}/{f2}",
                        "distance": dist,
                        "class1": c1,
                        "class2": c2
                    })
        return leaks

    t_check_0 = time.time()
    leak_train_val = check_leakage("train", train_hashes, "val", val_hashes, max_dist=4)
    leak_train_test = check_leakage("train", train_hashes, "test", test_hashes, max_dist=4)
    leak_val_test = check_leakage("val", val_hashes, "test", test_hashes, max_dist=4)
    t_check_1 = time.time()
    
    print(f"   Cross-split comparison completed in {t_check_1 - t_check_0:.2f}s")
    print(f"   • Leakage Train <-> Val:  {len(leak_train_val)}")
    print(f"   • Leakage Train <-> Test: {len(leak_train_test)}")
    print(f"   • Leakage Val <-> Test:   {len(leak_val_test)}")

    all_leaks = leak_train_val + leak_train_test + leak_val_test
    if all_leaks:
        print("\n❌ CRITICAL: Perceptual leakage detected across splits!")
        for lk in all_leaks[:10]:
            print(f"   - {lk['file1']} <-> {lk['file2']} (Hamming = {lk['distance']})")
        print("STOPPING as per instructions!")
        sys.exit(1)
    else:
        print("   ✓ ZERO PERCEPTUAL LEAKAGE DETECTED! (All pairwise distances > 4 across all splits).")

    # 4. Generate Final Breakdown Table
    table_rows = []
    print("\n" + "=" * 80)
    print(f"{'الفئة (Class)':<18} | {'Train (70%)':<14} | {'Val (15%)':<14} | {'Test (15%)':<14} | {'Total':<10}")
    print("-" * 80)

    total_tr, total_va, total_te, total_all = 0, 0, 0, 0
    for c in CLASSES:
        tr_c = sum(1 for item in train_hashes if item[0] == c)
        va_c = sum(1 for item in val_hashes if item[0] == c)
        te_c = sum(1 for item in test_hashes if item[0] == c)
        tot_c = tr_c + va_c + te_c
        total_tr += tr_c
        total_va += va_c
        total_te += te_c
        total_all += tot_c
        print(f"{c:<18} | {tr_c:<14} | {va_c:<14} | {te_c:<14} | {tot_c:<10}")

    print("=" * 80)
    print(f"{'الإجمالي الكلي':<18} | {total_tr:<14} | {total_va:<14} | {total_te:<14} | {total_all:<10}")
    print("=" * 80)

    # Save verification report
    report = {
        "status": "PASSED_100%",
        "total_images": total_all,
        "splits": {
            "train": total_tr,
            "val": total_va,
            "test": total_te
        },
        "classes": {
            c: {
                "train": sum(1 for item in train_hashes if item[0] == c),
                "val": sum(1 for item in val_hashes if item[0] == c),
                "test": sum(1 for item in test_hashes if item[0] == c),
                "total": sum(1 for item in train_hashes + val_hashes + test_hashes if item[0] == c)
            }
            for c in CLASSES
        },
        "perceptual_leakage": {
            "max_hamming_threshold": 4,
            "train_val_leaks": len(leak_train_val),
            "train_test_leaks": len(leak_train_test),
            "val_test_leaks": len(leak_val_test)
        }
    }
    
    with open(os.path.join(OUTPUT_DIR, "split_audit_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nAudit report saved to: {os.path.join(OUTPUT_DIR, 'split_audit_report.json')}")
    return report


if __name__ == "__main__":
    perform_split()
    audit_splits()
