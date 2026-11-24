# -*- coding: utf-8 -*-
"""
Build Final Dataset:
1. Deduplicate D:\JMuBEN_Combined -> D:\JMuBEN_Dedup (using pHash threshold <= 4)
2. Merge D:\JMuBEN_Dedup + D:\BRACOL_Sorted -> D:\Coffee_Dataset_Final
   - Prefixes: JM_ and BR_
   - Folder mapping:
     * Cerscospora / Cercospora -> Cerscospora
     * Healthy -> Healthy
     * Leaf rust / Rust -> Leaf rust
     * Miner -> Miner
     * Phoma / Brown_leaf_spot -> Phoma
"""

import os
import sys
import shutil
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl
from analyze_remaining_classes import deduplicate_pigeonhole, process_image

JMUBEN_SRC = r"D:\JMuBEN_Combined"
JMUBEN_DEDUP = r"D:\JMuBEN_Dedup"
BRACOL_SRC = r"D:\BRACOL_Sorted"
FINAL_DIR = r"D:\Coffee_Dataset_Final"

CLASSES_5 = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]


def step1_create_jmuben_dedup():
    print("=" * 70)
    print("🚀 Step 1: Creating D:\\JMuBEN_Dedup with strict pHash <= 4...")
    print("=" * 70)
    os.makedirs(JMUBEN_DEDUP, exist_ok=True)
    
    jm_classes = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
    counts = {}

    for cls in jm_classes:
        src_cls_dir = os.path.join(JMUBEN_SRC, cls)
        dst_cls_dir = os.path.join(JMUBEN_DEDUP, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)
        
        files = [
            os.path.join(src_cls_dir, f)
            for f in os.listdir(src_cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        print(f"Processing JMuBEN [{cls}] ({len(files)} files)...")
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(process_image, files))
            
        valid = [r for r in res if r[3] is None]
        phashes = [r[2] for r in valid]
        acc_idx, _ = deduplicate_pigeonhole(phashes, 4)
        
        # Copy accepted unique files
        for i in acc_idx:
            src_fp = valid[i][0]
            fname = os.path.basename(src_fp)
            dst_fp = os.path.join(dst_cls_dir, fname)
            shutil.copy2(src_fp, dst_fp)
            
        counts[cls] = len(acc_idx)
        print(f"  ✓ JMuBEN [{cls}]: {len(acc_idx)} unique files copied to {dst_cls_dir}")

    print("\nJMuBEN Dedup Summary:", counts)
    return counts


def step2_merge_to_final():
    print("\n" + "=" * 70)
    print("🚀 Step 2: Merging JMuBEN_Dedup + BRACOL_Sorted -> Coffee_Dataset_Final...")
    print("=" * 70)
    
    for cls in CLASSES_5:
        os.makedirs(os.path.join(FINAL_DIR, cls), exist_ok=True)

    final_counts = {cls: {"JM": 0, "BR": 0, "Total": 0} for cls in CLASSES_5}

    # 1. Copy from JMuBEN_Dedup
    for cls in CLASSES_5:
        src_dir = os.path.join(JMUBEN_DEDUP, cls)
        dst_dir = os.path.join(FINAL_DIR, cls)
        files = os.listdir(src_dir)
        for f in files:
            src_path = os.path.join(src_dir, f)
            # Prefix with JM_
            dst_name = f"JM_{f}"
            dst_path = os.path.join(dst_dir, dst_name)
            shutil.copy2(src_path, dst_path)
            final_counts[cls]["JM"] += 1

    # 2. Copy from BRACOL_Sorted
    bracol_map = {
        "Cercospora": "Cerscospora",
        "Healthy": "Healthy",
        "Miner": "Miner",
        "Rust": "Leaf rust",
        "Brown_leaf_spot": "Phoma"
    }

    for br_cls, final_cls in bracol_map.items():
        src_dir = os.path.join(BRACOL_SRC, br_cls)
        dst_dir = os.path.join(FINAL_DIR, final_cls)
        files = [
            os.path.join(src_dir, f)
            for f in os.listdir(src_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]
        
        # Ensure we only copy unique at D <= 4 (especially for Rust which had 5 dups)
        with ThreadPoolExecutor(max_workers=8) as ex:
            res = list(ex.map(process_image, files))
        valid = [r for r in res if r[3] is None]
        phashes = [r[2] for r in valid]
        acc_idx, _ = deduplicate_pigeonhole(phashes, 4)
        
        for i in acc_idx:
            src_fp = valid[i][0]
            fname = os.path.basename(src_fp)
            dst_name = f"BR_{fname}"
            dst_path = os.path.join(dst_dir, dst_name)
            shutil.copy2(src_fp, dst_path)
            final_counts[final_cls]["BR"] += 1

    for cls in CLASSES_5:
        final_counts[cls]["Total"] = final_counts[cls]["JM"] + final_counts[cls]["BR"]

    return final_counts


def main():
    t0 = time.time()
    
    # Check if JMuBEN_Dedup already has files
    need_jm_dedup = True
    if os.path.exists(JMUBEN_DEDUP):
        sub_cnt = sum(len(os.listdir(os.path.join(JMUBEN_DEDUP, d))) for d in os.listdir(JMUBEN_DEDUP) if os.path.isdir(os.path.join(JMUBEN_DEDUP, d)))
        if sub_cnt >= 2700:
            print(f"D:\\JMuBEN_Dedup already exists with {sub_cnt} files. Skipping Step 1.")
            need_jm_dedup = False

    if need_jm_dedup:
        step1_create_jmuben_dedup()

    # Step 2: Merge
    final_counts = step2_merge_to_final()
    
    t1 = time.time()
    print("\n" + "=" * 80)
    print(f"{'الفئة النهائية':<18} | {'من JMuBEN':<14} | {'من BRACOL':<14} | {'المجموع النهائي':<18}")
    print("-" * 80)
    total_jm = 0
    total_br = 0
    total_all = 0
    for cls in CLASSES_5:
        jm = final_counts[cls]["JM"]
        br = final_counts[cls]["BR"]
        tot = final_counts[cls]["Total"]
        total_jm += jm
        total_br += br
        total_all += tot
        print(f"{cls:<18} | {jm:<14} | {br:<14} | {tot:<18}")
    print("=" * 80)
    print(f"{'الإجمالي الكلي':<18} | {total_jm:<14} | {total_br:<14} | {total_all:<18}")
    print("=" * 80)
    print(f"Elapsed time: {t1-t0:.2f}s")


if __name__ == "__main__":
    main()
