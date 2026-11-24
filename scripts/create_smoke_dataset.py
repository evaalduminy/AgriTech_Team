# -*- coding: utf-8 -*-
"""
Create Mini Smoke Test Dataset
Source: D:\Coffee_Dataset_Split
Target: D:\Coffee_Dataset_Smoke
Limits per class:
  - train: min(100, available)
  - val:   min(20, available)
  - test:  min(20, available)
Seed: 42
"""

import os
import shutil
import random

SRC_DIR = r"D:\Coffee_Dataset_Split"
SMOKE_DIR = r"D:\Coffee_Dataset_Smoke"
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
SPLITS_LIMITS = {
    "train": 100,
    "val": 20,
    "test": 20
}

random.seed(42)

if os.path.exists(SMOKE_DIR):
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)

counts = {s: {} for s in SPLITS_LIMITS}

for s, limit in SPLITS_LIMITS.items():
    s_src = os.path.join(SRC_DIR, s)
    s_dst = os.path.join(SMOKE_DIR, s)
    os.makedirs(s_dst, exist_ok=True)
    
    for c in CLASSES:
        c_src = os.path.join(s_src, c)
        c_dst = os.path.join(s_dst, c)
        os.makedirs(c_dst, exist_ok=True)
        
        files = [f for f in os.listdir(c_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # Sample randomly
        sample_size = min(limit, len(files))
        sampled_files = random.sample(files, sample_size)
        
        for f in sampled_files:
            shutil.copy2(os.path.join(c_src, f), os.path.join(c_dst, f))
            
        counts[s][c] = len(sampled_files)

print("=" * 70)
print(f"{'الفئة (Class)':<18} | {'Train (max 100)':<16} | {'Val (max 20)':<14} | {'Test (max 20)':<14} | {'Total':<10}")
print("-" * 70)
tot_tr = sum(counts["train"].values())
tot_va = sum(counts["val"].values())
tot_te = sum(counts["test"].values())

for c in CLASSES:
    tr = counts["train"][c]
    va = counts["val"][c]
    te = counts["test"][c]
    print(f"{c:<18} | {tr:<16} | {va:<14} | {te:<14} | {tr+va+te:<10}")

print("=" * 70)
print(f"{'TOTAL':<18} | {tot_tr:<16} | {tot_va:<14} | {tot_te:<14} | {tot_tr+tot_va+tot_te:<10}")
print("=" * 70)
