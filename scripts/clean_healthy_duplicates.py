# -*- coding: utf-8 -*-
"""
Clean 58 cross-dataset contaminated Healthy images from BRACOL
Target: D:\Coffee_Dataset_Final\Healthy
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl

HEALTHY_DIR = r"D:\Coffee_Dataset_Final\Healthy"

all_files = [f for f in os.listdir(HEALTHY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
jm_files = [f for f in all_files if f.startswith("JM_")]
br_files = [f for f in all_files if f.startswith("BR_")]

print(f"Total files in Healthy: {len(all_files)}")
print(f"  • JM_ files: {len(jm_files)}")
print(f"  • BR_ files: {len(br_files)}")

# Compute pHashes
def hash_worker(fname):
    fp = os.path.join(HEALTHY_DIR, fname)
    return fname, hl.compute_perceptual_hash(fp)

with ThreadPoolExecutor(max_workers=8) as ex:
    jm_hashes = dict(ex.map(hash_worker, jm_files))
    br_hashes = dict(ex.map(hash_worker, br_files))

# Find the BR_ files that match any JM_ file (Hamming <= 4)
br_to_delete = set()
match_details = []

for br_f, br_h in br_hashes.items():
    for jm_f, jm_h in jm_hashes.items():
        dist = bin(br_h ^ jm_h).count("1")
        if dist <= 4:
            br_to_delete.add(br_f)
            match_details.append((br_f, jm_f, dist))
            break

print(f"\nFound {len(br_to_delete)} BR_ files matching JM_ files (Hamming <= 4).")

# Delete the matching BR_ files
for br_f in sorted(br_to_delete):
    fp = os.path.join(HEALTHY_DIR, br_f)
    os.remove(fp)

# Re-check remaining files in Healthy
remaining = [f for f in os.listdir(HEALTHY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"Remaining files in Healthy: {len(remaining)}")

FINAL_DIR = r"D:\Coffee_Dataset_Final"
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]

print("\n" + "=" * 60)
print(f"{'الفئة':<18} | {'العدد المتبقي':<15}")
print("-" * 60)
total_remaining = 0
for c in CLASSES:
    c_dir = os.path.join(FINAL_DIR, c)
    cnt = len([f for f in os.listdir(c_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total_remaining += cnt
    print(f"{c:<18} | {cnt:<15}")
print("=" * 60)
print(f"{'الإجمالي الكلي':<18} | {total_remaining:<15}")
print("=" * 60)
