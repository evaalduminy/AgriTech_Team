# -*- coding: utf-8 -*-
"""
Forensic Analysis of BRACOL Dataset (D:\BRACOL_Sorted)
======================================================
Analyzes:
1. Unique MD5 count (byte-level)
2. Unique pHash count at thresholds 0, 2, 4
3. Duplicate counts and Redundancy percentage at threshold <= 4
4. Checks for corrupt/zero-byte files
5. Pairwise Hamming distance metrics
"""

import os
import sys
import time
import hashlib
import json
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl

CLASSES = ["Cercospora", "Healthy", "Miner", "Rust", "Brown_leaf_spot"]
BASE_DIR = r"D:\BRACOL_Sorted"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def process_image(fp):
    try:
        if os.path.getsize(fp) == 0:
            return fp, None, None, "Zero-byte file"
        with Image.open(fp) as img:
            img.verify()
        with open(fp, "rb") as f:
            data = f.read()
            md5 = hashlib.md5(data).hexdigest()
        ph = hl.compute_perceptual_hash(fp)
        return fp, md5, ph, None
    except Exception as e:
        return fp, None, None, str(e)


def deduplicate_pigeonhole(hashes, threshold):
    """
    Deduplicates a list of 64-bit integer hashes at a given Hamming distance threshold.
    Returns:
      accepted_indices: list of indices of accepted unique images
      duplicate_matches: list of (dup_idx, matched_cand_idx, distance)
    """
    n_chunks = 8
    chunk_bits = 8
    bucket_index = [{} for _ in range(n_chunks)]

    accepted_indices = []
    accepted_hashes = []
    duplicate_matches = []

    for idx, h in enumerate(hashes):
        chunks = [(h >> (i * chunk_bits)) & 0xFF for i in range(n_chunks)]

        candidate_indices = set()
        for i, chk in enumerate(chunks):
            if chk in bucket_index[i]:
                candidate_indices.update(bucket_index[i][chk])

        is_dup = False
        best_cand = None
        best_dist = 999

        for c_idx in candidate_indices:
            cand_h = accepted_hashes[c_idx]
            dist = bin(h ^ cand_h).count("1")
            if dist < best_dist:
                best_dist = dist
                best_cand = c_idx
            if dist <= threshold:
                is_dup = True
                break

        if not is_dup and threshold >= 8:
            for c_idx, cand_h in enumerate(accepted_hashes):
                dist = bin(h ^ cand_h).count("1")
                if dist <= threshold:
                    is_dup = True
                    best_dist = dist
                    best_cand = c_idx
                    break

        if is_dup:
            duplicate_matches.append((idx, accepted_indices[best_cand], best_dist))
        else:
            accepted_idx = len(accepted_hashes)
            accepted_indices.append(idx)
            accepted_hashes.append(h)
            for i, chk in enumerate(chunks):
                bucket_index[i].setdefault(chk, []).append(accepted_idx)

    return accepted_indices, duplicate_matches


def analyze_class(class_name):
    class_dir = os.path.join(BASE_DIR, class_name)
    if not os.path.exists(class_dir):
        print(f"Directory not found: {class_dir}")
        return None

    files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTS)]
    total_files = len(files)

    print(f"\n{'='*70}")
    print(f"🔬 Starting Analysis for [{class_name}] (Total files: {total_files})")
    print(f"{'='*70}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(process_image, files))
    t1 = time.time()
    print(f"Hash extraction completed in {t1-t0:.2f}s")

    valid_results = [r for r in results if r[3] is None]
    corrupt_results = [r for r in results if r[3] is not None]

    if corrupt_results:
        print(f"⚠️ Corrupt/Unreadable files found: {len(corrupt_results)}")
        for cr in corrupt_results[:5]:
            print(f"   - {os.path.basename(cr[0])}: {cr[3]}")

    valid_paths = [r[0] for r in valid_results]
    valid_md5s = [r[1] for r in valid_results]
    valid_phashes = [r[2] for r in valid_results]

    unique_md5_set = set(valid_md5s)
    unique_md5_count = len(unique_md5_set)
    print(f"1. Unique MD5 Hashes (Byte-level): {unique_md5_count} / {len(valid_results)} ({len(valid_results) - unique_md5_count} exact byte duplicates)")

    threshold_results = {}
    for t in [0, 2, 4]:
        acc_idx, dup_matches = deduplicate_pigeonhole(valid_phashes, threshold=t)
        unique_cnt = len(acc_idx)
        dropped_cnt = len(valid_phashes) - unique_cnt
        drop_pct = (dropped_cnt / len(valid_phashes)) * 100.0 if valid_phashes else 0.0
        threshold_results[t] = {
            "unique": unique_cnt,
            "dropped": dropped_cnt,
            "drop_pct": drop_pct,
            "acc_idx": acc_idx,
            "dup_matches": dup_matches
        }
        print(f"   Threshold D <= {t}: Unique = {unique_cnt}, Dropped = {dropped_cnt} ({drop_pct:.2f}% redundant)")

    # Pairwise Hamming distance on unique accepted at D <= 4
    acc_4 = threshold_results[4]["acc_idx"]
    unique_hashes = [valid_phashes[i] for i in acc_4]
    n_u = len(unique_hashes)
    if n_u > 1:
        # Sample or compute pairwise
        if n_u <= 1000:
            dists = []
            for i in range(n_u):
                for j in range(i + 1, n_u):
                    dists.append(bin(unique_hashes[i] ^ unique_hashes[j]).count("1"))
            pairwise_stats = {
                "n_unique": n_u,
                "mean": float(np.mean(dists)),
                "min": int(np.min(dists)),
                "max": int(np.max(dists))
            }
        else:
            pairwise_stats = {"n_unique": n_u, "note": "sampled"}
    else:
        pairwise_stats = {"n_unique": n_u}

    return {
        "class_name": class_name,
        "total_files": total_files,
        "valid_files": len(valid_results),
        "corrupt_files": [cr[0] for cr in corrupt_results],
        "unique_md5": unique_md5_count,
        "thresholds": {
            str(t): {
                "unique": threshold_results[t]["unique"],
                "dropped": threshold_results[t]["dropped"],
                "drop_pct": threshold_results[t]["drop_pct"]
            }
            for t in [0, 2, 4]
        },
        "pairwise_unique_d4": pairwise_stats
    }


def main():
    print("🚀 Running BRACOL Forensic Analysis on D:\\BRACOL_Sorted...")
    overall_report = {}
    for cls in CLASSES:
        res = analyze_class(cls)
        if res:
            overall_report[cls] = res

    out_json = os.path.join(os.path.dirname(__file__), "forensic_report_bracol.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(overall_report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Forensic Report saved to: {out_json}")

    print("\n" + "="*80)
    print(f"{'فئة BRACOL':<18} | {'إجمالي الصور':<14} | {'MD5 فريدة':<12} | {'pHash فريدة (≤4)':<18} | {'نسبة التكرار':<14}")
    print("-" * 80)
    for cls in CLASSES:
        rep = overall_report.get(cls)
        if rep:
            tot = rep["total_files"]
            md5 = rep["unique_md5"]
            ph4 = rep["thresholds"]["4"]["unique"]
            dup_pct = rep["thresholds"]["4"]["drop_pct"]
            print(f"{cls:<18} | {tot:<14} | {md5:<12} | {ph4:<18} | {dup_pct:.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
