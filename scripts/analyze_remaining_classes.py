# -*- coding: utf-8 -*-
"""
Forensic Analysis of Remaining Classes: Leaf rust, Miner, Healthy, Phoma
========================================================================
Analyzes:
1. Unique MD5 count (byte-level)
2. Unique pHash count at thresholds 0, 2, 4, 8
3. Distance distribution (histogram) between duplicates and accepted representative (at D <= 4)
4. Pairwise Hamming distances between unique images (Mean, Min, Max)
5. Dropped vs Kept counts at D <= 4
6. Identifies any corrupt files without modifying/deleting them.
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

CLASSES = ["Leaf rust", "Miner", "Healthy", "Phoma"]
BASE_DIR = r"D:\JMuBEN_Combined"
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def process_image(fp):
    try:
        if os.path.getsize(fp) == 0:
            return fp, None, None, "Zero-byte file"
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
    chunk_bits = 8  # 8 chunks of 8 bits = 64 bits
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

        # Search candidates
        for c_idx in candidate_indices:
            cand_h = accepted_hashes[c_idx]
            dist = bin(h ^ cand_h).count("1")
            if dist < best_dist:
                best_dist = dist
                best_cand = c_idx
            if dist <= threshold:
                is_dup = True
                break

        # If not found in candidates, and threshold > 0, check all if candidate set missed something (safety)
        if not is_dup and threshold >= 8:
            # 8 chunks guarantees at least 1 exact match for up to threshold = 7.
            # For threshold >= 8, fallback check on all accepted if needed:
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

    # 2. Unique pHash at thresholds 0, 2, 4, 8
    print("2. Computing pHash uniqueness across thresholds...")
    threshold_results = {}
    for t in [0, 2, 4, 8]:
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
        print(f"   • Threshold <= {t}: Unique = {unique_cnt:<6} | Dropped = {dropped_cnt:<6} ({drop_pct:.2f}%)")

    # 3. Distance distribution at D <= 4
    acc_idx_4 = threshold_results[4]["acc_idx"]
    dup_matches_4 = threshold_results[4]["dup_matches"]
    dist_hist = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for _, _, d in dup_matches_4:
        if d in dist_hist:
            dist_hist[d] += 1
        else:
            dist_hist[d] = 1

    print("\n3. Histogram of Hamming distances between duplicates & representative (D <= 4):")
    for d in sorted(dist_hist.keys()):
        pct = (dist_hist[d] / len(dup_matches_4) * 100.0) if dup_matches_4 else 0.0
        print(f"   - Distance {d}: {dist_hist[d]:<6} duplicates ({pct:.2f}%)")

    # 4. Pairwise distances between unique images (at D <= 4)
    unique_hashes = [valid_phashes[i] for i in acc_idx_4]
    n_unique = len(unique_hashes)
    print(f"\n4. Pairwise distances between the {n_unique} unique images:")
    if n_unique <= 1:
        pw_mean, pw_min, pw_max = 0.0, 0, 0
    elif n_unique <= 2000:
        # Exact all-pairs
        u_arr = np.array(unique_hashes, dtype=np.uint64)
        pairs_dists = []
        for i in range(n_unique):
            xor_vals = u_arr[i] ^ u_arr[i+1:]
            for val in xor_vals:
                pairs_dists.append(bin(int(val)).count("1"))
        pairs_dists = np.array(pairs_dists)
        pw_mean = float(np.mean(pairs_dists))
        pw_min = int(np.min(pairs_dists))
        pw_max = int(np.max(pairs_dists))
    else:
        # Sample 100,000 random pairs
        rng = np.random.RandomState(42)
        idx1 = rng.randint(0, n_unique, size=100000)
        idx2 = rng.randint(0, n_unique, size=100000)
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]
        u_arr = np.array(unique_hashes, dtype=np.uint64)
        xor_vals = u_arr[idx1] ^ u_arr[idx2]
        pairs_dists = [bin(int(val)).count("1") for val in xor_vals]
        pw_mean = float(np.mean(pairs_dists))
        pw_min = int(np.min(pairs_dists))
        pw_max = int(np.max(pairs_dists))

    print(f"   - Mean Distance : {pw_mean:.2f}")
    print(f"   - Min Distance  : {pw_min}")
    print(f"   - Max Distance  : {pw_max}")

    # Alerts
    if n_unique > 500:
        print(f"📢 ALERT: [{class_name}] has > 500 unique leaves ({n_unique} unique)!")
    elif n_unique < 50:
        print(f"🚨 WARNING: [{class_name}] has < 50 unique leaves ({n_unique} unique)!")

    return {
        "class_name": class_name,
        "total_files": total_files,
        "valid_files": len(valid_results),
        "corrupt_files": [cr[0] for cr in corrupt_results],
        "unique_md5": unique_md5_count,
        "thresholds": {
            t: {
                "unique": threshold_results[t]["unique"],
                "dropped": threshold_results[t]["dropped"],
                "drop_pct": threshold_results[t]["drop_pct"]
            }
            for t in [0, 2, 4, 8]
        },
        "dist_hist_dup_d4": dist_hist,
        "pairwise_unique_d4": {
            "n_unique": n_unique,
            "mean": pw_mean,
            "min": pw_min,
            "max": pw_max
        }
    }


def main():
    all_reports = {}
    for c in CLASSES:
        res = analyze_class(c)
        all_reports[c] = res

    # Save summary report to JSON
    report_path = r"scripts/forensic_report_remaining_classes.json"
    with open(report_path, "w", encoding="utf-8") as f:
        # strip non-serializable elements
        json.dump(all_reports, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Completed all classes. Full report saved to {report_path}")


if __name__ == "__main__":
    main()
