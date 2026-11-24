import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl

JMUBEN_DIR = r"D:\JMuBEN_Dedup"
BRACOL_DIR = r"D:\BRACOL_Sorted"
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
BR_MAP = {
    "Cerscospora": "Cercospora",
    "Healthy": "Healthy",
    "Leaf rust": "Rust",
    "Miner": "Miner",
    "Phoma": "Brown_leaf_spot"
}

def get_hashes(base_dir, class_name):
    c_dir = os.path.join(base_dir, class_name)
    if not os.path.exists(c_dir):
        return []
    files = [os.path.join(c_dir, f) for f in os.listdir(c_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(hl.compute_perceptual_hash, files))
    return list(zip(files, res))

cross_leaks = {}
total_cross_dups = 0

print("="*75)
print("🔍 Checking cross-dataset contamination between JMuBEN_Dedup and BRACOL_Sorted...")
print("="*75)

for jm_cls, br_cls in BR_MAP.items():
    jm_data = get_hashes(JMUBEN_DIR, jm_cls)
    br_data = get_hashes(BRACOL_DIR, br_cls)
    
    dups = []
    for jm_fp, jm_h in jm_data:
        for br_fp, br_h in br_data:
            dist = bin(jm_h ^ br_h).count("1")
            if dist <= 4:
                dups.append((os.path.basename(jm_fp), os.path.basename(br_fp), dist))
                
    cross_leaks[jm_cls] = dups
    total_cross_dups += len(dups)
    print(f"• Class [{jm_cls}]: {len(dups)} cross-dataset duplicates found (Hamming <= 4)!")

print("="*75)
print(f"TOTAL CROSS-DATASET DUPLICATES BETWEEN JMuBEN AND BRACOL: {total_cross_dups}")
print("="*75)

for jm_cls, dups in cross_leaks.items():
    if dups:
        print(f"\n--- Samples in {jm_cls} (Total: {len(dups)}) ---")
        for jm_f, br_f, d in dups[:10]:
            print(f"  JM: {jm_f} <===> BR: {br_f} (Hamming = {d})")
