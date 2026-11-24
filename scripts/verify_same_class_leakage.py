import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import hardening_layer_v3 as hl

OUTPUT_DIR = r"D:\Coffee_Dataset_Split"
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def get_hashes(s_dir):
    data = []
    for c in CLASSES:
        c_dir = os.path.join(s_dir, c)
        if os.path.exists(c_dir):
            for f in os.listdir(c_dir):
                if f.lower().endswith(VALID_EXTS):
                    fp = os.path.join(c_dir, f)
                    data.append((c, f, fp))
    
    def worker(item):
        c, f, fp = item
        return c, f, hl.compute_perceptual_hash(fp)
        
    with ThreadPoolExecutor(max_workers=8) as ex:
        res = list(ex.map(worker, data))
    return res

train_hashes = get_hashes(os.path.join(OUTPUT_DIR, "train"))
val_hashes = get_hashes(os.path.join(OUTPUT_DIR, "val"))
test_hashes = get_hashes(os.path.join(OUTPUT_DIR, "test"))

print(f"Loaded hashes: Train={len(train_hashes)}, Val={len(val_hashes)}, Test={len(test_hashes)}")

# 1. Within-class cross-split check (CRITICAL FOR LEAKAGE)
same_class_leaks = []
for split1_name, split1_data, split2_name, split2_data in [
    ("train", train_hashes, "val", val_hashes),
    ("train", train_hashes, "test", test_hashes),
    ("val", val_hashes, "test", test_hashes)
]:
    for c1, f1, h1 in split1_data:
        for c2, f2, h2 in split2_data:
            if c1 == c2:
                dist = bin(h1 ^ h2).count("1")
                if dist <= 4:
                    same_class_leaks.append((split1_name, split2_name, c1, f1, f2, dist))

print("\n" + "="*75)
print(f"1. WITHIN-CLASS CROSS-SPLIT LEAKAGE (Same class in Train and Val/Test):")
print("="*75)
print(f"Total Same-Class Leaks found (Hamming <= 4): {len(same_class_leaks)}")
if same_class_leaks:
    for s1, s2, c, f1, f2, d in same_class_leaks:
        print(f"  ❌ {s1}/{c}/{f1} <===> {s2}/{c}/{f2} (Hamming={d})")
else:
    print("  ✅ ZERO (0) SAME-CLASS LEAKS DETECTED ACROSS ALL SPLITS!")

# 2. Cross-class collision check (different classes, incidental visual similarity)
diff_class_collisions = []
for split1_name, split1_data, split2_name, split2_data in [
    ("train", train_hashes, "val", val_hashes),
    ("train", train_hashes, "test", test_hashes),
    ("val", val_hashes, "test", test_hashes)
]:
    for c1, f1, h1 in split1_data:
        for c2, f2, h2 in split2_data:
            if c1 != c2:
                dist = bin(h1 ^ h2).count("1")
                if dist <= 4:
                    diff_class_collisions.append((split1_name, split2_name, c1, c2, f1, f2, dist))

print("\n" + "="*75)
print(f"2. CROSS-CLASS COLLISION (Different classes having similar global leaf outline):")
print("="*75)
print(f"Total Different-Class collisions (Hamming <= 4): {len(diff_class_collisions)}")
for s1, s2, c1, c2, f1, f2, d in diff_class_collisions:
    print(f"  ℹ️ {s1}/{c1}/{f1} vs {s2}/{c2}/{f2} (Hamming={d})")

# Let's inspect the files in diff_class_collisions to see what they are!
print("\n--- Inspecting the 4 cross-class collision pairs ---")
for s1, s2, c1, c2, f1, f2, d in diff_class_collisions:
    print(f"Pair: {c1} ({f1}) vs {c2} ({f2}), distance={d}")
