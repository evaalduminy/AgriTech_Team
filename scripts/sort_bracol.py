import os
import shutil
import pandas as pd

csv_path = r"D:\BRACOL_Extracted\dataset.csv"
images_dir = r"D:\BRACOL_Extracted\images"
sorted_base_dir = r"D:\BRACOL_Sorted"

# Class mapping based on predominant_stress in BRACOL
class_mapping = {
    0: "Healthy",
    1: "Miner",
    2: "Rust",
    3: "Brown_leaf_spot",
    4: "Cercospora"
}

# Create target directories
for folder_name in class_mapping.values():
    target_dir = os.path.join(sorted_base_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# Map id to predominant_stress and check image files
id_to_stress = dict(zip(df['id'], df['predominant_stress']))

copied_counts = {name: 0 for name in class_mapping.values()}
skipped_stress_5 = []
missing_in_csv = []
missing_files = []

all_img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print(f"Total images found in extracted dir: {len(all_img_files)}")

for fname in all_img_files:
    img_id_str, _ = os.path.splitext(fname)
    if not img_id_str.isdigit():
        missing_in_csv.append(fname)
        continue
    img_id = int(img_id_str)
    
    if img_id not in id_to_stress:
        missing_in_csv.append(fname)
        continue
    
    stress = id_to_stress[img_id]
    if stress == 5:
        skipped_stress_5.append(fname)
        continue
    
    if stress in class_mapping:
        target_folder = class_mapping[stress]
        src_path = os.path.join(images_dir, fname)
        dst_path = os.path.join(sorted_base_dir, target_folder, fname)
        shutil.copy2(src_path, dst_path)
        copied_counts[target_folder] += 1
    else:
        print(f"Unexpected stress value {stress} for file {fname}")

print("\n--- Sorting Summary ---")
total_sorted = 0
for cls_name, cnt in copied_counts.items():
    print(f"Class '{cls_name}': {cnt} images copied")
    total_sorted += cnt

print(f"\nTotal sorted into 5 classes: {total_sorted}")
print(f"Images with stress=5 (Undetermined/Mixed): {len(skipped_stress_5)}")
print(f"Total images accounted for: {total_sorted + len(skipped_stress_5)} / {len(all_img_files)}")
