import os
import pandas as pd

csv_path = r"D:\BRACOL_Extracted\dataset.csv"
images_dir = r"D:\BRACOL_Extracted\images"

df = pd.read_csv(csv_path)
files = os.listdir(images_dir)

print(f"Total files in images dir: {len(files)}")
img_ids = []
invalid_names = []
for f in files:
    name, ext = os.path.splitext(f)
    if name.isdigit():
        img_ids.append(int(name))
    else:
        invalid_names.append(f)

print(f"Files with numeric names: {len(img_ids)}")
if invalid_names:
    print(f"Non-numeric filenames: {invalid_names[:10]}")

img_id_set = set(img_ids)
csv_id_set = set(df['id'])

common_ids = img_id_set.intersection(csv_id_set)
missing_in_images = csv_id_set - img_id_set
missing_in_csv = img_id_set - csv_id_set

print(f"Common IDs: {len(common_ids)}")
print(f"IDs in CSV but missing from images: {len(missing_in_images)}")
print(f"IDs in images but missing from CSV: {len(missing_in_csv)}")

# Filter df to only existing images
df_existing = df[df['id'].isin(img_id_set)].copy()
print(f"\nExisting images dataframe shape: {df_existing.shape}")

print("\n--- Distribution of predominant_stress in EXISTING 1,401 IMAGES ---")
print(df_existing['predominant_stress'].value_counts().sort_index())

print("\n--- Cross-tabulation of predominant_stress with diseases in existing images ---")
print(pd.crosstab(df_existing['predominant_stress'], [df_existing['miner'], df_existing['rust'], df_existing['phoma'], df_existing['cercospora']]))

print("\n--- Value counts of each binary label in existing images ---")
for col in ['miner', 'rust', 'phoma', 'cercospora']:
    print(f"{col}: {df_existing[col].value_counts().to_dict()}")

print("\n--- Severity distribution in existing images ---")
print(df_existing['severity'].value_counts().sort_index())
