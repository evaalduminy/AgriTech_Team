import os
import pandas as pd

csv_path = r"D:\BRACOL_Extracted\dataset.csv"
images_dir = r"D:\BRACOL_Extracted\images"

print(f"Reading CSV from: {csv_path}")
df = pd.read_csv(csv_path)

print("\n--- Shape and Columns ---")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

print("\n--- First 5 rows ---")
print(df.head(5).to_string())

print("\n--- Summary Info ---")
print(df.info())

print("\n--- Descriptive Statistics ---")
print(df.describe().to_string())

print("\n--- Value Counts for 'predominant_stress' ---")
print(df['predominant_stress'].value_counts(dropna=False))

print("\n--- Value counts for each disease column ---")
for col in ['miner', 'rust', 'phoma', 'cercospora', 'severity']:
    if col in df.columns:
        print(f"\nValue counts for '{col}':")
        print(df[col].value_counts(dropna=False))

# Check images in directory vs ids in CSV
if os.path.exists(images_dir):
    files = os.listdir(images_dir)
    print(f"\nTotal files in images directory: {len(files)}")
    print(f"Sample 5 files: {files[:5]}")
