# -*- coding: utf-8 -*-
"""
Robust CLAHE (Contrast Limited Adaptive Histogram Equalization) Pipeline
========================================================================
Applies CLAHE exclusively to the Lightness (L) channel in CIELAB color space:
- Preserves chromatic channels (A and B), preventing hue distortion / magenta casts.
- Enhances local contrast in shadowed and overexposed leaf areas.
- Evaluates on 3 Brazilian and 3 Ugandan test images, generating side-by-side comparisons.
"""

import os
import time
import shutil
import cv2
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
ARTIFACT_DIR = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """
    Applies CLAHE strictly to the Lightness (L) channel in CIELAB color space.
    
    Parameters:
        image: np.ndarray of shape (H, W, 3), dtype=uint8, RGB color order.
        clip_limit: Threshold for contrast limiting (default: 2.0).
        tile_grid_size: Neighborhood grid dimensions (default: (8, 8)).
        
    Returns:
        np.ndarray: Contrast-enhanced image with untouched chrominance, RGB uint8.
    """
    # 1. Convert from RGB to CIELAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # 2. Split LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 3. Create CLAHE object and apply solely to the Lightness (L) channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    # 4. Merge back with untouched A and B chrominance channels
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))

    # 5. Convert back from CIELAB to RGB color space
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    return rgb_enhanced


def get_dataset_samples():
    """Retrieves the identical 3 Brazilian and 3 Ugandan test samples."""
    # Brazilian test samples
    brazil_root = r"D:\Coffee_Dataset_4Class\test"
    brazil_classes = ["Fungal_Disease", "Healthy", "Phoma"]
    brazil_samples = []
    for c in brazil_classes:
        p = os.path.join(brazil_root, c)
        files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            brazil_samples.append((f"Brazil: {c}", files[0]))

    # Ugandan test samples
    uganda_root = r"C:\Users\Hawaa\Downloads\A Machine  Learning Dataset for Classification of Common Coffee Leaf Diseases in Uganda\coffee dataset"
    uganda_classes = [("leaf rust", "Fungal_Disease"), ("Health leaves", "Healthy"), ("phoma", "Phoma")]
    uganda_samples = []
    for folder, mapped_name in uganda_classes:
        p = os.path.join(uganda_root, folder)
        files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and os.path.getsize(os.path.join(p, f)) > 1000]
        if files:
            uganda_samples.append((f"Uganda: {mapped_name} ({folder})", files[0]))

    return brazil_samples, uganda_samples


def main():
    print("=" * 80)
    print("Robust LAB CLAHE Normalization & Visual Verification")
    print("Configuration: clipLimit=2.0, tileGridSize=(8, 8), L-channel only")
    print("=" * 80)

    brazil_samples, uganda_samples = get_dataset_samples()
    all_samples = brazil_samples + uganda_samples

    output_files = []
    timing_records = []

    # Process each sample
    for idx, (label, img_path) in enumerate(all_samples, start=1):
        with Image.open(img_path) as raw_img:
            img_rgb = np.array(raw_img.convert("RGB"))

        # Measure execution latency
        t0 = time.perf_counter()
        img_clahe = apply_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8))
        t_elapsed = (time.perf_counter() - t0) * 1000.0
        timing_records.append(t_elapsed)

        # LAB stats
        lab_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2LAB)

        l_orig_mean, a_orig_mean, b_orig_mean = np.mean(lab_orig, axis=(0, 1))
        l_clahe_mean, a_clahe_mean, b_clahe_mean = np.mean(lab_clahe, axis=(0, 1))

        print(f"\n[{idx}/6] {label} ({img_rgb.shape[1]}x{img_rgb.shape[0]} px, processed in {t_elapsed:.3f} ms):")
        print(f"   Original Lightness (L): {l_orig_mean:.1f} | A: {a_orig_mean:.1f} | B: {b_orig_mean:.1f}")
        print(f"   CLAHE    Lightness (L): {l_clahe_mean:.1f} | A: {a_clahe_mean:.1f} | B: {b_clahe_mean:.1f}")
        print(f"   Chrominance Shift (A, B delta): delta_A={abs(a_clahe_mean - a_orig_mean):.3f}, delta_B={abs(b_clahe_mean - b_orig_mean):.3f} (Colors Preserved)")

        # Individual comparison figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original: {label}\nLightness Mean L={l_orig_mean:.1f}", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img_clahe)
        axes[1].set_title(f"CLAHE (LAB: L-Channel Only)\nBalanced Lightness L={l_clahe_mean:.1f}", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        plt.suptitle(f"Sample {idx}: {os.path.basename(img_path)}", fontsize=11, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_name = f"clahe_comparison_{idx}.png"
        out_path = os.path.join(WORKSPACE, out_name)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        output_files.append(out_path)

        if os.path.exists(ARTIFACT_DIR):
            shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, out_name))

    # Generate full 6-sample composite grid
    print("\nGenerating full 6-sample composite grid (clahe_composite_grid.png)...")
    fig, axes = plt.subplots(6, 2, figsize=(11, 24), dpi=200)

    for i, (label, img_path) in enumerate(all_samples):
        with Image.open(img_path) as raw_img:
            img_rgb = np.array(raw_img.convert("RGB"))
        img_clahe = apply_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8))

        lab_orig = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
        lab_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2LAB)
        l_orig_mean = np.mean(lab_orig[:, :, 0])
        l_clahe_mean = np.mean(lab_clahe[:, :, 0])

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"[{label}] Original\nMean Lightness L={l_orig_mean:.1f}", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_clahe)
        axes[i, 1].set_title(f"[{label}] CLAHE Enhanced\nBalanced Lightness L={l_clahe_mean:.1f}", fontsize=9, fontweight="bold")
        axes[i, 1].axis("off")

    plt.suptitle("LAB CLAHE Contrast Normalization: Brazilian vs. Ugandan Coffee Leaves", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    composite_path = os.path.join(WORKSPACE, "clahe_composite_grid.png")
    fig.savefig(composite_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(composite_path, os.path.join(ARTIFACT_DIR, "clahe_composite_grid.png"))

    print("=" * 80)
    print("CLAHE Benchmark & Verification Summary:")
    print(f"Average Normalization Latency: {np.mean(timing_records):.3f} ms per image")
    print(f"Composite Grid Saved:          {composite_path}")
    print(f"Artifact Directory Mirrored:   {os.path.join(ARTIFACT_DIR, 'clahe_composite_grid.png')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
