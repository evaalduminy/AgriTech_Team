# -*- coding: utf-8 -*-
"""
Fast NumPy Gray World Normalization & Visual Verification Pipeline
==================================================================
Applies the Gray World color constancy algorithm to sample images from both
the Brazilian (source domain) and Ugandan (target domain) coffee leaf datasets.
Generates side-by-side visual comparisons and channel statistics.
"""

import os
import glob
import time
import shutil
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
ARTIFACT_DIR = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff"

def gray_world_normalize(image: np.ndarray) -> np.ndarray:
    """
    Fast, fully-vectorized NumPy implementation of the Gray World algorithm.
    
    Assumes the average reflected scene color under neutral illuminant is achromatic.
    Estimates illuminant color by calculating the spatial mean of each color channel,
    then rescales each channel so that the scene average matches the global gray average.
    
    Parameters:
        image: np.ndarray of shape (H, W, 3), dtype=uint8, RGB color order.
        
    Returns:
        np.ndarray: Color-normalized image of shape (H, W, 3), dtype=uint8.
    """
    # Convert image to float32 to prevent overflow during gain scaling
    img_float = image.astype(np.float32)

    # Compute spatial mean per channel: shape (1, 1, 3)
    channel_means = np.mean(img_float, axis=(0, 1), keepdims=True)

    # Compute global achromatic gray target: (mean_R + mean_G + mean_B) / 3.0
    gray_mean = np.mean(channel_means)

    # Calculate channel scaling gains (epsilon avoids division by zero)
    channel_gains = gray_mean / (channel_means + 1e-8)

    # Rescale channels independently and clip to valid 8-bit dynamic range [0, 255]
    normalized = np.clip(img_float * channel_gains, 0.0, 255.0).astype(np.uint8)

    return normalized


def get_dataset_samples():
    """Selects 3 representative samples from Brazil and 3 from Uganda across classes."""
    # 1. Brazilian Samples (D:\Coffee_Dataset_4Class\test)
    brazil_root = r"D:\Coffee_Dataset_4Class\test"
    brazil_classes = ["Fungal_Disease", "Healthy", "Phoma"]
    brazil_samples = []
    for c in brazil_classes:
        p = os.path.join(brazil_root, c)
        files = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            brazil_samples.append((f"Brazil: {c}", files[0]))

    # 2. Ugandan Samples
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
    print("Fast NumPy Gray World Normalization & Side-by-Side Visualization")
    print("=" * 80)

    brazil_samples, uganda_samples = get_dataset_samples()
    all_samples = brazil_samples + uganda_samples

    print(f"Selected {len(all_samples)} total benchmark samples:")
    for label, path in all_samples:
        print(f"  • {label:<32} -> {os.path.basename(path)}")
    print("-" * 80)

    output_files = []
    timing_records = []

    # Process and save each comparison
    for idx, (label, img_path) in enumerate(all_samples, start=1):
        with Image.open(img_path) as raw_img:
            img_rgb = np.array(raw_img.convert("RGB"))

        # Measure speed
        t0 = time.perf_counter()
        img_norm = gray_world_normalize(img_rgb)
        t_elapsed = (time.perf_counter() - t0) * 1000.0  # ms
        timing_records.append(t_elapsed)

        orig_means = np.mean(img_rgb, axis=(0, 1))
        norm_means = np.mean(img_norm, axis=(0, 1))

        print(f"\n[{idx}/6] {label} ({img_rgb.shape[1]}x{img_rgb.shape[0]} px, processed in {t_elapsed:.3f} ms):")
        print(f"   Original RGB Means:   R={orig_means[0]:.1f}, G={orig_means[1]:.1f}, B={orig_means[2]:.1f}")
        print(f"   Normalized RGB Means: R={norm_means[0]:.1f}, G={norm_means[1]:.1f}, B={norm_means[2]:.1f}")

        # Plot individual side-by-side
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Original: {label}\nRGB Mean: ({orig_means[0]:.1f}, {orig_means[1]:.1f}, {orig_means[2]:.1f})", fontsize=10, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(img_norm)
        axes[1].set_title(f"Gray World Normalized\nRGB Mean: ({norm_means[0]:.1f}, {norm_means[1]:.1f}, {norm_means[2]:.1f})", fontsize=10, fontweight="bold")
        axes[1].axis("off")

        plt.suptitle(f"Sample {idx}: {os.path.basename(img_path)}", fontsize=11, fontweight="bold", y=0.98)
        plt.tight_layout()

        out_name = f"gray_world_comparison_{idx}.png"
        out_path = os.path.join(WORKSPACE, out_name)
        fig.savefig(out_path, bbox_inches="tight", dpi=200)
        plt.close(fig)

        output_files.append(out_path)

        # Copy to artifact directory
        if os.path.exists(ARTIFACT_DIR):
            shutil.copy2(out_path, os.path.join(ARTIFACT_DIR, out_name))

    # Generate a comprehensive 6-row comparison grid
    print("\nGenerating full 6-sample composite grid (gray_world_composite_grid.png)...")
    fig, axes = plt.subplots(6, 2, figsize=(11, 24), dpi=200)

    for i, (label, img_path) in enumerate(all_samples):
        with Image.open(img_path) as raw_img:
            img_rgb = np.array(raw_img.convert("RGB"))
        img_norm = gray_world_normalize(img_rgb)

        orig_means = np.mean(img_rgb, axis=(0, 1))
        norm_means = np.mean(img_norm, axis=(0, 1))

        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title(f"[{label}] Original\nMean RGB: ({orig_means[0]:.1f}, {orig_means[1]:.1f}, {orig_means[2]:.1f})", fontsize=9, fontweight="bold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(img_norm)
        axes[i, 1].set_title(f"[{label}] Gray World Normalized\nMean RGB: ({norm_means[0]:.1f}, {norm_means[1]:.1f}, {norm_means[2]:.1f})", fontsize=9, fontweight="bold")
        axes[i, 1].axis("off")

    plt.suptitle("Gray World Color Constancy: Brazilian vs. Ugandan Coffee Leaves", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    composite_path = os.path.join(WORKSPACE, "gray_world_composite_grid.png")
    fig.savefig(composite_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    if os.path.exists(ARTIFACT_DIR):
        shutil.copy2(composite_path, os.path.join(ARTIFACT_DIR, "gray_world_composite_grid.png"))

    print("=" * 80)
    print("Benchmark & Generation Summary:")
    print(f"Average Normalization Latency: {np.mean(timing_records):.3f} ms per image")
    print(f"Composite Grid Saved:          {composite_path}")
    print(f"Artifact Directory Mirrored:   {os.path.join(ARTIFACT_DIR, 'gray_world_composite_grid.png')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
