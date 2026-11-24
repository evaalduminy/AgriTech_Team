# -*- coding: utf-8 -*-
"""
Generate Academic Report for Thesis:
- Stratified Split Table & Percentages
- Imbalance Ratio
- Mathematical Class Weights (Formulation & Values)
"""

import os
import json

DATASET_SPLIT = r"D:\Coffee_Dataset_Split"
CLASSES = ["Cerscospora", "Healthy", "Leaf rust", "Miner", "Phoma"]

train_cnt = {c: len(os.listdir(os.path.join(DATASET_SPLIT, "train", c))) for c in CLASSES}
val_cnt = {c: len(os.listdir(os.path.join(DATASET_SPLIT, "val", c))) for c in CLASSES}
test_cnt = {c: len(os.listdir(os.path.join(DATASET_SPLIT, "test", c))) for c in CLASSES}
total_cnt = {c: train_cnt[c] + val_cnt[c] + test_cnt[c] for c in CLASSES}

N_train = sum(train_cnt.values())
N_val = sum(val_cnt.values())
N_test = sum(test_cnt.values())
N_total = sum(total_cnt.values())
K = len(CLASSES)

weights_train = {c: N_train / (K * train_cnt[c]) for c in CLASSES}
weights_total = {c: N_total / (K * total_cnt[c]) for c in CLASSES}

imb_ratio_total = total_cnt["Leaf rust"] / total_cnt["Cerscospora"]
imb_ratio_train = train_cnt["Leaf rust"] / train_cnt["Cerscospora"]

report_academic = {
    "dataset_root": DATASET_SPLIT,
    "total_samples": N_total,
    "splits": {
        "train": {"count": N_train, "pct": round(N_train / N_total * 100, 2)},
        "val": {"count": N_val, "pct": round(N_val / N_total * 100, 2)},
        "test": {"count": N_test, "pct": round(N_test / N_total * 100, 2)}
    },
    "imbalance_ratio": {
        "formula": "Count(Max_Class: Leaf rust) / Count(Min_Class: Cerscospora)",
        "ratio_total": round(imb_ratio_total, 4),
        "ratio_train": round(imb_ratio_train, 4),
        "ratio_jmuben_before_bracol": round(1038 / 80, 4)
    },
    "class_distribution": {
        c: {
            "train": train_cnt[c],
            "train_pct_within_class": round(train_cnt[c] / total_cnt[c] * 100, 2),
            "val": val_cnt[c],
            "val_pct_within_class": round(val_cnt[c] / total_cnt[c] * 100, 2),
            "test": test_cnt[c],
            "test_pct_within_class": round(test_cnt[c] / total_cnt[c] * 100, 2),
            "total": total_cnt[c],
            "prevalence_pct": round(total_cnt[c] / N_total * 100, 2),
            "weight_train": round(weights_train[c], 4),
            "weight_total": round(weights_total[c], 4)
        }
        for c in CLASSES
    }
}

out_path = os.path.join(DATASET_SPLIT, "academic_split_and_weights_report.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(report_academic, f, indent=2, ensure_ascii=False)

print(f"Report saved to {out_path}")
