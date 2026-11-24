import os
import glob
import numpy as np
from train_gatekeeper_and_pipeline import predict_with_gatekeeper

test_cases = [
    # 1. Real Test Set Coffee Leaves across classes
    ("Test Fungal Disease Leaf 1", glob.glob(r"D:\Coffee_Dataset_4Class\test\Fungal_Disease\*")[0]),
    ("Test Fungal Disease Leaf 2", glob.glob(r"D:\Coffee_Dataset_4Class\test\Fungal_Disease\*")[1]),
    ("Test Healthy Leaf 1", glob.glob(r"D:\Coffee_Dataset_4Class\test\Healthy\*")[0]),
    ("Test Miner Leaf 1", glob.glob(r"D:\Coffee_Dataset_4Class\test\Miner\*")[0]),
    ("Test Phoma Leaf 1", glob.glob(r"D:\Coffee_Dataset_4Class\test\Phoma\*")[0]),
    
    # 2. Foreign Out-of-Distribution (OOD) objects
    ("Foreign: 2.png", r"D:\probe_foreign\2.png"),
    ("Foreign: 1764100816328~2.jpg", r"D:\probe_foreign\1764100816328~2.jpg"),
    ("Foreign: download (6).jfif", r"D:\probe_foreign\download (6).jfif"),
    
    # 3. Web / Screen crop Euro Green
    ("Euro Green (Web Screenshot Crop)", r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg")
]

print("=" * 80)
print(f"{'Sample Label':<30} | {'Stage':<22} | {'Decision':<15} | {'Score / Conf'}")
print("=" * 80)

for label, path in test_cases:
    if os.path.exists(path):
        res = predict_with_gatekeeper(path)
        status = res["status"]
        stage = res["stage"]
        decision = res.get("decision", "N/A")
        if status == "Accepted":
            score_str = f"Conf: {res['confidence_percent']} (GK: {res['gatekeeper_score']:.3f})"
        elif stage == "Gatekeeper_OOD_Filter":
            score_str = f"GK Leaf Prob: {res['gatekeeper_score_percent']}"
        else:
            score_str = f"Max Conf: {res.get('confidence_percent', 'N/A')}"
        print(f"{label:<30} | {stage:<22} | {decision:<15} | {score_str}")
        print(f"   Message: {res['message']}")
        if "probabilities" in res:
            probs = res["probabilities"]
            prob_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in probs.items()])
            print(f"   Probabilities: {prob_str}")
        print("-" * 80)
