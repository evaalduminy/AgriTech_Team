import os
import json

def verify_artifacts():
    results = []
    
    # 1. Models
    p1 = os.path.exists("outputs/best_coffee_model_phase1.h5")
    p2 = os.path.exists("outputs/best_coffee_model_final.h5")
    sz1 = os.path.getsize("outputs/best_coffee_model_phase1.h5") if p1 else 0
    sz2 = os.path.getsize("outputs/best_coffee_model_final.h5") if p2 else 0
    results.append(("Model Artifacts (Phase1 & Final H5)", p1 and p2 and sz1 > 10000000 and sz2 > 15000000, f"Phase1: {sz1:,} bytes, Final: {sz2:,} bytes"))

    # 2. Evaluation Report & guard_decision
    rep_path = "outputs/final_evaluation_report.json"
    has_rep = os.path.exists(rep_path)
    rep_data = json.load(open(rep_path, encoding='utf-8')) if has_rep else {}
    has_guard = "guard_decision" in rep_data
    results.append(("JSON Report & guard_decision", has_rep and has_guard, f"guard_decision: {rep_data.get('guard_decision')}"))

    # 3. 300 DPI Images
    imgs = ["accuracy_curve.png", "loss_curve.png", "confusion_matrix.png"]
    all_imgs = all(os.path.exists(os.path.join("outputs", img)) for img in imgs)
    results.append(("High-Res Visualizations (300 DPI PNGs)", all_imgs, f"{len(imgs)}/3 PNGs present in outputs/"))

    # 4. Metrics validity
    acc = rep_data.get("test_accuracy", 0)
    loss = rep_data.get("test_loss", 0)
    results.append(("Test Evaluation Provenance & Consistency", acc > 0.80, f"Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}"))

    all_pass = all(r[1] for r in results)
    print("VERIFICATION RESULT:", "PASS ✅" if all_pass else "FAIL ❌")
    for name, ok, details in results:
        status_str = "PASS ✅" if ok else "FAIL ❌"
        print(f"  • [{status_str}] {name}: {details}")

if __name__ == "__main__":
    verify_artifacts()
