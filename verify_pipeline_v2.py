"""
Standalone Cascaded Pipeline Verifier v2
Stage 1: gatekeeper_leaf_filter_v2.tflite  (Leaf vs Not_Leaf, threshold=0.50)
Stage 2: coffee_model_final.tflite          (4-class: Fungal_Disease, Healthy, Miner, Phoma, threshold=0.85)
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = r"c:\Users\Hawaa\.gemini\antigravity-ide\scratch\coffee_leaf_classifier"
GATEKEEPER_V2  = os.path.join(WORKSPACE, "gatekeeper_leaf_filter_v2.tflite")
DISEASE_MODEL  = os.path.join(WORKSPACE, "coffee_model_final.tflite")

GATEKEEPER_THRESHOLD  = 0.50
CONFIDENCE_THRESHOLD  = 0.85

CLASS_NAMES_DISEASE = ["Fungal_Disease", "Healthy", "Miner", "Phoma"]

# ── Load TFLite interpreters ───────────────────────────────────────────────────
gk_interp = tf.lite.Interpreter(model_path=GATEKEEPER_V2)
gk_interp.allocate_tensors()
gk_in  = gk_interp.get_input_details()[0]
gk_out = gk_interp.get_output_details()[0]

dm_interp = tf.lite.Interpreter(model_path=DISEASE_MODEL)
dm_interp.allocate_tensors()
dm_in  = dm_interp.get_input_details()[0]
dm_out = dm_interp.get_output_details()[0]

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_mobilenet(path, size=(224, 224)):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    x = np.array(img, dtype=np.float32)
    x = (x / 127.5) - 1.0          # MobileNetV2 preprocess_input equivalent
    return np.expand_dims(x, axis=0)

# ── Cascaded predictor ─────────────────────────────────────────────────────────
def predict_cascaded(image_path):
    if not os.path.exists(image_path):
        return {"error": f"File not found: {image_path}"}

    x = preprocess_mobilenet(image_path)

    # Stage 1 — Gatekeeper
    gk_interp.set_tensor(gk_in["index"], x)
    gk_interp.invoke()
    gk_score = float(gk_interp.get_tensor(gk_out["index"])[0][0])  # sigmoid = P(Leaf)

    if gk_score < GATEKEEPER_THRESHOLD:
        return {
            "stage":           "Stage 1 — Gatekeeper",
            "stage1_result":   "Rejected",
            "gk_score":        gk_score,
            "stage2_pred":     "N/A",
            "stage2_conf":     None,
            "final_status":    "BLOCKED (OOD)",
        }

    # Stage 2 — Disease classifier
    dm_interp.set_tensor(dm_in["index"], x)
    dm_interp.invoke()
    probs = dm_interp.get_tensor(dm_out["index"])[0]
    pred_idx  = int(np.argmax(probs))
    pred_conf = float(probs[pred_idx])
    pred_name = CLASS_NAMES_DISEASE[pred_idx] if pred_idx < len(CLASS_NAMES_DISEASE) else f"class_{pred_idx}"

    if pred_conf >= CONFIDENCE_THRESHOLD:
        final_status = f"ACCEPTED — {pred_name}"
    else:
        final_status = f"LOW CONFIDENCE ({pred_conf*100:.1f}%) — {pred_name}"

    return {
        "stage":           "Stage 2 — Disease",
        "stage1_result":   "Passed",
        "gk_score":        gk_score,
        "stage2_pred":     pred_name,
        "stage2_conf":     pred_conf,
        "final_status":    final_status,
    }

# ── Test cases ─────────────────────────────────────────────────────────────────
import glob

fungal_imgs = sorted(glob.glob(r"D:\Coffee_Dataset_4Class\test\Fungal_Disease\*"))[:5]
healthy_imgs = sorted(glob.glob(r"D:\Coffee_Dataset_4Class\test\Healthy\*"))[:3]
euro_green = [
    r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg",
    r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg",
]
ood_imgs = [
    r"D:\probe_foreign\2.png",
    r"D:\probe_foreign\1764100816328~2.jpg",
    r"D:\probe_foreign\download (6).jfif",
]

test_cases = (
    [("Fungal_Disease/" + os.path.basename(p), p) for p in fungal_imgs] +
    [("Healthy/" + os.path.basename(p), p) for p in healthy_imgs] +
    [("Euro_Green_1", euro_green[0])] +
    [("Euro_Green_2", euro_green[1])] +
    [("OOD/" + os.path.basename(p), p) for p in ood_imgs]
)

# ── Run & Print ────────────────────────────────────────────────────────────────
print("=" * 110)
print(f"{'Image':<40} | {'GK Score':>9} | {'Stage1':>9} | {'Stage2 Pred':<18} | {'Stage2 Conf':>11} | Final Status")
print("=" * 110)

all_pass = True
for label, path in test_cases:
    r = predict_cascaded(path)
    if "error" in r:
        print(f"{label:<40} | ERROR: {r['error']}")
        continue

    gk_str   = f"{r['gk_score']:.4f}"
    s1_str   = r["stage1_result"]
    s2_pred  = r["stage2_pred"]
    s2_conf  = f"{r['stage2_conf']*100:.1f}%" if r["stage2_conf"] is not None else "N/A"
    final    = r["final_status"]

    # Determine PASS/FAIL for summary
    expected_pass  = not label.startswith("OOD/")
    actually_passed = r["stage1_result"] == "Passed"
    correct = (expected_pass == actually_passed)
    if not correct:
        all_pass = False
    verdict = "✓" if correct else "✗"

    print(f"{label:<40} | {gk_str:>9} | {s1_str:>9} | {s2_pred:<18} | {s2_conf:>11} | {verdict} {final}")

print("=" * 110)
print(f"\nOverall pipeline PASS/FAIL: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
print(f"Gatekeeper threshold: {GATEKEEPER_THRESHOLD}  |  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"Gatekeeper model: {GATEKEEPER_V2}")
print(f"Disease model:    {DISEASE_MODEL}")
