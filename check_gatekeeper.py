import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

model = tf.keras.models.load_model("gatekeeper_leaf_filter.h5")

def test_img(p):
    if not os.path.exists(p):
        print(f"File not found: {p}")
        return
    with Image.open(p) as img:
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        norm = preprocess_input(arr)
        pred = model.predict(np.expand_dims(norm, 0))[0][0]
        print(f"{os.path.basename(p):<35} -> Sigmoid: {pred:.6f}")

print("=== Leaf Train Images ===")
test_img(r"Gatekeeper_Dataset/train/Leaf/leaf_train_000.jpg")
test_img(r"Gatekeeper_Dataset/train/Leaf/leaf_train_001.jpg")
test_img(r"Gatekeeper_Dataset/train/Leaf/leaf_train_002.jpg")

print("\n=== Not_Leaf Train Images ===")
test_img(r"Gatekeeper_Dataset/train/Not_Leaf/not_leaf_train_000.jpg")
test_img(r"Gatekeeper_Dataset/train/Not_Leaf/not_leaf_train_001.jpg")
test_img(r"Gatekeeper_Dataset/train/Not_Leaf/not_leaf_train_002.jpg")

print("\n=== Euro Green & OOD Images ===")
test_img(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg")
test_img(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg")
test_img(r"D:\probe_foreign\2.png")
test_img(r"D:\probe_foreign\1764100816328~2.jpg")
