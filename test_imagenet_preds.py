import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Check standard MobileNetV2 with ImageNet top
img_net_model = tf.keras.applications.MobileNetV2(weights="imagenet")

def check_imagenet(p):
    with Image.open(p) as img:
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        norm = preprocess_input(arr)
        preds = img_net_model.predict(np.expand_dims(norm, 0))
        top3 = decode_predictions(preds, top=3)[0]
        print(f"\nImage: {p[-35:]}")
        for _, name, score in top3:
            print(f"  {name}: {score:.4f}")

check_imagenet(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg")
check_imagenet(r"Gatekeeper_Dataset/train/Leaf/leaf_train_000.jpg")
check_imagenet(r"D:\probe_foreign\2.png")
