import os
import numpy as np
from PIL import Image

for i in range(5):
    p = f"Gatekeeper_Dataset/train/Leaf/leaf_train_{i:03d}.jpg"
    img = Image.open(p)
    arr = np.array(img)
    print(f"leaf_{i}: size={img.size}, mean RGB={arr.mean(axis=(0,1))}")

for p in [r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg",
          r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg"]:
    img = Image.open(p)
    arr = np.array(img)
    print(f"Euro Green: size={img.size}, mean RGB={arr.mean(axis=(0,1))}")
