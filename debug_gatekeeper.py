import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model("gatekeeper_leaf_filter.h5")
# Look at the Dense layer weights
dense_layer = model.get_layer("leaf_probability")
weights, bias = dense_layer.get_weights()
print(f"Weights shape: {weights.shape}, Bias: {bias}")
print(f"Weights min: {weights.min():.4f}, max: {weights.max():.4f}, mean: {weights.mean():.4f}, std: {weights.std():.4f}")

gap_model = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("gap").output)

def get_gap(p):
    with Image.open(p) as img:
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        norm = preprocess_input(arr)
        gap = gap_model.predict(np.expand_dims(norm, 0))[0]
    logit = np.dot(gap, weights)[0] + bias[0]
    sigmoid = 1.0 / (1.0 + np.exp(-logit))
    return gap, logit, sigmoid

p_leaf = "Gatekeeper_Dataset/train/Leaf/leaf_train_000.jpg"
p_not_leaf = "Gatekeeper_Dataset/train/Not_Leaf/not_leaf_train_000.jpg"
p_euro1 = r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg"

_, l_leaf, s_leaf = get_gap(p_leaf)
_, l_not, s_not = get_gap(p_not_leaf)
_, l_euro, s_euro = get_gap(p_euro1)

print(f"Leaf train 000: Logit = {l_leaf:.4f}, Sigmoid = {s_leaf:.4f}")
print(f"Not Leaf train 000: Logit = {l_not:.4f}, Sigmoid = {s_not:.4f}")
print(f"Euro Green 1: Logit = {l_euro:.4f}, Sigmoid = {s_euro:.4f}")
