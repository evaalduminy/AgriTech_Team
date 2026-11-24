import os
import shutil
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Add Euro Green to Gatekeeper Leaf train
shutil.copy2(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg",
             r"Gatekeeper_Dataset/train/Leaf/euro_green_1.jpg")
shutil.copy2(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg",
             r"Gatekeeper_Dataset/train/Leaf/euro_green_2.jpg")

train_datagen = ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)
train_gen = train_datagen.flow_from_directory(
    "Gatekeeper_Dataset/train",
    target_size=(224, 224),
    batch_size=32,
    classes=["Not_Leaf", "Leaf"],
    class_mode="binary",
    shuffle=True
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
out = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(base.input, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(train_gen, epochs=5, verbose=1)

def test_p(p):
    with Image.open(p) as img:
        arr = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
        norm = preprocess_input(arr)
        pred = model.predict(np.expand_dims(norm, 0))[0][0]
        print(f"{os.path.basename(p):<35} -> Sigmoid: {pred:.4f}")

print("\n--- TEST AFTER 5 EPOCHS ---")
test_p(r"Gatekeeper_Dataset/train/Leaf/leaf_train_000.jpg")
test_p(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795060575456_80bfb5e1.jpg")
test_p(r"C:\Users\Hawaa\.gemini\antigravity\brain\c1ac15a1-8b0a-4609-9797-f8640de48dff\.user_uploaded\media_1795070946739_57a5d33b.jpg")
test_p(r"D:\probe_foreign\2.png")
test_p(r"D:\probe_foreign\1764100816328~2.jpg")
test_p(r"D:\probe_foreign\download (6).jfif")
