import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# -------- PATHS (FIXED) --------
BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "model", "cat_dog_best.keras")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
PRED_DIR = os.path.join(BASE_DIR, "predictions")

os.makedirs(PRED_DIR, exist_ok=True)


# -------- LOAD MODEL --------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")


IMG_SIZE = (224, 224)
class_names = {0: "cat", 1: "dog"}   # SAME as training


# -------- LOOP OVER TEST_IMAGES --------
for img_name in os.listdir(TEST_DIR):

    IMAGE_PATH = os.path.join(TEST_DIR, img_name)

    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    print(f"\nProcessing: {img_name}")

    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ Image not found: {IMAGE_PATH}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE)

    img_array = preprocess_input(img_resized)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    prob = pred[0][0]

    class_idx = int(prob > 0.5)
    label = class_names[class_idx]
    confidence = prob * 100 if class_idx == 1 else (1 - prob) * 100


    # -------- SHOW IMAGE --------
    plt.imshow(img_rgb)
    image_name = os.path.splitext(img_name)[0]
    plt.title(f"{image_name}\nPrediction: {label} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()


    # -------- SAVE IMAGE --------
    save_path = os.path.join(PRED_DIR, img_name)
    cv2.imwrite(save_path, img)
    print(f"📸 Saved → {save_path}")
