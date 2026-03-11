import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ------------------------
# Paths
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs"))
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

VAL_DIR = os.path.join(DATA_DIR, "val")

# ------------------------
# Load class names
# ------------------------
with open(os.path.join(MODEL_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

num_classes = len(class_names)

# ------------------------
# Dataset loader
# ------------------------
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
).prefetch(AUTOTUNE)

# ------------------------
# Load trained model
# ------------------------
model_path = os.path.join(MODEL_DIR, "hybrid_best.keras")
print(f"📂 Loading model from: {model_path}")
model = tf.keras.models.load_model(model_path)

# ------------------------
# Predict
# ------------------------
y_true = []
y_pred = []

for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ------------------------
# Metrics
# ------------------------
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

print("\n🔢 Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
