# src/hybrid_model.py
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# ------------------------
# Config
# ------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # src/
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "data"))
OUTPUT_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "outputs"))
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25
AUTOTUNE = tf.data.AUTOTUNE

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")  # optional

# ------------------------
# Helpers: build label map & file lists
# ------------------------
def build_file_list_and_labels(folder):
    """
    Walk folder/class subfolders and return lists of file paths and integer labels.
    """
    class_names = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))])
    if not class_names:
        raise FileNotFoundError(f"No class subfolders found in {folder}.")
    label_map = {name: idx for idx, name in enumerate(class_names)}
    file_paths = []
    labels = []
    for cname in class_names:
        class_folder = os.path.join(folder, cname)
        for fname in os.listdir(class_folder):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(os.path.join(class_folder, fname))
                labels.append(label_map[cname])
    return file_paths, labels, class_names

# ------------------------
# Safe loader (skip corrupted images)
# ------------------------
def py_load_and_preprocess(path, label):
    """
    This function runs in Python (via tf.py_function). It attempts to read and decode an image.
    If successful returns (image_array, label, ok_flag=1). If fails returns (zeros, -1, ok_flag=0).
    """
    path = path.decode() if isinstance(path, bytes) else path
    try:
        img_raw = tf.io.read_file(path)
        img = tf.image.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = img.numpy().astype(np.float32) / 255.0
        return img, np.int32(label), np.int32(1)
    except Exception as e:
        # return dummy image and ok_flag 0
        # Note: we return label as -1 to mark invalid
        dummy = np.zeros((IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        return dummy, np.int32(-1), np.int32(0)

def tf_load_and_preprocess(path, label):
    """
    Wraps py_load_and_preprocess into TF graph. Returns (img, label).
    We filter out entries with label == -1 later.
    """
    img, lbl, ok = tf.py_function(func=py_load_and_preprocess, inp=[path, label],
                                  Tout=[tf.float32, tf.int32, tf.int32])
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    lbl.set_shape(())
    ok.set_shape(())
    return img, lbl, ok

# ------------------------
# Build tf.data Dataset safely
# ------------------------
def build_safe_dataset_from_directory(folder, batch_size=BATCH_SIZE, shuffle=True):
    file_paths, labels, class_names = build_file_list_and_labels(folder)
    if len(file_paths) == 0:
        raise FileNotFoundError(f"No image files found in {folder}.")

    # Create dataset of file paths & labels
    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((paths_ds, labels_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_paths), seed=42)

    # Map to (img, label, ok_flag)
    ds = ds.map(lambda p, l: tf_load_and_preprocess(p, l), num_parallel_calls=AUTOTUNE)

    # Filter out corrupted images (where label == -1 or ok_flag == 0)
    ds = ds.filter(lambda img, lbl, ok: tf.logical_and(tf.not_equal(lbl, -1), tf.equal(ok, 1)))

    # Drop ok flag now
    ds = ds.map(lambda img, lbl, ok: (img, lbl), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, class_names

# ------------------------
# Build datasets
# ------------------------
print("🔄 Building safe datasets from files...")

train_ds, train_classes = build_safe_dataset_from_directory(TRAIN_DIR, batch_size=BATCH_SIZE, shuffle=True)
val_ds, val_classes = build_safe_dataset_from_directory(VAL_DIR, batch_size=BATCH_SIZE, shuffle=False)

# sanity check classes are identical
if train_classes != val_classes:
    print("⚠️ Warning: class names differ between train and val folders.")
class_names = train_classes
num_classes = len(class_names)

# Save class names json for later inference
with open(os.path.join(MODEL_DIR, "class_names.json"), "w") as f:
    json.dump(class_names, f, indent=2)
print(f"✅ Classes ({num_classes}): {class_names}")
print("✅ Datasets built. Examples per batch:", "train batch size =", BATCH_SIZE)

# ------------------------
# Build Hybrid Model (Functional API)
# ------------------------
def build_hybrid_model(input_shape=(224, 224, 3), num_classes=num_classes):
    inputs = layers.Input(shape=input_shape, name="image_input")

    # CNN branch (light)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    cnn_feat = layers.Dense(128, activation="relu")(x)

    # DenseNet121 branch
    # Use the same input tensor for DenseNet
    base = DenseNet121(weights="imagenet", include_top=False, input_tensor=inputs)
    base.trainable = False  # freeze for initial training
    dnet_feat = layers.GlobalAveragePooling2D()(base.output)
    dnet_feat = layers.Dense(128, activation="relu")(dnet_feat)

    # Concatenate features
    combined = layers.Concatenate()([cnn_feat, dnet_feat])
    x = layers.Dense(256, activation="relu")(combined)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="Hybrid_CNN_DenseNet121")
    return model

print("🔧 Building model...")
model = build_hybrid_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# ------------------------
# Callbacks
# ------------------------
checkpoint_path = os.path.join(MODEL_DIR, "hybrid_best.keras")
checkpoint_cb = ModelCheckpoint(checkpoint_path, save_best_only=True, monitor="val_accuracy", mode="max")
earlystop_cb = EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, mode="max")
tensorboard_cb = TensorBoard(log_dir=LOG_DIR)

# ------------------------
# Train
# ------------------------
print("🚀 Starting training...")
try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[checkpoint_cb, earlystop_cb, tensorboard_cb]
    )
except tf.errors.InvalidArgumentError as e:
    print("❌ Training failed due to an InvalidArgumentError (likely a corrupt image).")
    print("Error:", e)
    print("Suggestion: run clean_dataset.py or inspect dataset files manually.")
    raise

# ------------------------
# Save final model
# ------------------------
final_model_path = os.path.join(MODEL_DIR, "hybrid_final.keras")
model.save(final_model_path)
print(f"🎉 Training complete. Final model saved to: {final_model_path}")
