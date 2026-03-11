import tensorflow as tf
import os

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = r"c:\Users\suhas\sugarcane_pathology_detection\data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")
TEST_DIR = os.path.join(BASE_DIR, "test")

# ----------------------------
# Parameters
# ----------------------------
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# ----------------------------
# Load datasets
# ----------------------------
print("🔄 Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# ----------------------------
# Store class names
# ----------------------------
class_names = train_ds.class_names
print("\n✅ Classes found:", class_names)

# Save class names for later use
with open("class_names.txt", "w") as f:
    for cname in class_names:
        f.write(cname + "\n")

print("📁 Saved class names to class_names.txt")

# ----------------------------
# Optimize datasets
# ----------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ----------------------------
# Normalization
# ----------------------------
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

print("\n✅ Data preprocessing complete!")
print("Train batches:", tf.data.experimental.cardinality(train_ds).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_ds).numpy())
print("Test batches:", tf.data.experimental.cardinality(test_ds).numpy())
