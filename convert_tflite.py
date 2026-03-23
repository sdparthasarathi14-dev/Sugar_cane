import tensorflow as tf
import os

model_path = os.path.join("outputs", "models", "hybrid_final.keras")
tflite_path = os.path.join("outputs", "models", "hybrid_final.tflite")

print(f"Loading {model_path}...")
model = tf.keras.models.load_model(model_path)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open(tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model to {tflite_path}")
