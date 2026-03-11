import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to your trained model
MODEL_PATH = r"c:\Users\suhas\sugarcane_pathology_detection\outputs\models\hybrid_final.keras"

# Load the trained hybrid model
model = tf.keras.models.load_model(MODEL_PATH)

# Define your class names (same order as during training)
CLASS_NAMES = ['Brown_rust', 'Healthy', 'Mosaic', 'RedRot', 'Smut', 'Viral_disease', 'YellowLeaf', 'sett_rot']

def predict_image(img_path):
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"Predicted Class: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

# Example usage:
if __name__ == "__main__":
    test_img = r"C:\Users\suhas\sugarcane_pathology_detection\src\test_images\test2.jpeg"  # change this to your test image
    predict_image(test_img)
