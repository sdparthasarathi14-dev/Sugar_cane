import cv2
import numpy as np
import tensorflow as tf

# Load your trained model
MODEL_PATH = r"c:\Users\suhas\sugarcane_pathology_detection\outputs\models\hybrid_final.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Class names (use the same order as training dataset)
CLASS_NAMES = ['Brown_rust', 'Healthy', 'Mosaic', 'RedRot', 'Smut', 'Viral_disease', 'YellowLeaf', 'sett_rot']

def preprocess_frame(frame):
    # Resize to model input size
    img = cv2.resize(frame, (224, 224))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Open laptop camera
cap = cv2.VideoCapture(0)

print("📷 Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # Preprocess the frame for prediction
    img = preprocess_frame(frame)
    preds = model.predict(img, verbose=0)
    pred_class = np.argmax(preds[0])
    confidence = np.max(preds[0]) * 100
    label = f"{CLASS_NAMES[pred_class]} ({confidence:.2f}%)"

    # Show prediction on the frame
    cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Sugarcane Disease Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
