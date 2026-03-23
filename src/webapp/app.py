import os
import json
import numpy as np
from datetime import datetime

# Optimize TensorFlow memory usage for lightweight free tiers
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

# ---------------- Config ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))           # src/webapp
PROJECT_ROOT = os.path.normpath(os.path.join(BASE_DIR, "..", ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "outputs", "models", "hybrid_final.keras")
CLASS_JSON = os.path.join(PROJECT_ROOT, "outputs", "models", "class_names.json")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)   # model input

# ---------------- App ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# ---------------- Load model & classes ----------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded from:", MODEL_PATH)

if os.path.exists(CLASS_JSON):
    with open(CLASS_JSON, "r") as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = ['Brown_rust', 'Healthy', 'Mosaic', 'RedRot', 'Smut', 'Viral_disease', 'YellowLeaf', 'sett_rot']

# Remedies dictionary
REMEDIES = {

    "Brown_rust": """
    <ul>
        <li>Remove and destroy infected leaves to reduce spread.</li>
        <li>Avoid excess nitrogen fertilizer and improve air circulation.</li>
        <li>
            <b>🛒 Organic Neem Oil (Fungal Control):</b><br>
            <a href="https://www.amazon.in/s?k=neem+oil+for+plants" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=neem+oil+plants" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "Healthy": """
    <ul>
        <li>No disease detected. Crop is healthy.</li>
        <li>Maintain balanced irrigation and organic nutrition.</li>
        <li>
            <b>🛒 Vermicompost / Organic Growth Booster:</b><br>
            <a href="https://www.amazon.in/s?k=vermicompost" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=vermicompost" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "Mosaic": """
    <ul>
        <li>Remove infected plants immediately.</li>
        <li>Control aphids and other insect vectors.</li>
        <li>
            <b>🛒 Neem-Based Organic Insecticide:</b><br>
            <a href="https://www.amazon.in/s?k=neem+based+insecticide" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=neem+insecticide" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "RedRot": """
    <ul>
        <li>Uproot and destroy affected clumps.</li>
        <li>Treat seed setts before planting.</li>
        <li>
            <b>🛒 Trichoderma Biofungicide:</b><br>
            <a href="https://www.amazon.in/s?k=trichoderma+biofungicide" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=trichoderma+fungicide" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "Smut": """
    <ul>
        <li>Use disease-free seed setts.</li>
        <li>Maintain field sanitation and remove infected shoots.</li>
        <li>
            <b>🛒 Organic Fungicide (Preventive):</b><br>
            <a href="https://www.amazon.in/s?k=organic+fungicide+plants" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=organic+fungicide" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "Viral_disease": """
    <ul>
        <li>Remove infected plants to stop spread.</li>
        <li>Control insect vectors like aphids and whiteflies.</li>
        <li>
            <b>🛒 Organic Insect Control Solution:</b><br>
            <a href="https://www.amazon.in/s?k=organic+insecticide+plants" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=organic+insecticide" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "YellowLeaf": """
    <ul>
        <li>Improve soil nutrition and potassium levels.</li>
        <li>Avoid water stress and ensure proper drainage.</li>
        <li>
            <b>🛒 Organic Potash Fertilizer:</b><br>
            <a href="https://www.amazon.in/s?k=organic+potash+fertilizer" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=organic+potash" target="_blank">Flipkart</a>
        </li>
    </ul>
    """,

    "sett_rot": """
    <ul>
        <li>Use healthy seed setts only.</li>
        <li>Ensure proper soil drainage before planting.</li>
        <li>
            <b>🛒 Bio-Fungicide for Sett Treatment:</b><br>
            <a href="https://www.amazon.in/s?k=bio+fungicide+plants" target="_blank">Amazon</a> |
            <a href="https://www.flipkart.com/search?q=bio+fungicide" target="_blank">Flipkart</a>
        </li>
    </ul>
    """
}


# ---------------- Helpers ----------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_pil_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(pil_img):
    x = preprocess_pil_image(pil_img)
    preds = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(preds))
    conf = float(np.max(preds)) * 100.0
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    return label, conf, preds

def estimate_infection(pil_img):
    img = pil_img.convert("RGB").resize((512, 512))  # larger to compute area
    arr = np.asarray(img).astype(int)
    R = arr[..., 0]; G = arr[..., 1]; B = arr[..., 2]

    green_mask = (G > 80) & (G > R) & (G > B)
    leaf_area = green_mask.sum()
    if leaf_area == 0: leaf_area = arr.shape[0] * arr.shape[1]

    lesion_mask = ((R > G + 20) & (R > B)) | ((G < 100) & (R > 100) & (B < 120))
    infected_pixels = (lesion_mask & green_mask).sum()
    if infected_pixels == 0: infected_pixels = lesion_mask.sum()

    ratio = (infected_pixels / max(1, leaf_area)) * 100.0
    if ratio < 1.0: severity = "Low"
    elif ratio < 5.0: severity = "Moderate"
    else: severity = "High"

    return int(leaf_area), int(infected_pixels), round(ratio, 3), severity




# ---------------- Routes ----------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        pil_img = Image.open(save_path)
        label, confidence, raw_scores = predict_image(pil_img)
        leaf_area_px, infected_px, infected_ratio, severity = estimate_infection(pil_img)

        status = {
            "Severity": severity,
            "Total Number of Spots (approx)": int(max(0, infected_px // 30)),
            "Leaf Area (px)": leaf_area_px,
            "Infected Region (px)": infected_px,
            "Infected Ratio (%)": infected_ratio
        }

        remedy = REMEDIES.get(label, "No remedy available.")

      

        return render_template(
            "result.html",
            image_url=url_for("uploaded_file", filename=filename),
            disease=label,
            confidence=round(confidence, 2),
            status=status,
            remedy=remedy,
            
        )
    else:
        return "Unsupported file type", 400

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# ---------------- Run ----------------
if __name__ == "__main__":
    print("Starting Flask app... Visit http://127.0.0.1:5000")
    app.run(debug=True)
