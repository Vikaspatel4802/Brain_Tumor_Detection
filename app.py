from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import io
from PIL import Image
import urllib.request  # NEW: for downloading model from Drive

app = Flask(__name__)

# ===== Model load (lazy, with download support) =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Final_Model.h5")
MODEL_URL = "https://drive.google.com/uc?export=download&id=1rdGwASKMfK0YkvTzhjChbeOzk5dqD-VL"
model = None  # will be loaded on first use


def get_model():
    """
    Lazily download and load the model.
    - If Final_Model.h5 is not present, download it from MODEL_URL.
    - Then load it once and reuse.
    """
    global model

    if model is None:
        # Download if file is missing
        if not os.path.exists(MODEL_PATH):
            if not MODEL_URL:
                raise RuntimeError(
                    "MODEL_URL environment variable is not set and model file is missing."
                )

            print("Downloading model from:", MODEL_URL)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Model downloaded to:", MODEL_PATH)

        print("Loading model from:", MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.")

    return model


# Image size used during training
IMG_SIZE = 128


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="Please upload an MRI image.")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No file selected.")

    # Read uploaded image into PIL
    img = Image.open(io.BytesIO(file.read()))
    img = img.resize((IMG_SIZE, IMG_SIZE))  # 128 x 128

    # Convert to array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    print("DEBUG img_array.shape:", img_array.shape)

    # ----- MULTI-CLASS PREDICTION -----
    model_instance = get_model()           # <-- use lazy-loaded model
    preds = model_instance.predict(img_array)

    class_index = np.argmax(preds[0])
    confidence = round(np.max(preds[0]) * 100, 2)

    class_labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
    label = class_labels[class_index]

    # ===== EXTRA INFO: Tumor name & simple "size" category =====
    if label == "No Tumor":
        tumor_name = "No Tumor Detected"
        tumor_size = "N/A"
    else:
        tumor_name = f"{label} Tumor"

        # NOTE: This is NOT a real medical size calculation.
        # It’s just a rough, demo-style category based on model confidence.
        if confidence >= 85:
            tumor_size = "Large (high confidence)"
        elif confidence >= 60:
            tumor_size = "Medium"
        else:
            tumor_size = "Small / Early-stage (low confidence)"

    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        tumor_name=tumor_name,
        tumor_size=tumor_size,
    )


if __name__ == "__main__":
    app.run(debug=True)
