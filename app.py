from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import io
from PIL import Image
import requests  # make sure 'requests' is in requirements.txt

app = Flask(__name__)

# ===== Model load (lazy, Google Drive-aware) =====
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Final_Model.h5")
MODEL_URL = "https://drive.google.com/file/d/1rdGwASKMfK0YkvTzhjChbeOzk5dqD-VL/view?usp=sharing" # you already set this on Render
model = None  # will be loaded on first use


def extract_file_id(url: str) -> str:
    """
    Extract Google Drive file ID from various URL formats.
    - https://drive.google.com/file/d/<ID>/view?usp=sharing
    - https://drive.google.com/uc?export=download&id=<ID>
    """
    if "id=" in url:
        # ?id=<ID>
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "id" in qs:
            return qs["id"][0]

    if "/d/" in url:
        # /d/<ID>/
        return url.split("/d/")[1].split("/")[0]

    # If it's already just an ID, return as is
    return url


def get_confirm_token(response):
    """Get confirmation token for large Google Drive files."""
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def save_response_content(response, destination, chunk_size=32768):
    """Stream response content to a file."""
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def download_model_from_google_drive(url: str, destination: str):
    """Download large file from Google Drive handling confirm token."""
    if not url:
        raise RuntimeError("MODEL_URL is not set")

    file_id = extract_file_id(url)
    print("Using Google Drive file ID:", file_id, flush=True)

    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={"id": file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        print("Found confirm token, downloading with confirmation...", flush=True)
        params = {"id": file_id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    response.raise_for_status()
    save_response_content(response, destination)

    size_mb = os.path.getsize(destination) / (1024 * 1024)
    print(f"Model downloaded to {destination} ({size_mb:.2f} MB)", flush=True)


def download_model():
    """Download the model file from Google Drive if it does not exist."""
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        print(f"Model file already exists at: {MODEL_PATH} ({size_mb:.2f} MB)", flush=True)
        return

    print("Downloading model from Google Drive...", flush=True)
    download_model_from_google_drive(MODEL_URL, MODEL_PATH)


def get_model():
    """
    Lazily download and load the model.
    - If Final_Model.h5 is not present, download it from Google Drive.
    - Then load it once and reuse.
    """
    global model

    if model is None:
        download_model()
        print("Loading model from:", MODEL_PATH, flush=True)
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded.", flush=True)

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

    try:
        # Read uploaded image into PIL
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize((IMG_SIZE, IMG_SIZE))  # 128 x 128

        # Convert to array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        print("DEBUG img_array.shape:", img_array.shape, flush=True)

        # ----- MULTI-CLASS PREDICTION -----
        model_instance = get_model()
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

            # Demo-only "size" estimate based on confidence
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

    except Exception as e:
        # Log the error to Render logs
        print("ERROR during prediction:", repr(e), flush=True)
        return render_template(
            "index.html",
            prediction="Internal error while processing the image.",
            confidence=None,
            tumor_name=None,
            tumor_size=None,
        ), 500


if __name__ == "__main__":
    app.run(debug=True)
