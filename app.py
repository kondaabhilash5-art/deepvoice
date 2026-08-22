from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
import numpy as np
import librosa
import joblib

import os
import tempfile


# ============================================================
# PATHS
# ============================================================

# app.py is now in the ROOT of the GitHub repository

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "deepvoice_model.pkl"
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)


# ============================================================
# CONFIGURATION
# ============================================================

AI_THRESHOLD = 0.85


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

CORS(app)


# ============================================================
# LOAD MODEL
# ============================================================

print("================================")
print("DeepVoice Guard Backend")
print("================================")

print("Base directory:")
print(BASE_DIR)

print("Model path:")
print(MODEL_PATH)

print("Frontend path:")
print(FRONTEND_DIR)


try:

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]

    FEATURES = model_data["features"]

    print()
    print("Model loaded successfully")
    print(
        f"Features: {len(FEATURES)}"
    )

except Exception as e:

    print()
    print("ERROR loading model:")
    print(str(e))

    model = None
    FEATURES = []


# ============================================================
# FRONTEND
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# FRONTEND STATIC FILES
# ============================================================

@app.route("/<path:filename>")
def frontend_files(filename):

    file_path = os.path.join(
        FRONTEND_DIR,
        filename
    )

    if os.path.isfile(file_path):

        return send_from_directory(
            FRONTEND_DIR,
            filename
        )

    return jsonify({
        "success": False,
        "error": "File not found"
    }), 404


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "success": True,

        "status": "online",

        "service":
            "DeepVoice Guard",

        "model_loaded":
            model is not None,

        "features":
            len(FEATURES),

        "threshold":
            AI_THRESHOLD * 100

    })


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    # Load audio
    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(y) == 0:

        raise ValueError(
            "Audio file is empty"
        )


    # Chroma
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )


    # RMS
    rms = librosa.feature.rms(
        y=y
    )


    # Spectral centroid
    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )


    # Spectral bandwidth
    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )


    # Spectral rolloff
    rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )


    # Zero crossing rate
    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y
        )
    )


    # MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )


    # Create features
    features = {

        "chroma_stft":
            float(np.mean(chroma)),

        "rms":
            float(np.mean(rms)),

        "spectral_centroid":
            float(
                np.mean(
                    spectral_centroid
                )
            ),

        "spectral_bandwidth":
            float(
                np.mean(
                    spectral_bandwidth
                )
            ),

        "rolloff":
            float(
                np.mean(rolloff)
            ),

        "zero_crossing_rate":
            float(
                np.mean(
                    zero_crossing_rate
                )
            )
    }


    # MFCC 1-20
    for i in range(20):

        features[
            f"mfcc{i + 1}"
        ] = float(
            np.mean(
                mfcc[i]
            )
        )


    return features


# ============================================================
# PREDICTION
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    audio_path = None

    try:

        # ----------------------------------------------------
        # Check model
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "Model is not loaded"

            }), 500


        # ----------------------------------------------------
        # Check audio
        # ----------------------------------------------------

        if "audio" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "No audio file provided"

            }), 400


        audio = request.files[
            "audio"
        ]


        if not audio.filename:

            return jsonify({

                "success": False,

                "error":
                    "No file selected"

            }), 400


        # ----------------------------------------------------
        # Get extension
        # ----------------------------------------------------

        extension = os.path.splitext(
            audio.filename
        )[1].lower()

        if not extension:

            extension = ".wav"


        # ----------------------------------------------------
        # Temporary file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            audio_path = temp_file.name

            audio.save(
                audio_path
            )


        print()
        print("--------------------------------")
        print(
            "Analyzing:",
            audio.filename
        )
        print("--------------------------------")


        # ----------------------------------------------------
        # Extract features
        # ----------------------------------------------------

        features = extract_features(
            audio_path
        )


        # ----------------------------------------------------
        # DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [features]
        )


        # ----------------------------------------------------
        # Check model features
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in FEATURES

            if feature
            not in input_data.columns

        ]


        if missing_features:

            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    missing_features
                )
            )


        # Correct feature order
        input_data = input_data[
            FEATURES
        ]


        # ----------------------------------------------------
        # Prediction probabilities
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )


        # Class 0 = REAL
        # Class 1 = FAKE

        real_probability = float(
            probabilities[0]
        )

        fake_probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # Final decision
        # ----------------------------------------------------

        if (
            fake_probability
            >= AI_THRESHOLD
        ):

            prediction = "FAKE"

            display_label = (
                "AI-GENERATED"
            )

            confidence_level = (
                "HIGH"
            )

        else:

            prediction = "REAL"

            display_label = (
                "REAL"
            )

            confidence_level = (
                "MEDIUM"
            )


        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = max(
            real_probability,
            fake_probability
        )


        real_percentage = round(
            real_probability * 100,
            2
        )

        fake_percentage = round(
            fake_probability * 100,
            2
        )

        confidence_percentage = round(
            confidence * 100,
            2
        )


        # ----------------------------------------------------
        # Terminal
        # ----------------------------------------------------

        print(
            "Prediction:",
            display_label
        )

        print(
            "Real:",
            f"{real_percentage}%"
        )

        print(
            "AI:",
            f"{fake_percentage}%"
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify({

            "success":
                True,

            "prediction":
                prediction,

            "display_label":
                display_label,

            "real_probability":
                real_percentage,

            "fake_probability":
                fake_percentage,

            "confidence":
                confidence_percentage,

            "confidence_level":
                confidence_level,

            "threshold":
                AI_THRESHOLD * 100

        })


    except Exception as e:

        print()
        print("================================")
        print("ERROR")
        print("================================")
        print(str(e))


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


    finally:

        if (
            audio_path
            and
            os.path.exists(
                audio_path
            )
        ):

            try:

                os.remove(
                    audio_path
                )

            except Exception:

                pass


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    print()
    print("================================")
    print("Starting DeepVoice Guard")
    print("================================")

    print(
        f"Website port: {port}"
    )

    print(
        "AI threshold: 85%"
    )

    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
