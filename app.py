from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import tempfile

import numpy as np
import pandas as pd
import librosa
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "deepvoice_model_v2.pkl"
)

# If your actual file is named deepvoice_model.pkl,
# change the line above to:
#
# MODEL_PATH = os.path.join(
#     BASE_DIR,
#     "model",
#     "deepvoice_model.pkl"
# )


FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

# Your existing decision rule
AI_THRESHOLD = 0.85

# Memory-friendly audio settings
TARGET_SAMPLE_RATE = 16000
MAX_AUDIO_SECONDS = 30


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

print("================================")


model = None
FEATURES = []


try:

    model_data = joblib.load(
        MODEL_PATH
    )

    # Expected structure:
    # {
    #     "model": trained_model,
    #     "features": [...]
    # }

    model = model_data["model"]

    FEATURES = model_data["features"]

    print("Model loaded successfully")

    print(
        f"Features: {len(FEATURES)}"
    )

except Exception as e:

    print("ERROR loading model:")
    print(str(e))

    model = None
    FEATURES = []


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# FRONTEND FILES
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
            AI_THRESHOLD * 100,

        "sample_rate":
            TARGET_SAMPLE_RATE,

        "max_audio_seconds":
            MAX_AUDIO_SECONDS

    })


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    print("Loading audio...")

    # --------------------------------------------------------
    # MEMORY OPTIMIZATION
    # --------------------------------------------------------
    #
    # 16 kHz is enough for speech analysis.
    #
    # duration=30 prevents very long WhatsApp recordings
    # from consuming excessive RAM.
    #

    y, sr = librosa.load(
        audio_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True,
        duration=MAX_AUDIO_SECONDS
    )

    if y is None or len(y) == 0:

        raise ValueError(
            "Audio file is empty or could not be decoded."
        )


    print(
        f"Audio loaded: {len(y)} samples"
    )

    print(
        f"Sample rate: {sr}"
    )

    print(
        f"Duration: {len(y) / sr:.2f} seconds"
    )


    # ========================================================
    # CHROMA
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )


    # ========================================================
    # RMS
    # ========================================================

    rms = librosa.feature.rms(
        y=y
    )


    # ========================================================
    # SPECTRAL CENTROID
    # ========================================================

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )


    # ========================================================
    # SPECTRAL BANDWIDTH
    # ========================================================

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )


    # ========================================================
    # SPECTRAL ROLLOFF
    # ========================================================

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )


    # ========================================================
    # ZERO CROSSING RATE
    # ========================================================

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y
        )
    )


    # ========================================================
    # MFCC
    # ========================================================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )


    # ========================================================
    # CREATE FEATURE DICTIONARY
    # ========================================================

    features = {

        "chroma_stft":
            float(
                np.mean(chroma)
            ),

        "rms":
            float(
                np.mean(rms)
            ),

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
                np.mean(
                    rolloff
                )
            ),

        "zero_crossing_rate":
            float(
                np.mean(
                    zero_crossing_rate
                )
            )

    }


    # ========================================================
    # MFCC 1-20
    # ========================================================

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
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    audio_path = None

    try:

        print()
        print("================================")
        print("NEW AUDIO REQUEST")
        print("================================")


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "Model is not loaded on the server."

            }), 500


        # ----------------------------------------------------
        # CHECK AUDIO
        # ----------------------------------------------------

        if "audio" not in request.files:

            return jsonify({

                "success": False,

                "error":
                    "No audio file provided."

            }), 400


        audio = request.files[
            "audio"
        ]


        if not audio.filename:

            return jsonify({

                "success": False,

                "error":
                    "No audio file selected."

            }), 400


        print(
            "File:",
            audio.filename
        )


        # ----------------------------------------------------
        # GET FILE EXTENSION
        # ----------------------------------------------------

        extension = os.path.splitext(
            audio.filename
        )[1].lower()


        if not extension:

            extension = ".wav"


        # ----------------------------------------------------
        # SAVE TEMPORARY AUDIO
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp_file:

            audio_path = temp_file.name

            audio.save(
                audio_path
            )


        print(
            "Temporary file created."
        )


        # ----------------------------------------------------
        # EXTRACT FEATURES
        # ----------------------------------------------------

        features = extract_features(
            audio_path
        )


        print(
            "Features extracted."
        )


        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [features]
        )


        # ----------------------------------------------------
        # CHECK FEATURES
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
                +
                ", ".join(
                    missing_features
                )
            )


        # ----------------------------------------------------
        # CORRECT FEATURE ORDER
        # ----------------------------------------------------

        input_data = input_data[
            FEATURES
        ]


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        real_probability = float(
            probabilities[0]
        )

        fake_probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # DECISION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # This remains your existing rule:
        #
        # AI >= 85%  -> FAKE
        # AI < 85%   -> REAL
        #
        # The frontend may display the larger percentage,
        # but this backend decision does NOT change.
        #

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
        # CONFIDENCE
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
        # LOG RESULT
        # ----------------------------------------------------

        print()
        print("--------------------------------")
        print("RESULT")
        print("--------------------------------")

        print(
            "Prediction:",
            display_label
        )

        print(
            "Real probability:",
            f"{real_percentage}%"
        )

        print(
            "AI probability:",
            f"{fake_percentage}%"
        )

        print(
            "Displayed confidence:",
            f"{confidence_percentage}%"
        )

        print(
            "AI threshold:",
            "85%"
        )

        print("--------------------------------")


        # ----------------------------------------------------
        # RETURN JSON
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
        print("PREDICTION ERROR")
        print("================================")

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print("================================")


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


    finally:

        # ----------------------------------------------------
        # DELETE TEMP FILE
        # ----------------------------------------------------

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

            except Exception as cleanup_error:

                print(
                    "Temporary file cleanup failed:",
                    cleanup_error
                )


# ============================================================
# RUN SERVER
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
        f"Port: {port}"
    )

    print(
        "AI threshold: 85%"
    )

    print(
        f"Sample rate: {TARGET_SAMPLE_RATE}"
    )

    print(
        f"Maximum audio: {MAX_AUDIO_SECONDS} seconds"
    )

    print("================================")


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
