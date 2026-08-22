from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import tempfile

import pandas as pd
import numpy as np
import librosa
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# V2 model
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "deepvoice_model_v2.pkl"
)

# IMPORTANT:
# Your structure is:
#
# deepvoice/
# ├── app.py
# ├── requirements.txt
# ├── model/
# │   └── deepvoice_model_v2.pkl
# └── frontend/
#     ├── index.html
#     ├── script.js
#     └── style.css
#
# Therefore frontend is inside BASE_DIR.

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

INDEX_FILE = os.path.join(
    FRONTEND_DIR,
    "index.html"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# CONFIGURATION
# ============================================================

# Hackathon decision rule:
#
# AI probability >= 85% -> AI-GENERATED
# AI probability < 85%  -> REAL

AI_THRESHOLD = 0.85


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("========================================")
print("DeepVoice Guard Backend")
print("========================================")

print("BASE DIR:")
print(BASE_DIR)

print()

print("MODEL PATH:")
print(MODEL_PATH)

print()

print("FRONTEND PATH:")
print(FRONTEND_DIR)

print()

print("INDEX FILE:")
print(INDEX_FILE)

print()

print("Model exists:")
print(os.path.exists(MODEL_PATH))

print("Frontend exists:")
print(os.path.isdir(FRONTEND_DIR))

print("index.html exists:")
print(os.path.isfile(INDEX_FILE))

print("========================================")


# ============================================================
# LOAD MODEL V2
# ============================================================

model = None
FEATURES = []


try:

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]

    FEATURES = model_data["features"]

    print("Model V2 loaded successfully")

    print(
        "Features:",
        len(FEATURES)
    )

except Exception as e:

    print("========================================")
    print("MODEL LOAD ERROR")
    print("========================================")

    print(
        type(e).__name__
    )

    print(
        str(e)
    )

    print("========================================")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET"])
def home():

    print("HOME REQUEST")

    print(
        "Looking for:",
        INDEX_FILE
    )

    if not os.path.isfile(
        INDEX_FILE
    ):

        return jsonify({

            "success": False,

            "error":
                "frontend/index.html not found",

            "expected_path":
                INDEX_FILE,

            "frontend_directory":
                FRONTEND_DIR

        }), 500


    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# FRONTEND FILES
# ============================================================

@app.route(
    "/<path:filename>",
    methods=["GET"]
)
def frontend_file(filename):

    requested_file = os.path.join(
        FRONTEND_DIR,
        filename
    )


    if os.path.isfile(
        requested_file
    ):

        return send_from_directory(
            FRONTEND_DIR,
            filename
        )


    return jsonify({

        "success": False,

        "error":
            "Frontend file not found",

        "file":
            filename

    }), 404


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "success":
            True,

        "status":
            "online",

        "service":
            "DeepVoice Guard",

        "model":
            "V2",

        "model_loaded":
            model is not None,

        "features":
            len(FEATURES),

        "model_exists":
            os.path.isfile(
                MODEL_PATH
            ),

        "frontend_exists":
            os.path.isdir(
                FRONTEND_DIR
            ),

        "index_exists":
            os.path.isfile(
                INDEX_FILE
            ),

        "threshold":
            AI_THRESHOLD * 100

    })


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    print("Loading audio...")

    # IMPORTANT:
    # This matches your V2 training/testing code.
    #
    # sr=None
    # mono=True

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )


    if y is None or len(y) == 0:

        raise ValueError(
            "Audio file is empty or could not be decoded."
        )


    print(
        "Sample rate:",
        sr
    )

    print(
        "Duration:",
        round(
            len(y) / sr,
            2
        ),
        "seconds"
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
    # CREATE FEATURES
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
                np.mean(rolloff)
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
        print("========================================")
        print("NEW AUDIO REQUEST")
        print("========================================")


        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "success":
                    False,

                "error":
                    "V2 model is not loaded."

            }), 500


        # ----------------------------------------------------
        # CHECK AUDIO
        # ----------------------------------------------------

        if "audio" not in request.files:

            return jsonify({

                "success":
                    False,

                "error":
                    "No audio file provided."

            }), 400


        audio = request.files[
            "audio"
        ]


        if not audio.filename:

            return jsonify({

                "success":
                    False,

                "error":
                    "No audio file selected."

            }), 400


        print(
            "Analyzing:",
            audio.filename
        )


        # ----------------------------------------------------
        # FILE EXTENSION
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

            audio_path = (
                temp_file.name
            )

            audio.save(
                audio_path
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

        data = pd.DataFrame(
            [features]
        )


        # ----------------------------------------------------
        # CHECK FEATURES
        # ----------------------------------------------------

        missing_features = [

            feature

            for feature in FEATURES

            if feature
            not in data.columns

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
        # EXACT FEATURE ORDER
        # ----------------------------------------------------

        data = data[
            FEATURES
        ]


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction_number = (
            model.predict(data)[0]
        )


        probabilities = (
            model.predict_proba(
                data
            )[0]
        )


        # ====================================================
        # PROBABILITY MAPPING
        # ====================================================
        #
        # Your V2 test_model_v2.py uses:
        #
        # probabilities[0] = REAL
        # probabilities[1] = FAKE
        #
        # prediction 0 = REAL
        # prediction 1 = FAKE
        #

        real_probability = float(
            probabilities[0]
        )

        fake_probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # RAW MODEL PREDICTION
        # ----------------------------------------------------

        model_prediction = (
            "FAKE"
            if prediction_number == 1
            else "REAL"
        )


        # ====================================================
        # HACKATHON DECISION
        # ====================================================
        #
        # AI >= 85% -> AI-GENERATED
        # AI < 85%  -> REAL
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


        # ====================================================
        # DISPLAY CONFIDENCE
        # ====================================================
        #
        # Your website displays the larger probability.
        #

        confidence = max(
            real_probability,
            fake_probability
        )


        # ----------------------------------------------------
        # CONVERT TO PERCENTAGE
        # ----------------------------------------------------

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


        # ====================================================
        # PRINT RESULT
        # ====================================================

        print()
        print("----------------------------------------")

        print(
            "Model prediction:",
            model_prediction
        )

        print(
            "Final prediction:",
            prediction
        )

        print(
            "REAL probability:",
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

        print("----------------------------------------")


        # ====================================================
        # JSON RESPONSE
        # ====================================================

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
        print("========================================")
        print("PREDICTION ERROR")
        print("========================================")

        print(
            type(e).__name__
        )

        print(
            str(e)
        )

        print("========================================")


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


    finally:

        # ----------------------------------------------------
        # DELETE TEMPORARY FILE
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
    print("========================================")
    print("STARTING DEEPVOICE GUARD")
    print("========================================")

    print(
        "Port:",
        port
    )

    print(
        "Model:",
        "V2"
    )

    print(
        "Features:",
        len(FEATURES)
    )

    print(
        "Frontend:",
        FRONTEND_DIR
    )

    print(
        "index.html exists:",
        os.path.isfile(INDEX_FILE)
    )

    print()
    print("========================================")
    print("DECISION RULE")
    print("========================================")

    print(
        "AI probability >= 85% -> AI-GENERATED"
    )

    print(
        "AI probability < 85%  -> REAL"
    )

    print("========================================")


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
