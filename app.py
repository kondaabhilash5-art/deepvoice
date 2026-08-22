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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "deepvoice_model.pkl"
)

FRONTEND_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "frontend"
)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    static_url_path=""
)

CORS(app)


# ============================================================
# CONFIGURATION
# ============================================================

# Hackathon decision threshold
# AI probability >= 85% -> AI-GENERATED
# AI probability < 85%  -> REAL

AI_THRESHOLD = 0.85


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("================================")
print("DeepVoice Guard Backend")
print("================================")

print(
    "Model path:",
    MODEL_PATH
)

print(
    "Frontend path:",
    FRONTEND_DIR
)


try:

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]

    FEATURES = model_data["features"]

    print(
        "Model loaded successfully"
    )

    print(
        f"Features: {len(FEATURES)}"
    )

except Exception as e:

    print(
        "ERROR loading model:"
    )

    print(
        str(e)
    )

    model = None

    FEATURES = []


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    """
    Extract the same 26 features used
    during model training.
    """

    # --------------------------------------------------------
    # Load audio
    # --------------------------------------------------------

    y, sr = librosa.load(
        audio_path,
        sr=None,
        mono=True
    )

    if len(y) == 0:

        raise ValueError(
            "Audio file is empty"
        )


    # --------------------------------------------------------
    # Chroma
    # --------------------------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )


    # --------------------------------------------------------
    # RMS
    # --------------------------------------------------------

    rms = librosa.feature.rms(
        y=y
    )


    # --------------------------------------------------------
    # Spectral Centroid
    # --------------------------------------------------------

    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )


    # --------------------------------------------------------
    # Spectral Bandwidth
    # --------------------------------------------------------

    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )


    # --------------------------------------------------------
    # Spectral Rolloff
    # --------------------------------------------------------

    rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )


    # --------------------------------------------------------
    # Zero Crossing Rate
    # --------------------------------------------------------

    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y
        )
    )


    # --------------------------------------------------------
    # MFCC
    # --------------------------------------------------------

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )


    # --------------------------------------------------------
    # Create feature dictionary
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # MFCC 1-20
    # --------------------------------------------------------

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
# FRONTEND
# ============================================================

@app.route(
    "/",
    methods=["GET"]
)
def home():

    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# FRONTEND STATIC FILES
# ============================================================

@app.route(
    "/<path:filename>",
    methods=["GET"]
)
def frontend_files(filename):

    file_path = os.path.join(
        FRONTEND_DIR,
        filename
    )

    if os.path.isfile(
        file_path
    ):

        return send_from_directory(
            FRONTEND_DIR,
            filename
        )

    return jsonify({

        "success": False,

        "error":
            "File not found"

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

        "status":
            "online",

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
# PREDICTION API
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
        # Check audio field
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


        # ----------------------------------------------------
        # Check filename
        # ----------------------------------------------------

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
        # Save temporary audio
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
        # Convert to DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame(
            [features]
        )


        # ----------------------------------------------------
        # Make sure feature order matches
        # the trained model
        # ----------------------------------------------------

        missing_features = [
            feature
            for feature in FEATURES
            if feature not in input_data.columns
        ]


        if missing_features:

            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    missing_features
                )
            )


        input_data = input_data[
            FEATURES
        ]


        # ----------------------------------------------------
        # Model probabilities
        # ----------------------------------------------------

        probabilities = (
            model.predict_proba(
                input_data
            )[0]
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The trained model uses:
        #
        # Class 0 = REAL
        # Class 1 = FAKE
        # ----------------------------------------------------

        real_probability = float(
            probabilities[0]
        )

        fake_probability = float(
            probabilities[1]
        )


        # ====================================================
        # FINAL APPLICATION DECISION
        # ====================================================

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
        # Overall model confidence
        # ----------------------------------------------------

        confidence = max(
            real_probability,
            fake_probability
        )


        # ----------------------------------------------------
        # Convert to percentages
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


        # ----------------------------------------------------
        # Terminal output
        # ----------------------------------------------------

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
            "Confidence:",
            f"{confidence_percentage}%"
        )


        # ====================================================
        # RESPONSE
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


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        print()
        print("================================")
        print("ERROR")
        print("================================")

        print(
            str(e)
        )


        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


    # ========================================================
    # CLEANUP TEMPORARY FILE
    # ========================================================

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
# LOCAL DEVELOPMENT
# ============================================================

if __name__ == "__main__":

    # Render supplies PORT as an environment variable.
    # Local development falls back to 5000.

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
        f"Frontend: {FRONTEND_DIR}"
    )

    print(
        f"Website: http://127.0.0.1:{port}"
    )

    print(
        "API: /predict"
    )

    print()
    print("================================")
    print("DECISION RULE")
    print("================================")

    print(
        "AI probability >= 85% -> AI-GENERATED"
    )

    print(
        "AI probability < 85%  -> REAL"
    )

    print()

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )