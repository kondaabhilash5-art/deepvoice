from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import tempfile

import numpy as np
import pandas as pd
import librosa
import joblib


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "deepvoice_model_v2.pkl"
)

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "frontend"
)

INDEX_FILE = os.path.join(
    FRONTEND_DIR,
    "index.html"
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

CORS(app)


# Maximum uploaded file size: 25 MB
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


# ============================================================
# SETTINGS
# ============================================================

AI_THRESHOLD = 0.85

# IMPORTANT:
# Only analyze the first 30 seconds.
#
# This greatly reduces RAM usage on Render.
MAX_AUDIO_SECONDS = 30


# ============================================================
# STARTUP
# ============================================================

print("========================================")
print("DeepVoice Guard")
print("========================================")

print("BASE DIR:")
print(BASE_DIR)

print("MODEL:")
print(MODEL_PATH)

print("FRONTEND:")
print(FRONTEND_DIR)

print("INDEX:")
print(INDEX_FILE)

print()

print(
    "Model exists:",
    os.path.isfile(MODEL_PATH)
)

print(
    "Frontend exists:",
    os.path.isdir(FRONTEND_DIR)
)

print(
    "Index exists:",
    os.path.isfile(INDEX_FILE)
)

print("========================================")


# ============================================================
# LOAD MODEL
# ============================================================

model = None
FEATURES = []


try:

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]

    FEATURES = model_data["features"]

    print("V2 MODEL LOADED")

    print(
        "Features:",
        len(FEATURES)
    )

except Exception as e:

    print("MODEL LOAD ERROR")

    print(
        type(e).__name__,
        str(e)
    )


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if not os.path.isfile(
        INDEX_FILE
    ):

        return jsonify({

            "success": False,

            "error":
                "frontend/index.html not found",

            "expected":
                INDEX_FILE

        }), 500


    return send_from_directory(
        FRONTEND_DIR,
        "index.html"
    )


# ============================================================
# FRONTEND FILES
# ============================================================

@app.route(
    "/<path:filename>"
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
            "File not found",

        "file":
            filename

    }), 404


# ============================================================
# HEALTH
# ============================================================

@app.route("/health")
def health():

    return jsonify({

        "success": True,

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
            85,

        "max_audio_seconds":
            MAX_AUDIO_SECONDS

    })


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(audio_path):

    print(
        "Loading first",
        MAX_AUDIO_SECONDS,
        "seconds..."
    )


    # ========================================================
    # IMPORTANT
    # ========================================================
    #
    # sr=None is preserved because this is how your V2
    # model was trained/tested.
    #
    # duration=30 prevents very long recordings from
    # consuming huge amounts of RAM.
    #

    y, sr = librosa.load(

        audio_path,

        sr=None,

        mono=True,

        duration=MAX_AUDIO_SECONDS

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
        "Samples:",
        len(y)
    )

    print(
        "Analyzed duration:",
        round(
            len(y) / sr,
            2
        ),
        "seconds"
    )


    # ========================================================
    # FEATURES
    # ========================================================

    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )


    rms = librosa.feature.rms(
        y=y
    )


    spectral_centroid = (
        librosa.feature.spectral_centroid(
            y=y,
            sr=sr
        )
    )


    spectral_bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )
    )


    rolloff = (
        librosa.feature.spectral_rolloff(
            y=y,
            sr=sr
        )
    )


    zero_crossing_rate = (
        librosa.feature.zero_crossing_rate(
            y
        )
    )


    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=20
    )


    # ========================================================
    # FEATURE DICTIONARY
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


    # Free large arrays immediately

    del y
    del chroma
    del rms
    del spectral_centroid
    del spectral_bandwidth
    del rolloff
    del zero_crossing_rate
    del mfcc


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
        print("NEW PREDICTION")
        print("========================================")


        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        if model is None:

            return jsonify({

                "success": False,

                "error":
                    "V2 model is not loaded."

            }), 500


        # ----------------------------------------------------
        # AUDIO
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
        # FILE EXTENSION
        # ----------------------------------------------------

        extension = os.path.splitext(
            audio.filename
        )[1].lower()


        if not extension:

            extension = ".wav"


        # ----------------------------------------------------
        # TEMP FILE
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension
        ) as temp:

            audio_path = temp.name

            audio.save(
                audio_path
            )


        # ----------------------------------------------------
        # FILE SIZE
        # ----------------------------------------------------

        file_size = os.path.getsize(
            audio_path
        )

        print(
            "File size:",
            round(
                file_size / 1024 / 1024,
                2
            ),
            "MB"
        )


        # ----------------------------------------------------
        # FEATURE EXTRACTION
        # ----------------------------------------------------

        features = extract_features(
            audio_path
        )


        # ----------------------------------------------------
        # DATAFRAME
        # ----------------------------------------------------

        data = pd.DataFrame(
            [features]
        )


        # ----------------------------------------------------
        # FEATURE CHECK
        # ----------------------------------------------------

        missing = [

            feature

            for feature in FEATURES

            if feature
            not in data.columns

        ]


        if missing:

            raise ValueError(
                "Missing features: "
                +
                ", ".join(missing)
            )


        # Exact model feature order

        data = data[
            FEATURES
        ]


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction_number = (
            model.predict(data)[0]
        )


        probabilities = (
            model.predict_proba(data)[0]
        )


        # V2 mapping:
        #
        # [0] = REAL
        # [1] = FAKE

        real_probability = float(
            probabilities[0]
        )

        fake_probability = float(
            probabilities[1]
        )


        # ----------------------------------------------------
        # RAW MODEL RESULT
        # ----------------------------------------------------

        model_prediction = (

            "FAKE"

            if prediction_number == 1

            else "REAL"

        )


        # ----------------------------------------------------
        # HACKATHON RULE
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
        # RESULT
        # ----------------------------------------------------

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
            "REAL:",
            real_percentage,
            "%"
        )

        print(
            "AI:",
            fake_percentage,
            "%"
        )

        print(
            "Confidence:",
            confidence_percentage,
            "%"
        )

        print("----------------------------------------")


        # ----------------------------------------------------
        # RESPONSE
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
                85,

            "analyzed_seconds":
                MAX_AUDIO_SECONDS

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
        "Model: V2"
    )

    print(
        "Features:",
        len(FEATURES)
    )

    print(
        "Max audio:",
        MAX_AUDIO_SECONDS,
        "seconds"
    )

    print("========================================")


    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
