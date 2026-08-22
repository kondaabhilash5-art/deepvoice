// ==========================================
// DeepVoice Guard Frontend
// ==========================================

// Flask backend
const API_URL = "/predict";


// ==========================================
// Get HTML elements
// ==========================================

const audioFile =
    document.getElementById("audioFile");

const chooseButton =
    document.getElementById("chooseButton");

const dropZone =
    document.getElementById("dropZone");

const analyzeButton =
    document.getElementById("analyzeButton");

const selectedFile =
    document.getElementById("selectedFile");

const loading =
    document.getElementById("loading");

const errorBox =
    document.getElementById("error");

const resultCard =
    document.getElementById("resultCard");

const resultIcon =
    document.getElementById("resultIcon");

const resultLabel =
    document.getElementById("resultLabel");

const confidence =
    document.getElementById("confidence");

const realProbability =
    document.getElementById("realProbability");

const fakeProbability =
    document.getElementById("fakeProbability");

const realBar =
    document.getElementById("realBar");

const fakeBar =
    document.getElementById("fakeBar");

const confidenceLevel =
    document.getElementById("confidenceLevel");

const newAnalysis =
    document.getElementById("newAnalysis");


// ==========================================
// Selected audio
// ==========================================

let selectedAudio = null;


// ==========================================
// Choose button
// ==========================================

chooseButton.addEventListener(
    "click",
    () => {

        audioFile.click();

    }
);


// ==========================================
// File selected
// ==========================================

audioFile.addEventListener(
    "change",
    () => {

        if (audioFile.files.length > 0) {

            handleFile(
                audioFile.files[0]
            );

        }

    }
);


// ==========================================
// Handle selected file
// ==========================================

function handleFile(file) {

    selectedAudio = file;

    selectedFile.textContent =
        "Selected: " + file.name;

    analyzeButton.disabled = false;

    errorBox.style.display = "none";

    resultCard.style.display = "none";

}


// ==========================================
// Drag and drop
// ==========================================

dropZone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropZone.classList.add(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "dragover"
        );

    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );

        const files =
            event.dataTransfer.files;

        if (files.length > 0) {

            handleFile(files[0]);

        }

    }
);


// ==========================================
// Analyze button
// ==========================================

analyzeButton.addEventListener(
    "click",
    analyzeVoice
);


// ==========================================
// Analyze voice
// ==========================================

async function analyzeVoice() {

    if (!selectedAudio) {

        showError(
            "Please select an audio file first."
        );

        return;

    }


    // Hide previous result

    resultCard.style.display =
        "none";

    errorBox.style.display =
        "none";


    // Show loading

    loading.style.display =
        "block";

    analyzeButton.disabled =
        true;


    try {

        // ----------------------------------
        // Create form data
        // ----------------------------------

        const formData =
            new FormData();

        formData.append(
            "audio",
            selectedAudio
        );


        // ----------------------------------
        // Send audio to Flask
        // ----------------------------------

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",
                    body: formData
                }
            );


        // ----------------------------------
        // Read response
        // ----------------------------------

        const data =
            await response.json();


        // ----------------------------------
        // Check response
        // ----------------------------------

        if (
            !response.ok ||
            !data.success
        ) {

            throw new Error(
                data.error ||
                "Prediction failed."
            );

        }


        // ----------------------------------
        // Display result
        // ----------------------------------

        showResult(data);


    } catch (error) {

        console.error(
            "DeepVoice Guard error:",
            error
        );

        showError(
            "Could not analyze the audio. " +
            error.message
        );


    } finally {

        loading.style.display =
            "none";

        analyzeButton.disabled =
            false;

    }

}


// ==========================================
// Show result
// ==========================================

function showResult(data) {

    resultCard.style.display =
        "block";


    // ======================================
    // Prediction
    // ======================================

    if (
        data.prediction === "FAKE"
    ) {

        // AI-generated

        resultIcon.textContent =
            "⚠";

        resultIcon.classList.add(
            "fake"
        );

        resultLabel.textContent =
            "AI-GENERATED VOICE";


    } else {

        // REAL

        resultIcon.textContent =
            "✓";

        resultIcon.classList.remove(
            "fake"
        );

        resultLabel.textContent =
            "REAL VOICE";

    }


    // ======================================
    // Confidence
    // ======================================

    confidence.textContent =
        data.confidence + "%";


    // ======================================
    // Real probability
    // ======================================

    realProbability.textContent =
        data.real_probability + "%";


    // ======================================
    // AI probability
    // ======================================

    fakeProbability.textContent =
        data.fake_probability + "%";


    // ======================================
    // Reset bars before animation
    // ======================================

    realBar.style.width = "0%";

    fakeBar.style.width = "0%";


    // ======================================
    // Animate probability bars
    // ======================================

    setTimeout(
        () => {

            realBar.style.width =
                data.real_probability + "%";

            fakeBar.style.width =
                data.fake_probability + "%";

        },
        100
    );


    // ======================================
    // Confidence level
    // ======================================

    confidenceLevel.textContent =
        data.confidence_level;


    // ======================================
    // Scroll to result
    // ======================================

    resultCard.scrollIntoView({

        behavior: "smooth",

        block: "center"

    });

}


// ==========================================
// Show error
// ==========================================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";

}


// ==========================================
// New analysis
// ==========================================

newAnalysis.addEventListener(
    "click",
    () => {

        selectedAudio = null;

        audioFile.value = "";

        selectedFile.textContent = "";

        analyzeButton.disabled =
            true;

        resultCard.style.display =
            "none";

        errorBox.style.display =
            "none";

        realBar.style.width =
            "0%";

        fakeBar.style.width =
            "0%";

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    }
);