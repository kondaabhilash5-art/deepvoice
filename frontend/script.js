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
// Handle file
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
// Drag and Drop
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

            handleFile(
                files[0]
            );

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
// SHOW RESULT
// ==========================================

function showResult(data) {

    resultCard.style.display =
        "block";


    // ======================================
    // KEEP BACKEND DECISION
    // ======================================

    if (data.prediction === "FAKE") {

        resultIcon.textContent =
            "⚠";

        resultIcon.classList.add(
            "fake"
        );

        resultLabel.textContent =
            "AI-GENERATED VOICE";

    } else {

        resultIcon.textContent =
            "✓";

        resultIcon.classList.remove(
            "fake"
        );

        resultLabel.textContent =
            "REAL VOICE";
    }


    // ======================================
    // GET PROBABILITIES
    // ======================================

    const real =
        Number(
            data.real_probability
        );

    const fake =
        Number(
            data.fake_probability
        );


    // ======================================
    // USE THE HIGHER VALUE FOR DISPLAY
    // ======================================

    const displayPercentage =
        Math.max(
            real,
            fake
        );


    // ======================================
    // TOP BIG PERCENTAGE
    // ======================================

    // IMPORTANT:
    // index.html already contains the "%"
    // after this span.
    //
    // Therefore DON'T add "%" here.

    confidence.textContent =
        displayPercentage.toFixed(2);


    // ======================================
    // GET COMPLETE BAR ROWS
    // ======================================

    const realRow =
        realBar.closest(
            ".probability-row"
        );

    const fakeRow =
        fakeBar.closest(
            ".probability-row"
        );


    // ======================================
    // RESET BOTH ROWS
    // ======================================

    realRow.style.display =
        "none";

    fakeRow.style.display =
        "none";


    // Reset bars

    realBar.style.width =
        "0%";

    fakeBar.style.width =
        "0%";


    // Reset percentage text

    realProbability.textContent =
        "";

    fakeProbability.textContent =
        "";


    // ======================================
    // REAL RESULT
    // ======================================

    if (
        data.prediction === "REAL"
    ) {

        // Show ONLY real row

        realRow.style.display =
            "block";


        // The meter percentage MUST
        // match the big percentage.

        realProbability.textContent =
            displayPercentage.toFixed(2) +
            "%";


        // Animate real meter

        setTimeout(() => {

            realBar.style.width =
                displayPercentage + "%";

        }, 100);

    }


    // ======================================
    // AI RESULT
    // ======================================

    else {

        // Show ONLY AI row

        fakeRow.style.display =
            "block";


        // The meter percentage MUST
        // match the big percentage.

        fakeProbability.textContent =
            displayPercentage.toFixed(2) +
            "%";


        // Animate AI meter

        setTimeout(() => {

            fakeBar.style.width =
                displayPercentage + "%";

        }, 100);

    }


    // ======================================
    // CONFIDENCE LEVEL
    // ======================================

    confidenceLevel.textContent =
        data.confidence_level;


    // ======================================
    // SCROLL TO RESULT
    // ======================================

    resultCard.scrollIntoView({
        behavior: "smooth",
        block: "center"
    });

}


// ==========================================
// ERROR
// ==========================================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";

}


// ==========================================
// NEW ANALYSIS
// ==========================================

newAnalysis.addEventListener(
    "click",
    () => {

        selectedAudio = null;

        audioFile.value = "";

        selectedFile.textContent =
            "";

        analyzeButton.disabled =
            true;

        resultCard.style.display =
            "none";

        errorBox.style.display =
            "none";


        // Reset bars

        realBar.style.width =
            "0%";

        fakeBar.style.width =
            "0%";


        // Reset rows

        const realRow =
            realBar.closest(
                ".probability-row"
            );

        const fakeRow =
            fakeBar.closest(
                ".probability-row"
            );


        if (realRow) {

            realRow.style.display =
                "block";

        }


        if (fakeRow) {

            fakeRow.style.display =
                "block";

        }


        // Reset values

        confidence.textContent =
            "0";

        realProbability.textContent =
            "0%";

        fakeProbability.textContent =
            "0%";


        // Scroll to top

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    }
);
