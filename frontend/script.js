// ==========================================
// DeepVoice Guard Frontend
// ==========================================

const API_URL = "/predict";


// ==========================================
// HTML ELEMENTS
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
// SELECTED AUDIO
// ==========================================

let selectedAudio = null;


// ==========================================
// CHOOSE BUTTON
// ==========================================

chooseButton.addEventListener(
    "click",
    () => {

        audioFile.click();

    }
);


// ==========================================
// FILE SELECTED
// ==========================================

audioFile.addEventListener(
    "change",
    () => {

        if (
            audioFile.files &&
            audioFile.files.length > 0
        ) {

            handleFile(
                audioFile.files[0]
            );

        }

    }
);


// ==========================================
// HANDLE FILE
// ==========================================

function handleFile(file) {

    selectedAudio = file;

    selectedFile.textContent =
        "Selected: " + file.name;

    analyzeButton.disabled =
        false;

    errorBox.style.display =
        "none";

    resultCard.style.display =
        "none";

}


// ==========================================
// DRAG OVER
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


// ==========================================
// DRAG LEAVE
// ==========================================

dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "dragover"
        );

    }
);


// ==========================================
// DROP
// ==========================================

dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );

        const files =
            event.dataTransfer.files;

        if (
            files &&
            files.length > 0
        ) {

            handleFile(
                files[0]
            );

        }

    }
);


// ==========================================
// ANALYZE
// ==========================================

analyzeButton.addEventListener(
    "click",
    analyzeVoice
);


// ==========================================
// ANALYZE VOICE
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
        // FormData
        // ----------------------------------

        const formData =
            new FormData();

        formData.append(
            "audio",
            selectedAudio
        );


        console.log(
            "Uploading:",
            selectedAudio.name
        );


        // ----------------------------------
        // Request
        // ----------------------------------

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",
                    body: formData
                }
            );


        console.log(
            "HTTP status:",
            response.status
        );


        // ----------------------------------
        // Read response as text
        // ----------------------------------

        const responseText =
            await response.text();


        console.log(
            "Server response:",
            responseText
        );


        if (
            !responseText ||
            responseText.trim() === ""
        ) {

            throw new Error(
                "The server returned an empty response " +
                "(HTTP " +
                response.status +
                ")."
            );

        }


        // ----------------------------------
        // Parse JSON
        // ----------------------------------

        let data;

        try {

            data =
                JSON.parse(
                    responseText
                );

        } catch (parseError) {

            console.error(
                "JSON parse error:",
                parseError
            );

            throw new Error(
                "Server returned a non-JSON response " +
                "(HTTP " +
                response.status +
                ")."
            );

        }


        // ----------------------------------
        // Backend error
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
            "Analysis error:",
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
    // FINAL PREDICTION
    // ======================================

    const prediction =
        String(
            data.prediction || ""
        ).toUpperCase();


    // ======================================
    // PROBABILITIES
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
    // DISPLAY PERCENTAGE
    // ======================================
    //
    // IMPORTANT:
    //
    // This is the probability corresponding
    // to the final displayed result.
    //
    // The backend decides:
    //
    // AI >= 85% -> AI
    // AI < 85%  -> REAL
    //
    // ======================================

    let displayPercentage;


    if (
        prediction === "FAKE"
    ) {

        displayPercentage =
            fake;

    } else {

        displayPercentage =
            fake < 85
                ? fake
                : real;

    }


    // ======================================
    // RESULT ICON + LABEL
    // ======================================

    if (
        prediction === "FAKE"
    ) {

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
    // BIG PERCENTAGE
    // ======================================
    //
    // index.html already contains %.
    //
    // Therefore only put the number here.
    //

    confidence.textContent =
        displayPercentage.toFixed(2);


    // ======================================
    // BAR ROWS
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
    // RESET BARS
    // ======================================

    realBar.style.width =
        "0%";

    fakeBar.style.width =
        "0%";


    realProbability.textContent =
        "";

    fakeProbability.textContent =
        "";


    // ======================================
    // SHOW ONLY FINAL RESULT BAR
    // ======================================

    if (
        prediction === "FAKE"
    ) {

        if (realRow) {

            realRow.style.display =
                "none";

        }

        if (fakeRow) {

            fakeRow.style.display =
                "block";

        }


        fakeProbability.textContent =
            displayPercentage.toFixed(2) +
            "%";


        setTimeout(
            () => {

                fakeBar.style.width =
                    displayPercentage +
                    "%";

            },
            100
        );

    } else {

        if (fakeRow) {

            fakeRow.style.display =
                "none";

        }

        if (realRow) {

            realRow.style.display =
                "block";

        }


        realProbability.textContent =
            displayPercentage.toFixed(2) +
            "%";


        setTimeout(
            () => {

                realBar.style.width =
                    displayPercentage +
                    "%";

            },
            100
        );

    }


    // ======================================
    // CONFIDENCE LEVEL
    // ======================================

    confidenceLevel.textContent =
        data.confidence_level ||
        "MEDIUM";


    // ======================================
    // SCROLL
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

        audioFile.value =
            "";

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


        // Restore both rows for next analysis

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

        confidenceLevel.textContent =
            "";


        // Scroll top

        window.scrollTo({

            top: 0,

            behavior: "smooth"

        });

    }
);
