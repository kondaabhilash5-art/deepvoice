// ==========================================
// DeepVoice Guard Frontend
// ==========================================

// Flask backend endpoint
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
// Choose Audio
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
// Drag over
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
// Drag leave
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
// Drop
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
// Analyze button
// ==========================================

analyzeButton.addEventListener(
    "click",
    analyzeVoice
);


// ==========================================
// Analyze Voice
// ==========================================

async function analyzeVoice() {

    if (!selectedAudio) {

        showError(
            "Please select an audio file first."
        );

        return;

    }


    // --------------------------------------
    // Reset previous result
    // --------------------------------------

    resultCard.style.display =
        "none";

    errorBox.style.display =
        "none";


    // --------------------------------------
    // Loading
    // --------------------------------------

    loading.style.display =
        "block";

    analyzeButton.disabled =
        true;


    try {

        // ----------------------------------
        // Form data
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
        // Send to Flask
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
        // Read response as TEXT first
        // ----------------------------------
        //
        // IMPORTANT:
        // Do NOT immediately call response.json().
        //
        // If Render returns an empty response,
        // HTML error page, 502, 500, etc.,
        // response.json() causes:
        //
        // "Unexpected end of JSON input"
        //
        // Reading text first lets us see
        // what the server actually returned.
        //

        const responseText =
            await response.text();


        console.log(
            "Server response:",
            responseText
        );


        // ----------------------------------
        // Empty response
        // ----------------------------------

        if (
            !responseText ||
            responseText.trim() === ""
        ) {

            throw new Error(
                "The server returned an empty response " +
                "(HTTP " +
                response.status +
                "). Check the Render logs."
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
                "). " +
                "Response: " +
                responseText.substring(
                    0,
                    300
                )
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
        // Successful result
        // ----------------------------------

        console.log(
            "Prediction:",
            data
        );


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
// Show Result
// ==========================================

function showResult(data) {

    resultCard.style.display =
        "block";


    // ======================================
    // Determine result
    // ======================================

    const prediction =
        String(
            data.prediction || ""
        ).toUpperCase();


    // ======================================
    // AI-GENERATED
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

    }


    // ======================================
    // REAL
    // ======================================

    else {

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

    const confidenceValue =
        Number(
            data.confidence
        );


    if (
        Number.isFinite(
            confidenceValue
        )
    ) {

        confidence.textContent =
            confidenceValue.toFixed(2) + "%";

    } else {

        confidence.textContent =
            "--";

    }


    // ======================================
    // Probabilities
    // ======================================

    const realValue =
        Number(
            data.real_probability
        );

    const fakeValue =
        Number(
            data.fake_probability
        );


    // ======================================
    // YOUR DISPLAY RULE
    // ======================================
    //
    // You wanted the website to show only
    // the winning result's percentage.
    //
    // Example:
    //
    // AI = 59%
    // REAL = 41%
    //
    // Final result = REAL according to
    // your 85% backend rule.
    //
    // The large confidence number should
    // match the displayed result.
    //
    // The bars will also show only the
    // selected result.
    //
    // ======================================


    if (
        prediction === "FAKE"
    ) {

        // AI result

        fakeProbability.textContent =
            fakeValue.toFixed(2) + "%";

        realProbability.textContent =
            "";

        fakeBar.style.width =
            fakeValue + "%";

        realBar.style.width =
            "0%";

        fakeBar.style.display =
            "block";

        realBar.style.display =
            "none";

    }


    else {

        // REAL result

        realProbability.textContent =
            realValue.toFixed(2) + "%";

        fakeProbability.textContent =
            "";

        realBar.style.width =
            realValue + "%";

        fakeBar.style.width =
            "0%";

        realBar.style.display =
            "block";

        fakeBar.style.display =
            "none";

    }


    // ======================================
    // Confidence level
    // ======================================

    if (
        data.confidence_level
    ) {

        confidenceLevel.textContent =
            data.confidence_level;

    } else {

        confidenceLevel.textContent =
            "MEDIUM";

    }


    // ======================================
    // Animate bar
    // ======================================

    requestAnimationFrame(
        () => {

            if (
                prediction === "FAKE"
            ) {

                fakeBar.style.width =
                    fakeValue + "%";

            } else {

                realBar.style.width =
                    realValue + "%";

            }

        }
    );


    // ======================================
    // Scroll to result
    // ======================================

    resultCard.scrollIntoView({

        behavior: "smooth",

        block: "center"

    });

}


// ==========================================
// Error
// ==========================================

function showError(message) {

    errorBox.textContent =
        message;

    errorBox.style.display =
        "block";

}


// ==========================================
// New Analysis
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

        realBar.style.display =
            "block";

        fakeBar.style.display =
            "block";


        // Reset percentages

        realProbability.textContent =
            "0%";

        fakeProbability.textContent =
            "0%";


        // Reset confidence

        confidence.textContent =
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
