import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .gemini_fallback import analyze_with_gemini, gemini_enabled
from .image_prediction import (
    best_prediction,
    predictions_to_top_predictions,
    prepare_image_variants,
)
from .web_lookup import build_web_lookup


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_DIR / "models" / "tomato_disease_model.keras"
CLASS_INFO_PATH = PROJECT_DIR / "models" / "class_names.json"

API_KEY = os.getenv("TOMATO_API_KEY", "change-this-key")

app = FastAPI(title="Tomato Disease Detection API")

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_INFO_PATH, "r", encoding="utf-8") as file:
    class_info = json.load(file)


def verify_api_key(x_api_key: str | None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def predict_with_local_model(image_bytes):
    image_variants = prepare_image_variants(image_bytes, class_info["image_size"])
    predictions = model.predict(image_variants, verbose=0)
    averaged_predictions = np.mean(predictions, axis=0)
    return predictions_to_top_predictions(
        class_info["class_names"],
        averaged_predictions,
    )


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Tomato disease API is running",
        "gemini_fallback_enabled": gemini_enabled(),
    }


@app.get("/preview", response_class=HTMLResponse)
def preview_page():
    return """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tomato Disease Detector</title>
  <style>
    :root {
      color-scheme: light;
      --leaf: #176d3b;
      --leaf-dark: #0d3f26;
      --ink: #162018;
      --muted: #637065;
      --line: #d8e5dc;
      --surface: #ffffff;
      --wash: #f4f8f1;
      --accent: #d4552f;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(135deg, rgba(23, 109, 59, 0.14), rgba(212, 85, 47, 0.08)),
        var(--wash);
    }

    main {
      width: min(1040px, calc(100% - 32px));
      margin: 0 auto;
      padding: 32px 0;
    }

    .shell {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      min-height: calc(100vh - 64px);
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 18px 60px rgba(22, 32, 24, 0.12);
    }

    .workspace {
      padding: 32px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      font-weight: 700;
      color: var(--leaf-dark);
    }

    .mark {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: #fff;
      background: var(--leaf);
      font-size: 22px;
    }

    h1 {
      margin: 10px 0 0;
      max-width: 680px;
      font-size: clamp(32px, 5vw, 56px);
      line-height: 1.02;
      letter-spacing: 0;
      color: var(--ink);
    }

    .sub {
      max-width: 640px;
      margin: 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.55;
    }

    form {
      display: grid;
      gap: 16px;
      margin-top: auto;
      padding-top: 16px;
    }

    label {
      display: grid;
      gap: 8px;
      font-size: 13px;
      font-weight: 700;
      color: var(--leaf-dark);
    }

    input[type="text"],
    input[type="file"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px 14px;
      font: inherit;
      color: var(--ink);
      background: #fff;
    }

    input[type="file"] {
      padding: 10px;
    }

    button {
      width: fit-content;
      border: 0;
      border-radius: 8px;
      padding: 12px 18px;
      font: inherit;
      font-weight: 700;
      color: #fff;
      background: var(--leaf);
      cursor: pointer;
    }

    button:disabled {
      opacity: 0.6;
      cursor: wait;
    }

    .preview {
      width: 100%;
      aspect-ratio: 4 / 3;
      border: 1px dashed #a8bdad;
      border-radius: 8px;
      background: #f7faf8;
      display: grid;
      place-items: center;
      overflow: hidden;
      color: var(--muted);
      text-align: center;
      padding: 16px;
    }

    .preview img {
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    aside {
      background: #f8fbf6;
      border-left: 1px solid var(--line);
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .result {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 18px;
      min-height: 170px;
    }

    .status {
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }

    .disease {
      margin: 0;
      font-size: 30px;
      line-height: 1.15;
      letter-spacing: 0;
      color: var(--leaf-dark);
    }

    .confidence {
      margin: 10px 0 0;
      font-weight: 700;
      color: var(--accent);
    }

    .bars {
      display: grid;
      gap: 12px;
    }

    .lookup {
      display: none;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 16px;
    }

    .lookup h3 {
      margin: 0 0 10px;
      font-size: 16px;
      color: var(--leaf-dark);
    }

    .lookup p {
      margin: 0 0 12px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
    }

    .lookup-group {
      display: grid;
      gap: 8px;
      padding-top: 12px;
      border-top: 1px solid var(--line);
    }

    .lookup-group + .lookup-group {
      margin-top: 12px;
    }

    .lookup-title {
      font-weight: 700;
      color: var(--ink);
    }

    .lookup-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .lookup-links a {
      color: var(--leaf-dark);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      text-decoration: none;
      font-size: 13px;
      background: #f8fbf6;
    }

    .bar-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      color: var(--muted);
    }

    .track {
      height: 8px;
      background: #e5eee7;
      border-radius: 999px;
      overflow: hidden;
    }

    .fill {
      height: 100%;
      background: var(--leaf);
      width: 0%;
    }

    @media (max-width: 820px) {
      .shell {
        grid-template-columns: 1fr;
      }

      aside {
        border-left: 0;
        border-top: 1px solid var(--line);
      }

      .workspace,
      aside {
        padding: 22px;
      }
    }
  </style>
</head>
<body>
  <main>
    <section class="shell">
      <div class="workspace">
        <div class="brand">
          <div class="mark">T</div>
          Tomato Disease Detector
        </div>
        <h1>Upload a tomato leaf photo for disease prediction.</h1>
        <p class="sub">This preview uses your local FastAPI model and returns Healthy, Early Blight, Late Blight, or Unknown when confidence is low.</p>

        <div class="preview" id="imagePreview">Choose a tomato leaf image to preview it here.</div>

        <form id="predictForm">
          <label>
            API key
            <input id="apiKey" type="text" value="my-secret-key" autocomplete="off">
          </label>
          <label>
            Leaf image
            <input id="fileInput" type="file" accept="image/*" required>
          </label>
          <button id="submitButton" type="submit">Predict disease</button>
        </form>
      </div>

      <aside>
        <div class="result">
          <p class="status" id="statusText">Waiting for image</p>
          <h2 class="disease" id="diseaseText">No result yet</h2>
          <p class="confidence" id="confidenceText"></p>
        </div>
        <div class="bars" id="bars"></div>
        <div class="lookup" id="lookup"></div>
      </aside>
    </section>
  </main>

  <script>
    const form = document.getElementById("predictForm");
    const apiKey = document.getElementById("apiKey");
    const fileInput = document.getElementById("fileInput");
    const imagePreview = document.getElementById("imagePreview");
    const statusText = document.getElementById("statusText");
    const diseaseText = document.getElementById("diseaseText");
    const confidenceText = document.getElementById("confidenceText");
    const bars = document.getElementById("bars");
    const lookup = document.getElementById("lookup");
    const submitButton = document.getElementById("submitButton");

    fileInput.addEventListener("change", () => {
      const file = fileInput.files[0];
      if (!file) return;
      const imageUrl = URL.createObjectURL(file);
      imagePreview.innerHTML = `<img src="${imageUrl}" alt="Selected tomato leaf">`;
    });

    function percent(value) {
      return `${Math.round(value * 1000) / 10}%`;
    }

    function showBars(predictions) {
      bars.innerHTML = "";
      for (const prediction of predictions || []) {
        const item = document.createElement("div");
        item.innerHTML = `
          <div class="bar-label">
            <span>${prediction.disease}</span>
            <strong>${percent(prediction.confidence)}</strong>
          </div>
          <div class="track">
            <div class="fill" style="width:${prediction.confidence * 100}%"></div>
          </div>
        `;
        bars.appendChild(item);
      }
    }

    function showLookup(webLookup) {
      lookup.innerHTML = "";
      lookup.style.display = "none";

      if (!webLookup) return;

      const groups = (webLookup.candidates || []).map((candidate) => {
        return `
          <div class="lookup-group">
            <div class="lookup-title">${candidate.disease}</div>
            <div class="lookup-links">
              <a href="${candidate.links.google_images}" target="_blank" rel="noreferrer">Google Images</a>
              <a href="${candidate.links.google_web}" target="_blank" rel="noreferrer">Google Search</a>
              <a href="${candidate.links.bing_images}" target="_blank" rel="noreferrer">Bing Images</a>
            </div>
          </div>
        `;
      }).join("");

      lookup.innerHTML = `
        <h3>Web lookup</h3>
        <p>${webLookup.reason}</p>
        <p>${webLookup.note}</p>
        ${groups}
      `;
      lookup.style.display = "block";
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();

      const file = fileInput.files[0];
      if (!file) return;

      const data = new FormData();
      data.append("file", file);

      submitButton.disabled = true;
      statusText.textContent = "Analyzing leaf";
      diseaseText.textContent = "Working...";
      confidenceText.textContent = "";
      bars.innerHTML = "";
      showLookup(null);

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: {
            "x-api-key": apiKey.value
          },
          body: data
        });

        const result = await response.json();
        if (!response.ok) {
          throw new Error(result.detail || "Prediction failed");
        }

        statusText.textContent = result.supported ? "Prediction complete" : "Low confidence";
        diseaseText.textContent = result.disease;
        confidenceText.textContent = `Confidence: ${percent(result.confidence)}`;
        showBars(result.top_predictions);
        showLookup(result.web_lookup);
      } catch (error) {
        statusText.textContent = "Request failed";
        diseaseText.textContent = error.message;
      } finally {
        submitButton.disabled = false;
      }
    });
  </script>
</body>
</html>
    """


@app.post("/predict")
async def predict(file: UploadFile = File(...), x_api_key: str | None = Header(None)):
    verify_api_key(x_api_key)

    threshold = float(class_info["confidence_threshold"])
    image_bytes = await file.read()
    top_predictions = predict_with_local_model(image_bytes)
    best = best_prediction(top_predictions)
    confidence = float(best["confidence"])

    if confidence < threshold:
        web_lookup = build_web_lookup()
        gemini_result = analyze_with_gemini(image_bytes, file.content_type)

        if gemini_result.get("supported"):
            disease = gemini_result["disease"]
            return {
                "disease": disease,
                "confidence": gemini_result["confidence"],
                "top_predictions": top_predictions,
                "supported": True,
                "threshold": threshold,
                "prediction_method": "gemini_fallback",
                "local_prediction_method": "test_time_augmentation",
                "gemini_result": gemini_result,
                "web_lookup": web_lookup,
                "message": (
                    f"The local model was uncertain, but Gemini suggests: {disease}."
                ),
            }

        google_conclusion = (
            web_lookup.get("google_keyword_result", {})
            .get("conclusion", {})
        )

        if google_conclusion.get("supported"):
            disease = google_conclusion["disease"]
            return {
                "disease": disease,
                "confidence": confidence,
                "top_predictions": top_predictions,
                "supported": True,
                "threshold": threshold,
                "prediction_method": "google_keyword_fallback",
                "local_prediction_method": "test_time_augmentation",
                "web_lookup": web_lookup,
                "message": (
                    f"The local model was uncertain, but Google result keywords "
                    f"suggest: {disease}."
                ),
            }

        return {
            "disease": "Unknown",
            "confidence": confidence,
            "top_predictions": top_predictions,
            "supported": False,
            "threshold": threshold,
            "prediction_method": "test_time_augmentation",
            "gemini_result": gemini_result,
            "web_lookup": web_lookup,
            "message": "Image is unclear or the disease is not supported.",
        }

    disease = best["disease"]
    return {
        "disease": disease,
        "confidence": confidence,
        "top_predictions": top_predictions,
        "supported": True,
        "threshold": threshold,
        "prediction_method": "test_time_augmentation",
        "message": f"The tomato leaf appears to be: {disease}.",
    }
