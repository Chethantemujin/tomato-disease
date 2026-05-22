---
title: Tomato Disease Detector
emoji: 🍅
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tomato Disease Detector API & Web App

A FastAPI-based web service and application that leverages a custom-trained **EfficientNetB0** deep learning model in TensorFlow to accurately detect tomato leaf diseases.

## 🌟 Key Features
- **TensorFlow Inference:** Uses a local convolutional neural network for rapid predictions.
- **Test-Time Augmentation (TTA):** Evaluates multiple variants of uploaded leaf photos to maximize accuracy.
- **Gemini Fallback:** Leverages Google Gemini 2.5 Flash visual reasoning when the local model is uncertain.
- **Interactive Preview UI:** Provides an elegant web interface at `/preview` to upload photos and view predictions.
