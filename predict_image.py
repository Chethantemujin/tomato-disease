import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

from api.image_prediction import (
    best_prediction,
    predictions_to_top_predictions,
    prepare_image_variants,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "tomato_disease_model.keras"
CLASS_INFO_PATH = PROJECT_DIR / "models" / "class_names.json"


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_image.py path\\to\\leaf.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        sys.exit(1)

    with open(CLASS_INFO_PATH, "r", encoding="utf-8") as file:
        class_info = json.load(file)

    model = tf.keras.models.load_model(MODEL_PATH)
    image_bytes = image_path.read_bytes()
    image_variants = prepare_image_variants(image_bytes, class_info["image_size"])

    predictions = model.predict(image_variants, verbose=0)
    averaged_predictions = np.mean(predictions, axis=0)
    top_predictions = predictions_to_top_predictions(
        class_info["class_names"],
        averaged_predictions,
    )
    best = best_prediction(top_predictions)
    confidence = float(best["confidence"])
    threshold = float(class_info["confidence_threshold"])

    if confidence < threshold:
        disease = "Unknown"
        message = "Image is unclear or the disease is not supported."
    else:
        disease = best["disease"]
        message = f"The tomato leaf appears to be: {disease}."

    print(f"Disease: {disease}")
    print(f"Confidence: {confidence:.2%}")
    print("Prediction method: test-time augmentation")
    print(f"Message: {message}")
    print("Top predictions:")
    for prediction in top_predictions:
        print(f"- {prediction['disease']}: {prediction['confidence']:.2%}")


if __name__ == "__main__":
    main()
