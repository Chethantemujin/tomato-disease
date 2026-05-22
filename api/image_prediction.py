from io import BytesIO

import numpy as np
from PIL import Image, ImageEnhance


def prepare_image_variants(image_bytes, image_size):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(tuple(image_size))

    variants = [
        image,
        image.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(image).enhance(0.9),
        ImageEnhance.Brightness(image).enhance(1.1),
        ImageEnhance.Contrast(image).enhance(1.1),
        center_crop_zoom(image, zoom=0.9),
    ]

    return np.array(variants, dtype=np.float32)


def center_crop_zoom(image, zoom):
    width, height = image.size
    crop_width = int(width * zoom)
    crop_height = int(height * zoom)
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height))


def predictions_to_top_predictions(class_names, predictions):
    return sorted(
        [
            {"disease": disease, "confidence": float(score)}
            for disease, score in zip(class_names, predictions)
        ],
        key=lambda item: item["confidence"],
        reverse=True,
    )


def best_prediction(top_predictions):
    if not top_predictions:
        return {"disease": "Unknown", "confidence": 0.0}
    return top_predictions[0]
