import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ALLOWED_DISEASES = {"Early Blight", "Late Blight", "Healthy", "Unknown"}


def gemini_enabled():
    return bool(os.getenv("GEMINI_API_KEY"))


def analyze_with_gemini(image_bytes, mime_type):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {
            "enabled": False,
            "supported": False,
            "disease": "Unknown",
            "message": "Set GEMINI_API_KEY to enable Gemini fallback.",
        }

    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent"
    )

    prompt = """
You are helping classify tomato leaf disease from an uploaded image.

Return only valid JSON with this schema:
{
  "disease": "Early Blight" | "Late Blight" | "Healthy" | "Unknown",
  "confidence": number from 0 to 1,
  "keywords_found": ["early blight" or "late blight" or "healthy" if relevant],
  "reason": "short reason"
}

Rules:
- Choose Early Blight only if the image shows symptoms consistent with tomato early blight, such as brown circular lesions, concentric rings, or yellow halos.
- Choose Late Blight only if the image shows symptoms consistent with tomato late blight, such as water-soaked lesions, dark irregular patches, or pale/white mold-like growth.
- Choose Healthy only if the tomato leaf appears healthy.
- Choose Unknown if the image is unclear, not a tomato leaf, or symptoms do not clearly match Early Blight or Late Blight.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type or "image/jpeg",
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "response_mime_type": "application/json",
        },
    }

    request = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as error:
        return {
            "enabled": True,
            "supported": False,
            "disease": "Unknown",
            "error": str(error),
            "message": "Gemini fallback failed.",
        }

    result = parse_gemini_response(data)
    result["enabled"] = True
    return result


def parse_gemini_response(data):
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        result = json.loads(clean_json_text(text))
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        return {
            "supported": False,
            "disease": "Unknown",
            "confidence": 0.0,
            "keywords_found": [],
            "reason": "Could not parse Gemini response.",
            "raw_response": data,
        }

    disease = result.get("disease", "Unknown")
    if disease not in ALLOWED_DISEASES:
        disease = "Unknown"

    confidence = clamp_confidence(result.get("confidence", 0.0))
    supported = disease != "Unknown" and confidence >= 0.6

    return {
        "supported": supported,
        "disease": disease if supported else "Unknown",
        "confidence": confidence,
        "keywords_found": result.get("keywords_found", []),
        "reason": result.get("reason", ""),
    }


def clean_json_text(text):
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:]
    if value.startswith("```"):
        value = value[3:]
    if value.endswith("```"):
        value = value[:-3]
    return value.strip()


def clamp_confidence(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
