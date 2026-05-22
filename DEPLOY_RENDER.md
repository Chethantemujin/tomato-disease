# Deploy Tomato Disease API On Render

This deploys your FastAPI API so your friend's app can call it.

## Files To Upload

Upload the project to GitHub, but do not upload:

```text
dataset/
tomatoleaf.zip
```

These are ignored by `.gitignore`.

Do upload:

```text
api/
models/tomato_disease_model.keras
models/class_names.json
requirements.txt
render.yaml
runtime.txt
```

## Render Settings

Build command:

```text
pip install -r requirements.txt
```

Start command:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

```text
TOMATO_API_KEY=your_api_key_for_friend_app
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
```

`GEMINI_API_KEY` is optional, but keep it if you want Gemini fallback when your model returns `Unknown`.

## After Deploy

Your public API will look like:

```text
https://tomato-disease-api.onrender.com/predict
```

Your friend's app should call:

```text
POST https://your-render-url/predict
Header: x-api-key: your_api_key_for_friend_app
Body: multipart/form-data
Field: file
```
