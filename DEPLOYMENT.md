# Tomato Disease API Deployment

Your app needs two things:

1. A public API URL, for example `https://your-tomato-api.onrender.com/predict`
2. An API key, sent as the `x-api-key` header

The API key alone cannot make the model available. The API must be running on a public server.

## Generate Your API Key

```powershell
python generate_api_key.py
```

Copy the generated key. Use it as the server environment variable:

```text
TOMATO_API_KEY=your_generated_key
```

## Run Locally

```powershell
$env:TOMATO_API_KEY="your_generated_key"
uvicorn api.main:app --reload
```

Local endpoint:

```text
http://127.0.0.1:8000/predict
```

## Deploy Online

Upload this project to a hosting service that supports Python web apps.

Start command:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT
```

Environment variable:

```text
TOMATO_API_KEY=your_generated_key
```

The deployed endpoint will look like:

```text
https://your-server-domain/predict
```

## Request Format

Method:

```text
POST
```

URL:

```text
https://your-server-domain/predict
```

Header:

```text
x-api-key: your_generated_key
```

Body:

```text
multipart/form-data
file: tomato_leaf_image.jpg
```

Example response:

```json
{
  "disease": "Early Blight",
  "confidence": 0.94,
  "supported": true,
  "message": "The tomato leaf appears to be: Early Blight."
}
```
