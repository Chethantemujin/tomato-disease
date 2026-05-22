# Gemini Fallback

The API now supports this flow:

```text
Uploaded image
->
Your local TensorFlow model predicts first
->
If confidence is high: return local model result
->
If result is Unknown: ask Gemini to inspect the image
->
If Gemini is confident: return Gemini fallback result
->
Otherwise: return Unknown + web lookup links
```

## Enable Gemini

Use your Gemini API key:

```powershell
$env:GEMINI_API_KEY="your-gemini-api-key"
$env:TOMATO_API_KEY="my-secret-key"
uvicorn api.main:app --reload
```

Optional model override:

```powershell
$env:GEMINI_MODEL="gemini-2.5-flash"
```

## API Result

When Gemini is used, `/predict` can return:

```json
{
  "disease": "Late Blight",
  "confidence": 0.78,
  "supported": true,
  "prediction_method": "gemini_fallback",
  "local_prediction_method": "test_time_augmentation",
  "message": "The local model was uncertain, but Gemini suggests: Late Blight."
}
```

## Important

Gemini is not a replacement for your trained model. It is only used when your model returns `Unknown`.

Also, Gemini is doing visual reasoning from the uploaded image. It is not reading Google AI Overview or Google Images search results.
