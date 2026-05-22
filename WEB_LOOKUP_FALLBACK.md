# Web Lookup Fallback

The API now uses this flow:

```text
Uploaded image
↓
Local TensorFlow model prediction
↓
If confidence is high: return disease result
↓
If confidence is low: return Unknown + web lookup links
```

The fallback appears only when the model returns `Unknown`.

If Google Custom Search credentials are configured, the API also checks Google result titles/snippets for:

```text
early blight
late blight
```

If one keyword appears more often than the other, the API can conclude that disease using:

```json
"prediction_method": "google_keyword_fallback"
```

## Important

This does not automatically reverse-search the private uploaded image. Google Lens reverse image search and Google AI Overview are not available through an official public API for this project.

Instead, the API returns web lookup links for visual comparison:

```text
tomato early blight leaf concentric rings
tomato late blight leaf water soaked lesions
```

Your app can show these links when prediction is uncertain.

## Enable Automatic Google Keyword Checking

Create a Google Programmable Search Engine and Custom Search JSON API key, then set:

```powershell
$env:GOOGLE_SEARCH_API_KEY="your-google-search-api-key"
$env:GOOGLE_SEARCH_ENGINE_ID="your-search-engine-id"
```

Then restart:

```powershell
uvicorn api.main:app --reload
```

Google's official Custom Search JSON API provides limited free daily quota for eligible projects. It does not provide AI Overview text.

## Example Unknown Response

```json
{
  "disease": "Unknown",
  "confidence": 0.52,
  "supported": false,
  "web_lookup": {
    "reason": "The local model was not confident enough to make a final prediction.",
    "candidates": [
      {
        "disease": "Early Blight",
        "links": {
          "google_images": "https://www.google.com/search?tbm=isch&q=..."
        }
      }
    ]
  }
}
```
