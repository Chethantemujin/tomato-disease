import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import urlopen
from urllib.parse import quote_plus


DISEASE_LOOKUP_KEYWORDS = [
    {
        "disease": "Early Blight",
        "keywords": [
            "tomato early blight leaf concentric rings",
            "tomato early blight brown spots yellow halo",
            "early blight tomato leaf symptoms",
        ],
    },
    {
        "disease": "Late Blight",
        "keywords": [
            "tomato late blight leaf water soaked lesions",
            "tomato late blight white mold leaf underside",
            "late blight tomato leaf symptoms",
        ],
    },
]


def build_web_lookup():
    google_result = run_google_keyword_fallback()

    return {
        "reason": "The local model was not confident enough to make a final prediction.",
        "note": (
            "Google AI Overview and Google Lens reverse image search are not available "
            "through an official public API. This fallback uses Google Custom Search "
            "result text when API credentials are configured."
        ),
        "google_keyword_result": google_result,
        "candidates": [
            {
                "disease": item["disease"],
                "keywords": item["keywords"],
                "links": build_links(item["keywords"]),
            }
            for item in DISEASE_LOOKUP_KEYWORDS
        ],
    }


def build_links(keywords):
    primary_query = keywords[0]
    return {
        "google_images": search_url(
            "https://www.google.com/search?tbm=isch&q=",
            primary_query,
        ),
        "google_web": search_url("https://www.google.com/search?q=", primary_query),
        "bing_images": search_url("https://www.bing.com/images/search?q=", primary_query),
    }


def search_url(base_url, query):
    return f"{base_url}{quote_plus(query)}"


def run_google_keyword_fallback():
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

    if not api_key or not search_engine_id:
        return {
            "enabled": False,
            "message": (
                "Set GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_ENGINE_ID to enable "
                "automatic Google result keyword checking."
            ),
        }

    query = "tomato leaf disease brown spots yellowing leaf symptoms"
    try:
        results = google_custom_search(api_key, search_engine_id, query)
    except (HTTPError, URLError, TimeoutError) as error:
        return {
            "enabled": True,
            "error": str(error),
            "message": "Google keyword lookup failed.",
        }

    keyword_scores = score_keywords(results)
    conclusion = conclude_from_scores(keyword_scores)

    return {
        "enabled": True,
        "query": query,
        "keyword_scores": keyword_scores,
        "conclusion": conclusion,
        "results_checked": len(results),
        "results": results[:5],
    }


def google_custom_search(api_key, search_engine_id, query):
    url = (
        "https://www.googleapis.com/customsearch/v1"
        f"?key={quote_plus(api_key)}"
        f"&cx={quote_plus(search_engine_id)}"
        f"&q={quote_plus(query)}"
        "&num=5"
    )
    with urlopen(url, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))

    items = payload.get("items", [])
    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        }
        for item in items
    ]


def score_keywords(results):
    text = " ".join(
        f"{result.get('title', '')} {result.get('snippet', '')}"
        for result in results
    ).lower()

    return {
        "Early Blight": text.count("early blight"),
        "Late Blight": text.count("late blight"),
    }


def conclude_from_scores(scores):
    early_score = scores["Early Blight"]
    late_score = scores["Late Blight"]

    if early_score == 0 and late_score == 0:
        return {
            "disease": "Unknown",
            "supported": False,
            "reason": "Neither early blight nor late blight appeared in the checked result text.",
        }

    if early_score > late_score:
        return {
            "disease": "Early Blight",
            "supported": True,
            "reason": "Early blight appeared more often in Google result text.",
        }

    if late_score > early_score:
        return {
            "disease": "Late Blight",
            "supported": True,
            "reason": "Late blight appeared more often in Google result text.",
        }

    return {
        "disease": "Unknown",
        "supported": False,
        "reason": "Early blight and late blight keyword counts were tied.",
    }
