from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def predict_text_emotion(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        return {"label": "neutral", "raw_label": "NEUTRAL", "score": 0.0}

    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    # Heuristic mapping for moods
    if compound >= 0.35:
        label = "happy"
    elif compound <= -0.35:
        label = "sad"
    else:
        label = "neutral"

    return {
        "label": label,
        "raw_label": f"compound={compound}",
        "score": compound
    }
