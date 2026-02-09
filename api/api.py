from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

import joblib
import os
import re
from collections import Counter
from urllib.parse import urlparse, parse_qs
from datetime import datetime

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="MindPulse AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------
pipeline = joblib.load("Model/mental_health_model.pkl")

LABELS = {
    0: "Anxiety",
    1: "Bipolar",
    2: "Depression",
    3: "Normal",
    4: "Personality_disorder",
    5: "Stress",
    6: "Suicidal",
}

# --------------------------------------------------
# NLP
# --------------------------------------------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in stop_words
    ]
    return " ".join(tokens)

def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False

# --------------------------------------------------
# YOUTUBE
# --------------------------------------------------
youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

def extract_video_id(url: str):
    parsed = urlparse(url)
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    if "youtube.com" in parsed.netloc:
        return parse_qs(parsed.query).get("v", [None])[0]
    return None

# --------------------------------------------------
# HOME
# --------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open("UI/index.html", "r", encoding="utf-8") as f:
        return f.read()

# --------------------------------------------------
# MAIN ANALYSIS ENDPOINT
# --------------------------------------------------
@app.get("/analyze")
def analyze_video(
    youtube_url: str = Query(...),
    max_comments: int = 100,
    confidence_threshold: float = 0.55
):
    video_id = extract_video_id(youtube_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    # ---------- VIDEO METADATA ----------
    try:
        video_resp = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        snippet = video_resp["items"][0]["snippet"]
        title = snippet["title"]
        thumbnail = snippet["thumbnails"]["high"]["url"]

    except Exception:
        return {"error": "Failed to fetch video metadata"}

    # ---------- COMMENTS ----------
    category_counter = Counter()
    recent_comments = []

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            order="relevance",
            textFormat="plainText"
        )

        response = request.execute()

        for item in response["items"]:
            raw_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]

            if not is_english(raw_text):
                continue

            clean = preprocess_text(raw_text)

            probs = pipeline.predict_proba([clean])[0]
            best_score = float(max(probs))
            pred_idx = int(probs.argmax())

            if best_score < confidence_threshold:
                label = "Normal"
            else:
                label = LABELS[pred_idx]

            category_counter[label] += 1

            recent_comments.append({
                "text": raw_text,
                "label": label,
                "confidence": round(best_score, 2),
                "processed_time": datetime.now().strftime("%H:%M:%S")
            })

    except HttpError as e:
        return {"error": str(e)}

    total = sum(category_counter.values())

    # ---------- PERCENTAGES ----------
    percentages = {
        k: round((v / total) * 100, 2) if total else 0
        for k, v in category_counter.items()
    }

    # ensure all 7 labels exist (for UI cards)
    for lbl in LABELS.values():
        percentages.setdefault(lbl, 0)

    return {
        "video": {
            "id": video_id,
            "title": title,
            "thumbnail": thumbnail,
            "total_comments": total
        },
        "community_profile": percentages,
        "recent_comments": recent_comments[:20]
    }

