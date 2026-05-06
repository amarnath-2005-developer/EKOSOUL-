# EkoSoul Prototype

An end-to-end, lightweight prototype for multimodal mood detection (text + voice) and mood-based music recommendations using Streamlit.

## Features
- Text Emotion Detection with a HuggingFace DistilBERT sentiment model (mapped to Happy/Sad/Neutral).
- Voice Emotion Lite: Upload a short audio (WAV/MP3). We extract MFCC + basic energy/pitch features and infer Sad/Calm/Happy via simple rules.
- Mood Match vs Mood Boost modes for recommendations.
- Pre-curated playlists (JSON) shown with thumbnails + links.

## Quickstart
```bash
# 1) Create and activate a virtual environment (Windows PowerShell shown)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Upgrade pip (fixes many install issues on Windows)
python -m pip install --upgrade pip

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run the app
streamlit run app.py
```

## Notes
- The text model maps Positive→Happy, Negative→Sad; borderline/short text→Neutral.
- For audio, keep uploads to ~3–5 seconds for the demo.
- You can replace playlist JSONs with your own links (Spotify/YouTube/Apple Music).
- If you want mic recording in the browser, consider adding `streamlit-webrtc`; this prototype keeps it simple with file upload.
