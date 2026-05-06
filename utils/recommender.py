import json
from pathlib import Path

PLAYLIST_DIR = Path(__file__).resolve().parent.parent / "playlists"

def _load_playlist(name: str):
    p = PLAYLIST_DIR / f"{name}.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def pick_playlist(mood: str, mode: str = "match") -> list:
    """
    mood: 'happy' | 'sad' | 'neutral' | 'calm'
    mode: 'match' | 'boost'
    Returns a list of tracks [{title, artist, url, thumbnail}]
    """
    mood = (mood or "neutral").lower()
    mode = (mode or "match").lower()

    if mode == "match":
        if mood in ("happy",):
            return _load_playlist("happy_playlist")
        elif mood in ("sad",):
            return _load_playlist("sad_playlist")
        else:  # neutral/calm
            return _load_playlist("calm_playlist")
    else:  # boost
        if mood == "sad":
            return _load_playlist("happy_playlist")
        elif mood == "calm":
            return _load_playlist("happy_playlist")
        elif mood == "neutral":
            return _load_playlist("happy_playlist")
        else:  # already happy
            return _load_playlist("calm_playlist")
