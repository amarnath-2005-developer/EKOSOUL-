import io
import streamlit as st
import numpy as np
import soundfile as sf
import av

from streamlit_webrtc import webrtc_streamer, WebRtcMode

from utils.text_emotion import predict_text_emotion
from utils.audio_emotion import predict_audio_emotion
from utils.recommender import pick_playlist

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="EkoSoul • Mood AI", page_icon="🎶", layout="centered")

st.title("🎶 EkoSoul – Mood AI (Prototype)")
st.caption("Multimodal mood detection (text + voice) • Mood Match or Boost • Curated playlists")

# -----------------------
# Sidebar – Mode selection
# -----------------------
with st.sidebar:
    st.header("Mode")
    mode = st.radio("Recommendation Mode", ["Match", "Boost"], index=0, horizontal=True)
    st.markdown(
        "**Match**: recommends music that reflects your mood.\n\n**Boost**: recommends music to lift/soothe your mood."
    )

# -----------------------
# 1) TEXT EMOTION
# -----------------------
st.subheader("1) Text Emotion Detection")
text = st.text_area("Type how you feel (English works best):", placeholder="I'm feeling great today!")

if st.button("Analyze Mood (Text)"):
    res = predict_text_emotion(text)
    st.success(f"Detected mood: **{res['label'].upper()}**  \n(model: {res['raw_label']} • score={res['score']:.2f})")
    playlist = pick_playlist(res["label"], mode.lower())

    st.markdown("---")
    st.markdown("### 🎧 Recommended Playlist")
    for track in playlist:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(track["thumbnail"], use_container_width=True)
        with col2:
            st.markdown(f"**{track['title']}** — {track['artist']}  \n[Open Link]({track['url']})")

st.markdown("---")

# -----------------------
# Prepare session state for audio buffer & sample rate
# -----------------------
if "audio_buffer" not in st.session_state:
    st.session_state["audio_buffer"] = []  # list of numpy arrays
if "audio_sample_rate" not in st.session_state:
    st.session_state["audio_sample_rate"] = None

# -----------------------
# 2) VOICE EMOTION – HYBRID MODE (UPLOAD + RECORD)
# -----------------------
st.subheader("2) Voice Emotion Detection")

audio_input_mode = st.radio(
    "Choose Audio Input Method:",
    ["Upload File", "Record from Microphone"],
    horizontal=True
)

# File uploader (works when user chooses Upload)
uploaded_audio = None
if audio_input_mode == "Upload File":
    uploaded_audio = st.file_uploader("Upload WAV/MP3 (3–10 seconds)", type=["wav", "mp3", "ogg", "flac"])

# -----------------------
# Audio callback (append to session_state buffer)
# -----------------------
def audio_frame_callback(frame: av.AudioFrame):
    """
    Called by streamlit-webrtc for each incoming audio frame.
    Store numpy chunks and remember sample rate from the frame.
    """
    try:
        arr = frame.to_ndarray().astype(np.float32).flatten()
    except Exception:
        # If to_ndarray fails, skip the frame
        return frame

    # Append
    st.session_state["audio_buffer"].append(arr)

    # Store last seen sample rate (frames are usually consistent)
    try:
        sr = int(frame.sample_rate)
        st.session_state["audio_sample_rate"] = sr
    except Exception:
        # fallback if sample_rate not present
        pass

    return frame

# -----------------------
# Initialize WebRTC once (always) to avoid crash on re-runs
# -----------------------
webrtc_container = st.container()
with webrtc_container:
    webrtc_ctx = webrtc_streamer(
        key="mic_recorder_fixed_v1",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_frame_callback,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
        },
    )

# Show helpful UI depending on mode
if audio_input_mode == "Upload File":
    st.info("Upload an audio file and then click **Analyze Mood (Audio)**.")
else:
    st.success("🎤 Microphone ready. Click **Start** in the player, speak, then click **Stop** and then **Analyze Mood (Audio)**.")
    # show captured frames count for debugging / confirmation
    st.write("Audio frames captured so far:", len(st.session_state["audio_buffer"]))

# -----------------------
# Analyze button (works for both modes)
# -----------------------
if st.button("Analyze Mood (Audio)"):
    # Upload path
    if audio_input_mode == "Upload File":
        if uploaded_audio is None:
            st.warning("Please upload an audio file first.")
            st.stop()
        audio_bytes = uploaded_audio.read()
        audio_data = io.BytesIO(audio_bytes)

    # Microphone path
    else:
        if len(st.session_state["audio_buffer"]) == 0:
            st.warning("No audio recorded yet. Click Start → Speak → Stop in the mic widget.")
            st.stop()

        # Concatenate all chunks
        try:
            audio_np = np.concatenate(st.session_state["audio_buffer"])
        except Exception:
            st.warning("Failed to concatenate recorded audio frames.")
            st.stop()

        # Determine sample rate (fallback to 16000)
        sr = st.session_state.get("audio_sample_rate") or 16000

        wav_io = io.BytesIO()
        # Write buffer as WAV using the detected sample rate
        sf.write(wav_io, audio_np, sr, format="WAV")
        wav_io.seek(0)
        audio_data = wav_io

        # Clear buffer after capturing to avoid reusing the same audio
        st.session_state["audio_buffer"].clear()
        st.session_state["audio_sample_rate"] = None

    # Run the user's existing prediction function
    try:
        res = predict_audio_emotion(audio_data)
    except Exception as e:
        st.error(f"Audio prediction failed: {e}")
        st.stop()

    st.success(f"Detected voice mood: **{res['label'].upper()}**")

    with st.expander("See audio features"):
        st.json(res.get("features", {}))

    playlist = pick_playlist(res["label"], mode.lower())

    st.markdown("---")
    st.markdown("### 🎧 Recommended Playlist")
    for track in playlist:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(track["thumbnail"], use_container_width=True)
        with col2:
            st.markdown(f"**{track['title']}** — {track['artist']}  \n[Open Link]({track['url']})")

st.markdown("---")
st.caption("Powered by Streamlit + WebRTC • Add Spotify API later for dynamic playlists.")
