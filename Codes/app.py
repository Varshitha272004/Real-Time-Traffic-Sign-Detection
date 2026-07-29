
import streamlit as st
from PIL import Image
import cv2
import tempfile
import pyttsx3
import threading
import time
from detect import detect  # your existing YOLO detection function

# ----------------------------
# Streamlit Page Setup
# ----------------------------
st.set_page_config(page_title="Intelli Traffic Sign Detection", layout="wide")
st.title("🛑 Intelli Traffic Sign Detection")

# Display placeholders
frame_display = st.image([])
sign_display = st.empty()
processing_text = st.empty()

# File uploader for image/video
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "avi"])

# Determine source path
source = None
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1]
    tfile = tempfile.NamedTemporaryFile(suffix=f'.{file_ext}', delete=False)
    tfile.write(uploaded_file.read())
    tfile.flush()
    source = tfile.name

# ----------------------------
# TTS Function
# ----------------------------
def speak_text_nonblocking(text):
    """Speak text in a separate thread without blocking main loop"""
    def run_tts():
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run_tts, daemon=True).start()

# ----------------------------
# Real-Time Detection Loop
# ----------------------------
if source:
    processing_text.info("Processing... Please wait.")
    last_announced_time = {}  # track last announcement time per sign
    COOLDOWN = 5  # seconds between repeating the same sign

    try:
        # detect() should yield (frame, current_signs)
        for frame, current_signs in detect(source=source, view_img=False):
            # Convert BGR to RGB for Streamlit display
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_display.image(Image.fromarray(img_rgb), use_container_width=True)  # ✅ fixed here

            if current_signs:
                display_text = f"Detected Sign(s): {', '.join(current_signs)}"
                sign_display.success(display_text)

                # Announce ALL detected signs continuously (with cooldown)
                now = time.time()
                for sign in current_signs:
                    if sign not in last_announced_time or (now - last_announced_time[sign]) > COOLDOWN:
                        speak_text_nonblocking(f"{sign} detected ahead")
                        last_announced_time[sign] = now
            else:
                sign_display.info("No signs detected")
                last_announced_time.clear()  # reset when no signs

    except AssertionError as e:
        st.error(f"Error: {e}")

    processing_text.empty()
