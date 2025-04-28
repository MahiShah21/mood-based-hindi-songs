import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import webbrowser
from PIL import Image
import time

# Load your models and data (This should be in your main app file)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_model = load_model('emotion_recognition_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
hindi_song_links = {
    "happy": [
        "https://youtu.be/Cc_cNEjAh_Y?si=i0PKZ9aK9q9RydFh",  # Example Hindi Happy Song 1
        "https://youtu.be/Xf1922kJPfU?si=2GZj52VKSd-Wc9ts",  # Example Hindi Happy Song 2
        "hhttps://youtu.be/ZHsKQ_R0ZqI?si=nHGGH1FrvYXuXxEG",  # Example Hindi Happy Song 3
        "https://youtu.be/WuMWwPHTSoY?si=TjdYH6xuDZd1PWxI"   # Example Hindi Happy Song 4
    ],
    "sad": [
        "https://youtu.be/lN1m7zLBbSU?si=yBq7JWKTgGoTiA17",    # Example Hindi Sad Song 1
        "https://youtu.be/pon8irRa8II?si=A3uVtYkSA0DL8cHw",    # Example Hindi Sad Song 2
        "https://youtu.be/sVRwZEkXepg?si=538LGQErCKOM4c7k",    # Example Hindi Sad Song 3
        "https://youtu.be/jE3deqmhZy4?si=U8JetAZkg7yU4UaF"     # Example Hindi Sad Song 4
    ],
    "angry": [
        "https://youtu.be/jFGKJBPFdUA?si=IocpaNTI3_yQLszf",  # Example Hindi Angry Song 1
        "https://youtu.be/zLVZxHWL0ro?si=Mn9wVqvK2wsmOn7P",  # Example Hindi Angry Song 2
        "https://youtu.be/O6VbrzF79zI?si=9XdpksO1VVkF46QJ",  # Example Hindi Angry Song 3
        "https://youtu.be/p9DQINKZxWE?si=Xmi19_eD20Oa8XrF"   # Example Hindi Angry Song 4
    ],
    "surprise": [
        "https://youtu.be/HoCwa6gnmM0?si=BC2R7rC8YTPS-Rbl", # Example Hindi Surprise Song 1
        "https://youtu.be/FLz2eQtI_1w?si=QHIc5wQ1ablKi-U3", # Example Hindi Surprise Song 2
        "https://youtu.be/N5bELC8MXeU?si=yBSa4yPBmavxDBOV", # Example Hindi Surprise Song 3
        "https://youtu.be/NTHz9ephYTw?si=oomYezxkLG6Ub-IL"  # Example Hindi Surprise Song 4
    ],
    "neutral": [
        "https://youtu.be/Iy-6jmQCcrI?si=Vsmk353ih-W03k-u",  # Example Hindi Neutral Song 1
        "https://youtu.be/AsguumsKgBI?si=F7D_QqBc7We-F3Yp",  # Example Hindi Neutral Song 2
        "https://youtu.be/npao6yCXcGs?si=sjiGq8lPJNv3zEff",  # Example Hindi Neutral Song 3
        "https://youtu.be/NTHz9ephYTw?si=oomYezxkLG6Ub-IL"   # Example Hindi Neutral Song 4
    ],
    "fear": [
        "https://youtu.be/AsguumsKgBI?si=F7D_QqBc7We-F3Yp",     # Example Hindi Fear Song 1 (Instrumental)
        "https://youtu.be/WRoLW48WOBg?si=WhyAmLlrWTbxmkZs",     # Example Hindi Fear Song 2 (Instrumental)
        "https://youtu.be/lBvbNxiVmZA?si=eHI0QeGqkOy23Pn-",     # Example Hindi Fear Song 3 (Instrumental)
        "https://youtu.be/Oj484P4OXD8?si=zgJYO3jvu25p8GQl"      # Example Hindi Fear Song 4 (Instrumental)
    ],
    "disgust": [
        "https://youtu.be/TfWmovvr0cw?si=XLhZl0PJ4ZR_5pNV",    # Example Hindi Disgust Song 1 (Figurative)
        "https://youtu.be/7vnfdXAi5_Y?si=l0v_s2nMT2y-a9nM",    # Example Hindi Disgust Song 2 (Figurative)
        "https://youtu.be/FZLadzn5i6Q?si=2x0RdgmfxZb_91TP",    # Example Hindi Disgust Song 3 (Figurative)
        "https://youtu.be/WVMSjMtPbq0?si=NlVHrVHQ1QSRJmei"     # Example Hindi Disgust Song 4 (Figurative)
    ]
}

def recommend_music(facial_expression):
    """Recommends a Hindi song link based on the detected facial expression."""
    expression = facial_expression.lower()
    if expression in hindi_song_links and hindi_song_links[expression]:
        return random.choice(hindi_song_links[expression])
    else:
        return None  # Return None if no link found

# --- Playful Styling (Apply this to both pages)---
st.set_page_config(
    page_title="MoodTune Magic",
    page_icon="🎶",
)

st.markdown(
    """
    <style>
    .big-font {
        font-size:2.5rem !important;
        color:#FF4B4B; /* A vibrant red */
        font-weight: bold;
    }
    .emotion-text {
        font-size: 1.8rem !important;
        color: #1C83E1; /* A cheerful blue */
    }
    .recommendation-text {
        font-size: 1.2rem !important;
        color: #2E8B57; /* A friendly green */
    }
    .play-button {
        background-color: #FFA07A !important; /* A warm salmon color */
        color: white !important;
        padding: 0.5em 1.5em !important;
        border-radius: 10px !important;
        font-weight: bold !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Home Page ---
def home_page():
    st.markdown("<p class='big-font'>MoodTune Magic ✨</p>", unsafe_allow_html=True)
    st.subheader("Unleash your emotion, discover your sonic journey!")
    if st.button("Detect My Emotion! 📸", type="primary"):
        st.session_state.page = "detect"
        st.rerun()

# --- Detect Page ---
def detect_page():
    st.markdown("<p class='big-font'>Detecting Your Mood 🎵</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])
    video_placeholder = col1.empty()
    emotion_placeholder = col2.empty()
    recommendation_placeholder = col2.empty()
    play_button_placeholder = col2.empty()

    if st.button("Start Emotion Detection 📸", key="start_detection"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open webcam.")
            return

        with st.spinner("Detecting your emotion..."):
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(f"Grayscale image dtype before detectMultiScale: {gray.dtype}") # <--- THIS LINE

                faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
                detected_emotion = "No face detected"
                recommended_link = None  # Initialize to None
                detected_frame = frame.copy()  # To draw on

                if faces is not None and len(faces) > 0:
                    for (x, y, w, h) in faces:
                        face_roi = gray[y:y + h, x:x + w]
                        resized_face = cv2.resize(face_roi, (48, 48))
                        normalized_face = resized_face / 255.0
                        reshaped_face = np.expand_dims(normalized_face, axis=0)
                        reshaped_face = np.expand_dims(reshaped_face, axis=-1)
                        predictions = emotion_model.predict(reshaped_face)
                        predicted_emotion_index = np.argmax(predictions[0])
                        predicted_emotion = emotion_labels[predicted_emotion_index]
                        detected_emotion = f"<p class='emotion-text'>{predicted_emotion.capitalize()}!</p>"
                        recommended_link = recommend_music(predicted_emotion)
                        cv2.rectangle(detected_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(detected_frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        break # Process only the first detected face

                frame_rgb = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                video_placeholder.image(img, caption="Detected Face")
                emotion_placeholder.markdown(f"**Your Mood:** {detected_emotion}", unsafe_allow_html=True)
                if recommended_link:
                    recommendation_placeholder.markdown(f"<p class='recommendation-text'>**Tune Suggestion:** {recommended_link}</p>", unsafe_allow_html=True)
                    webbrowser.open_new_tab(recommended_link) # Directly open the link
                    play_button_placeholder.empty() # Remove the play button
                else:
                    recommendation_placeholder.markdown("<p class='recommendation-text'>**Tune Suggestion:** No specific song found.</p>", unsafe_allow_html=True)
                    play_button_placeholder.empty()

            else:
                with col1:
                    st.write("Click 'Start Emotion Detection' to begin!")
                with col2:
                    st.empty()  # Clear any previous results

        cap.release() # Release the webcam after detection

    if st.button("Back to Home", key="back_button"):
        st.session_state.page = "home"
        st.rerun()

# --- Main App Logic ---
if "page" not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "detect":
    detect_page()