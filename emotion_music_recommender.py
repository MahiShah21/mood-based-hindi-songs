import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import webbrowser

# Load the pre-trained face detection model (Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load your trained emotion recognition model
emotion_model = load_model('emotion_recognition_model.h5')
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

hindi_song_links = {
    "happy": [
        "https://youtu.be/Cc_cNEjAh_Y?si=i0PKZ9aK9q9RydFh",  # Hindi Happy Song 1
        "https://youtu.be/Xf1922kJPfU?si=2GZj52VKSd-Wc9ts",  # Hindi Happy Song 2
        "https://youtu.be/ZHsKQ_R0ZqI?si=nHGGH1FrvYXuXxEG",  # Hindi Happy Song 3
        "https://youtu.be/WuMWwPHTSoY?si=TjdYH6xuDZd1PWxI"   # Hindi Happy Song 4
    ],
    "sad": [
        "https://youtu.be/AGsn2ycFRqI?si=tnvf0CkaG6cG0au5",   # Hindi Sad Song 1
        "https://youtu.be/lmhKXQBgEQU?si=dJ22_MKgPRAEGix6",   # Hindi Sad Song 2
        "https://youtu.be/6MgsHSAcI9k?si=giYRVzGPShebbE1O",   # Hindi Sad Song 3
        "https://youtu.be/lN1m7zLBbSU?si=nBWMDBaW6mTl0gkm"    # Hindi Sad Song 4
    ],
    "angry": [
        "https://www.youtube.com/watch?v=ESCjZui9ybM",  # Hindi Angry Song 1
        "https://www.youtube.com/watch?v=TD6kW4oCcZY",  # Hindi Angry Song 2
        "https://www.youtube.com/watch?v=T5Aqa1KjIqQ",  # Hindi Angry Song 3
        "https://www.youtube.com/watch?v=mWRsgZuwf7w3"   # Hindi Angry Song 4
    ],
    "surprise": [
        "https://www.youtube.com/watch?v=mWRsgZuwf7w4", # Hindi Surprise Song 1
        "https://www.youtube.com/watch?v=mWRsgZuwf7w5",  # Hindi Surprise Song 2
        "https://www.youtube.com/watch?v=mWRsgZuwf7w6",  # Hindi Surprise Song 3
        "https://www.youtube.com/watch?v=mWRsgZuwf7w7"   # Hindi Surprise Song 4
    ],
    "neutral": [
        "https://www.youtube.com/watch?v=mWRsgZuwf7w8",  # Hindi Neutral Song 1
        "https://www.youtube.com/watch?v=mWRsgZuwf7w9",  # Hindi Neutral Song 2
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ0",  # Hindi Neutral Song 3
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ1"   # Hindi Neutral Song 4
    ],
    "fear": [
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ2",  # Hindi Fear Song 1 (Soundtrack)
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ3",  # Hindi Fear Song 2 (Soundtrack)
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ4",  # Hindi Fear Song 3 (Soundtrack)
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ5"   # Hindi Fear Song 4 (Soundtrack)
    ],
    "disgust": [
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ6", # Hindi Experimental Song 1 (Subjective)
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ7",  # Hindi Experimental Song 2 (Subjective)
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ8",  # Hindi Experimental Song 3 (Subjective)
        "https://www.youtube.com/watch?v=f4Gu-4W_peQ9"   # Hindi Experimental Song 4 (Subjective)
    ]
}

def recommend_music(facial_expression):
    """Recommends a Hindi song link based on the detected facial expression."""
    expression = facial_expression.lower()
    if expression in hindi_song_links and hindi_song_links[expression]:
        return random.choice(hindi_song_links[expression])  # Shuffles the songs
    else:
        return "No specific Hindi song link for this emotion."

# Open the default webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    cv2.imshow('Press "s" to capture and get song link, "q" to quit', frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        # 's' key pressed, capture the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if faces is not None and len(faces) > 0:
            (x, y, w, h) = faces[0]  # Consider only the first detected face
            face_roi = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face_roi, (48, 48))
            normalized_face = resized_face / 255.0
            reshaped_face = np.expand_dims(normalized_face, axis=0)
            reshaped_face = np.expand_dims(reshaped_face, axis=-1)

            # Predict emotion
            predictions = emotion_model.predict(reshaped_face)
            predicted_emotion_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[predicted_emotion_index]

            # Get music recommendation (song link)
            print("Predicted Emotion:", predicted_emotion)
            recommended_song_link = recommend_music(predicted_emotion)
            print("Recommended Song Link:", recommended_song_link)

            # Open the link in the default web browser
            if recommended_song_link and recommended_song_link.startswith("http"):
                webbrowser.open_new_tab(recommended_song_link)
                cv2.putText(frame, "Link opened in browser", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "No valid link", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {predicted_emotion}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Detected Emotion', frame) # Show the frame with detection
        else:
            print("No face detected.")

        break  # Exit the loop after one detection

    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()