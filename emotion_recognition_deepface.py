import cv2
from deepface import DeepFace
import random

def recommend_music(facial_expression):
    """Recommends a music genre based on the detected facial expression."""
    expression = facial_expression.lower()
    if expression == "happy":
        genres = ["Pop", "Upbeat Electronic", "Indie Pop"]
    elif expression == "sad":
        genres = ["Classical", "Acoustic", "Lo-fi Hip Hop"]
    elif expression == "angry":
        genres = ["Rock", "Metal", "Punk"]
    elif expression == "surprise":
        genres = ["Electronic", "Dance", "World Music"]
    elif expression == "neutral":
        genres = ["Ambient", "Jazz", "Folk"]
    elif expression == "fear":
        genres = ["Suspense", "Dark Ambient"]
    elif expression == "disgust":
        genres = ["Experimental", "Noise"]
    else:
        return "No specific recommendation for this expression."
    return random.choice(genres)

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

    if frame is not None:
        try:
            # Analyze the frame for emotions
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

            if analysis:
                for face_data in analysis:
                    dominant_emotion = face_data['dominant_emotion']
                    region = face_data['region']
                    x, y, w, h = region['x'], region['y'], region['w'], region['h']

                    cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Get music recommendation
                    recommended_genre = recommend_music(dominant_emotion)
                    cv2.putText(frame, f"Genre: {recommended_genre}", (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        except Exception as e:
            print(f"Error during analysis: {e}")

    cv2.imshow('Emotion-Based Music Recommendation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()