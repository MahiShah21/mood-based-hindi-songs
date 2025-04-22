import random

def recommend_music(facial_expression):
    """Recommends a Hindi music genre based on the detected facial expression."""
    expression = facial_expression.lower()
    if expression == "happy":
        genres = ["Bollywood Dance", "Hindi Pop", "Punjabi Pop"]
    elif expression == "sad":
        genres = ["Bollywood Sad Songs", "Hindi Ghazals", "Indian Classical (Vocal)"]
    elif expression == "angry":
        genres = ["Bollywood Rock", "Hindi Rap", "Sufi Rock"]
    elif expression == "surprise":
        genres = ["Bollywood Remix", "Indian Electronic Dance", "Indi Pop"]
    elif expression == "neutral":
        genres = ["Bollywood Instrumental", "Indian Folk (Relaxed)", "Ambient"]
    elif expression == "fear":
        genres = ["Bollywood Thriller Songs", "Dark Ambient (Indian Influence)"]
    elif expression == "disgust":
        genres = ["Experimental Indian", "Noise (if applicable in Indian context)"]
    else:
        return "No specific recommendation for this expression."
    return random.choice(genres)