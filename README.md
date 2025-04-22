# mood-based-hindi-songs

# Emotion-Based Hindi Music Recommender

This project detects facial emotions from a webcam feed and recommends a Hindi music genre, opening a relevant song link in your default web browser.

## Overview

The system uses:

* **Face Detection:** OpenCV's Haar cascade classifier to detect faces in the webcam feed.
* **Emotion Recognition:** A Convolutional Neural Network (CNN) model trained on a facial emotion dataset (specifically the FER-2013 dataset).
* **Music Recommendation:** A mapping of detected emotions to Hindi music genres, with links to example songs on platforms like YouTube.

## Setup

### Prerequisites

* **Python 3.12 or higher**
* **pip** (Python package installer)

### Installation
        TensorFlow
        Keras
        NumPy
        Potentially deepface and its dependencies


        
1.  **Clone the repository:**
    ```bash
    git clone <- link ->
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv moodtunes_env
    source moodtunes_env/bin/activate  # On Linux/macOS
    moodtunes_env\Scripts\activate  # On Windows
    ```

3.  **Install the required Python libraries:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create a `requirements.txt` file listing dependencies like `opencv-python`, `tensorflow`, `numpy`, `keras` (if you installed it separately), and `webbrowser`)*

    To create `requirements.txt`, you can run:
    ```bash
    pip freeze > requirements.txt
    ```
    Make sure you run this command within your activated virtual environment after installing all the necessary libraries.

### Trained Emotion Recognition Model

* The trained emotion recognition model (`emotion_recognition_model.h5`) is included in this repository. This file is necessary for the emotion detection to work without retraining.

### Training Data (For Retraining - Optional)

* The emotion recognition model was trained on the [FER-2013 dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).
* If you wish to retrain the model or explore the training process, you will need to download this dataset from Kaggle (you might need to create an account).
* The training script (`train_emotion_model.py`) expects the data to be organized in a specific way (e.g., a `fer2013.csv` file or folders containing images). Please refer to the comments in the training script for data preparation details.
* The `archive` folder in the repository might contain a subset or the original structure of the training data used.

## Usage

1.  **Ensure your webcam is connected and accessible.**

2.  **Run the main script:**
    ```bash
    python emotion_music_recommender.py
    ```

3.  **A window will open showing your webcam feed.**
    * Press the **'s'** key to capture a frame and detect your emotion. The detected emotion and a link to a relevant Hindi song will be printed in the terminal, and the song link will open in your default web browser.
    * Press the **'q'** key to quit the application.

## Notes

* The accuracy of emotion detection can vary depending on lighting conditions, facial expressions, and individual differences.
* The Hindi song recommendations are based on a predefined mapping of emotions to song links. You can modify the `hindi_song_links` dictionary in `emotion_music_recommender.py` to customize the song selections.

## Further Development (Optional)

* Improving emotion detection accuracy by training on more data or using a more complex model.
* Enhancing the music recommendation system with more nuanced genre selections or integration with music streaming service APIs.
* Adding a user interface for a more interactive experience.

## License

[Your License (e.g., MIT License)]