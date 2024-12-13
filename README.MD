# Emotion-Based Game Recommendation System

This project is a Flask application that detects emotions from images and recommends games based on the detected emotion.

## Features
- Emotion detection from facial images using a pre-trained deep learning model.
- Game recommendations based on emotion-to-cluster mapping and a dataset of games.
- REST API endpoint for emotion detection and recommendations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Leon24k/Emotion-Based-Game-Recommendation-System.git
   cd EmotionGameRecommendation
   ```
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```
3. Install dependencies:
    ```bash
   pip install -r requirements.txt
    ```
4. Ensure the uploads directory exists:
    ```bash
    mkdir uploads
    ```
5. Place the following files in the project directory:
   - emotion_detection.h5: Pre-trained emotion detection model.
   - dataset.pkl: Pre-processed game recommendation dataset.

## Usage
1. Start the Flask server:
    ```bash
   python app.py
    ```
2. Send a POST request to the /recommend endpoint:
   - Endpoint: http://127.0.0.1:5000/recommend
   - Form-data:
      - Key: file
      - Value: (Upload an image file)
   - The server will return a JSON response containing
     - Detected emotion
     - Recommended games

## Example Response
```bash
{
  "emotion": "happy",
  "recommendations": [
    {"appid": 12345, "name": "Game 1", "price": 19.99},
    {"appid": 67890, "name": "Game 2", "price": 9.99}
  ]
}
```
## Dependencies
- Flask
- TensorFlow
- Pillow
- Pandas
- NumPy