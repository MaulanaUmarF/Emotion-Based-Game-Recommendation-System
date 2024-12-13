from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
import pickle
import os
import PIL

# Initialize Flask app
app = Flask(__name__)

# Load emotion detection model
emotion_model = load_model("emotion_detection.h5")  # Replace with your model path

# Load the game recommendation dataset
with open('dataset.pkl', 'rb') as file:
    df_filtered1 = pickle.load(file)

# Mapping emotion labels to cluster values
emotion_to_cluster = {
    "happy": 0,
    "joy": 0,
    "love": 0,
    "sad": 1,
    "fear": 1,
    "anger": 1,
    "neutral": 2,
    "disgust": 2,
    "surprise": 2
}

# List of emotions in the model
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def map_emotion_to_cluster(emotion):
    """
    Maps an emotion to a corresponding cluster value.

    Args:
        emotion (str): The predicted emotion.

    Returns:
        int: Cluster number for game recommendation.
    """
    return emotion_to_cluster.get(emotion.lower(), 2)  # Default to cluster 2


def recommend_games(cluster_value, num_recommendations=5):
    """
    Recommend games based on cluster value.

    Args:
        cluster_value (int): The cluster value for game recommendation.
        num_recommendations (int): Number of games to recommend.

    Returns:
        DataFrame: Top recommended games.
    """
    filtered_games = df_filtered1[df_filtered1['cluster'] == cluster_value]
    if 'positive_ratings' in filtered_games.columns:
        filtered_games = filtered_games.sort_values(by='positive_ratings', ascending=False)
    recommended_games = filtered_games.head(num_recommendations)
    return recommended_games[['appid', 'name', 'price']]


@app.route('/recommend', methods=['POST'])
def detect_emotion_and_recommend():
    """
    Detects emotion from an uploaded image and provides game recommendations based on emotion.

    Returns:
        JSON: Detected emotion and recommended games.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file and file.filename != '':
        # Save the uploaded file
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        try:
            # Preprocess image
            image = load_img(file_path, target_size=(48, 48), color_mode="rgb")  # Ubah color_mode menjadi "rgb"
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0) / 255.0

            # Predict emotion
            predictions = emotion_model.predict(image)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotion_labels[emotion_index]

            # Map detected emotion to cluster
            cluster_value = map_emotion_to_cluster(detected_emotion)

            # Get game recommendations
            recommended_games = recommend_games(cluster_value=cluster_value)
            recommendations = recommended_games.to_dict(orient='records')

            # Clean up the uploaded file
            os.remove(file_path)

            # Return response
            return jsonify({
                "emotion": detected_emotion,
                "recommendations": recommendations
            })

        except Exception as e:
            os.remove(file_path)
            return jsonify({"error": str(e)}), 500

    else:
        return jsonify({"error": "Invalid file"}), 400


if __name__ == '__main__':
    # Ensure the uploads directory exists
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    app.run(debug=True)