from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

app = Flask(__name__)

# Example data
videos = pd.DataFrame([
    {"video_id": 1, "title": "Dance Moves", "tags": "dance, music, fun", "categories": "Entertainment"},
    {"video_id": 2, "title": "Cooking Tips", "tags": "cooking, food, recipes", "categories": "Food"},
    {"video_id": 3, "title": "Fitness Routine", "tags": "workout, fitness, health", "categories": "Fitness"}
])

users = pd.DataFrame([
    {"user_id": 1, "liked_videos": [1], "preferences": "dance, music, fun"},
    {"user_id": 2, "liked_videos": [2], "preferences": "cooking, food, recipes"}
])

# Vectorize tags and categories
vectorizer = TfidfVectorizer()
videos["combined_features"] = videos["tags"] + " " + videos["categories"]
video_vectors = vectorizer.fit_transform(videos["combined_features"])

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    user_id = data.get("user_id")

    # Get user preferences
    user = users[users["user_id"] == user_id]
    if user.empty:
        return jsonify({"error": "User not found"}), 404

    user_preferences = user["preferences"].values[0]
    user_vector = vectorizer.transform([user_preferences])

    # Calculate similarity
    similarity_scores = cosine_similarity(user_vector, video_vectors).flatten()
    recommended_indices = similarity_scores.argsort()[::-1]

    # Get top recommended videos
    recommendations = videos.iloc[recommended_indices].head(5).to_dict("records")

    return jsonify(recommendations)

if __name__ == "__main__":
    app.run(debug=True)
