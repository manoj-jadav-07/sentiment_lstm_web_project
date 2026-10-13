import json
import os
import re
import io
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_lstm.keras")
WORD_INDEX_PATH = os.path.join(BASE_DIR, "models", "word_index.json")
MAX_LENGTH = 200
VOCAB_SIZE = 20000
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB upload limit

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

model = None
word_index = {}

def load_artifacts():
    global model, word_index
    if os.path.exists(MODEL_PATH) and os.path.exists(WORD_INDEX_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(WORD_INDEX_PATH, "r", encoding="utf-8") as f:
            word_index = json.load(f)
        print("LSTM model loaded.")
    else:
        print("Model files are missing. Run: python train_model.py")

def text_to_sequence(text):
    # The Keras IMDb dataset reserves 0 for padding, 1 for START, and 2 for OOV.
    tokens = re.findall(r"[a-zA-Z']+", str(text).lower())
    sequence = [1]
    for token in tokens:
        index = word_index.get(token)
        if index is None:
            sequence.append(2)
        else:
            sequence.append(index + 3)
    sequence = sequence[:MAX_LENGTH]
    sequence += [0] * (MAX_LENGTH - len(sequence))
    return sequence

def predict_review(text):
    if model is None:
        raise RuntimeError("The trained model is not available. Run python train_model.py first.")
    sequence = np.array([text_to_sequence(text)], dtype=np.int32)
    probability = float(model.predict(sequence, verbose=0)[0][0])
    sentiment = "Positive" if probability >= 0.5 else "Negative"
    confidence = probability if probability >= 0.5 else 1.0 - probability
    return {
        "sentiment": sentiment,
        "confidence": round(confidence * 100, 2),
        "positive_probability": round(probability * 100, 2),
        "negative_probability": round((1 - probability) * 100, 2)
    }

@app.route("/")
def home():
    ready = model is not None
    return render_template("index.html", model_ready=ready)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    review = str(data.get("review", "")).strip()
    if not review:
        return jsonify({"error": "Please enter a review first."}), 400
    try:
        result = predict_review(review)
        result["review"] = review
        return jsonify(result)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503

@app.route("/analyze_csv", methods=["POST"])
def analyze_csv():
    if model is None:
        return jsonify({"error": "Train the model first by running python train_model.py."}), 503
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "Choose a CSV file to upload."}), 400
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400
    try:
        df = pd.read_csv(file)
        if df.empty:
            return jsonify({"error": "The CSV file has no rows."}), 400

        # Use a column named 'review' when present; otherwise use the first column.
        column_lookup = {str(c).strip().lower(): c for c in df.columns}
        review_column = column_lookup.get("review", df.columns[0])
        reviews = df[review_column].dropna().astype(str).head(1000).tolist()
        if not reviews:
            return jsonify({"error": "No review text was found in the CSV."}), 400

        results = []
        for review in reviews:
            item = predict_review(review)
            results.append({
                "review": review[:300],
                "sentiment": item["sentiment"],
                "confidence": item["confidence"]
            })

        positive = sum(1 for r in results if r["sentiment"] == "Positive")
        negative = len(results) - positive
        return jsonify({
            "total": len(results),
            "positive": positive,
            "negative": negative,
            "results": results
        })
    except Exception as exc:
        return jsonify({"error": f"Could not read this CSV: {exc}"}), 400

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "File is too large. Maximum upload size is 5 MB."}), 413

if __name__ == "__main__":
    load_artifacts()
    app.run(debug=True)
