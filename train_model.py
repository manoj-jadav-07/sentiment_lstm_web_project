import json
import os
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

VOCAB_SIZE = 20000
MAX_LENGTH = 200
MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_lstm.keras")
WORD_INDEX_PATH = os.path.join(MODEL_DIR, "word_index.json")

def main():
    print("Downloading/loading the IMDb dataset (first run needs internet)...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post", truncating="post")

    model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True),
        LSTM(64, dropout=0.2, recurrent_dropout=0.0),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    print("\nTraining model...")
    model.fit(
        x_train, y_train,
        epochs=3,
        batch_size=128,
        validation_split=0.2,
        verbose=1
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest accuracy: {accuracy * 100:.2f}%")
    model.save(MODEL_PATH)

    word_index = imdb.get_word_index()
    with open(WORD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(word_index, f)

    print("\nTraining complete!")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Word index saved to: {WORD_INDEX_PATH}")
    print("Now run: python app.py")

if __name__ == "__main__":
    main()
