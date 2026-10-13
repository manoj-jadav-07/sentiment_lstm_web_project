# Sentiment Analysis Using LSTM — Flask Web Application

A student-friendly deep-learning project that classifies text reviews as **Positive** or **Negative**. It includes single-review prediction, confidence display, CSV batch analysis, and summary statistics.

## Requirements

- Windows 10/11
- Python 3.10 or 3.11 (recommended for a straightforward TensorFlow setup)
- Internet connection for the first install and IMDb dataset download
- Around 2–4 GB of free disk space is recommended

## Project structure

```text
sentiment_lstm_web_project/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── sample_reviews.csv
├── models/
│   └── (generated after training)
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Step 1 — Install Python

Install Python 3.10 or 3.11 from https://www.python.org/downloads/windows/ .
During installation, select **Add Python to PATH**.

Open Command Prompt (CMD) and check:

```bat
python --version
pip --version
```

## Step 2 — Open the project folder

Extract the ZIP. Open CMD in the extracted folder. In File Explorer, open the folder, click the address bar, type `cmd`, and press Enter.

## Step 3 — Create and activate a virtual environment

```bat
python -m venv venv
venv\Scripts\activate
```

If activation is blocked in PowerShell, use **Command Prompt (CMD)** as shown above instead.

## Step 4 — Install dependencies

```bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow can take several minutes to install.

## Step 5 — Train the LSTM model

```bat
python train_model.py
```

The script downloads the IMDb movie-review dataset through Keras, trains an LSTM model, evaluates it on test data, and saves these files:

- `models/sentiment_lstm.keras`
- `models/word_index.json`

Training can take several minutes or longer depending on your computer. You only need to train once unless you change the model or dataset.

## Step 6 — Start the web application

```bat
python app.py
```

Open this address in Chrome or Edge:

http://127.0.0.1:5000

Keep the CMD window open while using the website. Press `Ctrl+C` in CMD to stop the server.

## Step 7 — Try the application

- Enter a movie review in the text box and click **Analyze sentiment**.
- Upload `sample_reviews.csv` in the batch analysis section.
- The CSV should contain a column called `review`. If it doesn't, the app uses the first column.
- The app analyzes up to 1,000 reviews per upload and limits file uploads to 5 MB.

## How it works

1. IMDb provides labelled positive and negative movie reviews.
2. The text is represented as integer word sequences.
3. An embedding layer converts token IDs into learned vectors.
4. LSTM processes the sequence and learns contextual patterns.
5. A sigmoid output predicts a probability from 0 to 1.
6. A probability of 0.5 or more is labelled Positive; otherwise it is labelled Negative.

## Common problems

### `python` is not recognized
Reinstall Python and select **Add Python to PATH**, or use the Python launcher `py`.

### Model files are missing
Run `python train_model.py` first and wait until training finishes.

### TensorFlow installation fails
Use 64-bit Python 3.10 or 3.11, create a fresh virtual environment, then rerun `pip install -r requirements.txt`. Read the error message carefully; TensorFlow compatibility depends on Python version and system setup.

### Port 5000 is already in use
Change the last line in `app.py` to:
```python
app.run(debug=True, port=5001)
```
Then open http://127.0.0.1:5001 .

### Predictions look incorrect
The model is trained on movie reviews, so it may not work well on sarcasm, mixed-language text, short phrases, or reviews from a different domain. Confidence is not a guarantee. For a stronger project, compare test accuracy and try a larger dataset or an LSTM/GRU model comparison.

## Academic note

This project is intended for local learning and demonstration. Before deploying it publicly, turn off Flask debug mode, add appropriate security controls, and review privacy requirements for uploaded reviews.
