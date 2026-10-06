"""Local browser interface for the phishing email detector."""

from pathlib import Path

import joblib
from flask import Flask, render_template, request

from phishing_detector import (
    generate_synthetic_dataset,
    predict_email,
    preprocess_dataset,
    train_model,
)


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"
VECTORIZER_PATH = BASE_DIR / "tfidf_vectorizer.joblib"

app = Flask(__name__)
_classifier = None


def get_classifier():
    """Load saved artifacts or train a small in-memory model on first use."""
    global _classifier
    if _classifier is not None:
        return _classifier

    if MODEL_PATH.is_file() and VECTORIZER_PATH.is_file():
        _classifier = (joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH))
    else:
        emails = preprocess_dataset(generate_synthetic_dataset())
        labels = emails["label"].astype(int).to_numpy()
        _classifier = train_model(emails["raw_text"].to_numpy(), labels)
    return _classifier


@app.route("/", methods=["GET", "POST"])
def index():
    email_text = ""
    result = None
    error = None

    if request.method == "POST":
        email_text = request.form.get("email_text", "")
        if not email_text.strip():
            error = "Enter an email message to check."
        else:
            model, vectorizer = get_classifier()
            label, confidence = predict_email(email_text, model, vectorizer)
            result = {"label": label, "confidence": confidence}

    return render_template(
        "index.html",
        email_text=email_text,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)