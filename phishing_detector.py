"""Beginner-friendly phishing email classifier using text and URL features."""

import argparse
import html
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split


MODEL_PATH = Path("phishing_model.joblib")
VECTORIZER_PATH = Path("tfidf_vectorizer.joblib")
CONFUSION_MATRIX_PATH = Path("confusion_matrix.png")

SUSPICIOUS_KEYWORDS = (
    "urgent",
    "verify",
    "password",
    "account",
    "suspended",
    "click",
    "login",
    "bank",
    "winner",
    "confirm",
)
URL_PATTERN = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
IP_PATTERN = re.compile(r"(?:\d{1,3}\.){3}\d{1,3}")


def clean_email(text):
    """Remove HTML markup, lowercase the message, and normalize whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def preprocess_dataset(dataframe):
    """Validate, clean, and de-duplicate a dataframe with text and label columns."""
    missing_columns = {"text", "label"} - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {sorted(missing_columns)}")

    emails = dataframe[["text", "label"]].dropna().copy()
    emails["raw_text"] = emails["text"].astype(str)
    emails["text"] = emails["raw_text"].map(clean_email)
    emails["label"] = pd.to_numeric(emails["label"], errors="coerce")
    emails = emails.dropna(subset=["label"])
    emails = emails[emails["label"].isin([0, 1])]
    emails = emails.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if emails.empty:
        raise ValueError("No usable rows found. Check that text is present and labels are 0 or 1.")
    if emails["label"].nunique() != 2:
        raise ValueError("The dataset must contain both labels: 0 (Safe) and 1 (Phishing).")
    return emails


def generate_synthetic_dataset(number_per_class=100):
    """Create a small illustrative dataset so the project runs without external data."""
    phishing_templates = [
        "URGENT: Verify your bank account password now at {url} or access will be suspended!",
        "Your account is suspended. Click {url} to confirm your login details immediately.",
        "Congratulations! You have won a prize. Click this link immediately to claim your reward.",
        "Security alert: verify your password at {url}. Urgent action is required!",
        "Click {url} to login and confirm your bank information before your account is locked.",
    ]
    safe_templates = [
        "Hello, the team meeting is scheduled for {day} at 10 AM. Notes are at {url}.",
        "Your monthly statement is ready in the company portal: {url}. Thank you.",
        "Here is the project update for {day}. Please review the shared document at {url}.",
        "Thanks for your order. The delivery status is available at {url}.",
        "The office will be closed on {day}. See the staff notice at {url}.",
    ]
    rows = []
    for index in range(number_per_class):
        phishing_url = f"http://login.example.test/action/{index}"
        safe_url = f"https://portal.example.test/updates/{index}"
        rows.append(
            {
                "text": phishing_templates[index % len(phishing_templates)].format(
                    url=phishing_url
                )
                + f" Case reference PH-{index:03d}.",
                "label": 1,
            }
        )
        rows.append(
            {
                "text": safe_templates[index % len(safe_templates)].format(
                    day=f"weekday {index % 7 + 1}", url=safe_url
                )
                + f" Message reference SAFE-{index:03d}.",
                "label": 0,
            }
        )
    return pd.DataFrame(rows)


def extract_handcrafted_features(emails):
    """Return sparse URL and message statistics for each raw email."""
    feature_rows = []
    for email_text in emails:
        raw_text = str(email_text)
        urls = [match.rstrip(".,!?;:)") for match in URL_PATTERN.findall(raw_text)]
        lower_text = raw_text.lower()
        keyword_count = sum(
            len(re.findall(rf"\b{re.escape(word)}\b", lower_text))
            for word in SUSPICIOUS_KEYWORDS
        )
        capital_words = re.findall(r"\b[A-Z]{2,}\b", raw_text)
        url_lengths = [len(url) for url in urls]

        feature_rows.append(
            [
                len(urls),
                int(any(IP_PATTERN.search(url) for url in urls)),
                int(any("@" in url for url in urls)),
                int(any("-" in url for url in urls)),
                (float(np.mean(url_lengths)) / 100) if url_lengths else 0.0,
                max(url_lengths, default=0) / 100,
                keyword_count,
                raw_text.count("!"),
                len(capital_words),
            ]
        )
    return csr_matrix(np.asarray(feature_rows, dtype=float))


def combine_features(emails, vectorizer, fit_vectorizer=False):
    """Combine TF-IDF text vectors with handcrafted numeric features."""
    cleaned_emails = [clean_email(email) for email in emails]
    if fit_vectorizer:
        text_features = vectorizer.fit_transform(cleaned_emails)
    else:
        text_features = vectorizer.transform(cleaned_emails)
    handcrafted_features = extract_handcrafted_features(emails)
    return hstack([text_features, handcrafted_features], format="csr")


def train_model(texts, labels):
    """Fit the vectorizer and classifier for reuse by the CLI and web app."""
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    train_features = combine_features(texts, vectorizer, fit_vectorizer=True)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(train_features, labels)
    return model, vectorizer


def train_and_evaluate(emails):
    """Split the data, train logistic regression, report metrics, and save artifacts."""
    raw_texts = emails["raw_text"].to_numpy()
    labels = emails["label"].astype(int).to_numpy()
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        raw_texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    model, vectorizer = train_model(train_texts, train_labels)
    test_features = combine_features(test_texts, vectorizer)
    predictions = model.predict(test_features)

    print("\nEvaluation on the held-out test set")
    print(f"Accuracy: {accuracy_score(test_labels, predictions):.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=[0, 1],
            target_names=["Safe", "Phishing"],
            zero_division=0,
        )
    )

    matrix = confusion_matrix(test_labels, predictions, labels=[0, 1])
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
    )
    plt.title("Phishing Email Detection")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {CONFUSION_MATRIX_PATH}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Vectorizer saved to: {VECTORIZER_PATH}")
    return model, vectorizer


def predict_email(text, model=None, vectorizer=None):
    """Predict an email label and return it with the model's confidence score."""
    model = model or joblib.load(MODEL_PATH)
    vectorizer = vectorizer or joblib.load(VECTORIZER_PATH)
    features = combine_features([text], vectorizer)
    probabilities = model.predict_proba(features)[0]
    predicted_label = int(model.classes_[int(np.argmax(probabilities))])
    label_name = "Phishing" if predicted_label == 1 else "Safe"
    confidence = float(np.max(probabilities))
    return label_name, confidence


def main():
    parser = argparse.ArgumentParser(description="Train and demo a phishing email detector.")
    parser.add_argument(
        "--data",
        type=Path,
        help="CSV path with text and label columns; omit to generate synthetic data.",
    )
    args = parser.parse_args()

    if args.data:
        if not args.data.is_file():
            parser.error(f"Dataset file not found: {args.data}")
        dataframe = pd.read_csv(args.data)
        print(f"Loaded dataset: {args.data}")
    else:
        dataframe = generate_synthetic_dataset()
        synthetic_path = Path("synthetic_emails.csv")
        dataframe.to_csv(synthetic_path, index=False)
        print(f"Generated {len(dataframe)} synthetic emails: {synthetic_path}")

    emails = preprocess_dataset(dataframe)
    print(f"Usable rows after cleaning: {len(emails)}")
    print("Label counts (0 = Safe, 1 = Phishing):")
    print(emails["label"].value_counts().sort_index().to_string())

    model, vectorizer = train_and_evaluate(emails)
    examples = [
        "URGENT! Verify your bank password now at http://192.168.1.20/login",
        "The weekly project notes are ready on the team portal at https://intranet.example.test/notes",
        "Your account is suspended. Click https://secure-check.example.test/confirm to login.",
    ]
    print("\nSample predictions")
    for index, text in enumerate(examples, start=1):
        label, confidence = predict_email(text, model, vectorizer)
        print(f"{index}. {label} (confidence: {confidence:.1%})")
        print(f"   Email: {text}")


if __name__ == "__main__":
    main()