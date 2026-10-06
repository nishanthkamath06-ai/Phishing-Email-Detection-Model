# Phishing Email Detection

A beginner-friendly scikit-learn project that classifies email text as `Phishing` or `Safe`. It combines TF-IDF text features with simple URL and message statistics, trains logistic regression, prints evaluation metrics, saves a confusion-matrix heatmap, and demonstrates predictions with confidence scores.

## Dataset

The CSV must have two columns:

- `text`: the email body
- `label`: `1` for Phishing or `0` for Safe

If no CSV is provided, the script generates 200 illustrative synthetic examples and writes them to `synthetic_emails.csv`. Synthetic data is only for demonstrating the workflow; its evaluation scores do not indicate real-world performance.

Real datasets can be found on Kaggle by searching for [Phishing Email Dataset](https://www.kaggle.com/datasets?search=phishing+email+dataset). Check the dataset's license and map its columns and labels to the required `text` and `label` format before using it.

## Run

From this folder, create and activate a virtual environment, then install the requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Start the browser interface:

```powershell
python web_app.py
```

Open <http://127.0.0.1:5000> in your browser. The first prediction trains a small model in memory when saved model files are not available.

To print evaluation metrics and save the model and confusion-matrix image, run the command-line version with generated sample data:

```powershell
python phishing_detector.py
```

Run with your own CSV:

```powershell
python phishing_detector.py --data path\to\emails.csv
```

The script creates `phishing_model.joblib`, `tfidf_vectorizer.joblib`, and `confusion_matrix.png` in the current folder. It also prints the accuracy, classification report, and three sample predictions. In Python, call `predict_email("email text here")` after training to reload the saved model and get a `(label, confidence)` result.