# 📩 SMS Spam Detector

An end-to-end machine learning project: it cleans text data, compares three
classifiers, tunes the best one with cross-validation, evaluates it properly on
imbalanced data, and serves predictions through a command-line tool and a
Streamlit web app.

## Results (held-out test set, 1,034 messages)

| Model               | Precision | Recall | F1    |
|---------------------|-----------|--------|-------|
| Naive Bayes         | 1.000     | 0.740  | 0.851 |
| Logistic Regression | 0.906     | 0.885  | 0.896 |
| Random Forest       | 1.000     | 0.824  | 0.904 |
| **Tuned Logistic Regression (final)** | 0.92 | 0.90 | **0.91** |

Overall accuracy of the final model: **98%**.

## Tech stack
Python, pandas, scikit-learn (TF-IDF, Pipeline, GridSearchCV), matplotlib,
joblib, Streamlit.

## How to run

### 1. Setup Environment
```bash
# Using standard venv
python -m venv .venv

# Activate on Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Or on Linux/macOS:
# source .venv/bin/activate

# Install dependencies:
pip install -r requirements.txt
# (or with uv: uv pip install -r requirements.txt)
```

### 2. Run Predictions & Web App
```bash
# Command-line prediction
python predict.py "Free prize! Claim your reward now"

# Launch Streamlit web app
streamlit run app.py
# (or if streamlit command isn't in PATH: python -m streamlit run app.py)
```

### 3. (Optional) Retrain Model
```bash
python train.py                   # trains, evaluates, saves the model + plots
```
The dataset (SMS Spam Collection, ~5,500 messages) downloads automatically
on the first run.

## Project structure
```
train.py          full training + evaluation pipeline
predict.py        command-line predictions
app.py            Streamlit web demo
models/           saved model (created by train.py)
reports/          plots + results.txt (created by train.py)
requirements.txt  project dependencies
```

## Recent Improvements & Fixes
- **Robust Path Resolution**: Replaced relative string paths with `pathlib.Path(__file__).parent` across `app.py`, `predict.py`, and `train.py` to prevent `FileNotFoundError` when executing from different working directories.
- **Clean Output & Warning Suppression**: Filtered cross-version scikit-learn unpickling warnings (`InconsistentVersionWarning`) for cleaner logs and UI presentation.
- **UI Validation**: Added input validation in `app.py` to prompt the user if the check button is pressed with an empty message field.
- **Cross-Platform Execution**: Documented reliable execution commands for Windows and Unix environments.


## Key design decisions
- **Imbalanced data (~13% spam):** accuracy alone is misleading, because
  predicting "ham" for everything already scores ~87%. I evaluated with
  precision, recall, and F1 on the spam class.
- **Stratified split:** train and test sets keep the same spam ratio.
- **Pipeline:** TF-IDF and the classifier are bundled, so preprocessing is
  fit only on training data (no data leakage) and the saved model accepts raw text.
- **class_weight="balanced":** makes the model pay more attention to the rare class.
- **Adjustable threshold in the app:** shows the precision/recall trade-off
  (a lower threshold catches more spam but flags more real messages).
- **Interpretability:** `reports/top_spam_words.png` shows which words push a
  message toward spam.

## Limitations and future work
- Trained on 2000s-era English SMS, so modern spam (WhatsApp, UPI scams,
  Hinglish messages) would need new data.
- Possible upgrades: sentence embeddings or a fine-tuned transformer, a
  Hinglish dataset, and deployment on Hugging Face Spaces or Google Cloud Run.
