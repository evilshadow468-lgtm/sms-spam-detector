"""
Train and compare spam classifiers on the SMS Spam Collection dataset.

Run:  python train.py
Output: models/spam_model.joblib, reports/*.png, reports/results.txt
"""
import os
import urllib.request
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")
DATA_PATH = DATA_DIR / "sms.tsv"
RANDOM_STATE = 42


def load_data():
    """Load the dataset, downloading it the first time."""
    if not DATA_PATH.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print("Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, str(DATA_PATH))
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "text"])

    df = df.drop_duplicates().dropna()
    df["target"] = (df["label"] == "spam").astype(int)  # spam=1, ham=0
    return df


def make_pipeline(classifier):
    """TF-IDF turns text into numbers; the classifier learns from them."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english")),
        ("clf", classifier),
    ])


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    print(f"Messages: {len(df)}  |  spam share: {df['target'].mean():.1%}")

    # Plot 1: class balance (the data is imbalanced, so accuracy alone misleads)
    df["label"].value_counts().plot(kind="bar", color=["#4C72B0", "#C44E52"])
    plt.title("Class distribution")
    plt.ylabel("Messages")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "class_distribution.png")
    plt.close()


    # stratify keeps the same spam ratio in train and test
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"], test_size=0.2,
        stratify=df["target"], random_state=RANDOM_STATE)

    candidates = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
    }

    lines, scores, fitted = [], {}, {}
    for name, clf in candidates.items():
        model = make_pipeline(clf).fit(X_train, y_train)
        preds = model.predict(X_test)
        scores[name] = {
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
        }
        fitted[name] = model
        lines.append(f"{name}: " + ", ".join(
            f"{k}={v:.3f}" for k, v in scores[name].items()))
        print(lines[-1])

    # Tune the best baseline family with cross-validation
    grid = GridSearchCV(
        make_pipeline(LogisticRegression(max_iter=1000,
                                         class_weight="balanced")),
        param_grid={"tfidf__ngram_range": [(1, 1), (1, 2)],
                    "clf__C": [1, 5, 10]},
        scoring="f1", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    preds = best.predict(X_test)
    lines.append(f"\nTuned Logistic Regression best params: {grid.best_params_}")
    lines.append("\nFinal test report (tuned model):\n" +
                 classification_report(y_test, preds,
                                       target_names=["ham", "spam"]))
    print(lines[-2])
    print(lines[-1])

    # Plot 2: confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, preds, display_labels=["ham", "spam"], cmap="Blues")
    plt.title("Confusion matrix (tuned model)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()

    # Plot 3: words the model finds most spammy
    vec = best.named_steps["tfidf"]
    coefs = best.named_steps["clf"].coef_[0]
    words = vec.get_feature_names_out()
    top = pd.Series(coefs, index=words).nlargest(15)[::-1]
    top.plot(kind="barh", color="#C44E52")
    plt.title("Top 15 words pushing a message toward SPAM")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "top_spam_words.png")
    plt.close()

    # Plot 4: model comparison
    pd.DataFrame(scores).T.plot(kind="bar", ylim=(0.6, 1.0))
    plt.title("Model comparison on test set")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png")
    plt.close()

    with open(REPORTS_DIR / "results.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    model_path = MODELS_DIR / "spam_model.joblib"
    joblib.dump(best, model_path)
    print(f"\nSaved model to {model_path}")



if __name__ == "__main__":
    main()
