"""Classify messages from the command line.

Usage:
    python predict.py "Congratulations! You won a free prize, call now"
    python predict.py            (interactive mode)
"""
import sys
import warnings
from pathlib import Path

import joblib

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"
model = joblib.load(MODEL_PATH)


def classify(text):
    prob = model.predict_proba([text])[0][1]  # probability of spam
    return ("SPAM" if prob >= 0.5 else "HAM"), prob


if len(sys.argv) > 1:
    label, prob = classify(" ".join(sys.argv[1:]))
    print(f"{label}  (spam probability: {prob:.1%})")
else:
    print("Type a message (blank line to quit).")
    while True:
        msg = input("> ").strip()
        if not msg:
            break
        label, prob = classify(msg)
        print(f"  {label}  (spam probability: {prob:.1%})")
