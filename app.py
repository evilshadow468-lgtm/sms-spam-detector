"""Web demo.  Run:  streamlit run app.py"""
import warnings
from pathlib import Path
import joblib
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")

MODEL_PATH = Path(__file__).parent / "models" / "spam_model.joblib"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("📩 SMS Spam Detector")
st.write("Paste a message and the model will estimate whether it is spam.")

text = st.text_area("Message", height=120,
                    placeholder="e.g. WINNER!! Claim your free prize now...")
threshold = st.slider("Spam threshold", 0.1, 0.9, 0.5, 0.05,
                      help="Lower = catches more spam but flags more real "
                           "messages by mistake.")

if st.button("Check"):
    if not text.strip():
        st.warning("Please enter a message to check.")
    else:
        prob = model.predict_proba([text])[0][1]
        if prob >= threshold:
            st.error(f"🚨 Likely SPAM ({prob:.1%})")
        else:
            st.success(f"✅ Looks legitimate ({prob:.1%} spam probability)")
        st.progress(float(prob))

