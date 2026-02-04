import streamlit as st
import pandas as pd
import pickle
from src.clean_text import clean_text

# ---------- Page Configuration ----------
st.set_page_config(
    page_title="TruthSeeker | Fake News Detection",
    page_icon="🕵️‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    with open("model/fake_news_model.pkl", "rb") as f:
        return pickle.load(f)

model, vectorizer = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🕵️‍♀️ TruthSeeker")
    st.info("This tool uses advanced AI to analyze news articles and detect potential misinformation.")
    
    st.markdown("### How it works")
    st.markdown("1. Paste the news text.")
    st.markdown("2. Click **Analyze**.")
    st.markdown("3. View the prediction and confidence score.")
    
    st.markdown("---")
    st.caption("Developed by Bethala Vinay")

# ---------- Main Content ----------
st.title("Fake News & Misinformation Detector")
st.markdown("#### Verify the authenticity of news articles in seconds.")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Input News")
    news_text = st.text_area(
        "Paste the content below:",
        height=300,
        placeholder="Type or paste the news article here to check its authenticity...",
        label_visibility="collapsed"
    )
    
    analyze_btn = st.button("🔍 Analyze Veracity", type="primary")

with col2:
    st.markdown("### 📊 Analysis Results")
    
    if analyze_btn:
        if not news_text.strip():
            st.warning("⚠️ Please provide some text to analyze.")
        else:
            with st.spinner("Analyzing content patterns..."):
                # Prediction Logic
                cleaned_text = clean_text(news_text)
                vector = vectorizer.transform([cleaned_text])
                prediction = model.predict(vector)[0]
                confidence = model.predict_proba(vector).max()
                
                # Display Results
                if prediction == "REAL":
                    st.success("✅ **Result: REAL NEWS**")
                    st.toast("Analysis Complete: Real News Detected", icon="✅")
                else:
                    st.error("🚨 **Result: FAKE NEWS**")
                    st.toast("Analysis Complete: Fake News Detected", icon="🚨")
                
                st.metric(label="Confidence Score", value=f"{confidence:.1%}")
                st.progress(confidence, text="Confidence Level")
                
                st.markdown("---")
                if prediction == "REAL":
                     st.markdown("This article contains patterns consistent with reliable reporting.")
                else:
                     st.markdown("This article contains patterns often found in misinformation.")
    else:
        st.info("👈 Enter text and click Analyze to see results here.")

# Optional: Keep the CSV upload feature but put it in an expander to keep UI clean

