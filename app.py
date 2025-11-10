import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from transformers import pipeline
import numpy as np
import PyPDF2
from newspaper import Article

# Load DL model and tokenizer
model = load_model("news_category_model_dl.h5")
tokenizer = joblib.load("tokenizer.pkl")
le = joblib.load("label_encoder.pkl")
max_len = 20

# Load summarizer
summarizer = pipeline("summarization", model="t5-small")

# --- CSS Styling ---
st.markdown("""
<style>
body {background-color: #f0f2f6; font-family: 'Segoe UI', sans-serif;}
h1 {color: #1f77b4; text-align: center;}
textarea {border-radius: 10px; padding: 10px; font-size: 16px;}
.stButton>button {background-color: #1f77b4; color: white; border-radius: 8px; padding: 10px 20px; font-size: 16px; font-weight: bold;}
.stButton>button:hover {background-color: #105488; color: #fff;}
.result {background-color: #ffffff; border-radius: 10px; padding: 15px; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); margin-top: 10px;}
</style>
""", unsafe_allow_html=True)

# --- App Title ---
st.title("🗞️  Smart News Analyzer")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Paste Article", "Upload PDF", "Enter URL"])

# --- Tab 1: Text Input ---
with tab1:
    user_input = st.text_area("Enter news headline or full article:")
    if st.button("Analyze Article"):
        if user_input.strip():
            # Process text
            seq = tokenizer.texts_to_sequences([user_input])
            padded = pad_sequences(seq, maxlen=max_len, padding='post')
            pred = model.predict(padded)
            category = le.inverse_transform([np.argmax(pred)])[0]
            polarity = TextBlob(user_input).sentiment.polarity
            sentiment = "Positive" if polarity>0 else "Negative" if polarity<0 else "Neutral"
            try:
                summary = summarizer(user_input, max_length=100, min_length=30, do_sample=False)
                summary_text = summary[0]['summary_text']
            except:
                summary_text = "Summary could not be generated."
            st.markdown(f"""
            <div class="result">
                <h3>Category:</h3> {category}<br>
                <h3>Sentiment:</h3> {sentiment}<br>
                <h3>Summary:</h3> {summary_text}
            </div>
            """, unsafe_allow_html=True)

# --- Tab 2: PDF Upload ---
with tab2:
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if st.button("Analyze PDF"):
        if pdf_file is not None:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            pdf_text = "".join([page.extract_text() + " " for page in pdf_reader.pages]).strip()
            if pdf_text:
                seq = tokenizer.texts_to_sequences([pdf_text])
                padded = pad_sequences(seq, maxlen=max_len, padding='post')
                pred = model.predict(padded)
                category = le.inverse_transform([np.argmax(pred)])[0]
                polarity = TextBlob(pdf_text).sentiment.polarity
                sentiment = "Positive" if polarity>0 else "Negative" if polarity<0 else "Neutral"
                try:
                    summary = summarizer(pdf_text, max_length=150, min_length=50, do_sample=False)
                    summary_text = summary[0]['summary_text']
                except:
                    summary_text = "Summary could not be generated."
                st.markdown(f"""
                <div class="result">
                    <h3>Category:</h3> {category}<br>
                    <h3>Sentiment:</h3> {sentiment}<br>
                    <h3>Summary:</h3> {summary_text}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Could not extract text from PDF.")

# --- Tab 3: URL Input ---
with tab3:
    url_input = st.text_input("Enter news article URL:")
    if st.button("Analyze URL"):
        if url_input.strip():
            try:
                article = Article(url_input)
                article.download()
                article.parse()
                url_text = article.text
                seq = tokenizer.texts_to_sequences([url_text])
                padded = pad_sequences(seq, maxlen=max_len, padding='post')
                pred = model.predict(padded)
                category = le.inverse_transform([np.argmax(pred)])[0]
                polarity = TextBlob(url_text).sentiment.polarity
                sentiment = "Positive" if polarity>0 else "Negative" if polarity<0 else "Neutral"
                try:
                    summary = summarizer(url_text, max_length=150, min_length=50, do_sample=False)
                    summary_text = summary[0]['summary_text']
                except:
                    summary_text = "Summary could not be generated."
                st.markdown(f"""
                <div class="result">
                    <h3>Category:</h3> {category}<br>
                    <h3>Sentiment:</h3> {sentiment}<br>
                    <h3>Summary:</h3> {summary_text}
                </div>
                """, unsafe_allow_html=True)
            except:
                st.warning("Could not extract article from URL.")
