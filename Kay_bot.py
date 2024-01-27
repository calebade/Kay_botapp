# streamlit_app.py
import streamlit as st
from textblob import TextBlob
from transformers import pipeline
import matplotlib.pyplot as plt
import numpy as np

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_bert(text):
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline(text)
    return result[0]['label']

def plot_sentiment_distribution(scores):
    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color=['red', 'gray', 'green'])
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)

def main():
    st.title("Sophisticated Sentiment Analysis App")
    
    # User input text area
    user_input = st.text_area("Enter text:", height=150)

    # Analyze button
    if st.button("Analyze"):
        if user_input:
            # Sentiment analysis using TextBlob
            sentiment_textblob = analyze_sentiment_textblob(user_input)

            # Sentiment analysis using BERT
            sentiment_bert = analyze_sentiment_bert(user_input)

            # Display sentiments
            st.write(f"TextBlob Sentiment: {sentiment_textblob}")
            st.write(f"BERT Sentiment: {sentiment_bert}")

            # Display a sentiment distribution plot
            scores = [0, 0, 0]  # Negative, Neutral, Positive
            if sentiment_textblob == "Negative":
                scores[0] += 1
            elif sentiment_textblob == "Neutral":
                scores[1] += 1
            else:
                scores[2] += 1

            if sentiment_bert == "NEGATIVE":
                scores[0] += 1
            elif sentiment_bert == "NEUTRAL":
                scores[1] += 1
            else:
                scores[2] += 1

            # Normalize scores to percentages
            scores = np.array(scores) / sum(scores) * 100

            # Plot sentiment distribution
            st.write("Sentiment Distribution:")
            plot_sentiment_distribution(scores)

if __name__ == "__main__":
    main()
