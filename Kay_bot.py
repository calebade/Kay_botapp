import streamlit as st
from textblob import TextBlob
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

def analyze_custom_sentiment(text):
    # Simple custom sentiment analysis function
    keywords_positive = ["happy", "joy", "excited"]
    keywords_negative = ["sad", "angry", "disappointed"]

    if any(keyword in text.lower() for keyword in keywords_positive):
        return "Positive"
    elif any(keyword in text.lower() for keyword in keywords_negative):
        return "Negative"
    else:
        return "Neutral"

def plot_sentiment_distribution(scores):
    labels = ["Negative", "Neutral", "Positive"]
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color=['red', 'gray', 'green'])
    ax.set_ylabel('Score')
    ax.set_title('Sentiment Distribution')
    st.pyplot(fig)

def main():
    st.title("Sentiment Analysis App")
    
    # User input text area
    user_input = st.text_area("Enter text:", height=150)

    # Analyze button
    if st.button("Analyze"):
        if user_input:
            # Sentiment analysis using TextBlob
            sentiment_textblob = analyze_sentiment_textblob(user_input)

            # Custom sentiment analysis function
            sentiment_custom = analyze_custom_sentiment(user_input)

            # Display sentiments
            st.write(f"TextBlob Sentiment: {sentiment_textblob}")
            st.write(f"Custom Sentiment: {sentiment_custom}")

            # Display a sentiment distribution plot
            scores = [0, 0, 0]  # Negative, Neutral, Positive
            if sentiment_textblob == "Negative":
                scores[0] += 1
            elif sentiment_textblob == "Neutral":
                scores[1] += 1
            else:
                scores[2] += 1

            if sentiment_custom == "Negative":
                scores[0] += 1
            elif sentiment_custom == "Neutral":
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
