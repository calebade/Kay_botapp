import streamlit as st
from textblob import TextBlob

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    st.title("Sentiment Analysis App")
    user_input = st.text_area("Enter text:")
    
    if st.button("Analyze"):
        if user_input:
            sentiment = analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
