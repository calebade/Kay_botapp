import streamlit as st
from textblob import TextBlob

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return "Positive"
    elif sentiment_score < 0:
        return "Negative"
    else:
        return "Neutral"

def collect_user_feedback(sentiment, user_feedback):
    # Store or process user feedback as needed
    st.write(f"User Feedback - Sentiment: {sentiment}, Feedback: {user_feedback}")

def main():
    st.title("Sentiment Analysis App")
    
    # User input text area
    user_input = st.text_area("Enter text:", height=150)

    # Analyze button
    if st.button("Analyze"):
        if user_input:
            # Sentiment analysis using TextBlob
            sentiment_textblob = analyze_sentiment_textblob(user_input)

            # Display sentiment
            st.write(f"TextBlob Sentiment: {sentiment_textblob}")

            # User feedback collection
            user_feedback = st.text_input("Provide feedback:")
            if st.button("Submit Feedback"):
                collect_user_feedback(sentiment_textblob, user_feedback)

if __name__ == "__main__":
    main()
