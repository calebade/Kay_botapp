# streamlit_app.py
import streamlit as st
import pandas as pd
from textblob import TextBlob

FEEDBACK_CSV_PATH = 'user_feedback.csv'

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
    feedback_data = {'Sentiment': [sentiment], 'Feedback': [user_feedback]}
    feedback_df = pd.DataFrame(feedback_data)

    # Append feedback to the CSV file
    feedback_df.to_csv(FEEDBACK_CSV_PATH, mode='a', header=not st.session_state.feedback_csv_exists, index=False)
    st.session_state.feedback_csv_exists = True

    st.write(f"User Feedback - Sentiment: {sentiment}, Feedback: {user_feedback}")
    st.success("Feedback submitted successfully!")

def main():
    st.title("Sentiment Analysis App with User Feedback")
    
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
    # Initialize a session state variable to check if the CSV file exists
    st.session_state.feedback_csv_exists = False

    # Run the Streamlit app
    main()

