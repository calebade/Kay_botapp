import streamlit as st
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from faker import Faker
import random

# Function to predict sentiment using the trained model
def predict_sentiment(text, model):
    sentiment = model.predict([text])[0]
    return sentiment

# Load synthetic data
synthetic_data = generate_synthetic_data(num_samples=1000)

# Split the data into training and testing sets
train_data, test_data = train_test_split(synthetic_data, test_size=0.2, random_state=42)

# Extract features and labels
train_texts, train_labels = zip(*train_data)

# Create and train the model
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(train_texts, train_labels)

# Streamlit app
st.title("Sentiment Analysis App")

# User input text area
user_input = st.text_area("Enter text:", "")

# Prediction button
if st.button("Predict"):
    if user_input:
        # Predict sentiment using the model
        sentiment_prediction = predict_sentiment(user_input, model)
        st.write(f"Predicted Sentiment: {sentiment_prediction}")

# Display 5 samples of synthetic data
st.subheader("5 Samples of Synthetic Data:")
for i in range(5):
    st.write(f"Sample {i+1}:")
    st.write(f"Text: {synthetic_data[i][0]}")
    st.write(f"Sentiment: {synthetic_data[i][1]}")
    st.write("-" * 30)

