# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D, Embedding
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def train_model():
    df = pd.read_csv("Tweets.csv")
    df = df[df['airline_sentiment'] != 'neutral']
    df = df[['text', 'airline_sentiment']]

    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['airline_sentiment'])

    tokenizer = Tokenizer(num_words=5000, lower=True)
    tokenizer.fit_on_texts(df['text'])
    sequences = tokenizer.texts_to_sequences(df['text'])
    padded = pad_sequences(sequences, maxlen=100)

    X_train, X_test, y_train, y_test = train_test_split(padded, df['label'], test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Embedding(5000, 128, input_length=100))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=3, batch_size=64, validation_data=(X_test, y_test), verbose=1)

    return model, tokenizer

# UI
st.title("👍Sentence Sentiment Analyzer💁")
st.write("Enter a sentence below to analyze its sentiment (Positive or Negative):")

user_input = st.text_input("Enter Here:")
if user_input:
    model, tokenizer = train_model()
    seq = tokenizer.texts_to_sequences([user_input])
    padded_input = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded_input)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😠"
    st.subheader(f"Sentiment: {sentiment}")
    st.write(f"Confidence: `{prediction:.2f}`")
