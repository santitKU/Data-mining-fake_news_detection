import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the tokenizer
tokenizer = None
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except FileNotFoundError:
    st.error("Error: tokenizer.pickle file not found. Make sure to run the data preprocessing step first.")
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")

maxlen = 100

# Load the trained model
model = None
try:
    model = load_model("model.h5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
except FileNotFoundError:
    st.error("Error: model.h5 file not found. Make sure to train and save the model first.")
except Exception as e:
    st.error(f"Error loading model: {e}")

def predict_news(news):
    if tokenizer is None or model is None:
        st.error("Error: Model or tokenizer not loaded.")
        return None
    
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([news])
    X = pad_sequences(sequence, maxlen=maxlen)
    # Make predictions
    prediction = model.predict(X)[0][0]
    if prediction >= 0.9:
        return 'Fake News'
    else:
        return 'Real News'

def main():
    st.title('Fake News Detection')

    # Text input for user to enter news
    news_input = st.text_area('Enter News:', '')

    if st.button('Detect'):
        if news_input:
            prediction = predict_news(news_input)
            if prediction:
                st.write('Predicted Class:', prediction)
        else:
            st.write('Please enter news to detect.')

if __name__ == '__main__':
    main()
