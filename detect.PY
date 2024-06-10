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

maxlen = 100

# Load the trained model
model = None
try:
    model = load_model(r"C:\Users\KU_AI_6\Desktop\fakenews\your_model.h5")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
except FileNotFoundError:
    st.error("Error: your_model.h5 file not found. Make sure to train and save the model first.")

print(model)


def predict_news(news):
    # Tokenize and pad the input text
    sequence = tokenizer.texts_to_sequences([news])
    X = pad_sequences(sequence, maxlen=maxlen)
    # Make predictions
    prediction = model.predict(X)[0][0]
    print(prediction)
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
            if tokenizer is None or model is None:
                st.error("Error: Model or tokenizer not loaded. Please make sure to run the data preprocessing and model training steps first.")
            else:
                prediction = predict_news(news_input)
                st.write('Predicted Class:', prediction)
        else:
            st.write('Please enter news to detect.')

if __name__ == '__main__':
    main()
