import streamlit as st
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model
model = tf.keras.models.load_model('my_model2.h5')

# Function to preprocess the input text
def preprocess_text(text):
    # Tokenization and padding
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequences = pad_sequences(sequences, padding='post', maxlen=max_length)
    return padded_sequences

# Function to predict spam or ham
def predict_spam(text):
    preprocessed_text = preprocess_text(text)
    prediction = model.predict(preprocessed_text)[0][0]
    return 'Spam' if prediction >= 0.5 else 'Ham'

# Load the dataset and tokenizer
df = pd.read_csv('finaldata.csv')
texts = df['text'].astype(str).values
labels = df['labels'].map({'ham': 0, 'spam': 1}).values
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
max_length = max([len(s.split()) for s in texts])

# Streamlit app
def main():
    st.title("Spam Detection")
    user_input = st.text_input("Enter a message:")
    if st.button("Predict"):
        prediction = predict_spam(user_input)
        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
