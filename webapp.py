import streamlit as st
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle

# Function to preprocess user input
def preprocess_input(user_input, lang):
    if lang == 'hindi':
        with open('/Users/baqir/jupyter codes/myproject/tokenizer.pkl', 'rb') as f:
             tokenizer = pickle.load(f)
        with open('/Users/baqir/jupyter codes/myproject/max_length.pkl', 'rb') as f:
             max_length = pickle.load(f)
         
    elif lang == 'english':
        with open('/Users/baqir/jupyter codes/myproject/tokenizer2.pkl', 'rb') as f:
             tokenizer = pickle.load(f)
        with open('/Users/baqir/jupyter codes/myproject/max_length2.pkl', 'rb') as f:
             max_length = pickle.load(f)
    else:
        raise ValueError("Invalid language selection.")
          
    user_sequence = tokenizer.texts_to_sequences([user_input])
    user_padded_sequence = pad_sequences(user_sequence, padding='post', maxlen=max_length)
    return user_padded_sequence


# Function to load the model
def loading_model(lang):
    if lang == 'english':
        model = load_model('/Users/baqir/jupyter codes/myproject/model2.h5')
    elif lang == 'hindi':
        model = load_model('/Users/baqir/jupyter codes/myproject/model1.h5')
    else:
        raise ValueError("Invalid language selection.")
    return model

# Function to predict spam or not spam
def predict_spam(user_input, lang):
    processed_input = preprocess_input(user_input, lang)
    model = loading_model(lang)
    prediction = model.predict(processed_input)
    return prediction

# Streamlit app
def main():
    st.title("Spam Detection Web App")
    
    language = st.selectbox("Select Language", options=["English", "Hindi"])
    lang = language.lower()
    
    user_input = st.text_input("Enter a message:")
    
    if st.button("Predict"):
        prediction = predict_spam(user_input, lang)
        if prediction >= 0.5:
            st.write("The message is predicted as spam.")
        else:
            st.write("The message is predicted as not spam.")

if __name__ == '__main__':
    main()
