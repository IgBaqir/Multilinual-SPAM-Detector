import streamlit as st
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout

# Load the dataset
df = pd.read_csv('finaldata.csv')

# Preprocessing
texts = df['text'].astype(str).values
labels = df['labels'].map({'ham': 0, 'spam': 1}).values

# Tokenization and padding
max_words = 1000
tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

# Define the model
model = Sequential()
model.add(Embedding(max_words, 32, input_length=padded_sequences.shape[1]))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Split the dataset into training and testing sets
train_size = int(0.8 * len(padded_sequences))
train_texts = padded_sequences[:train_size]
train_labels = labels[:train_size]
test_texts = padded_sequences[train_size:]
test_labels = labels[train_size:]

# Train the model
epochs = 10
batch_size = 32
model.fit(train_texts, train_labels, epochs=epochs, batch_size=batch_size, validation_data=(test_texts, test_labels))

# Streamlit app
def main():
    st.title("Spam Detection")
    user_input = st.text_input("Enter a message:")
    if st.button("Predict"):
        user_sequence = tokenizer.texts_to_sequences([user_input])
        user_padded_sequence = pad_sequences(user_sequence, padding='post', maxlen=padded_sequences.shape[1])
        prediction = model.predict(user_padded_sequence)
        if prediction >= 0.5:
            st.write("The message is predicted as spam.")
        else:
            st.write("The message is predicted as not spam.")

if __name__ == "__main__":
    main()
