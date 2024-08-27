import streamlit as st 
import numpy as np 
import pickle 
from keras.models import load_model 
from keras.preprocessing.sequence import pad_sequences 

# Load The Model 
model = load_model('Next_word_lstm.h5')

# Load the tokenizer 
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word 
def predict_the_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None    

# Streamlit App 
st.title('Next Word Prediction with LSTM and Early Stop')

input_text = st.text_input("Enter the sequence of words", 'Barn. In the same')

if st.button("Predict the Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_the_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')
