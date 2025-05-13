import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set page config
st.set_page_config(page_title="Next Word Prediction", layout="centered")

# Initialize session state for theme
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Toggle theme if button clicked
if st.button("🌙" if not st.session_state.dark_mode else "☀️", key="theme_toggle", help="Toggle Light/Dark Mode"):
    st.session_state.dark_mode = not st.session_state.dark_mode

# CSS for positioning the toggle button to top-right
st.markdown("""
    <style>
    div[data-testid="stButton"][aria-label="theme_toggle"] {
        position: fixed;
        top: 10px;
        right: 15px;
        z-index: 9999;
    }
    </style>
""", unsafe_allow_html=True)

# Apply theme colors
if st.session_state.dark_mode:
    bg_color = "#121212"
    text_color = "#FFFFFF"
    card_color = "#1e1e1e"
else:
    bg_color = "#FFFFFF"
    text_color = "#000000"
    card_color = "#F9F9F9"

# Apply background and text color styling
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .prediction-box {{
        background-color: {card_color};
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }}
    </style>
""", unsafe_allow_html=True)

# Load model and tokenizer
model = load_model('D:/LSTM Project/models/next_word_lstm.h5')
with open('D:/LSTM Project/models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# App content
st.title("🔮 Next Word Prediction (LSTM + Early Stopping)")
st.write("Enter a sequence of words, and let the LSTM model predict the most likely next word.")

input_text = st.text_input("💬 Your Text", "To be or not to")

if st.button("🚀 Predict Next Word"):
    def predict_next_word(model, tokenizer, text, max_sequence_len):
        token_list = tokenizer.texts_to_sequences([text])[0]
        if len(token_list) >= max_sequence_len:
            token_list = token_list[-(max_sequence_len - 1):]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted, axis=1)[0]
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                return word
        return None

    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    if next_word:
        st.markdown(f"""
            <div class="prediction-box">
                <h3>🧠 Predicted Next Word:</h3>
                <h2 style="color:#4a90e2;">{next_word}</h2>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Couldn't predict the next word. Try with a different input.")
