import streamlit as st
import torch
import pickle
import numpy as np
import re
import os
import gdown  

from model.base_models import BidirectionalRNN, AttentionClassifier
from model.bahdanau import BahdanauAttention

# Google Drive File IDs
EMBEDDING_FILE_ID = "113PyjIdTNwuIZbvjEcUcbdNolMzHmIKK"
MODEL_FILE_ID = "1RkVRAfOLD3HCwbaimCuZm2nrNIrBVgV9"

# Preprocessing
def preprocess_text(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    text = text.lower()
    text = re.sub(r"[^a-z0-9' ]", '', text)
    tokens = text.strip().split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return tokens

def encode_text(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

@st.cache_resource
def load_resources():
    os.makedirs("model", exist_ok=True)
    embedding_path = "model/embedding_matrix.pt"
    model_path = "model/BidirectionalRNN_Bahdanau.pth"

    if not os.path.exists(embedding_path):
        with st.spinner(" Downloading embedding matrix..."):
            gdown.download(f"https://drive.google.com/uc?id={EMBEDDING_FILE_ID}", embedding_path, quiet=False)

    if not os.path.exists(model_path):
        with st.spinner(" Downloading model weights..."):
            gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", model_path, quiet=False)

    with open("model/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    embedding_matrix = torch.load(embedding_path)

    hidden_dim = 128
    output_dim = 2
    base_model = BidirectionalRNN(embedding_matrix, hidden_dim, output_dim)
    attention = BahdanauAttention(
        encoder_hidden_dim=hidden_dim * 2,
        decoder_hidden_dim=hidden_dim * 2,
        attention_dim=64
    )
    model = AttentionClassifier(base_model, attention, hidden_dim * 2, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    return model, vocab

# Load model & vocab
model, vocab = load_resources()

# Streamlit UI
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")

st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #FF4B4B;'>🎬 IMDB Sentiment Analysis</h1>
        <p style='font-size: 18px;'>Built with Bidirectional RNN + Bahdanau Attention by leveraging Natural Language Processing techniques</p>
    </div>
""", unsafe_allow_html=True)

user_input = st.text_area("Enter a movie review below:", height=150, placeholder="e.g. The movie was absolutely fantastic with stunning performances...")

if st.button("Analyze Sentiment"):
    if not user_input.strip():
        st.warning("Please enter a review to analyze.")
    else:
        tokens = preprocess_text(user_input)
        encoded = encode_text(tokens, vocab)
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        lengths = torch.tensor([len(encoded)], dtype=torch.long)

        with torch.no_grad():
            logits, attn_weights = model(input_tensor, lengths)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()

        sentiment = "Positive" if pred_class == 1 else "Negative"
        st.markdown(f"### Prediction: **{sentiment}**")
        st.progress(confidence)
        st.markdown(f"**Model Confidence:** `{confidence:.4f}`")

        with st.expander(" View Attention Weights"):
            attn_weights = attn_weights[0][:len(tokens)].numpy()
            st.markdown("#### Attention Heatmap:")

            # Inject dual theme CSS
            st.markdown("""
    <style>
        .attention-word {
            padding: 6px 12px;
            margin: 6px;
            border-radius: 10px;
            font-weight: bold;
            display: inline-block;
            transition: all 0.2s ease-in-out;
        }

        /* Light mode styles */
        @media (prefers-color-scheme: light) {
            .attention-word {
                color: black;
                text-shadow: none;
            }
        }

        /* Dark mode styles */
        @media (prefers-color-scheme: dark) {
            .attention-word {
                color: white;
                text-shadow: 0 0 4px rgba(0,0,0,0.5);
            }
        }
    </style>
""", unsafe_allow_html=True)

# Then generate the HTML heatmap
            html_blocks = "".join([
    f"""
    <div class='attention-word' style='
        background: radial-gradient(circle, rgba(255,0,0,{weight:.3f}) 0%, rgba(100,0,0,0.2) 100%);
        box-shadow: 0 0 {6 + int(weight*12)}px rgba(255,0,0,{0.4 + weight/2});
    '>{word}</div>
    """
    for word, weight in zip(tokens, attn_weights)
])

            st.markdown(
    f"<div style='display: flex; flex-wrap: wrap;'>{html_blocks}</div>",
    unsafe_allow_html=True
)




            st.markdown(
            f"<div style='display: flex; flex-wrap: wrap;'>{html_blocks}</div>",
            unsafe_allow_html=True
    )



# Clean footer and hide Streamlit watermark
st.markdown("""
    <style>
        #MainMenu, footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <hr style='border: 1px solid #ddd; margin-top: 40px;'>
    <div style='text-align: center; color: gray; font-size: 14px;'>
        &copy; 2025 &nbsp; | &nbsp; Made with ❤️ by <strong>P. Vidya Praveen</strong> @ <em>Epoch, IIT Hyderabad</em>
    </div>
""", unsafe_allow_html=True)
