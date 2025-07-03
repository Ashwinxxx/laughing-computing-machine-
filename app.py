import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import re

# --- Config ---
SEQUENCE_LENGTH = 10
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
TRAINING_DATA_LIMIT = 10000
TEXT_URL = 'https://www.gutenberg.org/files/1661/1661-0.txt'

# --- NLTK Setup ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    st.success("✅ NLTK data ready!")

# --- Data Loading & Preprocessing ---
@st.cache_resource
def load_and_preprocess_data(text_url, sequence_length, training_data_limit):
    st.info("📚 Loading & preprocessing Sherlock Holmes text...")
    try:
        response = requests.get(text_url)
        response.raise_for_status()
        text = response.content.decode('utf-8').lower()
    except Exception as e:
        st.error(f"❌ Failed to download dataset: {e}")
        st.stop()

    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)

    word_counts = Counter(tokens)
    vocab = ['<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(vocab)

    def encode(seq):
        return [word2idx.get(w, word2idx['<unk>']) for w in seq]

    data = [(tokens[i:i+sequence_length-1], tokens[i+sequence_length-1])
            for i in range(len(tokens)-sequence_length)]
    encoded_data = [(torch.tensor(encode(x)), torch.tensor(word2idx.get(y, word2idx['<unk>'])))
                    for x, y in data[:training_data_limit]]

    st.success(f"✅ Data ready: {len(tokens)} tokens, Vocab Size: {vocab_size}")
    return vocab_size, word2idx, idx2word, encoded_data, encode

# --- LSTM Model ---
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# --- Training ---
@st.cache_resource
def train_model(vocab_size, encoded_data, embed_dim, hidden_dim, epochs):
    st.info("🚀 Training LSTM model...")
    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    progress = st.progress(0)
    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(encoded_data)
        for i, (inp, tgt) in enumerate(encoded_data):
            inp = inp.unsqueeze(0)
            out = model(inp)
            loss = criterion(out, tgt.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 100 == 0:
                progress.progress((epoch + i/len(encoded_data)) / epochs)
        st.write(f"Epoch {epoch+1} complete. Avg loss: {total_loss/len(encoded_data):.4f}")
    progress.empty()
    st.success("✅ Training complete!")
    return model

# --- Prediction ---
def suggest_next_words(model, text_prompt, encode_func, word2idx, idx2word, sequence_length, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())[-(sequence_length - 1):]
    input_tensor = torch.tensor(encode_func(tokens)).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, top_k).indices.tolist()
    return [idx2word[i] for i in top_indices]

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Predictive Keyboard", layout="wide")
st.title("✍️ Smart Predictive Keyboard")
st.markdown("Train an LSTM on Sherlock Holmes and get next-word predictions!")

# Download NLTK
download_nltk_data()

# Load and Preprocess
vocab_size, word2idx, idx2word, encoded_data, encode = load_and_preprocess_data(
    TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT)

# Train Model
model = train_model(vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS)

# Input UI
st.markdown("---")
st.header("Type Your Sentence:")

if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

user_input = st.text_area("Enter sentence:", st.session_state.current_text, height=150)
if user_input != st.session_state.current_text:
    st.session_state.current_text = user_input

# Suggestion UI
st.subheader("Predicted Next Words:")
placeholder = st.empty()
if st.session_state.current_text:
    try:
        suggestions = suggest_next_words(model, st.session_state.current_text, encode, word2idx, idx2word, SEQUENCE_LENGTH)
        cols = placeholder.columns(len(suggestions))
        for i, s in enumerate(suggestions):
            if cols[i].button(s, key=f'sug_{i}'):
                st.session_state.current_text += ' ' + s
                st.experimental_rerun()
    except Exception as e:
        st.error(f"Error: {e}")
else:
    placeholder.info("Start typing to get suggestions.")

# Clear Text
if st.button("🧹 Clear"):
    st.session_state.current_text = ""
    st.experimental_rerun()

# Explanation
st.markdown("---")
st.markdown("### 🔍 How It Works")
st.markdown("""
- Simple LSTM-based language model using PyTorch
- Trained on Sherlock Holmes text corpus
- Predicts the most likely next word based on past few words
- Demo is limited for performance, accuracy may vary
""")
