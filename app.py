import streamlit as st
import nltk
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import re
from collections import Counter
from nltk.tokenize import TreebankWordTokenizer

# --- Configuration ---
SEQUENCE_LENGTH = 10
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
TRAINING_DATA_LIMIT = 10000
TEXT_URL = 'https://www.gutenberg.org/files/1661/1661-0.txt'

# --- NLTK Downloads ---
@st.cache_resource
def download_nltk_data():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    st.success("✅ NLTK resources downloaded!")

# --- Tokenizer ---
tokenizer = TreebankWordTokenizer()

# --- Data Loading and Preprocessing ---
@st.cache_resource
def load_and_preprocess_data(text_url, sequence_length, training_data_limit):
    st.info("📚 Loading & preprocessing Sherlock Holmes text...")

    try:
        response = requests.get(text_url)
        response.raise_for_status()
        text = response.content.decode('utf-8').lower()
    except Exception as e:
        st.error(f"❌ Failed to load text: {e}")
        st.stop()

    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = tokenizer.tokenize(text)
    word_counts = Counter(tokens)
    vocab = ['<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)

    data = []
    for i in range(len(tokens) - sequence_length):
        input_seq = tokens[i:i + sequence_length - 1]
        target = tokens[i + sequence_length - 1]
        data.append((input_seq, target))

    def encode(seq):
        return [word2idx.get(word, word2idx['<unk>']) for word in seq]

    encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx.get(target, word2idx['<unk>'])))
                    for inp, target in data]

    st.success(f"✅ Data ready. Tokens: {len(tokens)}, Vocab Size: {vocab_size}")
    return vocab_size, word2idx, idx2word, encoded_data, encode

# --- Model ---
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- Train Model ---
@st.cache_resource
def train_model(vocab_size, encoded_data, embed_dim, hidden_dim, epochs, training_data_limit):
    st.info("🧠 Training model...")
    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    data_subset = encoded_data[:training_data_limit]

    bar = st.progress(0)
    msg = st.empty()

    for epoch in range(epochs):
        random.shuffle(data_subset)
        total_loss = 0
        for i, (input_seq, target) in enumerate(data_subset):
            input_seq = input_seq.unsqueeze(0)
            output = model(input_seq)
            loss = criterion(output, target.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                pct = int((i+1) / len(data_subset) * 100)
                bar.progress(min(pct, 100))
                msg.text(f"Epoch {epoch+1}/{epochs} | Step {i+1}/{len(data_subset)} | Loss: {loss.item():.4f}")
        st.write(f"✅ Epoch {epoch+1} completed. Avg Loss: {total_loss / len(data_subset):.4f}")

    bar.empty()
    msg.empty()
    st.success("✅ Training complete!")
    return model

# --- Prediction ---
def suggest_next_words(model, prompt, encode_func, word2idx, idx2word, sequence_length, top_k=3):
    model.eval()
    tokens = tokenizer.tokenize(prompt.lower())
    if len(tokens) == 0:
        return []
    input_seq = tokens[-(sequence_length - 1):]
    input_tensor = torch.tensor(encode_func(input_seq)).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, top_k).indices.tolist()
    return [idx2word[i] for i in top_indices]

# --- Streamlit UI ---
st.set_page_config(page_title="Predictive Keyboard", layout="wide")
st.title("✍️ Smart Predictive Keyboard")
st.markdown("""
Train an LSTM on Sherlock Holmes and get next-word predictions!
""")

download_nltk_data()

vocab_size, word2idx, idx2word, encoded_data, encode_func = load_and_preprocess_data(
    TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT
)

model = train_model(
    vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS, TRAINING_DATA_LIMIT
)

st.markdown("---")
st.header("Start Typing Here:")

if "current_text" not in st.session_state:
    st.session_state.current_text = ""

user_input = st.text_area("Your Sentence:", value=st.session_state.current_text, height=150)

if user_input != st.session_state.current_text:
    st.session_state.current_text = user_input

st.subheader("Next Word Suggestions:")
suggestions_container = st.empty()

def append_suggestion(suggestion):
    st.session_state.current_text += " " + suggestion
    st.experimental_rerun()

if st.session_state.current_text:
    try:
        suggestions = suggest_next_words(
            model, st.session_state.current_text, encode_func, word2idx, idx2word, SEQUENCE_LENGTH
        )
        if suggestions:
            cols = suggestions_container.columns(len(suggestions))
            for i, word in enumerate(suggestions):
                cols[i].button(word, on_click=append_suggestion, args=(word,), key=f"suggest_{i}")
        else:
            suggestions_container.info("Try typing a longer sentence.")
    except Exception as e:
        suggestions_container.error(f"⚠️ Error: {e}")
else:
    suggestions_container.info("Type something above to get predictions.")

if st.button("Clear Text"):
    st.session_state.current_text = ""
    st.experimental_rerun()

st.markdown("---")
st.markdown("### How it Works:")
st.markdown("""
- LSTM model trained on public domain Sherlock Holmes text
- Predicts the most likely next word based on your sentence
- Uses PyTorch and NLTK (safe tokenizer used to avoid punkt_tab error)
""")
