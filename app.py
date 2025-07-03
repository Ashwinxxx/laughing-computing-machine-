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

# --- Configuration ---
SEQUENCE_LENGTH = 10
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5
TRAINING_DATA_LIMIT = 10000
TEXT_URL = 'https://www.gutenberg.org/files/1661/1661-0.txt'  # Sherlock Holmes text

# --- Download NLTK data ---
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    st.success("✅ NLTK data downloaded successfully!")

download_nltk_data()

# --- Load and preprocess data ---
@st.cache_resource
def load_and_preprocess_data(text_url, sequence_length, training_data_limit):
    st.info("Loading and preprocessing text data...")

    try:
        response = requests.get(text_url)
        response.raise_for_status()
        text = response.content.decode('utf-8').lower()
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading text: {e}")
        st.stop()

    # Clean text
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)

    # Build vocabulary
    word_counts = Counter(tokens)
    vocab = ['<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)

    # Build sequences
    data = []
    for i in range(len(tokens) - sequence_length):
        input_seq = tokens[i:i + sequence_length - 1]
        target = tokens[i + sequence_length - 1]
        data.append((input_seq, target))

    def encode(seq):
        return [word2idx.get(word, word2idx['<unk>']) for word in seq]

    encoded_data = [
        (torch.tensor(encode(inp)), torch.tensor(word2idx.get(target, word2idx['<unk>'])))
        for inp, target in data
    ]

    st.success(f"Preprocessed! Tokens: {len(tokens)}, Vocab size: {vocab_size}")
    return vocab_size, word2idx, idx2word, encoded_data, encode

vocab_size, word2idx, idx2word, encoded_data, encode_func = load_and_preprocess_data(
    TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT
)

# --- Model ---
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])  # Only last timestep

# --- Training ---
@st.cache_resource
def train_model(vocab_size, encoded_data, embed_dim, hidden_dim, epochs, training_data_limit):
    st.info("Training model...")
    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    train_data = encoded_data[:training_data_limit]
    progress = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(train_data)

        for i, (inp, tgt) in enumerate(train_data):
            inp = inp.unsqueeze(0)
            tgt = tgt.unsqueeze(0)
            output = model(inp)
            loss = criterion(output, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                progress.progress((i + 1) / len(train_data))
                status.text(f"Epoch {epoch+1}/{epochs}, Step {i+1}, Loss: {loss.item():.4f}")

        st.write(f"✅ Epoch {epoch+1}: Avg Loss = {total_loss / len(train_data):.4f}")

    progress.empty()
    status.empty()
    st.success("Model trained!")
    return model

model = train_model(
    vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS, TRAINING_DATA_LIMIT
)

# --- Prediction ---
def suggest_next_words(model, text_prompt, encode_func, word2idx, idx2word, sequence_length, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) == 0:
        return []
    elif len(tokens) < sequence_length - 1:
        input_seq = tokens
    else:
        input_seq = tokens[-(sequence_length - 1):]

    input_tensor = torch.tensor(encode_func(input_seq)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, top_k).indices.tolist()

    return [idx2word[i] for i in top_indices]

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Predictive Keyboard")
st.title("✍️ Smart Predictive Keyboard")

st.markdown("""
This demo showcases a predictive keyboard trained on Sherlock Holmes stories.
Start typing a sentence, and it will suggest the next word.
""")

if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

user_input = st.text_area(
    "Your Sentence:",
    value=st.session_state.current_text,
    height=150,
    placeholder="Type your sentence here..."
)

if user_input != st.session_state.current_text:
    st.session_state.current_text = user_input

st.subheader("Next Word Suggestions:")
suggestions_placeholder = st.empty()

def append_suggestion(suggestion):
    st.session_state.current_text += " " + suggestion
    st.experimental_rerun()

if st.session_state.current_text:
    try:
        suggestions = suggest_next_words(
            model, st.session_state.current_text, encode_func,
            word2idx, idx2word, SEQUENCE_LENGTH
        )
        if suggestions:
            cols = suggestions_placeholder.columns(len(suggestions))
            for i, word in enumerate(suggestions):
                cols[i].button(word, on_click=append_suggestion, args=(word,), key=f"suggest_btn_{i}")
        else:
            suggestions_placeholder.info("No suggestions. Try typing more words.")
    except Exception as e:
        suggestions_placeholder.error(f"Error: {e}")
else:
    suggestions_placeholder.info("Start typing above to get suggestions.")

if st.button("Clear Text"):
    st.session_state.current_text = ""
    st.experimental_rerun()

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
- The model is a basic LSTM trained to predict the next word based on the previous 9.
- Data is sourced from public domain Sherlock Holmes stories.
- Trained on a limited dataset for speed and interactivity.
""")
