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
TEXT_URL = 'https://www.gutenberg.org/files/1661/1661-0.txt'  # Sherlock Holmes

# --- Download NLTK Data ---
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
    st.success("✅ NLTK data ready!")

# --- Load & Preprocess Text ---
@st.cache_resource
def load_and_preprocess_data(text_url, sequence_length, training_data_limit):
    st.info("📚 Loading & preprocessing Sherlock Holmes text...")

    try:
        response = requests.get(text_url)
        response.raise_for_status()
        text = response.content.decode('utf-8').lower()
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Failed to download text: {e}")
        st.stop()

    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)

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

    st.success(f"📊 Data ready! Tokens: {len(tokens)} | Vocab Size: {vocab_size}")
    return vocab_size, word2idx, idx2word, encoded_data, encode

# --- Model ---
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(PredictiveKeyboard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])
        return output

# --- Train Model ---
@st.cache_resource
def train_model(vocab_size, encoded_data, embed_dim, hidden_dim, epochs, training_data_limit):
    st.info("⚙️ Training LSTM model... Please wait...")

    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    training_data_subset = encoded_data[:training_data_limit]

    progress_bar = st.progress(0)
    status = st.empty()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(training_data_subset)

        for i, (input_seq, target) in enumerate(training_data_subset):
            input_seq = input_seq.unsqueeze(0)
            output = model(input_seq)
            loss = criterion(output, target.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                progress_bar.progress(min((i + 1) / len(training_data_subset), 1.0))
                status.text(f"Epoch {epoch+1}/{epochs} Step {i+1}/{len(training_data_subset)} | Loss: {loss.item():.4f}")

        st.write(f"✅ Epoch {epoch+1} done. Avg Loss: {total_loss / len(training_data_subset):.4f}")

    progress_bar.empty()
    status.empty()
    st.success("🎉 Model training completed!")
    return model

# --- Suggestion Function ---
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
        top_indices = torch.topk(probs, min(top_k, len(idx2word))).indices.tolist()

    return [idx2word[idx] for idx in top_indices]

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="✍️ Predictive Keyboard")
st.title("✍️ Smart Predictive Keyboard")
st.markdown("Train an LSTM on Sherlock Holmes and get next-word predictions!")

# Step 1: Download NLTK data
download_nltk_data()

# Step 2: Preprocess data
vocab_size, word2idx, idx2word, encoded_data, encode_func = load_and_preprocess_data(
    TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT
)

# Step 3: Train model
model = train_model(
    vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS, TRAINING_DATA_LIMIT
)

# Text input + suggestions
st.header("Try it out:")
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

user_input = st.text_area("Start typing your sentence:", value=st.session_state.current_text, height=150)

if user_input != st.session_state.current_text:
    st.session_state.current_text = user_input

st.subheader("Top Predictions:")

if st.session_state.current_text.strip():
    suggestions = suggest_next_words(model, st.session_state.current_text, encode_func, word2idx, idx2word, SEQUENCE_LENGTH)
    if suggestions:
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion):
                st.session_state.current_text += " " + suggestion
                st.experimental_rerun()
    else:
        st.info("No suggestions. Try more words.")
else:
    st.info("Start typing above to get suggestions.")

# Optional: Clear input
if st.button("🗑️ Clear"):
    st.session_state.current_text = ""
    st.experimental_rerun()
