import streamlit as st
import nltk
from nltk.tokenize import TreebankWordTokenizer
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
TEXT_URL = "https://www.gutenberg.org/files/1661/1661-0.txt"

# --- NLTK Prep ---
@st.cache_resource
def prepare_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    st.success("✅ NLTK data ready!")

prepare_nltk()

# Use tokenizer that doesn’t trigger punkt_tab bug
tokenizer = TreebankWordTokenizer()

# --- Data Loader ---
@st.cache_resource
def load_and_preprocess_data(url, sequence_len, data_limit):
    st.info("📚 Loading & preprocessing Sherlock Holmes text...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.content.decode('utf-8').lower()
    except Exception as e:
        st.error(f"Failed to load text: {e}")
        st.stop()
    
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = tokenizer.tokenize(text)
    word_counts = Counter(tokens)
    vocab = ['<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    vocab_size = len(vocab)

    def encode(seq):
        return [word2idx.get(w, word2idx['<unk>']) for w in seq]

    data = []
    for i in range(len(tokens) - sequence_len):
        input_seq = tokens[i:i + sequence_len - 1]
        target = tokens[i + sequence_len - 1]
        data.append((input_seq, target))

    encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx.get(tgt, word2idx['<unk>'])))
                    for inp, tgt in data]
    
    st.success(f"✅ Tokens: {len(tokens)}, Vocab: {vocab_size}")
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
def train_model(vocab_size, data, embed_dim, hidden_dim, epochs, limit):
    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    subset = data[:limit]
    
    bar = st.progress(0)
    text = st.empty()

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(subset)
        for i, (inp, tgt) in enumerate(subset):
            inp = inp.unsqueeze(0)
            output = model(inp)
            loss = criterion(output, tgt.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 100 == 0:
                bar.progress((i + 1) / len(subset))
                text.text(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(subset)}, Loss: {loss.item():.4f}")

        st.write(f"✅ Epoch {epoch+1} Avg Loss: {total_loss/len(subset):.4f}")
    bar.empty()
    return model

# --- Suggest ---
def suggest_next_words(model, prompt, encode, word2idx, idx2word, sequence_len, top_k=3):
    model.eval()
    tokens = tokenizer.tokenize(prompt.lower())

    if len(tokens) == 0:
        return []

    seq = tokens[-(sequence_len - 1):]
    inp = torch.tensor(encode(seq)).unsqueeze(0)

    with torch.no_grad():
        output = model(inp)
        probs = F.softmax(output, dim=1).squeeze()
        top_ids = torch.topk(probs, top_k).indices.tolist()

    return [idx2word[i] for i in top_ids]

# --- UI ---
st.set_page_config(layout="centered", page_title="Smart Predictive Keyboard")
st.title("✍️ Smart Predictive Keyboard")
st.markdown("Train an LSTM on Sherlock Holmes and get next-word predictions!")

vocab_size, word2idx, idx2word, encoded_data, encode_func = load_and_preprocess_data(
    TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT
)

model = train_model(
    vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS, TRAINING_DATA_LIMIT
)

# Text box
if "current_text" not in st.session_state:
    st.session_state.current_text = ""

st.header("Start Typing")
user_input = st.text_area("Your Sentence:", value=st.session_state.current_text, height=150)

if user_input != st.session_state.current_text:
    st.session_state.current_text = user_input

st.subheader("Next Word Suggestions")

def append_suggestion(word):
    st.session_state.current_text += " " + word
    st.experimental_rerun()

suggestions_area = st.empty()

if st.session_state.current_text:
    try:
        suggestions = suggest_next_words(
            model, st.session_state.current_text, encode_func, word2idx, idx2word, SEQUENCE_LENGTH
        )
        cols = suggestions_area.columns(len(suggestions))
        for i, word in enumerate(suggestions):
            cols[i].button(word, on_click=append_suggestion, args=(word,), key=f"btn_{i}")
    except Exception as e:
        st.error(f"⚠️ Suggestion Error: {e}")
else:
    suggestions_area.info("Type something to get next-word suggestions!")

if st.button("Clear Text"):
    st.session_state.current_text = ""
    st.experimental_rerun()

st.markdown("---")
st.markdown("Built with PyTorch, NLTK, and Streamlit.")

