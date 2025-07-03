import streamlit as st
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

# --- Simple Tokenization (No NLTK needed) ---
def word_tokenize(text):
    """Simple word tokenization using regex - no NLTK required"""
    # Remove extra whitespace and convert to lowercase
    text = text.lower().strip()
    # Extract words (alphabetic characters only)
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens

# --- Data Loading and Preprocessing ---
@st.cache_resource
def load_and_preprocess_data(text_url, sequence_length, training_data_limit):
    st.info("📚 Loading & preprocessing Sherlock Holmes text...")
    try:
        response = requests.get(text_url, timeout=30)
        response.raise_for_status()
        text = response.content.decode('utf-8').lower()
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading text: {e}")
        st.stop()

    # Clean the text
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize using our simple function
    tokens = word_tokenize(text)
    
    # Build vocabulary
    word_counts = Counter(tokens)
    vocab = ['<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)

    # Create training data
    data = []
    for i in range(len(tokens) - sequence_length):
        input_seq = tokens[i:i + sequence_length - 1]
        target = tokens[i + sequence_length - 1]
        data.append((input_seq, target))

    def encode(seq):
        return [word2idx.get(word, word2idx['<unk>']) for word in seq]

    encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx.get(target, word2idx['<unk>'])))
                    for inp, target in data[:training_data_limit]]

    st.success(f"✅ Data ready. Tokens: {len(tokens)} | Vocab size: {vocab_size}")
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
        output, _ = self.lstm(x)
        return self.fc(output[:, -1, :])

# --- Training ---
@st.cache_resource
def train_model(vocab_size, encoded_data, embed_dim, hidden_dim, epochs):
    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    st.info("🧠 Training the model...")
    progress_bar = st.progress(0)

    for epoch in range(epochs):
        total_loss = 0
        random.shuffle(encoded_data)

        for i, (input_seq, target) in enumerate(encoded_data):
            input_seq = input_seq.unsqueeze(0)
            output = model(input_seq)
            loss = criterion(output, target.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if i % 100 == 0:
                progress_bar.progress((i + 1) / len(encoded_data))

        st.write(f"Epoch {epoch+1}: Loss = {total_loss / len(encoded_data):.4f}")

    progress_bar.empty()
    st.success("✅ Training complete!")
    return model

# --- Prediction ---
def suggest_next_words(model, text_prompt, encode_func, word2idx, idx2word, sequence_length, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())

    if len(tokens) == 0:
        return []
    
    input_seq = tokens[-(sequence_length - 1):] if len(tokens) >= sequence_length - 1 else tokens
    
    # Handle empty sequence
    if len(input_seq) == 0:
        return []
        
    input_tensor = torch.tensor(encode_func(input_seq)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        top_indices = torch.topk(probs, min(top_k, len(idx2word))).indices.tolist()

    return [idx2word[idx] for idx in top_indices]

# --- Streamlit UI ---
st.set_page_config(page_title="Smart Predictive Keyboard", layout="wide")
st.title("✍️ Smart Predictive Keyboard")

st.markdown("""
Train an LSTM on Sherlock Holmes and get next-word predictions!  
*Uses simple regex tokenization - no external dependencies.*
""")

# Initialize session state
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

# Function to add word to current text
def add_word_to_text(word):
    if st.session_state.current_text:
        st.session_state.current_text += " " + word
    else:
        st.session_state.current_text = word

# Run setup - no external downloads needed
try:
    vocab_size, word2idx, idx2word, encoded_data, encode_func = load_and_preprocess_data(TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT)
    model = train_model(vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS)
    
    st.markdown("---")
    st.subheader("Start Typing:")

    user_input = st.text_area("Your text:", value=st.session_state.current_text, height=150, key="text_input")

    # Update session state when user types
    if user_input != st.session_state.current_text:
        st.session_state.current_text = user_input

    st.subheader("Suggestions:")
    suggestions_container = st.empty()

    if st.session_state.current_text:
        try:
            suggestions = suggest_next_words(model, st.session_state.current_text, encode_func, word2idx, idx2word, SEQUENCE_LENGTH)
            if suggestions:
                cols = suggestions_container.columns(len(suggestions))
                for i, suggestion in enumerate(suggestions):
                    if cols[i].button(suggestion, key=f"suggestion_{i}_{suggestion}"):
                        add_word_to_text(suggestion)
                        st.rerun()
            else:
                suggestions_container.info("Try typing a longer sentence.")
        except Exception as e:
            suggestions_container.error(f"Error generating suggestions: {e}")
    else:
        suggestions_container.info("Start typing above to get suggestions.")

    if st.button("Clear Text"):
        st.session_state.current_text = ""
        st.rerun()

except Exception as e:
    st.error(f"Error loading the application: {e}")
    st.info("Please check your internet connection and try refreshing the page.")
