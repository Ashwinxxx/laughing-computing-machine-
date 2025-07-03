
import streamlit as st
import nltk

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download("punkt", quiet=True)
    except Exception as e:
        st.warning(f"⚠️ Failed to download 'punkt': {e}")

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.download("stopwords", quiet=True)
    except Exception as e:
        st.warning(f"⚠️ Failed to download 'stopwords': {e}")

    st.success("✅ NLTK data downloaded successfully!")

# Now you can use nltk functions like:
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import re # For cleaning text

# --- Configuration ---
SEQUENCE_LENGTH = 10
EMBED_DIM = 64
HIDDEN_DIM = 128
EPOCHS = 5 # Number of training epochs
TRAINING_DATA_LIMIT = 10000 # Limit data for faster training in demo
TEXT_URL = 'https://www.gutenberg.org/files/1661/1661-0.txt' # Sherlock Holmes text

# --- NLTK Downloads (Cached) ---
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except nltk.downloader.DownloadError:
        nltk.download('punkt_tab', quiet=True)
    st.success("NLTK data downloaded!")

# --- Data Loading and Preprocessing (Cached) ---
@st.cache_resource
def load_and_preprocess_data(text_url, sequence_length, training_data_limit):
    """
    Loads text, tokenizes, builds vocabulary, and prepares encoded data.
    This function is cached to run only once.
    """
    st.info("Loading and preprocessing text data... This might take a moment.")
    
    # Download the text file
    try:
        response = requests.get(text_url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        text = response.content.decode('utf-8').lower()
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading text from {text_url}: {e}")
        st.stop() # Stop the app if data can't be loaded
    
    # Basic text cleaning
    text = re.sub(r'[^a-z\s]', '', text) # Remove non-alphabetic characters
    text = re.sub(r'\s+', ' ', text).strip() # Replace multiple spaces with single space

    tokens = word_tokenize(text)
    
    # Build vocabulary
    word_counts = Counter(tokens)
    # Add a special token for unknown words
    vocab = ['<unk>'] + sorted(word_counts, key=word_counts.get, reverse=True)
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    vocab_size = len(vocab)

    # Prepare input-output sequences
    data = []
    for i in range(len(tokens) - sequence_length):
        input_seq = tokens[i:i + sequence_length - 1]
        target = tokens[i + sequence_length - 1]
        data.append((input_seq, target))

    # Convert words to indices, handling unknown words
    def encode(seq):
        return [word2idx.get(word, word2idx['<unk>']) for word in seq]

    encoded_data = [(torch.tensor(encode(inp)), torch.tensor(word2idx.get(target, word2idx['<unk>'])))
                    for inp, target in data]
    
    st.success(f"Data loaded. Total Tokens: {len(tokens)}, Vocabulary Size: {vocab_size}")
    return vocab_size, word2idx, idx2word, encoded_data, encode

# --- Model Architecture ---
class PredictiveKeyboard(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(PredictiveKeyboard, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  # last LSTM output
        return output

# --- Model Training (Cached) ---
@st.cache_resource
def train_model(vocab_size, encoded_data, embed_dim, hidden_dim, epochs, training_data_limit):
    """
    Trains the PredictiveKeyboard model. This function is cached to run only once.
    """
    st.info("Training the predictive keyboard model... This will take a few minutes.")
    
    model = PredictiveKeyboard(vocab_size, embed_dim, hidden_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Limit data for training in the demo to speed it up
    training_data_subset = encoded_data[:training_data_limit]

    progress_text = st.empty()
    progress_bar = st.progress(0)

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
            
            # Update progress bar
            if i % 100 == 0: # Update every 100 steps to avoid too frequent updates
                progress_bar.progress((i + 1) / len(training_data_subset))
                progress_text.text(f"Epoch {epoch+1}/{epochs}, Step {i+1}/{len(training_data_subset)}, Loss: {loss.item():.4f}")

        st.write(f"Epoch {epoch+1} finished. Average Loss: {total_loss / len(training_data_subset):.4f}")
    
    progress_bar.empty() # Clear progress bar
    progress_text.empty() # Clear progress text
    st.success("Model training complete!")
    return model

# --- Prediction Function ---
def suggest_next_words(model, text_prompt, encode_func, word2idx, idx2word, sequence_length, top_k=3):
    model.eval()
    tokens = word_tokenize(text_prompt.lower())
    
    # Handle short inputs
    if len(tokens) == 0:
        return []
    elif len(tokens) < sequence_length - 1:
        # Pad with <unk> if too short, or just use available tokens
        # For a predictive keyboard, using available tokens is more natural
        input_seq = tokens
        st.warning(f"Input is short ({len(tokens)} words). Model works best with at least {sequence_length - 1} words.")
    else:
        input_seq = tokens[-(sequence_length - 1):] # Use the last N-1 words

    # Encode input, handling unknown words
    input_tensor = torch.tensor(encode_func(input_seq)).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1).squeeze()
        
        # Ensure top_k does not exceed vocab_size
        actual_top_k = min(top_k, len(idx2word))
        top_indices = torch.topk(probs, actual_top_k).indices.tolist()

    return [idx2word[idx] for idx in top_indices]

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Predictive Keyboard")

st.title("✍️ Smart Predictive Keyboard")

st.markdown("""
This interactive demo showcases a simple predictive keyboard model built with PyTorch and trained on Sherlock Holmes stories.
Type a sentence in the text area below, and the model will suggest the next few words based on your input.
""")

# Download NLTK data first
download_nltk_data()

# Load and preprocess data
vocab_size, word2idx, idx2word, encoded_data, encode_func = load_and_preprocess_data(
    TEXT_URL, SEQUENCE_LENGTH, TRAINING_DATA_LIMIT
)

# Train the model
model = train_model(
    vocab_size, encoded_data, EMBED_DIM, HIDDEN_DIM, EPOCHS, TRAINING_DATA_LIMIT
)

st.markdown("---")

st.header("Start Typing Here:")

# Initialize session state for the text input
if 'current_text' not in st.session_state:
    st.session_state.current_text = ""

# Text input area
user_input = st.text_area(
    "Your Sentence:",
    value=st.session_state.current_text,
    height=150,
    placeholder="Type your sentence here...",
    key="text_input_area" # Use a key to manage state
)

# Update session state when text area changes
if user_input != st.session_state.current_text:
    st.session_state.current_text = user_input

st.subheader("Next Word Suggestions:")

suggestions_container = st.empty() # Placeholder for suggestions

# Function to handle appending suggestions
def append_suggestion(suggestion):
    st.session_state.current_text += " " + suggestion
    # Rerun the app to update the text area and re-generate suggestions
    st.experimental_rerun()

# Generate and display suggestions
if st.session_state.current_text:
    try:
        suggestions = suggest_next_words(
            model, st.session_state.current_text, encode_func, word2idx, idx2word, SEQUENCE_LENGTH
        )
        
        if suggestions:
            cols = suggestions_container.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                cols[i].button(suggestion, on_click=append_suggestion, args=(suggestion,), key=f"suggest_btn_{i}")
        else:
            suggestions_container.info("No suggestions available. Try typing more words.")
    except ValueError as e:
        suggestions_container.error(f"Error: {e}")
    except Exception as e:
        suggestions_container.error(f"An unexpected error occurred during suggestion: {e}")
else:
    suggestions_container.info("Start typing in the box above to get suggestions!")

st.markdown("---")
st.markdown("### How it works:")
st.markdown("""
- The model is a simple LSTM (Long Short-Term Memory) neural network.
- It's trained on a dataset derived from the public domain text of Sherlock Holmes stories.
- It learns to predict the next word based on a sequence of preceding words.
- **Note:** For demonstration purposes, the model is trained on a limited subset of the data and for a small number of epochs, so its accuracy might not be perfect.
""")

# Add a "Clear Text" button
if st.button("Clear Text"):
    st.session_state.current_text = ""
    st.experimental_rerun()
