An interactive Streamlit app that demonstrates a simple LSTM-based predictive text model, trained on Sherlock Holmes stories (public domain text from Project Gutenberg 📖).

Users can type any sentence, and the model will suggest the next few words just like a predictive keyboard ✍️.

🚀 Demo Features
✅ Real-time next-word suggestions
✅ Type-based autocomplete for sentences
✅ Simple LSTM neural network built with PyTorch
✅ Training on Sherlock Holmes text
✅ Configurable training and prediction settings via sidebar
✅ Beautiful Streamlit-based UI with emoji headers and interactive buttons

🖥️ App Preview
Home Screen:
📥 Loads and preprocesses the Sherlock Holmes text

🚀 Trains a small LSTM model on the fly

✏️ Text area for user input

💬 Shows clickable word suggestions (next word predictions)Python 3.8+

Streamlit

PyTorch

NLTK

Requests

Other Python standard packages

Data Source: Sherlock Holmes stories (from Project Gutenberg)

Preprocessing: Basic tokenization, cleaning, and vocabulary building

Model Architecture:

Embedding Layer

LSTM Layer

Fully Connected (Linear) Output Layer

Training: On small data subset to keep demo fast

Prediction: Top-K next word suggestions using softma

