# data_utils.py
import os
import urllib.request
import re
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
import config

def download_dataset():
    """Downloads the Shakespeare dataset if it doesn't already exist."""
    if not os.path.exists(config.DATA_PATH):
        print(f"Downloading dataset from {config.DATA_URL}...")
        urllib.request.urlretrieve(config.DATA_URL, config.DATA_PATH)
        print("Download complete.")

def preprocess_data():
    """
    Reads the text file, cleans punctuation, tokenizes words, 
    and generates X (inputs) and y (targets) sequences.
    """
    with open(config.DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Clean Text: Lowercase and remove punctuation (keep basic letters and spaces)
    text = text.lower()
    text = re.sub(r'[^a-z \n]', '', text)

    # 2. Tokenize Text
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=config.VOCAB_SIZE, oov_token='<oov>')
    tokenizer.fit_on_texts([text])

    # Save tokenizer for text generation later
    with open(config.TOKENIZER_PATH, 'wb') as f:
        pickle.dump(tokenizer, f)

    # Convert entire text to a single sequence of integers
    full_sequence = tokenizer.texts_to_sequences([text])[0]
    
    # Limit dataset size for memory constraints (optional)
    full_sequence = full_sequence[:config.DATA_LIMIT]

    # 3. Prepare input-output pairs using a sliding window
    print("Generating sequences...")
    sequences = []
    for i in range(config.SEQ_LENGTH, len(full_sequence)):
        # Take a slice of SEQ_LENGTH + 1 (the last token is the target 'y')
        seq = full_sequence[i - config.SEQ_LENGTH : i + 1]
        sequences.append(seq)
        
    sequences = np.array(sequences)

    # X is all columns except the last, y is the last column
    X = sequences[:, :-1]
    y = sequences[:, -1]
    
    return X, y, tokenizer.word_index
