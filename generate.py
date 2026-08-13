# generate.py
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import config

def generate_text(seed_text, next_words=50):
    """
    Generates new text given a starting seed sequence.
    """
    # Load model and tokenizer
    try:
        model = load_model(config.MODEL_PATH)
        with open(config.TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
    except FileNotFoundError:
        return "Error: Model or Tokenizer not found. Please run train.py first."

    # Clean the seed text
    seed_text = seed_text.lower()
    in_text = seed_text
    generated_sequence = []

    for _ in range(next_words):
        # 1. Tokenize the current input text
        token_list = tokenizer.texts_to_sequences([in_text])[0]
        
        # 2. Pad to match the sequence length the model was trained on
        token_list = pad_sequences([token_list], maxlen=config.SEQ_LENGTH, padding='pre')

        # 3. Predict the probabilities of the next word
        predicted_probs = model.predict(token_list, verbose=0)
        
        # Add temperature scaling here if desired (using argmax for deterministic output)
        predicted_index = np.argmax(predicted_probs, axis=-1)[0]

        # 4. Map index back to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break

        # Append to inputs for the next iteration
        in_text += " " + output_word
        generated_sequence.append(output_word)

    return seed_text + " " + " ".join(generated_sequence)

if __name__ == "__main__":
    print("--- LSTM Text Generator ---")
    seeds = [
        "to be or not to be that",
        "alas poor yorick i knew him",
        "a horse a horse my kingdom for"
    ]
    
    for seed in seeds:
        print(f"\nSeed: '{seed}'")
        print(f"Generated: {generate_text(seed, next_words=20)}")
