
# Dataset configurations
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "shakespeare.txt"
MODEL_PATH = "lstm_model.keras"
TOKENIZER_PATH = "tokenizer.pkl"

# Model & Preprocessing Hyperparameters
VOCAB_SIZE = 5000       # Maximum number of words in the vocabulary
SEQ_LENGTH = 20         # Number of previous words to use as context
EMBEDDING_DIM = 100     # Dimension of word embeddings
LSTM_UNITS = 128        # Number of hidden units in LSTM layers
DROPOUT_RATE = 0.2      # Dropout to prevent overfitting

# Training configurations
BATCH_SIZE = 128
EPOCHS = 30             # High epoch count; early stopping will halt it if it plateaus
TEST_SIZE = 0.1         # 10% of data used for validation
DATA_LIMIT = 200000     # Limit total tokens to prevent OOM errors on local machines
