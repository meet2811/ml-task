# train.py
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import config
from data_utils import download_dataset, preprocess_data
from model import build_model

def train():
    # 1. Fetch and process data
    download_dataset()
    X, y, word_index = preprocess_data()

    # Ensure vocab size matches the actual tokenized vocabulary (accounting for limit)
    actual_vocab_size = min(config.VOCAB_SIZE, len(word_index) + 1)

    # 2. Split into training and validation sets
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=42)

    # 3. Build Model
    model = build_model(actual_vocab_size)
    model.summary()

    # 4. Define Callbacks
    callbacks = [
        # Stop training if validation loss doesn't improve for 3 epochs
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        # Save the best version of the model automatically
        ModelCheckpoint(config.MODEL_PATH, monitor='val_loss', save_best_only=True)
    ]

    # 5. Train Model
    print("Starting training...")
    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=config.EPOCHS,
              batch_size=config.BATCH_SIZE,
              callbacks=callbacks)
              
    print(f"Training complete. Best model saved to {config.MODEL_PATH}")

if __name__ == "__main__":
    train()
