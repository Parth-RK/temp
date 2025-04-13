# main.py
import torch
import torch.optim as optim
import torch.nn as nn
import os

from data_processor import load_and_prepare_data, PAD_TOKEN
import model as mt
import trainer as tt
import predictor as pt
import nltk
nltk.download('punkt')

# --- Configuration ---
# Data paths
DATA_FILEPATH = os.path.join(os.path.dirname(__file__), 'emotion_data.csv')
OUTPUT_DIR = "output"
MODEL_SAVE_NAME = "best_emotion_lstm.pt"
VOCAB_SAVE_NAME = "vocabulary.pkl"
ENCODER_SAVE_NAME = "label_encoder.pkl"

# Model Hyperparameters (tune these!)
CONFIG = {
    'embedding_dim': 100,
    'hidden_dim': 256,      # Size of LSTM hidden state
    'n_layers': 2,          # Number of LSTM layers
    'bidirectional': True,  # Use bidirectional LSTM
    'dropout': 0.5,         # Dropout rate
}

# Training Hyperparameters
LEARNING_RATE = 0.001 # Common starting point for Adam
N_EPOCHS = 10         # Increase for better results, monitor validation loss
BATCH_SIZE = 128      # Adjust based on GPU memory
VOCAB_FREQ_THRESHOLD = 3 # Ignore words appearing less than this many times

# --- Main Execution ---
if __name__ == "__main__":
    print("--- PyTorch Emotion Classification Pipeline ---")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, MODEL_SAVE_NAME)
    VOCAB_SAVE_PATH = os.path.join(OUTPUT_DIR, VOCAB_SAVE_NAME)
    ENCODER_SAVE_PATH = os.path.join(OUTPUT_DIR, ENCODER_SAVE_NAME)

    # 1. Load and Prepare Data (using PyTorch processor)
    print("\n[Phase 1] Loading and preparing data...")
    train_loader, val_loader, vocab, label_encoder, num_classes = load_and_prepare_data(
        filepath=DATA_FILEPATH,
        batch_size=BATCH_SIZE,
        vocab_freq_threshold=VOCAB_FREQ_THRESHOLD,
        output_dir=OUTPUT_DIR # Save vocab/encoder here
    )

    if train_loader is None:
        print("Data loading/preparation failed. Exiting.")
        exit()

    # 2. Define Model, Optimizer, Criterion, Device
    print("\n[Phase 2] Setting up model, optimizer, loss, and device...")
    PAD_IDX = vocab.stoi[PAD_TOKEN]
    INPUT_DIM = len(vocab)
    OUTPUT_DIM = num_classes

    device = tt.get_device() # Get GPU or CPU

    model = mt.create_model(INPUT_DIM, OUTPUT_DIM, CONFIG, PAD_IDX)
    model.to(device) # Move model to device

    # Using Adam optimizer - often works well
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # CrossEntropyLoss is suitable for multi-class classification
    # It combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss()
    criterion.to(device) # Move criterion to device (if it has parameters, although CEL doesn't)

    # 3. Train Model
    print("\n[Phase 3] Starting Training...")
    trained_model = tt.train_model(
        model, train_loader, val_loader, optimizer, criterion, device,
        n_epochs=N_EPOCHS, label_encoder=label_encoder, model_save_path=MODEL_SAVE_PATH
    )

    # --- Example Prediction ---
    print("\n[Phase 4] Testing prediction with loaded components...")

    # Need to recreate the model architecture to load the state dict
    # Make sure parameters match the saved model
    print("Recreating model structure for loading...")
    prediction_model = mt.create_model(INPUT_DIM, OUTPUT_DIM, CONFIG, PAD_IDX)

    # Load all components needed for prediction
    loaded_state_dict, loaded_vocab, loaded_encoder = pt.load_prediction_components(
        model_path=MODEL_SAVE_PATH,
        vocab_path=VOCAB_SAVE_PATH,
        encoder_path=ENCODER_SAVE_PATH,
        device=device # Load onto the correct device
    )

    if loaded_state_dict and loaded_vocab and loaded_encoder:
        try:
            prediction_model.load_state_dict(loaded_state_dict)
            prediction_model.to(device) # Move loaded model to device
            print("Model state loaded successfully into structure.")

            # Now use the predictor function
            test_texts = [
                "I am overjoyed with this wonderful news!",
                "Feeling so down and lonely today.",
                "This lecture is incredibly dull.",
                "Wow, I did not expect that ending!",
                "I'm really nervous about the presentation tomorrow.",
                "@someuser thanks for the follow, much appreciated!",
                "Get out of my way you idiot!"
            ]

            print("\nRunning predictions:")
            for text in test_texts:
                predicted_emotion = pt.predict_emotion(
                    text, prediction_model, loaded_vocab, loaded_encoder, device
                )
                print(f"Input: '{text}'")
                print(f"Predicted Emotion: {predicted_emotion}")

        except RuntimeError as e:
             print(f"Error loading state dict, possibly due to model structure mismatch: {e}")
        except Exception as e:
             print(f"An unexpected error occurred during prediction setup: {e}")

    else:
        print("Could not load necessary components for prediction test.")

    print("\n--- PyTorch Pipeline Finished ---")