# config.py
"""
Central configuration file for the emotion classification project.
"""

import torch
import os

# --- Data Paths ---
BASE_DIR = "."
TRAIN_PATH = os.path.join(BASE_DIR, "training.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

# --- Artifact Paths ---
ARTIFACTS_DIR = "artifacts" # Directory to save model and vocab
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.pt")
VOCAB_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "vocab.pt")
PREPROCESSOR_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_config.json") # To save config like max_length etc.
RESULTS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_plots.png")

# --- Preprocessing Parameters ---
MAX_LENGTH = 40  # Maximum sequence length after tokenization (+2 for SOS/EOS)
MIN_FREQ = 2     # Minimum frequency for words to be included in vocabulary

# --- Special Tokens ---
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"

# --- Model Selection ---
# Choose 'LSTM' or 'ANN'
MODEL_TYPE = 'LSTM' # Change this to 'ANN' to train the ANN model

# --- Model Hyperparameters ---
# Common
N_CLASS = 6        # Number of emotion classes
# LSTM Specific
EMBEDDING_DIM = 64 # Embedding dimension for LSTM
HIDDEN_DIM = 128   # Number of hidden units in LSTM layers
N_LAYERS = 2       # Number of LSTM layers
# ANN Specific (Input size depends on MAX_LENGTH + 2)
ANN_INPUT_SIZE = MAX_LENGTH + 2

# --- Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 10 # Reduced for quicker demonstration, increase for better results (e.g., 40)
LEARNING_RATE_LSTM = 0.005 # Adjusted learning rate
LEARNING_RATE_ANN = 0.001
SHUFFLE_DATA = True

# --- Application Settings ---
# Mapping from label index to emotion name (adjust if your dataset has different mapping)
INT_TO_LABEL = {
    0: 'sadness',
    1: 'joy',
    2: 'love',
    3: 'anger',
    4: 'fear',
    5: 'surprise'
}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}