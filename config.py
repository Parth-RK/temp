# config.py
"""
Central configuration file for the emotion classification project.
(TorchText Legacy Independent Version)
"""

import torch
import os

# --- Data Paths ---
BASE_DIR = "." # Or your local path
TRAIN_PATH = os.path.join(BASE_DIR, "training.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

# --- Artifact Paths ---
ARTIFACTS_DIR = "artifacts" # Directory to save model and vocab
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.pt")
# Changed VOCAB path to reflect saving dict, not torchtext object
VOCAB_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
PREPROCESSOR_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_config.json")
RESULTS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_plots.png")

# --- Preprocessing Parameters ---
MAX_LENGTH = 128  # Maximum sequence length
MIN_FREQ = 2     # Minimum frequency for words to be included in vocabulary

# --- Special Tokens ---
# Define indices explicitly for consistency
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# --- Model Selection ---
# Choose 'LSTM' or 'ANN' (ANN is less suitable without embedding now)
MODEL_TYPE = 'LSTM'

# --- Model Hyperparameters ---
# Common
N_CLASS = 6        # Number of emotion classes
# LSTM Specific
EMBEDDING_DIM = 300 # Increased embedding dim
HIDDEN_DIM = 256   # Increased hidden dim
N_LAYERS = 2       # Number of LSTM layers
# ANN Specific (Input size calculation needs review if not using embeddings)
# ANN_INPUT_SIZE = MAX_LENGTH + 2 # This is now less meaningful

# --- Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64 # Increased batch size
EPOCHS = 15 # Adjusted epochs
LEARNING_RATE_LSTM = 0.001 # Adjusted LR
LEARNING_RATE_ANN = 0.001
SHUFFLE_DATA = True

# --- Application Settings ---
INT_TO_LABEL = {
    0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'
}
LABEL_TO_INT = {v: k for k, v in INT_TO_LABEL.items()}

# --- NLP Configuration ---
SPACY_MODEL = "en_core_web_sm"  # Options: "en_core_web_sm", "en_core_web_md", "en_core_web_lg"