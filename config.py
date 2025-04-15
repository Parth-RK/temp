# config.py
"""
Central configuration file for the emotion classification project.
(TorchText Legacy Independent Version - Flexible Classes)
"""

import torch
import os

# --- Data Paths ---
BASE_DIR = "." # Or your local path
TRAIN_PATH = os.path.join(BASE_DIR, "training.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

# --- Artifact Paths ---
ARTIFACTS_DIR = "artifacts"
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.pt")
VOCAB_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
LABEL_MAP_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "label_map.json")
PREPROCESSOR_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_config.json")
RESULTS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_plots.png")

# --- Preprocessing Parameters ---
MAX_LENGTH = 128
MIN_FREQ = 2

# --- Special Tokens & Indices ---
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# --- Model Selection ---
MODEL_TYPE = 'LSTM' # Keep focused on LSTM as ANN needs significant change

# --- Model Hyperparameters ---
# N_CLASS removed - will be determined dynamically
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 2

# --- Training Parameters ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE_LSTM = 0.001
SHUFFLE_DATA = True

# --- NLP Configuration ---
SPACY_MODEL = "en_core_web_sm"

# --- Inference ---
TOP_K_PREDICTIONS = 3 # How many top predictions to show