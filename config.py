import torch
import os

BASE_DIR = "."
TRAIN_PATH = os.path.join(BASE_DIR, "training.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

ARTIFACTS_DIR = "artifacts"
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.pt")
VOCAB_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
LABEL_MAP_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "label_map.json")
PREPROCESSOR_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_config.json") # Keep if preprocessor state needed
RESULTS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_plots.png")
TEST_REPORT_PATH = os.path.join(ARTIFACTS_DIR, "test_classification_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, "test_confusion_matrix.png")

# --- Data Handling ---
MAX_LENGTH = 128 # Max sequence length (consider increasing if CNN/Attention helps)
MIN_FREQ = 2     # Minimum word frequency for vocabulary

# --- Vocabulary Special Tokens ---
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# --- Model Selection ---
# MODEL_TYPE = 'LSTM' # Old simple LSTM
MODEL_TYPE = 'CNN_RNN_Attention' # Use the new combined model

# --- Shared Model Hyperparameters ---
EMBEDDING_DIM = 300
DROPOUT_PROB = 0.4 # General dropout probability

# --- CNN Specific Hyperparameters (used if MODEL_TYPE is CNN_RNN_Attention) ---
CNN_OUT_CHANNELS = 100 # Number of output channels for EACH kernel size
CNN_KERNEL_SIZES = [3, 4, 5] # List of kernel sizes for CNN layers

# --- RNN Specific Hyperparameters ---
# Used for BOTH simple LSTM/GRU and the RNN part of CNN_RNN_Attention
RNN_TYPE = 'lstm' # Choose 'lstm' or 'gru'. Used ONLY if MODEL_TYPE='CNN_RNN_Attention'
RNN_HIDDEN_DIM = 256
RNN_LAYERS = 2

# --- Training ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 15 # Adjust as needed
LEARNING_RATE = 0.001 # Single LR for now, might need tuning
WEIGHT_DECAY = 1e-5 # Weight decay for AdamW optimizer
SHUFFLE_DATA = True
NUM_WORKERS = 2 # Dataloader workers

# --- Preprocessing ---
SPACY_MODEL = "en_core_web_md" # Use 'md' or 'lg' for better vectors if using spaCy vectors later

# --- Prediction ---
TOP_K_PREDICTIONS = 3 # For app.py if used

# --- Flag for simple LSTM ---
# If you want to quickly switch back to the old LSTM model without changing
# MODEL_TYPE constantly, you could use a flag, but changing MODEL_TYPE is clearer.
# USE_SIMPLE_LSTM = False