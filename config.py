# --- config.py ---
import torch
import os
import glob
import json
from pathlib import Path

# --- Core Settings ---
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Selection ---
# Options: 'Transformer', 'CNN_RNN_Attention', 'LSTM'
# This determines which configuration settings are primarily used
# and where artifacts for THIS run will be saved.
MODEL_TYPE = 'Transformer'
# MODEL_TYPE = 'LSTM'
# MODEL_TYPE = 'CNN_RNN_Attention'

# --- Data Configuration ---
DATA_DIR = "."
# *** Specify Train path (Mandatory) ***
TRAIN_FILE = "training.csv"
TRAIN_FILE_PATH = os.path.join(DATA_DIR, TRAIN_FILE)

# *** Specify Validation and Test paths (Optional) ***
# Set to None if you want to split from the training data
VALID_FILE = "validation.csv"
# VALID_FILE = None
VALID_FILE_PATH = os.path.join(DATA_DIR, VALID_FILE) if VALID_FILE else None

TEST_FILE = "test.csv"
# TEST_FILE = None
TEST_FILE_PATH = os.path.join(DATA_DIR, TEST_FILE) if TEST_FILE else None

# --- Data Format & Columns (Applied to all files if loaded) ---
INPUT_FILE_FORMAT = "csv" # Options: "csv", "tsv", "jsonl"
TEXT_COLUMN_INDEX = 0
LABEL_COLUMN_INDEX = 1
COLUMN_NAMES = ['text', 'label']
HAS_HEADER = True

# --- Data Splitting (Used if VALID_FILE_PATH or TEST_FILE_PATH is None) ---
VALIDATION_SPLIT_SIZE = 0.15 # Proportion of data for validation (from train)
TEST_SPLIT_SIZE = 0.15 # Proportion of data for test (from train, after val split)
STRATIFY_SPLIT = True # Stratify splits based on labels if splitting from train

# --- Artifacts & Output ---
ARTIFACTS_DIR = "artifacts"

# *** Define Fixed Artifact Directory based on MODEL_TYPE ***
# No more numbered runs, artifacts saved directly into the model type folder
MODEL_TYPE_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, MODEL_TYPE)

# Define paths relative to the fixed model type directory
MODEL_SAVE_DIR = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, "model") # Keep model in subfolder
BEST_MODEL_FILENAME = "best_model.pt"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME)

# Global label map path (outside model-specific dirs)
LABEL_MAP_FILENAME = "label_map.json"
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, LABEL_MAP_FILENAME)

# Model-specific vocab path (inside model type dir)
VOCAB_FILENAME = "vocab.json" # Only used for non-transformer models
VOCAB_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, VOCAB_FILENAME)

# Model-specific output files (inside model type dir)
TRAINING_PLOTS_FILENAME = "training_plots.png"
TRAINING_PLOTS_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, TRAINING_PLOTS_FILENAME)
TEST_REPORT_FILENAME = "test_report.txt"
TEST_REPORT_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, TEST_REPORT_FILENAME)
CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png"
CONFUSION_MATRIX_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, CONFUSION_MATRIX_FILENAME)
RUN_CONFIG_FILENAME = "run_config.json"
# Path for saving the config of the *current* run
RUN_CONFIG_PATH = os.path.join(MODEL_TYPE_ARTIFACTS_DIR, RUN_CONFIG_FILENAME)

# Ensure artifact directories for the current model type exist
os.makedirs(MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
# Ensure base artifacts dir exists (for global label map)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# --- Preprocessing ---
# Adjust preprocessor based on model type (common pattern)
PREPROCESSOR_TYPE = 'basic' if MODEL_TYPE == 'Transformer' else 'spacy'
SPACY_MODEL_NAME = "en_core_web_sm" # Smaller default model
REMOVE_STOPWORDS = False if MODEL_TYPE == 'Transformer' else False # Keep stopwords for transformers? Often yes.

# --- Transformer Model Specific ---
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased" # Example

# --- RNN/CNN Model Specific ---
EMBEDDING_DIM = 100 # Smaller default
VOCAB_MIN_FREQ = 3
CNN_OUT_CHANNELS = 64
CNN_KERNEL_SIZES = [3, 4, 5]
RNN_TYPE = 'lstm'
RNN_HIDDEN_DIM = 128
RNN_LAYERS = 1 # Simpler default
DROPOUT_PROB = 0.3 # Explicit dropout probability
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# --- Training Configuration ---
MAX_LEN = 128 # Keep consistent for now
TRAIN_BATCH_SIZE = 32 if MODEL_TYPE == 'Transformer' else 64
VALID_BATCH_SIZE = 64 if MODEL_TYPE == 'Transformer' else 128
EPOCHS = 4 if MODEL_TYPE == 'Transformer' else 8
LEARNING_RATE = 2e-5 if MODEL_TYPE == 'Transformer' else 1e-3
WEIGHT_DECAY = 0.01 # Often used with AdamW for regularization
OPTIMIZER_TYPE = 'AdamW' # Good default, works for all types
SCHEDULER_TYPE = 'linear_warmup' if MODEL_TYPE == 'Transformer' else 'reduce_on_plateau'
WARMUP_PROPORTION = 0.1 # Only relevant for linear_warmup
GRADIENT_CLIP_VALUE = 1.0 # Often useful for RNNs and Transformers

# --- Evaluation & Plotting ---
PLOT_TRAINING_HISTORY = True
GENERATE_TEST_REPORT = True
GENERATE_CONFUSION_MATRIX = True
METRIC_FOR_BEST_MODEL = 'accuracy' # or 'f1_weighted' or 'loss'

# --- Logging & Config Saving ---
def save_run_config(filepath=RUN_CONFIG_PATH):
    """Saves the current configuration values to a JSON file for the run."""
    import json
    import inspect
    config_vars = {}
    # Filter variables (same logic as before)
    current_module = __import__(__name__)
    for name, obj in inspect.getmembers(current_module):
        # Include only 'global' variables (typically uppercase)
        # and check if serializable
        if name.isupper() and not name.startswith("__") and not inspect.ismodule(obj) and \
           not inspect.isfunction(obj) and not inspect.isclass(obj) and \
           isinstance(obj, (str, int, float, bool, list, tuple, dict, type(None))):

            # Convert Path objects to string for serialization
            if isinstance(obj, Path):
                config_vars[name] = str(obj)
            elif isinstance(obj, (list, tuple)):
                 # Convert Path objects within lists/tuples
                 config_vars[name] = [str(i) if isinstance(i, Path) else i for i in obj]
            else:
                 config_vars[name] = obj

    try:
        # Ensure the directory exists one last time before writing
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_vars, f, indent=4, default=str) # Use default=str for safety
        # print(f"Run configuration saved to {filepath}") # Optional print
    except Exception as e:
        print(f"Warning: Could not save run configuration to {filepath}. Error: {e}")

# --- Print Summary ---
print(f"--- Configuration Loaded ---")
print(f"Selected Model Type: {MODEL_TYPE}")
print(f"Artifacts Directory for this type: {MODEL_TYPE_ARTIFACTS_DIR}")
print(f"Device: {DEVICE}")
print(f"Training Data: {TRAIN_FILE_PATH}")
print(f"Validation Data: {'Provided (' + str(VALID_FILE_PATH) + ')' if VALID_FILE_PATH else 'Splitting from Train'}")
print(f"Test Data: {'Provided (' + str(TEST_FILE_PATH) + ')' if TEST_FILE_PATH else 'Splitting from Train'}")
print(f"Preprocessor: {PREPROCESSOR_TYPE}")
if MODEL_TYPE == 'Transformer':
    print(f"Transformer Model: {TRANSFORMER_MODEL_NAME}")
else:
    print(f"Vocab Path: {VOCAB_PATH}")
    print(f"Embedding Dim: {EMBEDDING_DIM}, RNN Type: {RNN_TYPE}, Hidden Dim: {RNN_HIDDEN_DIM}")
print("---------------------------")