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
MODEL_TYPE = 'Transformer'
# MODEL_TYPE = 'LSTM'

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
# *** Function to get the next run directory ***
def get_next_run_dir(base_artifacts_dir, model_type):
    model_dir = os.path.join(base_artifacts_dir, model_type)
    os.makedirs(model_dir, exist_ok=True)
    existing_runs = glob.glob(os.path.join(model_dir, "[0-9][0-9][0-9]*")) # Find numeric dirs
    if not existing_runs:
        next_run_num = 1
    else:
        max_run_num = 0
        for run_path in existing_runs:
            try:
                # Use Path object for reliable basename extraction
                run_num = int(Path(run_path).name)
                if run_num > max_run_num:
                    max_run_num = run_num
            except ValueError:
                continue # Ignore non-numeric directory names
        next_run_num = max_run_num + 1
    # Format with leading zeros (e.g., 001)
    run_dir_name = f"{next_run_num:03d}"
    return os.path.join(model_dir, run_dir_name)

# *** Define Run directory dynamically ***
RUN_ARTIFACTS_DIR = get_next_run_dir(ARTIFACTS_DIR, MODEL_TYPE)
# Add RUN_NAME derived from the directory
RUN_NAME = os.path.basename(RUN_ARTIFACTS_DIR) # e.g., "001", "002"
MODEL_SAVE_DIR = os.path.join(RUN_ARTIFACTS_DIR, "model")
BEST_MODEL_FILENAME = "best_model.pt"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME)
# Global label map path (outside run-specific dirs)
LABEL_MAP_FILENAME = "label_map.json"
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, LABEL_MAP_FILENAME)
# Run-specific vocab path
VOCAB_FILENAME = "vocab.json" # Only used for non-transformer models
VOCAB_PATH = os.path.join(RUN_ARTIFACTS_DIR, VOCAB_FILENAME)
# Run-specific output files
TRAINING_PLOTS_FILENAME = "training_plots.png"
TRAINING_PLOTS_PATH = os.path.join(RUN_ARTIFACTS_DIR, TRAINING_PLOTS_FILENAME)
TEST_REPORT_FILENAME = "test_report.txt"
TEST_REPORT_PATH = os.path.join(RUN_ARTIFACTS_DIR, TEST_REPORT_FILENAME)
CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png"
CONFUSION_MATRIX_PATH = os.path.join(RUN_ARTIFACTS_DIR, CONFUSION_MATRIX_FILENAME)
RUN_CONFIG_FILENAME = "run_config.json"
RUN_CONFIG_PATH = os.path.join(RUN_ARTIFACTS_DIR, RUN_CONFIG_FILENAME) # Save config for this run

# Ensure artifact directories for the current run exist
os.makedirs(RUN_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Preprocessing ---
PREPROCESSOR_TYPE = 'basic' if MODEL_TYPE == 'Transformer' else 'spacy'
SPACY_MODEL_NAME = "en_core_web_md"
REMOVE_STOPWORDS = False

# --- Transformer Model Specific ---
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"

# --- RNN/CNN Model Specific ---
EMBEDDING_DIM = 300
VOCAB_MIN_FREQ = 2
CNN_OUT_CHANNELS = 100
CNN_KERNEL_SIZES = [3, 4, 5]
RNN_TYPE = 'lstm'
RNN_HIDDEN_DIM = 256
RNN_LAYERS = 2
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# --- Training Configuration ---
MAX_LEN = 128
TRAIN_BATCH_SIZE = 16 if MODEL_TYPE == 'Transformer' else 64
VALID_BATCH_SIZE = 32 if MODEL_TYPE == 'Transformer' else 128
EPOCHS = 5 if MODEL_TYPE == 'Transformer' else 10
LEARNING_RATE = 3e-5 if MODEL_TYPE == 'Transformer' else 1e-3
WEIGHT_DECAY = 0.01 if MODEL_TYPE == 'Transformer' else 1e-5
OPTIMIZER_TYPE = 'AdamW'
SCHEDULER_TYPE = 'linear_warmup' if MODEL_TYPE == 'Transformer' else 'reduce_on_plateau'
WARMUP_PROPORTION = 0.1
GRADIENT_CLIP_VALUE = 1.0

# --- Evaluation & Plotting ---
PLOT_TRAINING_HISTORY = True
GENERATE_TEST_REPORT = True
GENERATE_CONFUSION_MATRIX = True
METRIC_FOR_BEST_MODEL = 'accuracy'

# --- Logging & Config Saving ---
# (Keep the save_run_config function as before, just ensure it uses RUN_CONFIG_PATH)
def save_run_config(filepath=RUN_CONFIG_PATH):
    """Saves the current configuration values to a JSON file for the run."""
    import json
    import inspect
    config_vars = {}
    # Filter variables (same logic as before)
    for name, obj in inspect.getmembers(__import__(__name__)):
        if not name.startswith("__") and not inspect.ismodule(obj) and \
           not inspect.isfunction(obj) and not inspect.isclass(obj) and \
           isinstance(obj, (str, int, float, bool, list, tuple, dict, type(None))):
            # Handle non-serializable types like paths if needed, or just store strings
            if isinstance(obj, (list, tuple)):
                 config_vars[name] = [str(i) if isinstance(i, type(Path(TRAIN_FILE_PATH))) else i for i in obj]
            elif isinstance(obj, type(Path(TRAIN_FILE_PATH))): # Requires 'from pathlib import Path'
                 config_vars[name] = str(obj)
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


print(f"--- Configuration Loaded ---")
print(f"Model Type: {MODEL_TYPE}")
# Use RUN_NAME for logging
print(f"Run Name / Directory: {RUN_NAME} (in {os.path.dirname(RUN_ARTIFACTS_DIR)})")
print(f"Device: {DEVICE}")
print(f"Run Artifacts Directory: {RUN_ARTIFACTS_DIR}")
print(f"Training Data: {TRAIN_FILE_PATH}")
print(f"Validation Data: {'Provided (' + str(VALID_FILE_PATH) + ')' if VALID_FILE_PATH else 'Splitting from Train'}")
print(f"Test Data: {'Provided (' + str(TEST_FILE_PATH) + ')' if TEST_FILE_PATH else 'Splitting from Train'}")
print("---------------------------")