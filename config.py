# --- config.py ---
import torch
import os
import uuid

# --- Core Settings ---
RUN_ID = str(uuid.uuid4())[:8] # Unique ID for this run/experiment
SEED = 42 # For reproducibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data Configuration ---
DATA_DIR = "data"
# INPUT_DATA_FILE = "emotion_dataset_full.csv" # Example: Large dataset
INPUT_DATA_FILE = "emotion_data_lite2.csv" # Example: Smaller dataset
INPUT_FILE_PATH = os.path.join(DATA_DIR, INPUT_DATA_FILE)
INPUT_FILE_FORMAT = "csv" # Options: "csv", "tsv", "jsonl" (json lines)
TEXT_COLUMN_INDEX = 1 # Index of the column containing text (0-based)
LABEL_COLUMN_INDEX = 0 # Index of the column containing labels (0-based)
COLUMN_NAMES = ['label', 'text'] # Optional: Provide names if CSV has no header or for clarity
HAS_HEADER = True # Does the input file have a header row?

# --- Artifacts & Output ---
ARTIFACTS_DIR = "artifacts"
RUN_ARTIFACTS_DIR = os.path.join(ARTIFACTS_DIR, f"run_{RUN_ID}") # Store run-specific artifacts
MODEL_SAVE_DIR = os.path.join(RUN_ARTIFACTS_DIR, "model")
BEST_MODEL_FILENAME = "best_model.pt"
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, BEST_MODEL_FILENAME) # Full path to best model
LABEL_MAP_FILENAME = "label_map.json"
LABEL_MAP_PATH = os.path.join(ARTIFACTS_DIR, LABEL_MAP_FILENAME) # Store globally or allow user override
VOCAB_FILENAME = "vocab.json" # Only used for non-transformer models
VOCAB_PATH = os.path.join(RUN_ARTIFACTS_DIR, VOCAB_FILENAME)
TRAINING_PLOTS_FILENAME = "training_plots.png"
TRAINING_PLOTS_PATH = os.path.join(RUN_ARTIFACTS_DIR, TRAINING_PLOTS_FILENAME)
TEST_REPORT_FILENAME = "test_report.txt"
TEST_REPORT_PATH = os.path.join(RUN_ARTIFACTS_DIR, TEST_REPORT_FILENAME)
CONFUSION_MATRIX_FILENAME = "test_confusion_matrix.png"
CONFUSION_MATRIX_PATH = os.path.join(RUN_ARTIFACTS_DIR, CONFUSION_MATRIX_FILENAME)
RUN_CONFIG_FILENAME = "run_config.json"
RUN_CONFIG_PATH = os.path.join(RUN_ARTIFACTS_DIR, RUN_CONFIG_FILENAME) # Save config for this run

# Ensure artifact directories exist
os.makedirs(RUN_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- Model Selection ---
# Options: 'Transformer', 'CNN_RNN_Attention', 'LSTM'
# MODEL_TYPE = 'Transformer'
# MODEL_TYPE = 'CNN_RNN_Attention'
MODEL_TYPE = 'LSTM'

# --- Preprocessing ---
# Options: 'basic', 'spacy' (requires spaCy and model like 'en_core_web_sm')
PREPROCESSOR_TYPE = 'basic' if MODEL_TYPE == 'Transformer' else 'spacy'
# PREPROCESSOR_TYPE = 'spacy' # Can force spaCy for transformers if desired
SPACY_MODEL_NAME = "en_core_web_md" # spaCy model for 'spacy' preprocessor
REMOVE_STOPWORDS = False # Generally False for Transformers, True for others optional

# --- Transformer Model Specific ---
# Ignored if MODEL_TYPE is not 'Transformer'
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased" # e.g., "bert-base-uncased", "roberta-base"

# --- RNN/CNN Model Specific ---
EMBEDDING_DIM = 300
VOCAB_MIN_FREQ = 2
# CNN Specific (for CNN_RNN_Attention)
CNN_OUT_CHANNELS = 100
CNN_KERNEL_SIZES = [3, 4, 5]
# RNN Specific (for CNN_RNN_Attention, LSTM)
RNN_TYPE = 'lstm' # 'lstm' or 'gru'
RNN_HIDDEN_DIM = 256
RNN_LAYERS = 2
# Special Token Indices (for non-transformer models)
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

# --- Training Configuration ---
MAX_LEN = 128 # Max sequence length (for tokenizers and padding)
TRAIN_BATCH_SIZE = 16 if MODEL_TYPE == 'Transformer' else 64
VALID_BATCH_SIZE = 32 if MODEL_TYPE == 'Transformer' else 128
EPOCHS = 5 if MODEL_TYPE == 'Transformer' else 10
LEARNING_RATE = 3e-5 if MODEL_TYPE == 'Transformer' else 1e-3
WEIGHT_DECAY = 0.01 if MODEL_TYPE == 'Transformer' else 1e-5
# Optimizer: AdamW is generally good for both
OPTIMIZER_TYPE = 'AdamW' # Options: 'AdamW', 'Adam', 'SGD'
# Scheduler: Linear warmup common for Transformers, ReduceLROnPlateau for others
SCHEDULER_TYPE = 'linear_warmup' if MODEL_TYPE == 'Transformer' else 'reduce_on_plateau'
WARMUP_PROPORTION = 0.1 # Proportion of training steps for warmup (for linear_warmup)
GRADIENT_CLIP_VALUE = 1.0 # Max norm for gradient clipping (common for Transformers)

# --- Data Splitting ---
VALIDATION_SPLIT_SIZE = 0.15 # Proportion of data for validation
TEST_SPLIT_SIZE = 0.15 # Proportion of data for testing (taken from remaining)
STRATIFY_SPLIT = True # Stratify splits based on labels

# --- Evaluation & Plotting ---
PLOT_TRAINING_HISTORY = True
GENERATE_TEST_REPORT = True
GENERATE_CONFUSION_MATRIX = True
METRIC_FOR_BEST_MODEL = 'loss' # 'accuracy', 'f1_weighted', 'loss'

# --- Logging ---
print(f"--- Configuration Loaded for Run ID: {RUN_ID} ---")
print(f"Device: {DEVICE}")
print(f"Model Type: {MODEL_TYPE}")
if MODEL_TYPE == 'Transformer':
    print(f"Transformer Model: {TRANSFORMER_MODEL_NAME}")
print(f"Using Preprocessor: {PREPROCESSOR_TYPE}")
print(f"Max Sequence Length: {MAX_LEN}")
print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}")
print(f"Run Artifacts Directory: {RUN_ARTIFACTS_DIR}")
print(f"Best Model Path: {BEST_MODEL_PATH}")
print(f"Label Map Path: {LABEL_MAP_PATH}")
print("-------------------------------------------------")

def save_run_config(filepath=RUN_CONFIG_PATH):
    """Saves the current configuration values to a JSON file."""
    import json
    import inspect
    config_vars = {}
    # Iterate over variables in the current module (config.py)
    for name, obj in inspect.getmembers(__import__(__name__)):
        # Filter out modules, functions, classes, builtins, etc.
        if not name.startswith("__") and not inspect.ismodule(obj) and \
           not inspect.isfunction(obj) and not inspect.isclass(obj) and \
           isinstance(obj, (str, int, float, bool, list, tuple, dict, type(None))):
             # Only save simple types that are JSON serializable
            config_vars[name] = obj
    try:
        with open(filepath, 'w') as f:
            json.dump(config_vars, f, indent=4)
        print(f"Run configuration saved to {filepath}")
    except Exception as e:
        print(f"Warning: Could not save run configuration to {filepath}. Error: {e}")

# Automatically save the config when the module is imported
# save_run_config() # Optional: uncomment to save config on import