Okay, I've modified the code to use fixed artifact directories per model type (`artifacts/Transformer`, `artifacts/LSTM`, `artifacts/CNN_RNN_Attention`) instead of numbered run directories (`001`, `002`, etc.).

Here's a summary of the key changes:

1.  **`config.py`:**
    *   Removed the `get_next_run_dir` function.
    *   Removed `RUN_NAME`.
    *   Introduced `MODEL_TYPE_ARTIFACTS_DIR` which points directly to `artifacts/{MODEL_TYPE}`.
    *   All artifact paths (`BEST_MODEL_PATH`, `VOCAB_PATH`, `TRAINING_PLOTS_PATH`, `TEST_REPORT_PATH`, `CONFUSION_MATRIX_PATH`, `RUN_CONFIG_PATH`) are now relative to `MODEL_TYPE_ARTIFACTS_DIR`.
    *   Directory creation (`os.makedirs`) now targets `MODEL_TYPE_ARTIFACTS_DIR`.
    *   Updated print statements to reflect the fixed directory structure.
    *   `save_run_config` now defaults to saving in `MODEL_TYPE_ARTIFACTS_DIR`.

2.  **`app.py`:**
    *   Removed the `find_latest_run_dir` function.
    *   The `main` function now directly constructs the expected `model_type_dir` (e.g., `artifacts/Transformer`) and checks for its existence.
    *   `load_prediction_artifacts` and `load_run_config` now take `model_type_dir` as input and construct paths relative to it.
    *   Updated print statements and variable names (`run_dir` -> `model_type_dir`).

3.  **Other Files (`train.py`, `data_handler.py`, `engine.py`, `plotter.py`, `main.py`):**
    *   These files primarily rely on the paths defined in `config.py`. Since `config.py` was updated correctly, these files should function as intended, saving/loading artifacts to/from the new fixed locations. Minor adjustments to print statements might occur implicitly due to config changes.

Let's look at the updated files:

--- START OF FILE app.py ---

```python
# --- app.py ---
import torch
import os
import json
import argparse
import sys
from operator import itemgetter
import glob

# Import necessary modules after potentially modifying path
try:
    import config # To get default paths and potentially model type if config not saved in run
    import data_handler
    import engine # To load model structure and weights
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure config.py, data_handler.py, and engine.py are accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)

# --- Helper Functions ---

def load_run_config(model_type_dir):
    """Loads the specific configuration saved for the given model type run."""
    config_path = os.path.join(model_type_dir, config.RUN_CONFIG_FILENAME) # Use default filename
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file not found at {config_path}. Using global config.py defaults.")
        # Fallback logic: Use global config values directly.
        # This might be inaccurate if global config changed since the run.
        class RunConfig:
             MODEL_TYPE = config.MODEL_TYPE # Note: This might not match the requested model_type_dir's type
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = getattr(config, 'TRANSFORMER_MODEL_NAME', None) # Use getattr for safety
             # Construct potential path relative to the model_type_dir
             VOCAB_PATH = os.path.join(model_type_dir, config.VOCAB_FILENAME)
             REMOVE_STOPWORDS = getattr(config, 'REMOVE_STOPWORDS', False)
             SPACY_MODEL_NAME = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        # Override MODEL_TYPE based on the directory we are trying to load
        RunConfig.MODEL_TYPE = os.path.basename(model_type_dir)
        return RunConfig()

    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        # Convert loaded dict to an object for easier access (optional)
        class RunConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                # Ensure necessary paths are relative to the loaded model_type_dir
                self.VOCAB_PATH = os.path.join(model_type_dir, config.VOCAB_FILENAME)

        print(f"Loaded run configuration from {config_path}")
        # Ensure the loaded config's model type matches the directory (sanity check)
        if loaded_config.get('MODEL_TYPE') != os.path.basename(model_type_dir):
             print(f"Warning: Loaded config MODEL_TYPE ({loaded_config.get('MODEL_TYPE')}) does not match directory ({os.path.basename(model_type_dir)})")
        return RunConfig(**loaded_config)
    except Exception as e:
        print(f"Error loading run config from {config_path}: {e}. Using global defaults.")
        # Fallback to global config if loading fails
        class RunConfig: # Duplicated fallback logic
             MODEL_TYPE = config.MODEL_TYPE # Again, might not match requested type
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = getattr(config, 'TRANSFORMER_MODEL_NAME', None)
             VOCAB_PATH = os.path.join(model_type_dir, config.VOCAB_FILENAME)
             REMOVE_STOPWORDS = getattr(config, 'REMOVE_STOPWORDS', False)
             SPACY_MODEL_NAME = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        RunConfig.MODEL_TYPE = os.path.basename(model_type_dir) # Override with dir name
        return RunConfig()

def load_prediction_artifacts(model_type_dir):
    """Loads all necessary artifacts for prediction based on the model type's directory."""
    print(f"\nLoading artifacts from model type directory: {model_type_dir}")
    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory not found at {model_type_dir}")
        return None, None, None, None, None

    run_cfg = load_run_config(model_type_dir)

    # Load Label Map (global)
    label_to_int, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_PATH)
    if not int_to_label:
        print("Warning: Label map not found or empty. Predictions will show integer labels.")
        # Create a dummy map if needed elsewhere, or handle None gracefully
        int_to_label = {} # Empty dict signals no mapping available

    n_classes = len(int_to_label) if int_to_label else 0
    if n_classes == 0:
        print("Warning: Cannot determine number of classes from label map.")
        # Might need to infer from model later if possible, or fail

    # Determine paths within the model_type_dir
    model_path = os.path.join(model_type_dir, "model", config.BEST_MODEL_FILENAME)
    vocab_path = run_cfg.VOCAB_PATH # Use path from loaded config (points to model_type_dir)

    vocab_size = None
    vocab_or_tokenizer = None

    if run_cfg.MODEL_TYPE != 'Transformer':
        # Load Vocabulary for non-transformer models
        try:
            vocab = data_handler.Vocabulary.load(vocab_path)
            vocab_size = len(vocab)
            vocab_or_tokenizer = vocab
            print(f"Vocabulary loaded (Size: {vocab_size}).")
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {vocab_path}. Cannot proceed for {run_cfg.MODEL_TYPE} model.")
            return None, None, None, None, None
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return None, None, None, None, None
    else:
         # Load Tokenizer for transformer models
         if data_handler.AutoTokenizer is None:
              print("Error: Transformers library not installed, cannot load tokenizer.")
              return None, None, None, None, None
         try:
              print(f"Loading tokenizer: {run_cfg.TRANSFORMER_MODEL_NAME}")
              # Check if tokenizer files exist within the model_type_dir (optional optimization)
              # If not, from_pretrained will download. If they exist, it should load locally.
              tokenizer = data_handler.AutoTokenizer.from_pretrained(
                  run_cfg.TRANSFORMER_MODEL_NAME,
                  # local_files_only=True # Uncomment to force local loading if desired
              )
              vocab_or_tokenizer = tokenizer
              vocab_size = tokenizer.vocab_size # Use tokenizer's vocab size info
         except Exception as e:
              print(f"Error loading tokenizer '{run_cfg.TRANSFORMER_MODEL_NAME}': {e}")
              return None, None, None, None, None


    # Now load model (needs n_classes, and vocab_size if not transformer)
    if n_classes == 0 and run_cfg.MODEL_TYPE == 'Transformer':
         # Try to infer n_classes from a loaded transformer config if label map failed
         try:
             from transformers import AutoConfig
             model_cfg = AutoConfig.from_pretrained(run_cfg.TRANSFORMER_MODEL_NAME)
             n_classes = model_cfg.num_labels
             print(f"Inferred n_classes={n_classes} from Transformer config.")
         except Exception:
              print("Error: Failed to infer n_classes. Cannot load model.")
              return None, None, None, None, None
    elif n_classes == 0:
         print(f"Error: Cannot determine n_classes for model type {run_cfg.MODEL_TYPE} without a label map.")
         return None, None, None, None, None

    try:
        model = engine.load_trained_model(model_path, run_cfg.MODEL_TYPE, n_classes, vocab_size)
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {model_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading trained model: {e}")
        return None, None, None, None, None

    # Initialize Preprocessor based on run config
    print(f"Initializing preprocessor: {run_cfg.PREPROCESSOR_TYPE}")
    if run_cfg.PREPROCESSOR_TYPE == 'spacy':
         try:
             # Pass specific options used during training if available in run_cfg
             preprocessor = data_handler.SpacyTextPreprocessor(
                  spacy_model_name=getattr(run_cfg, 'SPACY_MODEL_NAME', config.SPACY_MODEL_NAME),
                  remove_stopwords=getattr(run_cfg, 'REMOVE_STOPWORDS', config.REMOVE_STOPWORDS)
             )
         except ImportError as e:
              print(f"Error initializing Spacy Preprocessor: {e}")
              return None, None, None, None, None
         except Exception as e: # Catch other Spacy errors (e.g., model download)
              print(f"Error initializing Spacy Preprocessor: {e}")
              return None, None, None, None, None
    else:
        preprocessor = data_handler.BasicTextCleaner()


    return model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg

# Removed find_latest_run_dir function

# --- Predictor Class ---

class EmotionPredictor:
    def __init__(self, model, vocab_or_tokenizer, preprocessor, int_to_label, run_config):
        self.model = model
        self.vocab_or_tokenizer = vocab_or_tokenizer
        self.preprocessor = preprocessor
        self.int_to_label = int_to_label if int_to_label else {} # Ensure it's a dict
        self.run_config = run_config
        self.device = config.DEVICE # Use global device config for prediction
        self.model.to(self.device)
        self.model.eval()
        print("\nEmotionPredictor initialized.")

    def _preprocess_input(self, text):
        """Prepares raw text input for the specific model type."""
        if isinstance(self.preprocessor, data_handler.SpacyTextPreprocessor):
             # Spacy preprocessor might tokenize, but model might expect string or specific tokens
             if self.run_config.MODEL_TYPE == 'Transformer':
                  cleaned_text = " ".join(self.preprocessor.clean_and_tokenize(text)) # Join tokens back
             else:
                  cleaned_tokens = self.preprocessor.clean_and_tokenize(text) # Keep as tokens
                  return cleaned_tokens # Return tokens for non-transformer vocab
        else: # Basic Cleaner
            cleaned_text = self.preprocessor.clean(text)
            if self.run_config.MODEL_TYPE != 'Transformer':
                 # Simple split for non-transformer vocab
                 # Ensure consistency with how vocab was built (usually tokenized)
                 return self.preprocessor.tokenize(cleaned_text) # Use basic cleaner's tokenize
        return cleaned_text # Return cleaned string for Transformer tokenizer

    def predict(self, text):
        """Predicts emotion probabilities for the input text."""
        processed_input = self._preprocess_input(text)

        try:
            with torch.no_grad():
                if self.run_config.MODEL_TYPE == 'Transformer':
                    # Ensure input is string for tokenizer
                    input_text = processed_input if isinstance(processed_input, str) else " ".join(processed_input)
                    encoding = self.vocab_or_tokenizer.encode_plus(
                        input_text,
                        add_special_tokens=True,
                        max_length=self.run_config.MAX_LEN,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                    )
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                else: # Non-Transformer models
                    # Expect processed_input to be list of tokens
                    if not isinstance(processed_input, list):
                         print(f"Warning: Expected list of tokens for {self.run_config.MODEL_TYPE}, got {type(processed_input)}. Attempting split.")
                         processed_input = str(processed_input).split()

                    numericalized = self.vocab_or_tokenizer.numericalize(processed_input)
                    # Apply padding and SOS/EOS based on how data was prepared for training
                    # This assumes SOS/EOS were used; adjust if not
                    max_len_adjusted = self.run_config.MAX_LEN - 2 # Account for SOS/EOS
                    padded_numericalized = numericalized[:max_len_adjusted]
                    sequence = [config.SOS_IDX] + padded_numericalized + [config.EOS_IDX]

                    # Pad sequence manually if needed, or rely on model/batching if it handles variable lengths
                    # For single prediction, simpler to pad here to match MAX_LEN used in training config
                    padded_sequence = sequence + [config.PAD_IDX] * (self.run_config.MAX_LEN - len(sequence))
                    sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(self.device) # Add batch dim
                    lengths = torch.tensor([len(sequence)], dtype=torch.long).to(self.device) # Original length before padding

                    # Pass lengths, model forward should handle it (e.g., pack_padded_sequence)
                    logits = self.model(text_indices=sequence_tensor, sequence_lengths=lengths)


            probabilities = torch.softmax(logits, dim=1).squeeze()
            probabilities_np = probabilities.cpu().numpy()

            results = []
            for i, prob in enumerate(probabilities_np):
                label_index = i
                # Use label map if available, otherwise show index
                label_name = self.int_to_label.get(label_index, f"Label_{label_index}")
                results.append({'label': label_name, 'score': float(prob)}) # Use dict for clarity

            # Sort by probability descending
            results.sort(key=itemgetter('score'), reverse=True)
            return results

        except Exception as e:
            print(f"\nError during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- Main Application Logic ---

def run_interactive_app(predictor):
    """Handles the interactive command-line loop."""
    print("\n--- Interactive Emotion Prediction ---")
    print(f"Using model: {predictor.run_config.MODEL_TYPE}")
    print("Enter text to classify, or type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting.")
                break

            prediction_results = predictor.predict(user_input)

            if prediction_results:
                print("\nPrediction Results:")
                # Find max score for highlighting
                max_score = prediction_results[0]['score'] if prediction_results else 0
                for result in prediction_results:
                    indicator = " *" if result['score'] == max_score and max_score > 0 else ""
                    print(f"  - {result['label']}: {result['score']:.4f}{indicator}")
            else:
                print("  Prediction failed.")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the loop: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Emotion Prediction App (Loads Model)")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['Transformer', 'CNN_RNN_Attention', 'LSTM'],
        help="Specify the type of model artifacts to load."
    )
    args = parser.parse_args()

    # Construct the path to the model type's artifact directory
    model_type_dir = os.path.join(config.ARTIFACTS_DIR, args.model_type)

    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory for model type '{args.model_type}' not found at {model_type_dir}")
        print("Please ensure the model has been trained first using 'python main.py'.")
        sys.exit(1)

    print(f"Loading artifacts for model type: {args.model_type}")
    model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg = load_prediction_artifacts(model_type_dir)

    if model is None:
        print("Failed to load necessary artifacts. Exiting.")
        sys.exit(1)

    predictor = EmotionPredictor(model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg)
    run_interactive_app(predictor)


if __name__ == "__main__":
    main()
```
--- END OF FILE app.py ---

--- START OF FILE config.py ---

```python
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
```
--- END OF FILE config.py ---

--- START OF FILE data_handler.py ---

```python
# --- data_handler.py ---
import torch
import pandas as pd
import numpy as np
import json
import os
import re
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm.auto import tqdm
import torch.nn as nn
import pandas.api.types as ptypes

# Try importing necessary libraries, warn if unavailable for certain preprocessors
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None # Flag that transformers is not installed

try:
    import spacy
    from nltk.corpus import stopwords as nltk_stopwords
    # Download stopwords if not present, suppress output after first time
    try:
        nltk_stopwords.words('english')
    except LookupError:
        import nltk
        print("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords', quiet=True)

except ImportError:
    spacy = None # Flag that spacy/nltk is not installed
    nltk_stopwords = None

import config # Import configuration

# --- Text Preprocessing ---

class BasicTextCleaner:
    """Basic cleaning: lowercase, remove mentions/URLs, normalize whitespace."""
    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r'@\w+', '', text) # Remove user mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        # Optional: Keep basic punctuation or remove all non-alphanumeric
        # text = re.sub(r"[^a-z0-9\s']", '', text) # More aggressive
        return text

    def preprocess_batch(self, texts):
        """Applies cleaning to a list of texts."""
        return [self.clean(text) for text in texts]

    def tokenize(self, text): # Basic split for consistency if needed elsewhere
        """Tokenizes cleaned text by splitting on whitespace."""
        return self.clean(text).split()

class SpacyTextPreprocessor:
    """Advanced cleaning using spaCy: lemmatization, optional stopword removal."""
    def __init__(self, spacy_model_name=config.SPACY_MODEL_NAME, remove_stopwords=config.REMOVE_STOPWORDS):
        if spacy is None or nltk_stopwords is None:
            raise ImportError("SpacyTextPreprocessor requires 'spacy' and 'nltk' to be installed. Run 'pip install spacy nltk' and download resources (e.g., python -m spacy download en_core_web_sm).")
        self.nlp = self._load_spacy_model(spacy_model_name)
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(nltk_stopwords.words('english')) if remove_stopwords else set()
        print(f"SpacyTextPreprocessor initialized (Model: {spacy_model_name}, Stopwords: {'Enabled' if self.remove_stopwords else 'Disabled'})")

    def _load_spacy_model(self, model_name):
        try:
            # Efficient loading: disable components not needed for lemmatization/tokenization
            return spacy.load(model_name, disable=['parser', 'ner'])
        except OSError:
            print(f"Spacy model '{model_name}' not found. Attempting to download...")
            try:
                spacy.cli.download(model_name)
                print(f"Model '{model_name}' downloaded.")
                return spacy.load(model_name, disable=['parser', 'ner'])
            except Exception as e:
                print(f"Error: Failed to download or load spaCy model '{model_name}'.")
                print(f"Please ensure you have the necessary permissions and network access.")
                print(f"You might need to run: python -m spacy download {model_name}")
                raise OSError(f"Could not load or download spacy model '{model_name}'") from e

    def clean_and_tokenize(self, text):
        """Cleans and tokenizes a single text string, returning a list of lemmas."""
        text = str(text).lower()
        text = re.sub(r'@\w+', '', text) # Remove user mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        # Basic whitespace normalization before spacy
        text = re.sub(r'\s+', ' ', text).strip()

        # Process text with loaded spaCy model
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            # Keep alphanumeric tokens, optionally filter stopwords
            is_valid = token.is_alpha or token.is_digit
            is_stop = token.lemma_ in self.stopwords if self.remove_stopwords else False

            if is_valid and not is_stop:
                 tokens.append(token.lemma_) # Use lemma

        return tokens

    def preprocess_batch(self, texts):
        """Optimized batch processing for spaCy, returns list of token lists."""
        processed_texts_tokens = []
        # Basic cleaning first (more efficient than doing regex inside the loop)
        cleaned_texts = (re.sub(r'\s+', ' ', re.sub(r'http\S+|www\S+|https\S+', '', re.sub(r'@\w+', '', str(text).lower()))).strip() for text in texts)

        # Use nlp.pipe for efficiency
        total = len(texts) if isinstance(texts, list) else None # Estimate total for tqdm if possible
        for doc in tqdm(self.nlp.pipe(cleaned_texts, batch_size=50), total=total, desc="SpaCy Processing"):
            tokens = []
            for token in doc:
                is_valid = token.is_alpha or token.is_digit
                is_stop = token.lemma_ in self.stopwords if self.remove_stopwords else False
                if is_valid and not is_stop:
                    tokens.append(token.lemma_)
            # Append the list of tokens for this document
            processed_texts_tokens.append(tokens)
        return processed_texts_tokens


# --- Vocabulary (for non-Transformer models) ---

class Vocabulary:
    def __init__(self, freq_threshold=config.VOCAB_MIN_FREQ):
        self.itos = {config.PAD_IDX: config.PAD_TOKEN, config.UNK_IDX: config.UNK_TOKEN,
                     config.SOS_IDX: config.SOS_TOKEN, config.EOS_IDX: config.EOS_TOKEN}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        """Builds vocabulary from a list of tokenized sentences."""
        print("Building vocabulary...")
        frequencies = Counter()
        idx = len(self.itos) # Start indexing after special tokens

        # Expect sentence_list to be lists of tokens
        for sentence_tokens in tqdm(sentence_list, desc="Counting Token Frequencies"):
            # Ensure input is iterable (list of tokens)
            if isinstance(sentence_tokens, list):
                frequencies.update(sentence_tokens)
            else:
                print(f"Warning: Skipping non-list item during vocab build: {sentence_tokens}")


        # Sort by frequency and filter
        sorted_freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)

        for word, freq in tqdm(sorted_freq, desc="Creating Mappings"):
            if freq >= self.freq_threshold:
                if word not in self.stoi: # Avoid overwriting special tokens
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        print(f"Vocabulary built. Size: {len(self.itos)} (min freq: {self.freq_threshold})")

    def numericalize(self, text_tokens):
        """Converts a list of tokens to a list of numerical indices."""
        # Handle non-list input gracefully
        if not isinstance(text_tokens, list):
            print(f"Warning: Numericalize received non-list input: {text_tokens}. Attempting split.")
            text_tokens = str(text_tokens).split()
        return [self.stoi.get(token, config.UNK_IDX) for token in text_tokens]

    def save(self, filepath=config.VOCAB_PATH): # Default to config path
        """Saves vocabulary stoi and itos maps to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'stoi': self.stoi,
            'itos': self.itos, # Save both for easier loading/debugging
            'freq_threshold': self.freq_threshold
        }
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=4, ensure_ascii=False)
            print(f"Vocabulary saved to {filepath}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")

    @classmethod
    def load(cls, filepath=config.VOCAB_PATH): # Default to config path
        """Loads vocabulary from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            # Use saved freq_threshold if available, else default from config
            freq_threshold = loaded_data.get('freq_threshold', config.VOCAB_MIN_FREQ)
            vocab = cls(freq_threshold)
            # Important: Convert loaded itos keys back to integers
            vocab.itos = {int(k): v for k,v in loaded_data['itos'].items()}
            # Ensure stoi values are integers (should be, but verify)
            vocab.stoi = {k: int(v) for k, v in loaded_data['stoi'].items()}
            print(f"Vocabulary loaded from {filepath}. Size: {len(vocab)}")
            return vocab
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {filepath}: {e}")
            raise
        except Exception as e:
            print(f"Error loading vocabulary from {filepath}: {e}")
            raise


# --- Label Handling ---

def to_native_type(item):
    """Converts numpy types to native Python types for JSON serialization."""
    if isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.ndarray):
        return item.tolist()
    elif isinstance(item, np.bool_):
        return bool(item)
    elif isinstance(item, (pd.Timestamp, pd.Timedelta)): # Handle pandas time types if they appear
        return str(item)
    return item

def save_label_mappings(label_to_int, int_to_label, filepath=config.LABEL_MAP_PATH):
    """Saves label mappings to a JSON file (global location)."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Ensure keys and values are native Python types and keys are strings for JSON
    try:
        label_to_int_serializable = {str(k): to_native_type(v) for k, v in label_to_int.items()}
        int_to_label_serializable = {str(k): to_native_type(v) for k, v in int_to_label.items()}
    except Exception as e:
         print(f"Error converting label map items for serialization: {e}")
         # Fallback: attempt conversion using str() for problematic items
         label_to_int_serializable = {str(k): str(v) for k, v in label_to_int.items()}
         int_to_label_serializable = {str(k): str(v) for k, v in int_to_label.items()}

    save_data = {
        'label_to_int': label_to_int_serializable,
        'int_to_label': int_to_label_serializable
    }
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4, ensure_ascii=False)
        print(f"Label mappings saved to {filepath}")
    except Exception as e:
        print(f"Error saving label mappings: {e}")

def load_label_mappings(filepath=config.LABEL_MAP_PATH):
    """Loads label mappings from the JSON file (global location)."""
    if not os.path.exists(filepath):
        print(f"Label mapping file not found at {filepath}. Returning None.")
        return None, None # Return None if file doesn't exist

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        # Convert keys back to appropriate types (int for int_to_label keys)
        label_to_int = loaded_data.get('label_to_int', {}) # Keep keys as strings (original labels)
        int_to_label_str_keys = loaded_data.get('int_to_label', {})
        # Convert int_to_label keys to int, handle potential non-integer keys gracefully
        int_to_label = {}
        for k, v in int_to_label_str_keys.items():
            try:
                int_key = int(k)
                int_to_label[int_key] = v
            except ValueError:
                print(f"Warning: Skipping non-integer key '{k}' in int_to_label map from {filepath}")


        if not label_to_int or not int_to_label:
             print(f"Warning: Loaded label map from {filepath} seems incomplete or empty.")
             return None, None

        print(f"Label mappings loaded from {filepath}. Num classes: {len(int_to_label)}")
        return label_to_int, int_to_label
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}. File might be corrupted.")
        return None, None
    except Exception as e:
        print(f"Error loading label mappings from {filepath}: {e}")
        return None, None


# --- Dataset Loading and Preparation ---

def load_raw_data(filepath, # Now a mandatory argument
                  file_format=config.INPUT_FILE_FORMAT,
                  text_col_idx=config.TEXT_COLUMN_INDEX,
                  label_col_idx=config.LABEL_COLUMN_INDEX,
                  col_names=config.COLUMN_NAMES,
                  has_header=config.HAS_HEADER):
    """Loads raw data from a specific file path into a pandas DataFrame."""
    print(f"Attempting to load raw data from: {filepath} (Format: {file_format})")

    # Check if file exists *before* trying to load
    if not filepath or not os.path.exists(filepath):
         print(f"Warning: Data file not found or path is invalid: {filepath}")
         return None # Return None if file not found or path is None/empty

    try:
        read_opts = {'on_bad_lines': 'warn', 'low_memory': False}
        if file_format == "csv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(filepath, header=header, names=names, **read_opts)
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(filepath, sep='\t', header=header, names=names, **read_opts)
        elif file_format == "jsonl":
            df = pd.read_json(filepath, lines=True)
            # JSONL might not have consistent headers, rely on indices or provided names
            if col_names is None: col_names = ['text', 'label'] # Default if not provided
            has_header = False # Assume no header row for selection logic below
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Validate and select columns
        num_cols = len(df.columns)
        if label_col_idx >= num_cols or text_col_idx >= num_cols:
             raise IndexError(f"Column index out of bounds (Label: {label_col_idx}, Text: {text_col_idx}). File '{os.path.basename(filepath)}' has {num_cols} columns: {list(df.columns)}")

        # Determine column names to use for selection
        label_col_name = df.columns[label_col_idx]
        text_col_name = df.columns[text_col_idx]

        print(f"  Using columns - Label: '{label_col_name}' (Index {label_col_idx}), Text: '{text_col_name}' (Index {text_col_idx})")

        # Create DataFrame with standard names 'label' and 'text'
        df_std = pd.DataFrame({
            'label': df[label_col_name],
            'text': df[text_col_name]
        })

        # Drop rows with NaN in selected cols *after* selection
        original_rows = len(df_std)
        df_std = df_std.dropna(subset=['label', 'text']).reset_index(drop=True)
        rows_dropped = original_rows - len(df_std)
        if rows_dropped > 0:
            print(f"  Dropped {rows_dropped} rows with NaN values in 'label' or 'text' columns.")

        # Convert text column to string type to avoid issues later
        df_std['text'] = df_std['text'].astype(str)

        print(f"  Successfully loaded {len(df_std)} rows from {os.path.basename(filepath)}.")
        return df_std

    except FileNotFoundError:
        # This shouldn't happen due to the check at the start, but included for safety
        print(f"Error: Data file somehow not found at {filepath} despite existence check.")
        return None
    except IndexError as e:
         print(f"Error: Problem accessing columns by index in {filepath}. Check TEXT_COLUMN_INDEX/LABEL_COLUMN_INDEX config settings vs file structure. Details: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None

def prepare_data(df_train, df_val, df_test):
    """
    Handles label processing (mapping text labels to integers if needed)
    and determines the number of classes. Saves mappings if created.
    Returns processed dataframes and label information.
    Assumes input DataFrames have 'label' and 'text' columns.
    """
    print("\n--- Preparing Labels ---")
    label_col = 'label' # Standardized column name

    # Ensure labels are strings initially for consistent processing if needed
    try:
        df_train[label_col] = df_train[label_col].astype(str)
        df_val[label_col] = df_val[label_col].astype(str)
        df_test[label_col] = df_test[label_col].astype(str)
    except Exception as e:
        print(f"Warning: Could not convert label column to string. Attempting to proceed. Error: {e}")


    label_to_int, int_to_label = load_label_mappings() # Try loading existing map first

    n_classes = None

    if label_to_int and int_to_label:
        print(f"Using pre-loaded label map from {config.LABEL_MAP_PATH}")
        n_classes = len(int_to_label)
        print(f"Applying loaded mapping ({n_classes} classes)...")
        # Apply mapping to all datasets
        for df_name, df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
            original_labels = set(df[label_col].unique())
            # Map string labels using the loaded map
            df['label_int'] = df[label_col].map(label_to_int)
            # Handle labels present in data but not in map
            unmapped_mask = df['label_int'].isnull()
            if unmapped_mask.any():
                unmapped_labels = set(df.loc[unmapped_mask, label_col].unique())
                print(f"Warning ({df_name}): Found labels not in loaded map: {unmapped_labels}. Dropping {unmapped_mask.sum()} rows.")
                df.dropna(subset=['label_int'], inplace=True)

            df[label_col] = df['label_int'].astype(int) # Convert mapped labels to int
            df.drop(columns=['label_int'], inplace=True)

    else: # No map loaded, create one from training data
        print("No pre-loaded label map found or map was invalid. Creating new mappings based on training data.")
        unique_train_labels = sorted(df_train[label_col].unique())
        label_to_int = {label: i for i, label in enumerate(unique_train_labels)}
        int_to_label = {i: label for label, i in label_to_int.items()}
        n_classes = len(label_to_int)
        print(f"Created mapping for {n_classes} labels: {unique_train_labels}")

        # Apply new mapping to all sets
        for df_name, df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
             df['label_int'] = df[label_col].map(label_to_int)
             # Check for labels in val/test not seen in train
             unmapped_mask = df['label_int'].isnull()
             if unmapped_mask.any():
                 unmapped_labels = set(df.loc[unmapped_mask, label_col].unique())
                 print(f"Warning ({df_name}): Found labels not present in training data: {unmapped_labels}. Dropping {unmapped_mask.sum()} rows.")
                 df.dropna(subset=['label_int'], inplace=True)

             df[label_col] = df['label_int'].astype(int)
             df.drop(columns=['label_int'], inplace=True)

        # Save the newly created map to the global location
        save_label_mappings(label_to_int, int_to_label)

    if n_classes is None:
         raise ValueError("Could not determine the number of classes after label processing.")

    print(f"\nLabel preparation complete. Determined {n_classes} classes.")
    print(f"Final int_to_label mapping: {int_to_label}")

    # Final check for NaNs introduced by mapping issues (should be handled above, but belt-and-suspenders)
    df_train.dropna(subset=['label', 'text'], inplace=True)
    df_val.dropna(subset=['label', 'text'], inplace=True)
    df_test.dropna(subset=['label', 'text'], inplace=True)
    print(f"Final dataset sizes after label processing - Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")


    return df_train, df_val, df_test, label_to_int, int_to_label, n_classes


# --- PyTorch Dataset and DataLoader ---

class GenericDataset(Dataset):
    """
    A generic dataset class adaptable for different model types.
    Expects preprocessed texts and integer labels.
    """
    def __init__(self, texts, labels, tokenizer=None, vocab=None, max_len=config.MAX_LEN, model_type=config.MODEL_TYPE):
        self.texts = texts # List of preprocessed texts (strings for Transformers, list of tokens for others)
        self.labels = labels # List/array of integer labels
        self.tokenizer = tokenizer # HuggingFace tokenizer (for Transformers)
        self.vocab = vocab       # Custom Vocabulary object (for non-Transformers)
        self.max_len = max_len
        self.model_type = model_type

        if not isinstance(self.texts, list) or not isinstance(self.labels, (list, np.ndarray)):
             raise TypeError("Inputs 'texts' and 'labels' must be lists or numpy arrays.")
        if len(self.texts) != len(self.labels):
            raise ValueError(f"Length mismatch between texts ({len(self.texts)}) and labels ({len(self.labels)}).")


        if self.model_type == 'Transformer':
            if self.tokenizer is None:
                 raise ValueError("Transformer model type requires a HuggingFace tokenizer.")
            # Ensure texts are strings for Transformer tokenizer
            self.texts = [str(t) if not isinstance(t, str) else t for t in self.texts]
        elif self.model_type in ['CNN_RNN_Attention', 'LSTM']:
             if self.vocab is None:
                 raise ValueError(f"{self.model_type} model type requires a custom Vocabulary object.")
             # Ensure texts are lists of tokens for vocab numericalization
             if not all(isinstance(t, list) for t in self.texts):
                  # This can happen if BasicTextCleaner was used with LSTM/CNN; attempt basic split
                  print(f"Warning: Non-tokenized text detected for {self.model_type}. Using basic whitespace split.")
                  self.texts = [str(t).split() for t in self.texts]
        else:
             raise ValueError(f"Unsupported MODEL_TYPE '{self.model_type}' in GenericDataset.")


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        # Ensure label is integer before converting to tensor
        label_int = int(self.labels[index])
        label = torch.tensor(label_int, dtype=torch.long)

        if self.model_type == 'Transformer':
            # Input text should be a string here
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length', # Pad to max_len
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt', # Return PyTorch tensors
            )
            return {
                # Flatten tensors to remove the batch dimension (DataLoader will re-add it)
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': label
            }
        else: # CNN_RNN_Attention, LSTM, etc.
            # Input text should be a list of tokens here
            numericalized_tokens = self.vocab.numericalize(text)

            # Truncate considering SOS and EOS tokens
            max_len_adjusted = self.max_len - 2
            truncated_tokens = numericalized_tokens[:max_len_adjusted]

            # Add SOS and EOS tokens
            sequence = [config.SOS_IDX] + truncated_tokens + [config.EOS_IDX]
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)

            # We will pad sequences dynamically in the collate function
            return {
                'sequence': sequence_tensor,
                'labels': label
             }


def create_dataloaders(train_data, val_data, test_data, model_type=config.MODEL_TYPE,
                       batch_size=config.TRAIN_BATCH_SIZE, val_batch_size=config.VALID_BATCH_SIZE,
                       tokenizer=None, vocab=None): # Pass tokenizer/vocab explicitly
    """Creates DataLoaders for train, validation, and test sets."""

    if model_type == 'Transformer':
        # Default collate_fn works for Transformers as dataset returns dict of tensors
        collate_fn = None
    else:
        # Custom collate for non-transformers (padding sequences within a batch)
        def collate_non_transformer(batch):
            sequences = [item['sequence'] for item in batch]
            labels = torch.stack([item['labels'] for item in batch]) # Stack labels into a tensor

            # Get sequence lengths BEFORE padding
            lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)

            # Pad sequences within the batch to the length of the longest sequence in the batch
            padded_sequences = nn.utils.rnn.pad_sequence(
                sequences,
                batch_first=True,
                padding_value=config.PAD_IDX # Use the PAD index from config
            )

            # Return padded sequences, labels, and original lengths
            return padded_sequences, labels, lengths

        collate_fn = collate_non_transformer

    # Determine number of workers based on OS (multiprocessing issues on Windows)
    num_workers = 0 # Default to 0 for wider compatibility, especially Windows

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False, # No need to shuffle validation/test data
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=val_batch_size, # Use validation batch size for test
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False
    )

    print(f"\nDataLoaders created (Batch Size: Train={batch_size}, Val/Test={val_batch_size}).")
    return train_loader, val_loader, test_loader

# --- Main Data Pipeline Function ---

def get_data_pipeline(force_rebuild_vocab=False):
    """
    Orchestrates the entire data loading, preprocessing, and preparation pipeline
    based on the settings in config.py.

    Args:
        force_rebuild_vocab (bool): If True, rebuilds vocabulary even if a file exists.

    Returns:
        tuple: Contains:
            - train_loader (DataLoader)
            - val_loader (DataLoader)
            - test_loader (DataLoader)
            - label_to_int (dict): Mapping from string label to integer.
            - int_to_label (dict): Mapping from integer to string label.
            - n_classes (int): Number of unique classes.
            - vocab_or_tokenizer: Either a Vocabulary object or a HuggingFace tokenizer.
            - vocab_size (int): Size of the vocabulary or tokenizer vocab.
    """
    print("--- Starting Data Pipeline ---")

    # 1. Load Raw Data
    print("\n--- Loading Data ---")
    df_train = load_raw_data(filepath=config.TRAIN_FILE_PATH)
    if df_train is None or df_train.empty:
        raise FileNotFoundError(f"CRITICAL: Training data failed to load from {config.TRAIN_FILE_PATH}. Cannot proceed.")

    df_val = load_raw_data(filepath=config.VALID_FILE_PATH)
    df_test = load_raw_data(filepath=config.TEST_FILE_PATH)

    # 2. Split Data if Necessary
    # Use copies to avoid modifying original dataframes loaded above
    df_train_processed = df_train.copy()
    df_val_processed = df_val.copy() if df_val is not None else None
    df_test_processed = df_test.copy() if df_test is not None else None

    train_needs_split = df_val_processed is None or df_test_processed is None
    if train_needs_split:
        print("\n--- Splitting Training Data ---")
        # Base splitting on the initially loaded training data copy
        df_to_split = df_train_processed

        # Split validation set first if needed
        if df_val_processed is None:
            print("Validation data not provided. Splitting from train...")
            if len(df_to_split) < 2:
                print("Warning: Not enough training data to create a validation split. Validation set will be empty.")
                df_val_processed = pd.DataFrame(columns=df_to_split.columns)
                df_train_intermediate = df_to_split # All remaining data is for train/test split
            else:
                print(f"Splitting validation set ({config.VALIDATION_SPLIT_SIZE*100:.1f}%)...")
                stratify_col_val = df_to_split['label'] if config.STRATIFY_SPLIT and 'label' in df_to_split else None
                try:
                    df_train_intermediate, df_val_processed = train_test_split(
                        df_to_split,
                        test_size=config.VALIDATION_SPLIT_SIZE,
                        random_state=config.SEED,
                        stratify=stratify_col_val
                    )
                    print(f"  Intermediate Train size: {len(df_train_intermediate)}, Val size: {len(df_val_processed)}")
                except ValueError as e:
                     print(f"Warning: Stratified split for validation failed ({e}). Performing non-stratified split.")
                     df_train_intermediate, df_val_processed = train_test_split(
                        df_to_split,
                        test_size=config.VALIDATION_SPLIT_SIZE,
                        random_state=config.SEED,
                        stratify=None
                    )
                     print(f"  Intermediate Train size: {len(df_train_intermediate)}, Val size: {len(df_val_processed)}")
        else:
            # Validation data was provided, use all original train data for train/test split
            df_train_intermediate = df_to_split


        # Split test set from the remaining training data if needed
        if df_test_processed is None:
             print("Test data not provided. Splitting from remaining train...")
             if len(df_train_intermediate) < 2:
                 print("Warning: Not enough remaining training data to create a test split. Test set will be empty.")
                 df_test_processed = pd.DataFrame(columns=df_train_intermediate.columns)
                 df_train_final = df_train_intermediate # All remaining is train
             else:
                # Calculate test split size relative to the *original* train size
                # Split size needed from the *intermediate* training data
                current_train_fraction = len(df_train_intermediate) / len(df_to_split) if len(df_to_split) > 0 else 1.0
                # Avoid division by zero if intermediate train is empty
                if current_train_fraction <= 0: current_train_fraction = 1.0

                # Ensure effective_split_size is between 0 and almost 1
                effective_split_size = config.TEST_SPLIT_SIZE / current_train_fraction
                effective_split_size = min(max(0.0, effective_split_size), 1.0 - (1/len(df_train_intermediate)) if len(df_train_intermediate) > 1 else 0.0)


                if effective_split_size <= 0 or (1-effective_split_size) * len(df_train_intermediate) < 1:
                     print(f"Warning: Calculated test split size ({effective_split_size:.3f}) is too small or leaves no training data. Test set will be empty.")
                     df_test_processed = pd.DataFrame(columns=df_train_intermediate.columns)
                     df_train_final = df_train_intermediate
                else:
                     print(f"Splitting test set ({effective_split_size*100:.1f}% from remaining train)...")
                     stratify_col_test = df_train_intermediate['label'] if config.STRATIFY_SPLIT and 'label' in df_train_intermediate else None
                     try:
                         df_train_final, df_test_processed = train_test_split(
                             df_train_intermediate,
                             test_size=effective_split_size,
                             random_state=config.SEED,
                             stratify=stratify_col_test
                         )
                     except ValueError as e:
                         print(f"Warning: Stratified split for test failed ({e}). Performing non-stratified split.")
                         df_train_final, df_test_processed = train_test_split(
                             df_train_intermediate,
                             test_size=effective_split_size,
                             random_state=config.SEED,
                             stratify=None
                         )
                     print(f"  Final Train size: {len(df_train_final)}, Test size: {len(df_test_processed)}")
        else:
             # Test data was provided, the intermediate training data is the final training data
             df_train_final = df_train_intermediate

        # Assign final split dataframes
        df_train_processed = df_train_final
        # df_val_processed and df_test_processed are already assigned
        print("--- Data Splitting Finished ---")


    # 3. Handle Labels (Map to Integers, Create/Save Mappings)
    # Pass the potentially split dataframes
    df_train_processed, df_val_processed, df_test_processed, \
    label_to_int, int_to_label, n_classes = prepare_data(
        df_train_processed, df_val_processed, df_test_processed
    )

    # Ensure we have valid dataframes after potential splitting and label processing
    if df_train_processed.empty:
        raise ValueError("Training data is empty after processing. Cannot proceed.")
    if df_val_processed.empty:
        print("Warning: Validation data is empty after processing.")
        # Optionally handle this - e.g., skip validation? For now, allow empty loader.
    if df_test_processed.empty:
        print("Warning: Test data is empty after processing.")
        # Optionally handle this - e.g., skip testing? For now, allow empty loader.


    # 4. Initialize Preprocessor and Tokenizer/Vocabulary
    print(f"\nInitializing preprocessor: {config.PREPROCESSOR_TYPE}")
    if config.PREPROCESSOR_TYPE == 'spacy':
        if spacy is None: raise ImportError("spaCy selected but not installed/available.")
        try:
             preprocessor = SpacyTextPreprocessor(
                 spacy_model_name=config.SPACY_MODEL_NAME,
                 remove_stopwords=config.REMOVE_STOPWORDS
            )
        except OSError as e:
             print(f"Fatal Error: Failed to initialize SpacyTextPreprocessor: {e}")
             sys.exit(1) # Exit if spacy model fails to load/download
    elif config.PREPROCESSOR_TYPE == 'basic':
        preprocessor = BasicTextCleaner()
    else:
        raise ValueError(f"Unsupported PREPROCESSOR_TYPE: {config.PREPROCESSOR_TYPE}")

    vocab_or_tokenizer = None
    vocab_size = None

    if config.MODEL_TYPE == 'Transformer':
        if AutoTokenizer is None:
             raise ImportError("HuggingFace Transformers library not installed. Needed for MODEL_TYPE='Transformer'.")
        print(f"Loading HuggingFace Tokenizer: {config.TRANSFORMER_MODEL_NAME}")
        try:
             vocab_or_tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
             vocab_size = vocab_or_tokenizer.vocab_size
        except Exception as e:
             print(f"Error loading Transformer tokenizer '{config.TRANSFORMER_MODEL_NAME}': {e}")
             raise # Re-raise as this is critical
    else:
        # Build or load vocabulary for non-transformer models
        vocab_path = config.VOCAB_PATH # Path within the model type's artifact dir
        if os.path.exists(vocab_path) and not force_rebuild_vocab:
            print(f"Attempting to load existing vocabulary from: {vocab_path}")
            try:
                vocab_or_tokenizer = Vocabulary.load(vocab_path)
            except Exception as e:
                print(f"Failed to load vocabulary, rebuilding. Error: {e}")
                vocab_or_tokenizer = None # Force rebuild below
        else:
             if force_rebuild_vocab:
                 print("Vocabulary rebuild forced.")
             else:
                 print(f"No existing vocabulary found at {vocab_path}. Building new vocabulary.")

        if vocab_or_tokenizer is None:
            print("Preprocessing training text for vocabulary building...")
            # We need tokenized text for vocab building
            # Use the chosen preprocessor's batch method
            # Spacy preprocessor's preprocess_batch returns list of token lists
            # Basic preprocessor's preprocess_batch returns list of strings; need to tokenize them
            if isinstance(preprocessor, SpacyTextPreprocessor):
                 train_tokens_list = preprocessor.preprocess_batch(df_train_processed['text'].tolist())
            else: # Basic cleaner - needs tokenization step
                 cleaned_train_texts = preprocessor.preprocess_batch(df_train_processed['text'].tolist())
                 train_tokens_list = [preprocessor.tokenize(text) for text in tqdm(cleaned_train_texts, desc="Tokenizing Train")]


            vocab_or_tokenizer = Vocabulary(freq_threshold=config.VOCAB_MIN_FREQ)
            vocab_or_tokenizer.build_vocabulary(train_tokens_list)
            vocab_or_tokenizer.save(vocab_path) # Save the new vocab to its designated path

        vocab_size = len(vocab_or_tokenizer)
        print(f"Using Vocabulary. Size: {vocab_size}")


    # 5. Preprocess Text Data (Apply Cleaning and Tokenization)
    print("\nApplying text preprocessing to all datasets...")
    # Important: The output format depends on the preprocessor and model type needs
    # - Transformers need strings.
    # - LSTM/CNN need lists of tokens.

    if config.MODEL_TYPE == 'Transformer':
         # Use basic cleaner or join spacy tokens back into strings
         if isinstance(preprocessor, SpacyTextPreprocessor):
              print("Joining SpaCy tokens for Transformer model...")
              train_texts = [" ".join(tokens) for tokens in preprocessor.preprocess_batch(df_train_processed['text'].tolist())]
              val_texts = [" ".join(tokens) for tokens in preprocessor.preprocess_batch(df_val_processed['text'].tolist())]
              test_texts = [" ".join(tokens) for tokens in preprocessor.preprocess_batch(df_test_processed['text'].tolist())]
         else: # Basic cleaner already produces strings
              train_texts = preprocessor.preprocess_batch(df_train_processed['text'].tolist())
              val_texts = preprocessor.preprocess_batch(df_val_processed['text'].tolist())
              test_texts = preprocessor.preprocess_batch(df_test_processed['text'].tolist())
    else: # LSTM, CNN_RNN_Attention need lists of tokens
         # Use the preprocessor's batch method directly if it returns token lists (like Spacy's)
         # If basic cleaner, tokenize after cleaning
         if isinstance(preprocessor, SpacyTextPreprocessor):
              train_texts = preprocessor.preprocess_batch(df_train_processed['text'].tolist())
              val_texts = preprocessor.preprocess_batch(df_val_processed['text'].tolist())
              test_texts = preprocessor.preprocess_batch(df_test_processed['text'].tolist())
         else: # Basic cleaner
              train_texts_cleaned = preprocessor.preprocess_batch(df_train_processed['text'].tolist())
              val_texts_cleaned = preprocessor.preprocess_batch(df_val_processed['text'].tolist())
              test_texts_cleaned = preprocessor.preprocess_batch(df_test_processed['text'].tolist())
              train_texts = [preprocessor.tokenize(t) for t in train_texts_cleaned]
              val_texts = [preprocessor.tokenize(t) for t in val_texts_cleaned]
              test_texts = [preprocessor.tokenize(t) for t in test_texts_cleaned]

    print("Text preprocessing complete.")

    # 6. Create PyTorch Datasets
    print("\nCreating PyTorch Datasets...")
    # Use the preprocessed texts and integer labels
    # Handle potentially empty validation/test sets
    train_dataset = GenericDataset(
        texts=train_texts,
        labels=df_train_processed['label'].tolist(), # Use .tolist() for consistency
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    )

    val_dataset = GenericDataset(
        texts=val_texts,
        labels=df_val_processed['label'].tolist(),
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    ) if not df_val_processed.empty else None # Create dataset only if df is not empty

    test_dataset = GenericDataset(
        texts=test_texts,
        labels=df_test_processed['label'].tolist(),
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    ) if not df_test_processed.empty else None # Create dataset only if df is not empty


    # 7. Create DataLoaders
    # Pass the datasets (which might be None)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        model_type=config.MODEL_TYPE,
        batch_size=config.TRAIN_BATCH_SIZE,
        val_batch_size=config.VALID_BATCH_SIZE,
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None
    )

    print("\n--- Data Pipeline Finished ---")
    return train_loader, val_loader, test_loader, label_to_int, int_to_label, n_classes, vocab_or_tokenizer, vocab_size
```
--- END OF FILE data_handler.py ---

--- START OF FILE dataman.py ---

```python
# --- dataman.py ---
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import sys

# Attempt to import config for defaults, provide fallbacks if it fails
try:
    import config
except ImportError:
    print("Warning: config.py not found. Using hardcoded defaults in dataman.")
    # Define minimal defaults if config cannot be imported
    class ConfigFallback:
        DATA_DIR = "." # Use current directory as default data dir
        # Assume standard 'text', 'label' columns if config is missing
        TEXT_COLUMN_INDEX = 0 # Default if no header/names from config
        LABEL_COLUMN_INDEX = 1 # Default if no header/names from config
        COLUMN_NAMES = ['text', 'label'] # Default if no header and config missing
        HAS_HEADER = True # Default assumption
        SEED = 42
        # Define INPUT_FILE_PATH based on common names if config unavailable
        if os.path.exists("training.csv"):
             INPUT_FILE_PATH = "training.csv"
        elif os.path.exists("data.csv"):
             INPUT_FILE_PATH = "data.csv"
        else:
             INPUT_FILE_PATH = None # Cannot determine default input
    config = ConfigFallback()
except Exception as e:
     print(f"Warning: Error importing config.py: {e}. Using hardcoded defaults in dataman.")
     config = ConfigFallback() # Use fallback


def _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format):
    """Helper to load data based on format and standardize columns."""
    print(f"Loading data from: {input_path} (Format: {file_format})")
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None

    try:
        read_opts = {'on_bad_lines': 'warn', 'low_memory': False}
        if file_format == "csv":
            header = 0 if has_header else None
            # Provide default names only if no header AND col_names is None (use config default/fallback)
            names = None if has_header else (col_names if col_names else ConfigFallback.COLUMN_NAMES)
            df = pd.read_csv(input_path, header=header, names=names, **read_opts)
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else (col_names if col_names else ConfigFallback.COLUMN_NAMES)
            df = pd.read_csv(input_path, sep='\t', header=header, names=names, **read_opts)
        elif file_format == "jsonl":
            df = pd.read_json(input_path, lines=True)
            # JSONL often lacks headers, column order might vary. Indices are crucial.
            has_header = False # Assume no standard header row for selection logic
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Select and rename columns based on indices provided
        num_cols = len(df.columns)
        if text_col_idx >= num_cols or label_col_idx >= num_cols:
             raise IndexError(f"Column index out of bounds (Text: {text_col_idx}, Label: {label_col_idx}). File '{os.path.basename(input_path)}' has {num_cols} columns: {list(df.columns)}")

        text_col_name = df.columns[text_col_idx]
        label_col_name = df.columns[label_col_idx]

        # Create new DataFrame with standardized names
        df_std = pd.DataFrame({
            'label': df[label_col_name],
            'text': df[text_col_name]
        })
        df_std = df_std.dropna(subset=['label', 'text']).reset_index(drop=True)
        df_std['text'] = df_std['text'].astype(str) # Ensure text is string

        print(f"Loaded {len(df_std)} rows (after dropping NaNs).")
        print(f"Using columns: label='{label_col_name}' (idx {label_col_idx}), text='{text_col_name}' (idx {text_col_idx})")
        return df_std

    except FileNotFoundError:
        # Should be caught above, but for safety
        print(f"Error: Input file not found at {input_path}")
        return None
    except IndexError as e:
         print(f"Error: Column index out of bounds. Check --text_col ({text_col_idx}) and --label_col ({label_col_idx}) for file {input_path}. Details: {e}")
         return None
    except Exception as e:
        print(f"Error loading data from {input_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_balanced_subset(input_path, output_path, n_samples_per_class,
                           text_col_idx=config.TEXT_COLUMN_INDEX,
                           label_col_idx=config.LABEL_COLUMN_INDEX,
                           col_names=config.COLUMN_NAMES, # Use config default/fallback
                           has_header=config.HAS_HEADER, # Use config default/fallback
                           file_format="csv"):
    """
    Creates a balanced dataset subset with n samples from each label category.

    Args:
        input_path (str): Path to the input data file.
        output_path (str): Path to save the balanced dataset.
        n_samples_per_class (int): Number of samples per label.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header (overrides config).
        has_header (bool): If the file has a header (overrides config).
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    # Use provided args if they differ from defaults, otherwise use config/fallback values
    current_text_col = text_col_idx
    current_label_col = label_col_idx
    current_has_header = has_header
    current_col_names = col_names # May be None

    df = _load_data(input_path, current_text_col, current_label_col, current_col_names, current_has_header, file_format)
    if df is None:
        return

    print(f"\nOriginal dataset shape (after load & NaN drop): {df.shape}")

    # Ensure label column is treated as string for value counts and grouping
    df['label'] = df['label'].astype(str)

    print("\nOriginal Label Distribution:")
    print(df['label'].value_counts())

    balanced_dfs = []
    unique_labels = df['label'].unique()

    print(f"\nCreating balanced subset with {n_samples_per_class} samples per class...")
    for label in unique_labels:
        label_df = df[df['label'] == label]
        available_samples = len(label_df)
        sample_size = min(n_samples_per_class, available_samples)

        if sample_size < n_samples_per_class:
            print(f"  Warning: Label '{label}' has only {available_samples} samples (requested {n_samples_per_class}). Taking all {sample_size} available.")
        elif sample_size == 0:
             print(f"  Warning: Label '{label}' has 0 samples. Skipping.")
             continue # Skip labels with no samples

        # Use replace=False for sampling without replacement
        sampled_df = label_df.sample(n=sample_size, random_state=config.SEED, replace=False)
        balanced_dfs.append(sampled_df)

    if not balanced_dfs:
        print("Error: No data collected for balancing. Check input data and parameters.")
        return

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True) # Shuffle rows

    print(f"\nBalanced subset shape: {balanced_df.shape}")
    print("Balanced Subset Label Distribution:")
    print(balanced_df['label'].value_counts())

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
        # Save with header using standard column names 'label', 'text'
        balanced_df.to_csv(output_path, index=False, header=True)
        print(f"\nBalanced subset saved to {output_path}")
    except Exception as e:
        print(f"Error saving balanced subset: {e}")

def split_data(input_path, train_path, val_path, test_path,
               val_size=0.15, test_size=0.15, stratify=True,
               text_col_idx=config.TEXT_COLUMN_INDEX,
               label_col_idx=config.LABEL_COLUMN_INDEX,
               col_names=config.COLUMN_NAMES, # Use config default/fallback
               has_header=config.HAS_HEADER, # Use config default/fallback
               file_format="csv"):
    """
    Splits the data into train, validation, and test sets.

    Args:
        input_path (str): Path to the input data file.
        train_path (str): Path to save the training set.
        val_path (str): Path to save the validation set.
        test_path (str): Path to save the test set.
        val_size (float): Proportion for validation set (from original).
        test_size (float): Proportion for test set (from original).
        stratify (bool): Whether to stratify based on labels.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header (overrides config).
        has_header (bool): If the file has a header (overrides config).
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    # Use provided args if they differ from defaults, otherwise use config/fallback values
    current_text_col = text_col_idx
    current_label_col = label_col_idx
    current_has_header = has_header
    current_col_names = col_names # May be None

    df = _load_data(input_path, current_text_col, current_label_col, current_col_names, current_has_header, file_format)
    if df is None:
        return

    print(f"Total data for splitting (after load & NaN drop): {len(df)} rows")

    if len(df) < 3:
        print("Error: Not enough data (less than 3 rows) to perform train/val/test split.")
        return

    if (val_size + test_size) >= 1.0:
        print(f"Error: Sum of validation ({val_size}) and test ({test_size}) sizes must be less than 1.0")
        return
    if val_size < 0 or test_size < 0:
         print(f"Error: Validation ({val_size}) and test ({test_size}) sizes must be non-negative.")
         return


    # Ensure label column is suitable for stratification if requested
    stratify_col = None
    if stratify:
        # Convert label to string for robust stratification, handle potential errors
        try:
            df['label_str'] = df['label'].astype(str)
            # Check if stratification is possible (at least 2 samples per class, or at least 1 sample if n_splits=1)
            label_counts = df['label_str'].value_counts()
            if any(count < 2 for count in label_counts):
                 print("Warning: Stratification may not be possible due to classes with fewer than 2 samples. Attempting anyway, but sklearn might raise an error or fallback.")
                 # Sklearn handles this internally in recent versions, often with a warning.
            stratify_col = df['label_str']
        except Exception as e:
             print(f"Warning: Could not prepare label column for stratification ({e}). Stratification disabled.")
             stratify = False # Disable stratification


    # --- Splitting Logic ---
    # 1. Split off Test set first
    train_val_df = df
    test_df = pd.DataFrame(columns=df.columns) # Initialize empty
    if test_size > 0:
        try:
            train_val_df, test_df = train_test_split(
                df,
                test_size=test_size,
                random_state=config.SEED,
                stratify=stratify_col
            )
        except ValueError as e:
             # This might happen if stratification fails (e.g., single sample classes)
             print(f"Warning: Stratified split for test set failed ({e}). Performing non-stratified split.")
             train_val_df, test_df = train_test_split(
                 df, test_size=test_size, random_state=config.SEED, stratify=None)
    else:
        print("Test size is 0. Test set will be empty.")
        # train_val_df remains the full dataset (minus label_str if added)


    # 2. Split remaining into Train and Validation
    train_df = train_val_df
    val_df = pd.DataFrame(columns=df.columns) # Initialize empty
    if val_size > 0 and len(train_val_df) > 0:
        # Adjust val_size relative to the *remaining* data after test split
        # Avoid division by zero if original df had size 0 or test_size was 1
        denominator = (1.0 - test_size)
        if denominator > 0 and len(train_val_df) >= 2: # Need at least 2 samples to split further
            relative_val_size = val_size / denominator
            # Ensure relative size is valid (e.g., not > 1)
            relative_val_size = min(max(0.0, relative_val_size), 1.0 - (1 / len(train_val_df)))

            if relative_val_size > 0:
                 # Prepare stratification column for the train_val split
                 stratify_col_train_val = None
                 if stratify and 'label_str' in train_val_df:
                     # Check stratification possibility for the remaining data
                      label_counts_tv = train_val_df['label_str'].value_counts()
                      if any(count < 2 for count in label_counts_tv):
                           print("Warning: Stratification for validation split might fail (classes < 2 samples). Attempting anyway.")
                      stratify_col_train_val = train_val_df['label_str']

                 try:
                    train_df, val_df = train_test_split(
                        train_val_df,
                        test_size=relative_val_size,
                        random_state=config.SEED,
                        stratify=stratify_col_train_val
                    )
                 except ValueError as e:
                    print(f"Warning: Stratified split for validation set failed ({e}). Performing non-stratified split.")
                    train_df, val_df = train_test_split(
                        train_val_df, test_size=relative_val_size, random_state=config.SEED, stratify=None)
            else:
                print("Calculated relative validation size is 0. Validation set will be empty.")
                train_df = train_val_df # All remaining is train
        elif len(train_val_df) < 2:
             print("Not enough data remaining after test split to create validation split. Validation set will be empty.")
             train_df = train_val_df
        else:
            print("Validation size is 0 or cannot split further. Validation set will be empty.")
            train_df = train_val_df # All remaining is train
    else:
         print("Validation size is 0. Validation set will be empty.")
         train_df = train_val_df # All remaining is train


    # Remove temporary stratification column if it exists
    if 'label_str' in train_df.columns: train_df = train_df.drop(columns=['label_str'])
    if 'label_str' in val_df.columns: val_df = val_df.drop(columns=['label_str'])
    if 'label_str' in test_df.columns: test_df = test_df.drop(columns=['label_str'])


    print(f"\nSplit complete:")
    print(f"  Train set size:      {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")
    print(f"  Test set size:       {len(test_df)}")
    print(f"  (Total rows assigned: {len(train_df) + len(val_df) + len(test_df)} / Original: {len(df)})")


    # --- Saving Logic ---
    try:
        # Save only the 'label' and 'text' columns with standard headers
        cols_to_save = ['label', 'text']
        for pth, dframe in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
            out_dir = os.path.dirname(pth)
            if out_dir: # Ensure output directory exists
                os.makedirs(out_dir, exist_ok=True)

            if dframe is not None and not dframe.empty:
                 # Select standard columns, ensure they exist
                 if all(col in dframe.columns for col in cols_to_save):
                     dframe_to_save = dframe[cols_to_save]
                     dframe_to_save.to_csv(pth, index=False, header=True)
                     print(f"  Saved {os.path.basename(pth)} ({len(dframe_to_save)} rows)")
                 else:
                      print(f"  Warning: Could not save {os.path.basename(pth)}. Missing required columns 'label' or 'text'.")
            else:
                 # Optionally create an empty file with header for consistency?
                 print(f"  Skipping save for empty dataset: {os.path.basename(pth)}")
                 # pd.DataFrame(columns=cols_to_save).to_csv(pth, index=False, header=True) # Uncomment to save empty file

        print("\nData splitting and saving finished.")
    except Exception as e:
        print(f"Error saving split files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Manipulation Utility (Manual Use)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    # --- Common Arguments ---
    common_parser = argparse.ArgumentParser(add_help=False)
    # Input File (optional, try to use config default)
    default_input = config.INPUT_FILE_PATH if hasattr(config, 'INPUT_FILE_PATH') and config.INPUT_FILE_PATH else None
    input_help = "Path to the input data file." + (f" (Default from config: {default_input})" if default_input else " (Default: try 'training.csv' or 'data.csv')")
    common_parser.add_argument("-i", "--input", type=str, default=default_input, help=input_help)
    common_parser.add_argument("--format", type=str, default="csv", choices=["csv", "tsv", "jsonl"], help="Input file format (Default: csv).")
    # Use config defaults for column indices/header, allow override
    common_parser.add_argument("--text_col", type=int, default=config.TEXT_COLUMN_INDEX, help=f"Index of the text column (Default: {config.TEXT_COLUMN_INDEX}).")
    common_parser.add_argument("--label_col", type=int, default=config.LABEL_COLUMN_INDEX, help=f"Index of the label column (Default: {config.LABEL_COLUMN_INDEX}).")
    # HAS_HEADER from config controls default behavior; --no_header overrides it
    header_action = 'store_false' if config.HAS_HEADER else 'store_true'
    header_default = config.HAS_HEADER
    common_parser.add_argument("--header", action=header_action, default=header_default, help=f"Specify if input file has a header row (Default: {config.HAS_HEADER}). Use --no-header to disable if default is True, or --header to enable if default is False.")


    # --- Balance Subcommand ---
    parser_balance = subparsers.add_parser("balance", help="Create a balanced subset of the data.", parents=[common_parser])
    parser_balance.add_argument("-o", "--output", type=str, required=True, help="Path to save the balanced output file (CSV format).")
    parser_balance.add_argument("-n", "--num_samples", type=int, required=True, help="Number of samples per class.")

    # --- Split Subcommand ---
    parser_split = subparsers.add_parser("split", help="Split data into train, validation, and test sets.", parents=[common_parser])
    parser_split.add_argument("--train_out", type=str, required=True, help="Path to save the training set (CSV format).")
    parser_split.add_argument("--val_out", type=str, required=True, help="Path to save the validation set (CSV format).")
    parser_split.add_argument("--test_out", type=str, required=True, help="Path to save the test set (CSV format).")
    parser_split.add_argument("--val_size", type=float, default=0.15, help="Validation set proportion (from original data) (Default: 0.15).")
    parser_split.add_argument("--test_size", type=float, default=0.15, help="Test set proportion (from original data) (Default: 0.15).")
    parser_split.add_argument("--no_stratify", action="store_true", default=False, help="Disable stratification during split (Default: Stratify).")


    args = parser.parse_args()

    # Ensure input file is specified if default couldn't be found
    if not args.input:
         parser.error("Input file path (-i/--input) is required as no default could be determined.")

    # Handle header logic based on action ('store_false' means header=False if flag is present)
    has_header = args.header # This now correctly reflects the presence/absence of the flag relative to the default

    if args.command == "balance":
        print("--- Running Balance Data ---")
        create_balanced_subset(
            input_path=args.input,
            output_path=args.output,
            n_samples_per_class=args.num_samples,
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=has_header, # Use processed value
            file_format=args.format,
            col_names=config.COLUMN_NAMES # Pass default names from config/fallback
        )
    elif args.command == "split":
        print("--- Running Split Data ---")
        split_data(
            input_path=args.input,
            train_path=args.train_out,
            val_path=args.val_out,
            test_path=args.test_out,
            val_size=args.val_size,
            test_size=args.test_size,
            stratify=not args.no_stratify, # Stratify unless --no_stratify is given
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=has_header, # Use processed value
            file_format=args.format,
            col_names=config.COLUMN_NAMES # Pass default names from config/fallback
        )
    else:
        parser.print_help()

```
--- END OF FILE dataman.py ---

--- START OF FILE engine.py ---

```python
# --- engine.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import time # For timing epochs

# Try importing transformer-specific scheduler
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    get_linear_schedule_with_warmup = None
    print("Warning: HuggingFace Transformers library not installed. Linear warmup scheduler will not be available.")


import config # Import configuration

# --- Model Initialization ---

def initialize_model(model_type, n_classes, vocab_size=None):
    """Initializes the model based on the configuration."""
    print(f"\nInitializing model: {model_type} with {n_classes} classes")
    if model_type == 'Transformer':
        from models import TransformerClassifier # Local import
        if not hasattr(config, 'TRANSFORMER_MODEL_NAME'):
             raise ValueError("config.TRANSFORMER_MODEL_NAME must be set for Transformer model type.")
        model = TransformerClassifier(
            model_name=config.TRANSFORMER_MODEL_NAME,
            n_classes=n_classes
            # Dropout is handled within the model using AutoConfig
        )
    elif model_type == 'CNN_RNN_Attention':
        from models import CNN_RNN_Attention # Local import
        if vocab_size is None: raise ValueError("vocab_size required for CNN_RNN_Attention")
        # Ensure necessary configs are present
        for cfg_name in ['EMBEDDING_DIM', 'CNN_OUT_CHANNELS', 'CNN_KERNEL_SIZES', 'RNN_TYPE', 'RNN_HIDDEN_DIM', 'RNN_LAYERS', 'DROPOUT_PROB', 'PAD_IDX']:
            if not hasattr(config, cfg_name): raise ValueError(f"config.{cfg_name} must be set for CNN_RNN_Attention.")
        model = CNN_RNN_Attention(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            rnn_type=config.RNN_TYPE,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            rnn_layers=config.RNN_LAYERS,
            n_class=n_classes,
            dropout_prob=config.DROPOUT_PROB, # Use dedicated dropout config
            pad_idx=config.PAD_IDX
        )
    elif model_type == 'LSTM':
        from models import LSTMNetwork # Local import
        if vocab_size is None: raise ValueError("vocab_size required for LSTM")
         # Ensure necessary configs are present
        for cfg_name in ['EMBEDDING_DIM', 'RNN_HIDDEN_DIM', 'RNN_LAYERS', 'DROPOUT_PROB', 'PAD_IDX']:
             if not hasattr(config, cfg_name): raise ValueError(f"config.{cfg_name} must be set for LSTM.")
        model = LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.RNN_HIDDEN_DIM,
            n_class=n_classes,
            n_layers=config.RNN_LAYERS,
            pad_idx=config.PAD_IDX,
            dropout_prob=config.DROPOUT_PROB # Use dedicated dropout config
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE in config: {model_type}")

    model.to(config.DEVICE)
    print(f"Model '{model_type}' initialized and moved to {config.DEVICE}")
    # Print parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
    except Exception as e:
        print(f"  Could not calculate parameter count: {e}")
    return model

# --- Optimizer and Scheduler ---

def initialize_optimizer_scheduler(model, optimizer_type, scheduler_type, num_train_steps=None):
    """Initializes optimizer and scheduler based on config."""
    print(f"\nInitializing Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    lr = config.LEARNING_RATE
    wd = config.WEIGHT_DECAY

    optimizer = None
    if optimizer_type == 'AdamW':
        # Differentiate parameters for weight decay (common for Transformers, good default)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': wd},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        print(f"  Using AdamW with LR={lr}, Weight Decay={wd} (applied selectively)")
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        print(f"  Using Adam with LR={lr}, Weight Decay={wd}")
    elif optimizer_type == 'SGD':
         # Add momentum if desired for SGD
         momentum = getattr(config, 'MOMENTUM', 0.9) # Default momentum if not in config
         optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, momentum=momentum)
         print(f"  Using SGD with LR={lr}, Weight Decay={wd}, Momentum={momentum}")
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {optimizer_type}")

    scheduler = None
    if scheduler_type == 'linear_warmup':
        if get_linear_schedule_with_warmup is None:
             print("Warning: 'linear_warmup' scheduler selected, but Transformers library not installed. No scheduler used.")
        elif num_train_steps is None:
            raise ValueError("num_train_steps is required for linear_warmup scheduler")
        else:
            num_warmup_steps = int(num_train_steps * config.WARMUP_PROPORTION)
            print(f"  Using Linear Warmup scheduler: Total steps={num_train_steps}, Warmup steps={num_warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps
            )
    elif scheduler_type == 'reduce_on_plateau':
        # Monitors validation loss by default
        # Get patience from config or use a default
        patience = getattr(config, 'SCHEDULER_PATIENCE', 2)
        factor = getattr(config, 'SCHEDULER_FACTOR', 0.1)
        print(f"  Using ReduceLROnPlateau scheduler: Factor={factor}, Patience={patience}, Monitoring 'val_loss'")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    elif scheduler_type is None or scheduler_type.lower() == 'none':
         print("  No learning rate scheduler selected.")
    else:
        print(f"Warning: Scheduler type '{scheduler_type}' requested but not implemented or recognized. No scheduler used.")


    return optimizer, scheduler

# --- Loss Function ---
# Using CrossEntropyLoss, suitable for multi-class classification
criterion = nn.CrossEntropyLoss()
print(f"\nUsing Loss Function: CrossEntropyLoss")

# --- Training Step ---

def train_step(model, data_loader, optimizer, device, scheduler=None, grad_clip_value=None):
    """Performs a single training epoch."""
    model.train() # Set model to training mode
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Training", leave=False, unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad() # Clear gradients from previous batch

        # --- Input Handling based on Model Type ---
        try:
            if config.MODEL_TYPE == 'Transformer':
                # Assumes batch is a dictionary from GenericDataset/DataLoader
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else: # Non-transformer models (LSTM, CNN_RNN)
                # Assumes batch is a tuple (sequences, labels, lengths) from collate_non_transformer
                sequences = batch[0].to(device)
                labels = batch[1].to(device)
                lengths = batch[2] # Lengths stay on CPU for pack_padded_sequence
                # Forward pass - model expects lengths
                outputs = model(text_indices=sequences, sequence_lengths=lengths)

        except Exception as e:
             print(f"\nError during forward pass in training batch {batch_idx}: {e}")
             print(f"Batch keys/type: {type(batch)}")
             if isinstance(batch, dict): print(f"Keys: {batch.keys()}")
             elif isinstance(batch, (list, tuple)): print(f"Length: {len(batch)}")
             # Optionally: print shapes or skip batch
             # continue # Skip this batch if error occurs
             raise # Re-raise the error to stop training

        # --- Loss Calculation & Backpropagation ---
        loss = criterion(outputs, labels)
        loss.backward() # Compute gradients

        # --- Gradient Clipping (Optional) ---
        if grad_clip_value is not None and grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        # --- Optimizer & Scheduler Step ---
        optimizer.step() # Update model weights
        # Step linear warmup scheduler *after* optimizer step
        if scheduler and config.SCHEDULER_TYPE == 'linear_warmup':
            scheduler.step()

        # --- Logging & Progress Bar ---
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}' # Get current LR
        })

    # --- Epoch End ---
    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time
    print(f"  Train Avg. Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
    return avg_loss

# --- Evaluation Step ---

def evaluate_step(model, data_loader, device):
    """Performs evaluation on a dataset (validation or test)."""
    if data_loader is None or len(data_loader) == 0:
        print("  Evaluation skipped: DataLoader is empty or None.")
        # Return default/empty metrics to avoid errors downstream
        return {
            'loss': float('nan'), 'accuracy': 0.0, 'precision_weighted': 0.0,
            'recall_weighted': 0.0, 'f1_weighted': 0.0,
            'predictions': [], 'true_labels': []
        }

    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch")

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch_idx, batch in enumerate(progress_bar):
            # --- Input Handling (same logic as train_step) ---
            try:
                if config.MODEL_TYPE == 'Transformer':
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else: # Non-transformer
                    sequences = batch[0].to(device)
                    labels = batch[1].to(device)
                    lengths = batch[2] # CPU
                    outputs = model(text_indices=sequences, sequence_lengths=lengths)

            except Exception as e:
                 print(f"\nError during forward pass in evaluation batch {batch_idx}: {e}")
                 # Decide how to handle: skip batch or raise error
                 # continue
                 raise

            # --- Loss Calculation ---
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # --- Predictions ---
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # --- Progress Bar ---
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
             })

    # --- Epoch End ---
    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time

    # --- Calculate Metrics ---
    # Ensure labels/preds are numpy arrays
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    # Calculate weighted precision, recall, F1 - use zero_division=0 to handle cases with no preds/labels for a class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average='weighted', zero_division=0
    )

    print(f"  Eval Avg. Loss:  {avg_loss:.4f} | Accuracy: {accuracy:.4f} | F1 (W): {f1:.4f} | Time: {elapsed_time:.2f}s")

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'predictions': all_preds, # Return predictions for detailed analysis (e.g., confusion matrix)
        'true_labels': all_labels # Return true labels
    }
    return metrics


# --- Training Loop ---

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_save_path, metric_for_best=config.METRIC_FOR_BEST_MODEL):
    """The main training loop."""
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_weighted': []}
    # Initialize best metric based on whether higher is better (accuracy, f1) or lower is better (loss)
    best_metric_value = -float('inf') if metric_for_best != 'loss' else float('inf')
    # Determine gradient clipping value from config
    grad_clip_value = getattr(config, 'GRADIENT_CLIP_VALUE', None) # Use None if not defined

    print(f"\n--- Starting Training ---")
    print(f"Model Type: {config.MODEL_TYPE}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Optimizer: {config.OPTIMIZER_TYPE}, Scheduler: {config.SCHEDULER_TYPE}")
    print(f"Monitoring validation '{metric_for_best}' for best model.")
    if grad_clip_value: print(f"Using gradient clipping: {grad_clip_value}")
    print(f"Model checkpoints will be saved to: {model_save_path}")

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # --- Training Phase ---
        train_loss = train_step(model, train_loader, optimizer, device, scheduler, grad_clip_value)
        history['train_loss'].append(train_loss)

        # --- Validation Phase ---
        val_metrics = evaluate_step(model, val_loader, device)

        # Handle case where validation loader was empty
        if val_metrics['loss'] is float('nan'):
             print("  Skipping validation metrics recording and best model check due to empty validation set.")
             continue # Proceed to next epoch

        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_weighted']
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_weighted'].append(val_f1)
        # Note: The print statement for eval metrics is now inside evaluate_step

        # --- Scheduler Step (for ReduceLROnPlateau) ---
        if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler.step(val_loss) # Pass the validation loss

        # --- Check for Best Model ---
        current_metric_value = val_metrics.get(metric_for_best, None)
        if current_metric_value is None:
             print(f"Warning: Metric '{metric_for_best}' not found in validation metrics. Cannot determine best model.")
             continue # Skip saving if metric is missing

        is_better = False
        if metric_for_best == 'loss':
            # Lower loss is better
            is_better = current_metric_value < best_metric_value
        else:
            # Higher accuracy/f1 is better
            is_better = current_metric_value > best_metric_value

        if is_better:
            print(f"  ✨ Validation '{metric_for_best}' improved ({best_metric_value:.4f} --> {current_metric_value:.4f}). Saving model...")
            best_metric_value = current_metric_value
            try:
                 # Ensure the directory exists (config.py should handle this, but double check)
                 os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                 # Save model state dictionary
                 torch.save(model.state_dict(), model_save_path)
                 print(f"     Model saved to {model_save_path}")
            except Exception as e:
                 print(f"     Error saving model: {e}")
                 # Decide whether to continue training or stop if saving fails
        else:
            print(f"  Validation '{metric_for_best}' ({current_metric_value:.4f}) did not improve from best ({best_metric_value:.4f}).")

    # --- Training End ---
    end_training_time = time.time()
    total_training_time = end_training_time - start_training_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Best validation '{metric_for_best}' achieved: {best_metric_value:.4f}")
    print(f"Model artifacts saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")
    return history

# --- Model Loading ---
def load_trained_model(model_path, model_type, n_classes, vocab_size=None):
    """
    Loads a pre-trained model's state dict.

    Args:
        model_path (str): Path to the saved .pt file (state_dict).
        model_type (str): Type of model ('Transformer', 'LSTM', 'CNN_RNN_Attention').
        n_classes (int): Number of output classes the model was trained for.
        vocab_size (int, optional): Vocabulary size, required for non-Transformer models.

    Returns:
        torch.nn.Module: The loaded model, in evaluation mode.

    Raises:
        FileNotFoundError: If the model_path does not exist.
        ValueError: If configuration mismatch (e.g., missing vocab_size).
        Exception: For other PyTorch loading errors.
    """
    print(f"\nAttempting to load model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 1. Initialize a model instance with the same architecture
    # This requires n_classes and potentially vocab_size
    try:
        model = initialize_model(model_type, n_classes, vocab_size)
        # Note: initialize_model already prints details and moves to device
    except ValueError as e:
         print(f"Error initializing model structure before loading weights: {e}")
         raise
    except Exception as e:
         print(f"Unexpected error initializing model structure: {e}")
         raise

    # 2. Load the state dictionary
    try:
        # Load state dict, ensuring it's loaded onto the correct device specified in config
        state_dict = torch.load(model_path, map_location=torch.device(config.DEVICE))
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully onto {config.DEVICE}.")
        model.eval() # Set to evaluation mode
        return model
    except FileNotFoundError:
         # Should be caught earlier, but defensive check
         print(f"Error: Model file disappeared before loading: {model_path}")
         raise
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("This could be due to:")
        print("  - Corrupted model file.")
        print("  - Mismatch between the saved weights and the current model architecture.")
        print("    (Check config.py settings like hidden dimensions, layers, etc., match the trained model).")
        print("  - Issues during file reading.")
        raise # Re-raise the exception after providing context
```
--- END OF FILE engine.py ---

--- START OF FILE main.py ---

```python
# --- main.py ---
import sys
import os
import argparse

# Dynamically add project root to path if needed (if running main.py from within project dir)
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# Import necessary modules after potentially modifying path
try:
    import config # Imports first to set up paths etc.
    import train
except ImportError as e:
     print(f"Error importing core modules: {e}")
     print("Ensure config.py and train.py are in the Python path or the same directory.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Emotion Classification Framework - Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
        )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=['Transformer', 'CNN_RNN_Attention', 'LSTM'],
        default=None,
        help='''Override the MODEL_TYPE set in config.py for this run.
If not provided, the value from config.py will be used.
This determines the model architecture and the artifact directory used.'''
    )
    # Add other potential command-line overrides here if needed
    # e.g., --epochs, --batch_size, --data_dir etc.
    # Example:
    # parser.add_argument('--epochs', type=int, default=None, help='Override config.EPOCHS')
    # parser.add_argument('--train_file', type=str, default=None, help='Override config.TRAIN_FILE_PATH')

    args = parser.parse_args()

    # --- Override Config if Specified ---
    # Store original config value if needed
    original_model_type = config.MODEL_TYPE
    config_overridden = False

    if args.model_type and args.model_type != config.MODEL_TYPE:
        print(f"Overriding config.MODEL_TYPE: '{config.MODEL_TYPE}' -> '{args.model_type}'")
        config.MODEL_TYPE = args.model_type
        config_overridden = True
        # IMPORTANT: Re-evaluate paths in config that depend on MODEL_TYPE
        config.MODEL_TYPE_ARTIFACTS_DIR = os.path.join(config.ARTIFACTS_DIR, config.MODEL_TYPE)
        config.MODEL_SAVE_DIR = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, "model")
        config.BEST_MODEL_PATH = os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_FILENAME)
        config.VOCAB_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.VOCAB_FILENAME)
        config.TRAINING_PLOTS_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.TRAINING_PLOTS_FILENAME)
        config.TEST_REPORT_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.TEST_REPORT_FILENAME)
        config.CONFUSION_MATRIX_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.CONFUSION_MATRIX_FILENAME)
        config.RUN_CONFIG_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.RUN_CONFIG_FILENAME)
        # Re-create directories for the potentially new model type
        os.makedirs(config.MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        print(f"Artifact directory updated to: {config.MODEL_TYPE_ARTIFACTS_DIR}")
        # Adjust other MODEL_TYPE dependent configs if necessary (e.g., BATCH_SIZE, LR)
        # This part requires careful handling - you might want to define defaults per model type
        # or require users to set all relevant configs if overriding MODEL_TYPE.
        # For now, we rely on the user ensuring config.py has sensible defaults for the chosen type
        # or using more CLI args to override them.
        print("Note: Other config settings (like BATCH_SIZE, LR, etc.) were NOT automatically adjusted based on the overridden model type. Ensure they are appropriate.")


    # Add similar blocks here if overriding other configs like epochs, paths, etc.
    # if args.epochs:
    #    print(f"Overriding config.EPOCHS: {config.EPOCHS} -> {args.epochs}")
    #    config.EPOCHS = args.epochs
    #    config_overridden = True


    print("========================================")
    print("=== Emotion Classification Framework ===")
    print("========================================")
    print(f"Starting training run for MODEL_TYPE: {config.MODEL_TYPE}")
    print(f"Artifacts will be saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")


    # Ensure artifact directories exist (config.py already does this, but check again)
    try:
        os.makedirs(config.MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        # Ensure base artifacts dir exists (for global label map)
        if os.path.dirname(config.LABEL_MAP_PATH):
             os.makedirs(os.path.dirname(config.LABEL_MAP_PATH), exist_ok=True)
    except OSError as e:
        print(f"Error creating artifact directories: {e}")
        sys.exit(1)


    # --- Save Configuration ---
    # Save the potentially overridden configuration *before* starting training
    try:
        # Use the correct save path based on the (potentially overridden) MODEL_TYPE
        config.save_run_config(filepath=config.RUN_CONFIG_PATH)
        print(f"Current run configuration saved to {config.RUN_CONFIG_PATH}")
    except Exception as e:
        # Non-fatal warning if config save fails
        print(f"Warning: Failed to save run configuration. {e}")


    # --- Execute Training ---
    try:
        # run_training_pipeline now uses the current state of the config module
        train.run_training_pipeline()
        print("\n--- Training Pipeline Completed Successfully ---")
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n--- File Not Found Error ---")
        print(f"Error: {e}")
        print("Please check the file paths specified in config.py (e.g., TRAIN_FILE_PATH).")
        sys.exit(1)
    except ImportError as e:
        print(f"\n--- Import Error ---")
        print(f"Error: {e}")
        print("Please ensure all required libraries are installed (e.g., transformers, torch, spacy, nltk, sklearn, pandas).")
        print("Check specific messages above for missing libraries like 'transformers' or 'spacy'.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- An Unhandled Error Occurred During Training ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("----------------------------------------------------")
        # Optionally print traceback for debugging
        import traceback
        traceback.print_exc()
        print("----------------------------------------------------")
        print("Training failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
```
--- END OF FILE main.py ---

--- START OF FILE models.py ---

```python
# --- models.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys # For error messages

# Try importing transformer components, raise clear error if missing
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    # Set flags to None, allowing other models to work if transformers not installed
    AutoModel = None
    AutoConfig = None
    # Print a warning but don't exit immediately, only raise error if TransformerClassifier is used
    print("Warning: HuggingFace Transformers library not installed or import failed.")
    print("         Transformer model type ('TransformerClassifier') will not be available.")

import config # Import configuration

# --- Attention Mechanism (for CNN_RNN_Attention) ---
class Attention(nn.Module):
    """ Simple Bahdanau-style attention mechanism. """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_dim = hidden_dim
        # Linear layer for query (e.g., final RNN hidden state - though not used here)
        # Linear layer for keys (RNN outputs)
        # Input is (batch, seq_len, hidden_dim * 2) because RNN is bidirectional
        self.W_k = nn.Linear(hidden_dim * 2, self.attention_dim, bias=False)
        # Score vector
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, rnn_outputs, sequence_lengths=None):
        """
        Calculates attention weights and context vector.

        Args:
            rnn_outputs (torch.Tensor): Outputs from RNN (batch_size, seq_len, hidden_dim * 2).
            sequence_lengths (torch.Tensor, optional): Original lengths of sequences (batch_size).

        Returns:
            tuple: (context_vector, attention_weights)
                   context_vector shape: (batch_size, hidden_dim * 2)
                   attention_weights shape: (batch_size, seq_len)
        """
        # Project RNN outputs into attention space
        # energy shape: (batch_size, seq_len, attention_dim)
        energy = torch.tanh(self.W_k(rnn_outputs))

        # Calculate attention scores
        # attention_scores shape: (batch_size, seq_len)
        attention_scores = self.v(energy).squeeze(2)

        # Apply mask based on sequence lengths *before* softmax
        if sequence_lengths is not None:
            max_len = rnn_outputs.size(1)
            # Create mask: True for padding positions, False otherwise
            # Ensure sequence_lengths is on the same device as attention_scores
            mask = torch.arange(max_len, device=attention_scores.device)[None, :] >= sequence_lengths.to(attention_scores.device)[:, None]
            # Fill masked positions with negative infinity so they get zero probability after softmax
            attention_scores = attention_scores.masked_fill(mask, -1e9) # Use large negative number

        # Compute attention weights (probabilities)
        # attention_weights shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate context vector (weighted sum of RNN outputs)
        # Unsqueeze attention_weights for batch matrix multiplication: (batch_size, 1, seq_len)
        # Context vector calculation: (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_dim * 2)
        # Result shape: (batch_size, 1, hidden_dim * 2) -> squeeze -> (batch_size, hidden_dim * 2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)

        return context_vector, attention_weights

# --- Transformer Model ---
class TransformerClassifier(nn.Module):
    """
    Generic Transformer-based classifier using HuggingFace's AutoModel.
    Loads a pre-trained transformer model and adds a classification head.
    """
    def __init__(self, model_name, n_classes):
        super().__init__()
        # Check if transformers library was imported successfully
        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library is required to use TransformerClassifier. Please install it (`pip install transformers`).")

        try:
            self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        except OSError as e:
             print(f"\nError loading transformer model '{model_name}'.")
             print(f"Ensure the model name is correct and you have an internet connection if it needs downloading.")
             print(f"Or, if it's a local path, ensure the path is correct.")
             print(f"Original error: {e}")
             sys.exit(1) # Exit if model loading fails critically
        except Exception as e:
             print(f"An unexpected error occurred while loading the transformer model '{model_name}': {e}")
             sys.exit(1)

        # Use dropout probability defined in the loaded transformer's config, or default if not present
        dropout_prob = getattr(self.config, 'classifier_dropout', # Try classifier specific dropout
                               getattr(self.config, 'hidden_dropout_prob', 0.1)) # Fallback to hidden dropout or 0.1
        self.dropout = nn.Dropout(dropout_prob)

        # Classification layer
        self.classifier = nn.Linear(self.config.hidden_size, n_classes)

        print(f"  TransformerClassifier using '{model_name}' initialized.")
        print(f"  Dropout probability: {dropout_prob:.2f}")


    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the transformer and classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each class (batch_size, n_classes).
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract the representation for classification.
        # Common strategies:
        # 1. Use the pooler output if available (often trained for classification)
        # 2. Use the hidden state of the [CLS] token (first token) from the last layer
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Use the last hidden state of the first token ([CLS])
            pooled_output = outputs.last_hidden_state[:, 0]

        # Apply dropout and classify
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits

# --- CNN + RNN + Attention Model ---
class CNN_RNN_Attention(nn.Module):
    """
    A model combining CNNs for local feature extraction, an RNN (LSTM/GRU)
    for sequential context, and an Attention mechanism for focusing on relevant parts.
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 cnn_out_channels,
                 cnn_kernel_sizes, # Expect list/tuple e.g., [3, 4, 5]
                 rnn_type, # 'lstm' or 'gru'
                 rnn_hidden_dim,
                 rnn_layers,
                 n_class,
                 dropout_prob,
                 pad_idx):
        super().__init__()

        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if not isinstance(cnn_kernel_sizes, (list, tuple)):
             # If a single int is passed, wrap it in a list
             cnn_kernel_sizes = [cnn_kernel_sizes]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # CNN layers with different kernel sizes applied in parallel
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=cnn_out_channels,
                      kernel_size=k,
                      padding='same') # 'same' padding ensures output length matches input length
            for k in cnn_kernel_sizes
        ])

        # Calculate total output channels from all parallel CNNs
        cnn_total_out_channels = cnn_out_channels * len(cnn_kernel_sizes)

        # RNN layer (LSTM or GRU)
        self.rnn_type = rnn_type.lower()
        rnn_input_dim = cnn_total_out_channels # Output of CNNs feeds into RNN
        # Apply dropout between RNN layers only if n_layers > 1
        rnn_dropout = dropout_prob if rnn_layers > 1 else 0.0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=True)
        else: # gru
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=True)

        # Attention layer - input dimension matches the bidirectional RNN output dimension
        self.attention = Attention(rnn_hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected output layer
        # Input dimension matches the attention context vector dimension (bidirectional RNN)
        self.fc = nn.Linear(rnn_hidden_dim * 2, n_class)

        self.pad_idx = pad_idx
        print(f"  CNN_RNN_Attention ({rnn_type.upper()}) initialized:")
        print(f"    Embedding Dim: {embedding_dim}, CNN Channels: {cnn_out_channels} (Kernels: {cnn_kernel_sizes})")
        print(f"    RNN Hidden Dim: {rnn_hidden_dim}, RNN Layers: {rnn_layers}, Bidirectional: True")
        print(f"    Dropout: {dropout_prob:.2f}")


    def forward(self, text_indices, sequence_lengths=None):
        """
        Forward pass for the CNN-RNN-Attention model.

        Args:
            text_indices (torch.Tensor): Input tensor of token indices (batch_size, seq_len).
            sequence_lengths (torch.Tensor, optional): Original lengths of sequences in the batch (batch_size).
                                                      Required for correct masking in attention and potentially RNN packing.

        Returns:
            torch.Tensor: Logits for each class (batch_size, n_class).
        """
        # Ensure input is LongTensor
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        # 1. Embedding Layer + Dropout
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text_indices))

        # 2. CNN Layers
        # Conv1d expects input shape: (batch_size, channels, seq_len)
        # Permute embedded tensor: (batch_size, embedding_dim, seq_len)
        embedded_permuted = embedded.permute(0, 2, 1)

        # Apply each convolution layer and ReLU activation
        # Each cnn_output shape: (batch_size, cnn_out_channels, seq_len)
        cnn_outputs = [F.relu(conv(embedded_permuted)) for conv in self.conv_layers]

        # Concatenate the outputs of the parallel CNNs along the channel dimension
        # cnn_cat shape: (batch_size, cnn_total_out_channels, seq_len)
        cnn_cat = torch.cat(cnn_outputs, dim=1)

        # Prepare input for RNN: (batch_size, seq_len, features)
        # Permute cnn_cat: (batch_size, seq_len, cnn_total_out_channels)
        rnn_input = cnn_cat.permute(0, 2, 1)

        # 3. RNN Layer
        # Use packing/padding for efficiency if sequence lengths are provided
        if sequence_lengths is not None:
             # Ensure lengths are on CPU for pack_padded_sequence
             # Sort sequences by length (required by pack_padded_sequence before PyTorch 1.7)
             # Modern PyTorch allows enforce_sorted=False, but sorting is often good practice.
             # For simplicity here, we use enforce_sorted=False if available, assuming lengths are correct.
             packed_input = nn.utils.rnn.pack_padded_sequence(rnn_input, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             packed_outputs, _ = self.rnn(packed_input)
             # Unpack the sequence
             rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
             # rnn_outputs shape: (batch_size, seq_len, rnn_hidden_dim * 2)
        else:
             # Warning: Processing without lengths means RNN processes padding tokens, which is inefficient and might hurt performance.
             print("Warning: Running RNN without sequence lengths. Padding tokens will be processed.")
             rnn_outputs, _ = self.rnn(rnn_input) # Shape: (batch_size, seq_len, rnn_hidden_dim * 2)


        # 4. Attention Layer
        # Pass sequence_lengths to attention for proper masking of padding tokens
        # context_vector shape: (batch_size, rnn_hidden_dim * 2)
        context_vector, attention_weights = self.attention(rnn_outputs, sequence_lengths)
        # attention_weights can be optionally returned or used for visualization

        # 5. Final Classification Layer
        dropped_context = self.dropout(context_vector)
        out = self.fc(dropped_context) # Shape: (batch_size, n_class)
        return out

# --- Simple LSTM Model ---
class LSTMNetwork(nn.Module):
    """
    A simpler bidirectional LSTM model for text classification.
    Uses the final hidden states for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Apply dropout between LSTM layers only if n_layers > 1
        rnn_dropout = dropout_prob if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=True) # Use bidirectional LSTM

        # Dropout layer applied to the concatenated final hidden states
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer
        # Input dimension is hidden_dim * 2 because LSTM is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, n_class)

        self.pad_idx = pad_idx
        print(f"  LSTMNetwork initialized:")
        print(f"    Embedding Dim: {embedding_dim}, Hidden Dim: {hidden_dim}")
        print(f"    Layers: {n_layers}, Bidirectional: True")
        print(f"    Dropout: {dropout_prob:.2f}")

    def forward(self, text_indices, sequence_lengths=None):
        """
        Forward pass for the LSTM model.

        Args:
            text_indices (torch.Tensor): Input tensor of token indices (batch_size, seq_len).
            sequence_lengths (torch.Tensor, optional): Original lengths of sequences (batch_size).

        Returns:
            torch.Tensor: Logits for each class (batch_size, n_class).
        """
        # Ensure input is LongTensor
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        # 1. Embedding Layer + Dropout
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text_indices))

        # 2. LSTM Layer
        # Pack sequence for efficiency if lengths are provided
        if sequence_lengths is not None:
             packed_input = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             # We only need the final hidden state, not the outputs per time step
             _, (hidden, cell) = self.lstm(packed_input)
             # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
             # cell shape:   (num_layers * num_directions, batch_size, hidden_dim)
        else:
             # Process without packing (less efficient)
             print("Warning: Running LSTM without sequence lengths. Padding tokens will be processed.")
             _, (hidden, cell) = self.lstm(embedded)


        # 3. Concatenate Final Hidden States
        # We need the hidden state from the last layer, for both forward and backward directions.
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # Forward final hidden state: hidden[-2, :, :]
        # Backward final hidden state: hidden[-1, :, :]
        # Concatenate along the feature dimension (dim=1)
        # hidden_concat shape: (batch_size, hidden_dim * 2)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # 4. Final Classification Layer
        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped) # Shape: (batch_size, n_class)
        return out
```
--- END OF FILE models.py ---

--- START OF FILE plotter.py ---

```python
# --- plotter.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings

# Try importing sklearn, provide warning if unavailable
try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    classification_report = None
    confusion_matrix = None
    print("Warning: scikit-learn not installed. Classification report and confusion matrix generation will be unavailable.")
    print("         Install it using: pip install scikit-learn")

import config # For default save paths

# Apply a default style
sns.set_theme(style="whitegrid")

def plot_training_history(history, save_path=None):
    """
    Plots training and validation loss, accuracy, and F1 score over epochs.

    Args:
        history (dict): Dictionary containing lists of metrics per epoch
                        (e.g., 'train_loss', 'val_loss', 'val_accuracy', 'val_f1_weighted').
                        Keys must match the expected metric names.
        save_path (str, optional): Path to save the plot image. If None (default), uses
                                   config.TRAINING_PLOTS_PATH.
    """
    if not isinstance(history, dict) or not history:
        print("Plotter Warning: History dictionary is empty or invalid. Cannot plot training history.")
        return

    # Use default save path from config if none provided
    if save_path is None:
        save_path = getattr(config, 'TRAINING_PLOTS_PATH', None)
        if save_path is None:
             print("Plotter Error: Default save path (config.TRAINING_PLOTS_PATH) not found and no save_path provided.")
             return # Cannot save

    # Check for essential keys
    if 'train_loss' not in history or not history['train_loss']:
         print("Plotter Warning: 'train_loss' not found or empty in history. Cannot plot.")
         return
    if 'val_loss' not in history or not history['val_loss']:
         print("Plotter Warning: 'val_loss' not found or empty in history. Loss plot will only show training loss.")
         # Continue, but plot will be incomplete

    epochs = range(1, len(history['train_loss']) + 1)
    df = pd.DataFrame(history)
    df['epoch'] = epochs

    # Determine number of subplots needed
    num_plots = 0
    plot_config = {}
    if 'train_loss' in df and 'val_loss' in df:
        num_plots += 1
        plot_config['loss'] = {'train': 'train_loss', 'val': 'val_loss', 'title': 'Loss'}
    if 'val_accuracy' in df:
        num_plots += 1
        plot_config['accuracy'] = {'train': 'train_accuracy', 'val': 'val_accuracy', 'title': 'Accuracy'}
    if 'val_f1_weighted' in df:
         num_plots += 1
         plot_config['f1'] = {'train': 'train_f1_weighted', 'val': 'val_f1_weighted', 'title': 'Weighted F1 Score'}

    if num_plots == 0:
        print("Plotter Warning: No plottable validation metrics (accuracy, f1_weighted) found in history dict, besides loss.")
        # Optionally plot just loss if val_loss was missing earlier?
        if 'train_loss' in df:
            num_plots = 1
            plot_config = {'loss': {'train': 'train_loss', 'val': None, 'title': 'Training Loss'}}
        else:
             return # Nothing to plot


    plt.figure(figsize=(6 * num_plots, 5)) # Adjust figure size

    plot_idx = 1
    for metric_key, cfg in plot_config.items():
        plt.subplot(1, num_plots, plot_idx)
        has_train = cfg.get('train') and cfg['train'] in df
        has_val = cfg.get('val') and cfg['val'] in df

        if has_train:
            plt.plot(df['epoch'], df[cfg['train']], label=f"Train {cfg['title']}", marker='o', linestyle='-')
        if has_val:
            plt.plot(df['epoch'], df[cfg['val']], label=f"Validation {cfg['title']}", marker='x', linestyle='--')

        if not has_train and not has_val:
            print(f"Plotter Warning: No data found for {cfg['title']}. Skipping plot.")
            continue # Skip if somehow no data exists for this subplot

        plt.title(f"{cfg['title']} vs. Epoch")
        plt.xlabel('Epoch')
        plt.ylabel(cfg['title'])

        # Adjust y-limits for accuracy and F1 for better visualization
        if metric_key in ['accuracy', 'f1']:
            min_val = 0
            max_val = 1
            if has_val and not df[cfg['val']].empty:
                min_val = max(0, df[cfg['val']].min() - 0.05)
            if has_val and not df[cfg['val']].empty:
                max_val = min(1.05, df[cfg['val']].max() + 0.05)
            elif has_train and not df[cfg['train']].empty: # Fallback to train if no val
                 min_val = max(0, df[cfg['train']].min() - 0.05)
                 max_val = min(1.05, df[cfg['train']].max() + 0.05)
            plt.ylim(bottom=min_val, top=max_val)

        plt.legend()
        plt.grid(True)
        plot_idx += 1


    plt.tight_layout(pad=2.0) # Add padding between subplots

    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches='tight'
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Plotter Error: Could not save training plot to {save_path}. Error: {e}")
    else:
        # If no save path was determined, optionally show the plot
        # plt.show()
        pass

    plt.close() # Close the figure to free memory


def generate_classification_analysis(true_labels, predictions, int_to_label, report_path=None, cm_path=None, prefix=""):
    """
    Generates and saves a classification report and confusion matrix.

    Args:
        true_labels (list or np.array): Ground truth integer labels.
        predictions (list or np.array): Predicted integer labels.
        int_to_label (dict): Mapping from integer labels to string names.
        report_path (str, optional): Path to save the text classification report. If None,
                                     uses config.TEST_REPORT_PATH.
        cm_path (str, optional): Path to save the confusion matrix plot. If None,
                                 uses config.CONFUSION_MATRIX_PATH.
        prefix (str, optional): Prefix for report/plot titles (e.g., "Test Set").
    """
    # Check if sklearn is available
    if classification_report is None or confusion_matrix is None:
        print("Plotter Info: Skipping classification analysis because scikit-learn is not installed.")
        return

    # Use default save paths from config if none provided
    if report_path is None:
        report_path = getattr(config, 'TEST_REPORT_PATH', None)
    if cm_path is None:
        cm_path = getattr(config, 'CONFUSION_MATRIX_PATH', None)

    if not report_path and not cm_path:
        print("Plotter Info: No paths provided or found in config for classification report or confusion matrix. Analysis will only be printed.")


    if not isinstance(true_labels, (list, np.ndarray)) or not isinstance(predictions, (list, np.ndarray)):
        print("Plotter Error: true_labels and predictions must be lists or numpy arrays.")
        return
    if len(true_labels) != len(predictions):
        print(f"Plotter Error: Length mismatch between true_labels ({len(true_labels)}) and predictions ({len(predictions)}).")
        return
    if len(true_labels) == 0:
        print("Plotter Info: true_labels and predictions are empty. Skipping classification analysis.")
        return


    # Determine unique labels present in the data and map them to names
    unique_labels_present = sorted(list(set(true_labels) | set(predictions)))

    if not int_to_label:
        print("Plotter Warning: int_to_label mapping not provided or empty. Using integer labels as names.")
        # Use unique sorted integer labels present in the data
        label_names = [str(i) for i in unique_labels_present]
        target_labels_for_report = unique_labels_present # Use integers for report's labels arg
    else:
        # Ensure keys are integers and values are strings for safety
        try:
            int_to_label_clean = {int(k): str(v) for k, v in int_to_label.items()}
        except (ValueError, TypeError):
            print("Plotter Warning: Could not properly convert int_to_label keys/values. Using integer labels.")
            int_to_label_clean = {} # Force fallback

        if not int_to_label_clean: # If conversion failed or original was empty
             label_names = [str(i) for i in unique_labels_present]
             target_labels_for_report = unique_labels_present
        else:
             # Map labels present in data to names, handle missing mappings
             label_names = [int_to_label_clean.get(i, f"Unknown({i})") for i in unique_labels_present]
             target_labels_for_report = unique_labels_present # Use integers for report's labels arg


    # --- Classification Report ---
    try:
        # Use target_labels_for_report to ensure order and inclusion matches label_names
        report_str = classification_report(
            true_labels,
            predictions,
            labels=target_labels_for_report, # Specify labels to include/order
            target_names=label_names,         # Corresponding names
            zero_division=0,                  # Avoid warnings for undefined metrics
            digits=4                          # Number of digits for precision
        )
        title = f"{prefix} Classification Report" if prefix else "Classification Report"
        # Calculate overall accuracy separately for display
        accuracy = np.mean(np.array(true_labels) == np.array(predictions))
        full_report_output = f"\n--- {title} ---\n"
        full_report_output += f"Overall Accuracy: {accuracy:.4f}\n\n"
        full_report_output += report_str
        full_report_output += "\n-----------------------------------\n"

        print(full_report_output) # Print to console

        # Save report to file if path provided
        if report_path:
            try:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    # Write the same content as printed
                    f.write(full_report_output)
                print(f"Classification report saved to {report_path}")
            except Exception as e:
                print(f"Plotter Error: Could not save classification report to {report_path}. Error: {e}")

    except Exception as e:
        print(f"Plotter Error: Could not generate classification report. Error: {e}")
        import traceback
        traceback.print_exc()


    # --- Confusion Matrix ---
    if cm_path:
        try:
            # Generate confusion matrix using the ordered labels
            cm = confusion_matrix(true_labels, predictions, labels=target_labels_for_report)

            # Dynamic figure sizing based on number of labels
            fig_width = max(8, len(label_names) * 0.7)
            fig_height = max(6, len(label_names) * 0.6)
            plt.figure(figsize=(fig_width, fig_height))

            # Use seaborn heatmap for better visualization
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names,
                        annot_kws={"size": 10}) # Adjust font size if needed

            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            cm_title = f"{prefix} Confusion Matrix" if prefix else "Confusion Matrix"
            plt.title(cm_title, fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate x-labels if many classes
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout(pad=1.5) # Adjust layout

            # Save the confusion matrix plot
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            plt.savefig(cm_path, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {cm_path}")
            plt.close() # Close the plot figure to free memory

        except ValueError as e:
             print(f"Plotter Error: Could not generate confusion matrix, possibly due to label mismatch or empty data. Error: {e}")
        except Exception as e:
            print(f"Plotter Error: Could not generate or save confusion matrix plot. Error: {e}")
            import traceback
            traceback.print_exc()
    elif report_path: # If only report_path was defined, mention CM wasn't saved
        print("Plotter Info: Confusion matrix path not specified. Matrix plot not saved.")
```
--- END OF FILE plotter.py ---

--- START OF FILE train.py ---

```python
# --- train.py ---
import torch
import os
import sys
import numpy as np

# Import necessary modules from the project
try:
    import config
    import data_handler
    import engine
    import models # Although models are initialized in engine, importing helps ensure availability
    import plotter
except ImportError as e:
    print(f"Error importing necessary modules in train.py: {e}")
    print("Ensure all required files (config.py, data_handler.py, engine.py, models.py, plotter.py) are accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports in train.py: {e}")
    sys.exit(1)

def run_training_pipeline():
    """
    Executes the full training and evaluation pipeline.
    """
    print(f"\n--- Starting Training Pipeline for {config.MODEL_TYPE} ---")

    # --- 1. Data Loading and Preparation ---
    # This step loads data, handles splitting, processes labels,
    # preprocesses text, builds/loads vocab/tokenizer, and creates DataLoaders.
    try:
        train_loader, val_loader, test_loader, \
        label_to_int, int_to_label, n_classes, \
        vocab_or_tokenizer, vocab_size = data_handler.get_data_pipeline()

        # Basic validation after data loading
        if train_loader is None or len(train_loader) == 0:
            print("Error: Training DataLoader is empty or None. Cannot proceed.")
            sys.exit(1)
        if n_classes <= 1:
             print(f"Error: Only {n_classes} class(es) detected after data processing. Need at least 2 for classification.")
             sys.exit(1)
        if config.MODEL_TYPE != 'Transformer' and vocab_size is None:
             print(f"Error: Vocab size not determined for non-Transformer model ({config.MODEL_TYPE}).")
             sys.exit(1)

        print(f"\nData pipeline finished. Number of classes: {n_classes}")
        if config.MODEL_TYPE != 'Transformer':
             print(f"Vocabulary size: {vocab_size}")
        else:
             print(f"Tokenizer vocab size: {vocab_size}")

    except FileNotFoundError as e:
        print(f"\nData Loading Error: {e}")
        print("Please check the data file paths in config.py.")
        sys.exit(1)
    except ValueError as e:
         print(f"\nData Processing Error: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 2. Model Initialization ---
    try:
        model = engine.initialize_model(
            model_type=config.MODEL_TYPE,
            n_classes=n_classes,
            vocab_size=vocab_size # Pass vocab_size (None for Transformers)
        )
    except ValueError as e:
        print(f"\nModel Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during model initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 3. Optimizer and Scheduler Initialization ---
    try:
        # Calculate total training steps for warmup scheduler if needed
        num_train_steps = None
        if config.SCHEDULER_TYPE == 'linear_warmup':
            num_train_steps = len(train_loader) * config.EPOCHS
            if num_train_steps == 0:
                 print("Warning: Calculated num_train_steps is 0. Linear warmup scheduler may not work as expected.")


        optimizer, scheduler = engine.initialize_optimizer_scheduler(
            model=model,
            optimizer_type=config.OPTIMIZER_TYPE,
            scheduler_type=config.SCHEDULER_TYPE,
            num_train_steps=num_train_steps
        )
    except ValueError as e:
        print(f"\nOptimizer/Scheduler Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during optimizer/scheduler setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 4. Training ---
    # The train_model function handles the epoch loop, train/eval steps,
    # checkpointing the best model, and returns the training history.
    try:
        history = engine.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, # Pass the validation loader
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            epochs=config.EPOCHS,
            model_save_path=config.BEST_MODEL_PATH, # Path to save the best model
            metric_for_best=config.METRIC_FOR_BEST_MODEL
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 5. Plot Training History (Optional) ---
    if config.PLOT_TRAINING_HISTORY and history:
        print("\n--- Plotting Training History ---")
        try:
            plotter.plot_training_history(
                history=history,
                save_path=config.TRAINING_PLOTS_PATH # Use path from config
            )
        except Exception as e:
            print(f"Warning: Failed to plot training history. Error: {e}")
    elif not history:
         print("\nSkipping training history plot: History data not available.")


    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    if test_loader is None or len(test_loader) == 0:
        print("Test DataLoader is empty or None. Skipping final test evaluation.")
    else:
        # Load the *best* saved model for final evaluation
        try:
            print(f"Loading best model from: {config.BEST_MODEL_PATH}")
            # Re-initialize a model instance and load the saved state dict
            best_model = engine.load_trained_model(
                model_path=config.BEST_MODEL_PATH,
                model_type=config.MODEL_TYPE,
                n_classes=n_classes,
                vocab_size=vocab_size
            )
            best_model.to(config.DEVICE) # Ensure model is on the correct device

            # Evaluate the loaded best model on the test set
            test_metrics = engine.evaluate_step(
                model=best_model,
                data_loader=test_loader,
                device=config.DEVICE
            )

            # Print test metrics explicitly
            print("\n--- Test Set Performance (Best Model) ---")
            print(f"  Test Loss:      {test_metrics.get('loss', float('nan')):.4f}")
            print(f"  Test Accuracy:  {test_metrics.get('accuracy', 0.0):.4f}")
            print(f"  Test Precision: {test_metrics.get('precision_weighted', 0.0):.4f} (Weighted)")
            print(f"  Test Recall:    {test_metrics.get('recall_weighted', 0.0):.4f} (Weighted)")
            print(f"  Test F1-Score:  {test_metrics.get('f1_weighted', 0.0):.4f} (Weighted)")
            print("-------------------------------------------")


            # --- 7. Generate Test Report & Confusion Matrix (Optional) ---
            if config.GENERATE_TEST_REPORT or config.GENERATE_CONFUSION_MATRIX:
                if 'predictions' in test_metrics and 'true_labels' in test_metrics:
                    print("\n--- Generating Test Analysis ---")
                    plotter.generate_classification_analysis(
                        true_labels=test_metrics['true_labels'],
                        predictions=test_metrics['predictions'],
                        int_to_label=int_to_label, # Use the mapping from data pipeline
                        report_path=config.TEST_REPORT_PATH if config.GENERATE_TEST_REPORT else None,
                        cm_path=config.CONFUSION_MATRIX_PATH if config.GENERATE_CONFUSION_MATRIX else None,
                        prefix="Test Set"
                    )
                else:
                    print("Warning: Cannot generate test report/confusion matrix. Predictions or true labels missing from test metrics.")

        except FileNotFoundError:
            print(f"Error: Best model file not found at {config.BEST_MODEL_PATH}. Cannot perform final evaluation.")
            print("This might happen if training failed to save a model checkpoint.")
        except Exception as e:
            print(f"\nAn error occurred during final test evaluation or analysis: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n--- Training Pipeline for {config.MODEL_TYPE} Finished ---")

# This allows the script to be run directly
# if __name__ == "__main__":
#     run_training_pipeline()
# Note: Typically, main.py is the entry point that calls this function.
```
--- END OF FILE train.py ---
