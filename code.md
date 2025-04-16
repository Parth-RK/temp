Okay, this is an excellent goal! We'll synthesize the best parts of both approaches and structure them according to your specifications to create a flexible, robust, and potentially SOTA emotion classification framework.

Here are the Python files based on your requirements:

**1. `config.py`**

```python
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
MODEL_TYPE = 'Transformer'
# MODEL_TYPE = 'CNN_RNN_Attention'
# MODEL_TYPE = 'LSTM'

# --- Preprocessing ---
# Options: 'basic', 'spacy' (requires spaCy and model like 'en_core_web_sm')
PREPROCESSOR_TYPE = 'basic' if MODEL_TYPE == 'Transformer' else 'spacy'
# PREPROCESSOR_TYPE = 'spacy' # Can force spaCy for transformers if desired
SPACY_MODEL_NAME = "en_core_web_sm" # spaCy model for 'spacy' preprocessor
REMOVE_STOPWORDS = False # Generally False for Transformers, True for others optional

# --- Transformer Model Specific ---
# Ignored if MODEL_TYPE is not 'Transformer'
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased" # e.g., "bert-base-uncased", "roberta-base"

# --- RNN/CNN Model Specific ---
# Ignored if MODEL_TYPE is 'Transformer'
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
METRIC_FOR_BEST_MODEL = 'accuracy' # 'accuracy', 'f1_weighted', 'loss'

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
```

**2. `dataman.py`**

```python
# --- dataman.py ---
import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split
import sys

# Dynamically add project root to path if needed (adjust relative path)
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

try:
    import config # Use config for default paths/columns if available
except ImportError:
    print("Warning: config.py not found. Using hardcoded defaults in dataman.")
    # Define minimal defaults if config cannot be imported
    class ConfigFallback:
        DATA_DIR = "data"
        TEXT_COLUMN_INDEX = 1
        LABEL_COLUMN_INDEX = 0
        COLUMN_NAMES = ['label', 'text']
        HAS_HEADER = True
        SEED = 42
    config = ConfigFallback()

def _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format):
    """Helper to load data based on format."""
    print(f"Loading data from: {input_path}")
    try:
        if file_format == "csv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(input_path, header=header, names=names)
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(input_path, sep='\t', header=header, names=names)
        elif file_format == "jsonl":
            df = pd.read_json(input_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Select and rename columns based on indices provided
        text_col_name = df.columns[text_col_idx]
        label_col_name = df.columns[label_col_idx]
        df = df[[label_col_name, text_col_name]]
        df.columns = ['label', 'text'] # Standardize column names

        print(f"Loaded {len(df)} rows.")
        print(f"Using columns: label='{label_col_name}' (idx {label_col_idx}), text='{text_col_name}' (idx {text_col_idx})")
        return df

    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        return None
    except IndexError:
         print(f"Error: Column index out of bounds. Check TEXT_COLUMN_INDEX ({text_col_idx}) and LABEL_COLUMN_INDEX ({label_col_idx}) for file {input_path}")
         return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_balanced_subset(input_path, output_path, n_samples_per_class,
                           text_col_idx=config.TEXT_COLUMN_INDEX,
                           label_col_idx=config.LABEL_COLUMN_INDEX,
                           col_names=config.COLUMN_NAMES,
                           has_header=config.HAS_HEADER,
                           file_format="csv"):
    """
    Creates a balanced dataset subset with n samples from each label category.

    Args:
        input_path (str): Path to the input data file.
        output_path (str): Path to save the balanced dataset.
        n_samples_per_class (int): Number of samples per label.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header.
        has_header (bool): If the file has a header.
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    df = _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format)
    if df is None:
        return

    print(f"\nOriginal dataset shape: {df.shape}")
    df = df.dropna(subset=['label', 'text']).reset_index(drop=True)
    print(f"Shape after dropping NaNs: {df.shape}")

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
            print(f"  Warning: Label '{label}' has only {sample_size} samples (requested {n_samples_per_class}). Taking all available.")

        sampled_df = label_df.sample(n=sample_size, random_state=config.SEED, replace=False)
        balanced_dfs.append(sampled_df)

    if not balanced_dfs:
        print("Error: No data collected for balancing. Check input data and parameters.")
        return

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=config.SEED).reset_index(drop=True) # Shuffle

    print(f"\nBalanced subset shape: {balanced_df.shape}")
    print("Balanced Subset Label Distribution:")
    print(balanced_df['label'].value_counts())

    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        # Save with header using standard column names
        balanced_df.to_csv(output_path, index=False)
        print(f"\nBalanced subset saved to {output_path}")
    except Exception as e:
        print(f"Error saving balanced subset: {e}")

def split_data(input_path, train_path, val_path, test_path,
               val_size=0.15, test_size=0.15, stratify=True,
               text_col_idx=config.TEXT_COLUMN_INDEX,
               label_col_idx=config.LABEL_COLUMN_INDEX,
               col_names=config.COLUMN_NAMES,
               has_header=config.HAS_HEADER,
               file_format="csv"):
    """
    Splits the data into train, validation, and test sets.

    Args:
        input_path (str): Path to the input data file.
        train_path (str): Path to save the training set.
        val_path (str): Path to save the validation set.
        test_path (str): Path to save the test set.
        val_size (float): Proportion for validation set.
        test_size (float): Proportion for test set (from the original data).
        stratify (bool): Whether to stratify based on labels.
        text_col_idx (int): Index of the text column.
        label_col_idx (int): Index of the label column.
        col_names (list): Column names if no header.
        has_header (bool): If the file has a header.
        file_format (str): 'csv', 'tsv', or 'jsonl'.
    """
    df = _load_data(input_path, text_col_idx, label_col_idx, col_names, has_header, file_format)
    if df is None:
        return

    df = df.dropna(subset=['label', 'text']).reset_index(drop=True)
    print(f"Total data for splitting: {len(df)} rows")

    if len(df) < 3:
        print("Error: Not enough data to perform train/val/test split.")
        return

    stratify_col = df['label'] if stratify else None

    # Calculate test size relative to original, val size relative to remaining
    if (val_size + test_size) >= 1.0:
        print("Error: Sum of validation and test sizes must be less than 1.0")
        return

    # Split off test set first
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=config.SEED,
        stratify=stratify_col
    )

    # Split remaining into train and validation
    # Adjust val_size relative to the remaining data after test split
    relative_val_size = val_size / (1.0 - test_size)
    stratify_col_train_val = train_val_df['label'] if stratify else None

    if len(train_val_df) < 2:
         print("Warning: Very few samples remaining after test split. Validation split might be empty.")
         train_df = train_val_df
         val_df = pd.DataFrame(columns=df.columns) # Empty df
    else:
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=config.SEED,
            stratify=stratify_col_train_val
        )

    print(f"\nSplit complete:")
    print(f"  Train set size: {len(train_df)}")
    print(f"  Validation set size: {len(val_df)}")
    print(f"  Test set size: {len(test_df)}")

    try:
        for pth, dframe in [(train_path, train_df), (val_path, val_df), (test_path, test_df)]:
            out_dir = os.path.dirname(pth)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            # Save with header using standard column names
            dframe.to_csv(pth, index=False)
            print(f"  Saved {os.path.basename(pth)} ({len(dframe)} rows)")
        print("\nData splitting and saving finished.")
    except Exception as e:
        print(f"Error saving split files: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Manipulation Utility (Manual Use)")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Balance Subcommand ---
    parser_balance = subparsers.add_parser("balance", help="Create a balanced subset of the data.")
    parser_balance.add_argument("-i", "--input", type=str, default=config.INPUT_FILE_PATH, help="Path to the input data file.")
    parser_balance.add_argument("-o", "--output", type=str, required=True, help="Path to save the balanced output file.")
    parser_balance.add_argument("-n", "--num_samples", type=int, required=True, help="Number of samples per class.")
    parser_balance.add_argument("--format", type=str, default="csv", choices=["csv", "tsv", "jsonl"], help="Input file format.")
    parser_balance.add_argument("--text_col", type=int, default=config.TEXT_COLUMN_INDEX, help="Index of the text column.")
    parser_balance.add_argument("--label_col", type=int, default=config.LABEL_COLUMN_INDEX, help="Index of the label column.")
    parser_balance.add_argument("--no_header", action="store_true", help="Specify if input file has no header.")

    # --- Split Subcommand ---
    parser_split = subparsers.add_parser("split", help="Split data into train, validation, and test sets.")
    parser_split.add_argument("-i", "--input", type=str, default=config.INPUT_FILE_PATH, help="Path to the input data file.")
    parser_split.add_argument("--train_out", type=str, required=True, help="Path to save the training set.")
    parser_split.add_argument("--val_out", type=str, required=True, help="Path to save the validation set.")
    parser_split.add_argument("--test_out", type=str, required=True, help="Path to save the test set.")
    parser_split.add_argument("--val_size", type=float, default=0.15, help="Validation set proportion.")
    parser_split.add_argument("--test_size", type=float, default=0.15, help="Test set proportion.")
    parser_split.add_argument("--no_stratify", action="store_true", help="Disable stratification during split.")
    parser_split.add_argument("--format", type=str, default="csv", choices=["csv", "tsv", "jsonl"], help="Input file format.")
    parser_split.add_argument("--text_col", type=int, default=config.TEXT_COLUMN_INDEX, help="Index of the text column.")
    parser_split.add_argument("--label_col", type=int, default=config.LABEL_COLUMN_INDEX, help="Index of the label column.")
    parser_split.add_argument("--no_header", action="store_true", help="Specify if input file has no header.")


    args = parser.parse_args()

    if args.command == "balance":
        print("--- Running Balance Data ---")
        create_balanced_subset(
            input_path=args.input,
            output_path=args.output,
            n_samples_per_class=args.num_samples,
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=not args.no_header,
            file_format=args.format
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
            stratify=not args.no_stratify,
            text_col_idx=args.text_col,
            label_col_idx=args.label_col,
            has_header=not args.no_header,
            file_format=args.format
        )
    else:
        parser.print_help()
```

**3. `data_handler.py`**

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
        return [self.clean(text) for text in texts]

    def tokenize(self, text): # Basic split for consistency if needed elsewhere
        return self.clean(text).split()

class SpacyTextPreprocessor:
    """Advanced cleaning using spaCy: lemmatization, optional stopword removal."""
    def __init__(self, spacy_model_name=config.SPACY_MODEL_NAME, remove_stopwords=config.REMOVE_STOPWORDS):
        if spacy is None or nltk_stopwords is None:
            raise ImportError("SpacyTextPreprocessor requires 'spacy' and 'nltk' to be installed. Run 'pip install spacy nltk' and download resources.")
        self.nlp = self._load_spacy_model(spacy_model_name)
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            # Ensure stopwords are downloaded
            try:
                self.stopwords = set(nltk_stopwords.words('english'))
                print("NLTK stopwords loaded.")
            except LookupError:
                print("NLTK 'stopwords' resource not found. Downloading...")
                import nltk
                try:
                    nltk.download('stopwords')
                    self.stopwords = set(nltk_stopwords.words('english'))
                    print("NLTK stopwords downloaded and loaded.")
                except Exception as e:
                    print(f"Warning: Failed to download NLTK stopwords: {e}. Stopword removal disabled.")
                    self.stopwords = set()
                    self.remove_stopwords = False
        else:
             self.stopwords = set()
        print(f"SpacyTextPreprocessor initialized (Stopwords: {'Enabled' if self.remove_stopwords else 'Disabled'})")

    def _load_spacy_model(self, model_name):
        try:
            return spacy.load(model_name, disable=['ner', 'parser']) # Faster loading
        except OSError:
            print(f"Spacy model '{model_name}' not found. Downloading...")
            try:
                spacy.cli.download(model_name)
                return spacy.load(model_name, disable=['ner', 'parser'])
            except Exception as e:
                print(f"Error downloading/loading spaCy model '{model_name}': {e}")
                raise

    def clean_and_tokenize(self, text):
        text = str(text).lower()
        text = re.sub(r'@\w+', '', text) # Remove user mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        # Basic whitespace normalization before spacy
        text = re.sub(r'\s+', ' ', text).strip()

        doc = self.nlp(text)
        tokens = []
        for token in doc:
            # Keep alphanumeric, ignore punct/space unless needed
            is_valid = token.is_alpha or token.is_digit
            is_stop = token.lemma_ in self.stopwords if self.remove_stopwords else False

            if is_valid and not is_stop:
                 tokens.append(token.lemma_) # Use lemma

        return tokens

    def preprocess_batch(self, texts):
        """Optimized batch processing for spaCy."""
        processed_texts = []
        # Use nlp.pipe for efficiency
        cleaned_texts = (re.sub(r'\s+', ' ', re.sub(r'http\S+|www\S+|https\S+', '', re.sub(r'@\w+', '', str(text).lower()))).strip() for text in texts)
        for doc in tqdm(self.nlp.pipe(cleaned_texts, batch_size=50), total=len(texts), desc="SpaCy Processing"):
            tokens = []
            for token in doc:
                is_valid = token.is_alpha or token.is_digit
                is_stop = token.lemma_ in self.stopwords if self.remove_stopwords else False
                if is_valid and not is_stop:
                    tokens.append(token.lemma_)
            processed_texts.append(" ".join(tokens)) # Return space-separated string for consistency
        return processed_texts


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
        print("Building vocabulary...")
        frequencies = Counter()
        idx = len(self.itos) # Start indexing after special tokens

        # Expect sentence_list to be lists of tokens
        for sentence_tokens in tqdm(sentence_list, desc="Counting Token Frequencies"):
            frequencies.update(sentence_tokens)

        # Sort by frequency and filter
        sorted_freq = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)

        for word, freq in tqdm(sorted_freq, desc="Creating Mappings"):
            if freq >= self.freq_threshold:
                if word not in self.stoi: # Avoid overwriting special tokens
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        print(f"Vocabulary built. Size: {len(self.itos)}")

    def numericalize(self, text_tokens):
        return [self.stoi.get(token, config.UNK_IDX) for token in text_tokens]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {
            'stoi': self.stoi,
            'itos': self.itos, # Save both for easier loading/debugging
            'freq_threshold': self.freq_threshold
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=4)
            print(f"Vocabulary saved to {filepath}")
        except Exception as e:
            print(f"Error saving vocabulary: {e}")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            freq_threshold = loaded_data.get('freq_threshold', config.VOCAB_MIN_FREQ)
            vocab = cls(freq_threshold)
            # Important: Convert loaded itos keys back to integers
            vocab.itos = {int(k): v for k,v in loaded_data['itos'].items()}
            vocab.stoi = loaded_data['stoi'] # Assumes stoi values are already integers
            print(f"Vocabulary loaded from {filepath}. Size: {len(vocab)}")
            return vocab
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
    """Saves label mappings to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Ensure keys and values are native Python types and keys are strings for JSON
    label_to_int_serializable = {str(k): to_native_type(v) for k, v in label_to_int.items()}
    int_to_label_serializable = {str(k): to_native_type(v) for k, v in int_to_label.items()}

    save_data = {
        'label_to_int': label_to_int_serializable,
        'int_to_label': int_to_label_serializable
    }
    try:
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        print(f"Label mappings saved to {filepath}")
    except Exception as e:
        print(f"Error saving label mappings: {e}")

def load_label_mappings(filepath=config.LABEL_MAP_PATH):
    """Loads label mappings from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Label mapping file not found at {filepath}. Returning None.")
        return None, None # Return None if file doesn't exist

    try:
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        # Convert keys back to appropriate types (int for int_to_label keys)
        label_to_int = loaded_data.get('label_to_int', {}) # Keep keys as strings
        int_to_label_str_keys = loaded_data.get('int_to_label', {})
        int_to_label = {int(k): v for k, v in int_to_label_str_keys.items()} # Convert keys to int

        if not label_to_int or not int_to_label:
             print(f"Warning: Loaded label map from {filepath} seems incomplete.")
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

def load_raw_data(filepath=config.INPUT_FILE_PATH,
                  file_format=config.INPUT_FILE_FORMAT,
                  text_col_idx=config.TEXT_COLUMN_INDEX,
                  label_col_idx=config.LABEL_COLUMN_INDEX,
                  col_names=config.COLUMN_NAMES,
                  has_header=config.HAS_HEADER):
    """Loads raw data from file into a pandas DataFrame."""
    print(f"Loading raw data from: {filepath} (Format: {file_format})")
    try:
        if file_format == "csv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(filepath, header=header, names=names, on_bad_lines='warn')
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(filepath, sep='\t', header=header, names=names, on_bad_lines='warn')
        elif file_format == "jsonl":
            df = pd.read_json(filepath, lines=True)
             # Need to know column names for jsonl if they aren't standard
            if not col_names:
                 print("Warning: COLUMNS_NAMES in config might be needed for jsonl if keys vary.")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Validate and select columns
        if label_col_idx >= len(df.columns) or text_col_idx >= len(df.columns):
             raise IndexError(f"Column index out of bounds. File has {len(df.columns)} columns.")

        label_col_name = df.columns[label_col_idx]
        text_col_name = df.columns[text_col_idx]

        print(f"Identified columns - Label: '{label_col_name}' (Index {label_col_idx}), Text: '{text_col_name}' (Index {text_col_idx})")

        # Create DataFrame with standard names 'label' and 'text'
        df_std = pd.DataFrame({
            'label': df[label_col_name],
            'text': df[text_col_name]
        })

        print(f"Loaded {len(df_std)} rows.")
        return df_std.dropna().reset_index(drop=True) # Drop rows with NaN in selected cols

    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        raise # Re-raise critical error
    except IndexError as e:
         print(f"Error: Problem accessing columns by index. Check config settings (COLUMN_INDEX, HAS_HEADER). Details: {e}")
         raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        import traceback
        traceback.print_exc()
        raise

def prepare_data(df_train, df_val, df_test):
    """
    Handles label processing (mapping text labels to integers if needed)
    and determines the number of classes. Saves mappings if created.
    Returns processed dataframes and label information.
    """
    print("\n--- Preparing Labels ---")
    label_col = 'label' # Standardized column name
    label_to_int, int_to_label = load_label_mappings() # Try loading existing map first

    train_labels = df_train[label_col]
    is_numeric_training_labels = ptypes.is_numeric_dtype(train_labels)

    n_classes = None

    if label_to_int and int_to_label:
        print(f"Using pre-loaded label map from {config.LABEL_MAP_PATH}")
        n_classes = len(int_to_label)
        # Apply mapping even if training labels are numeric, to ensure consistency
        if is_numeric_training_labels:
            # Check if numeric labels are valid keys in the loaded map
            valid_keys = {int(k) for k in int_to_label.keys()}
            if not all(label in valid_keys for label in train_labels.unique()):
                print("Warning: Training data contains numeric labels not present in the loaded int_to_label map. Attempting to proceed, but this might cause issues.")
            # Assume numeric labels are already the desired integers
            df_train[label_col] = df_train[label_col].astype(int)
            df_val[label_col] = df_val[label_col].astype(int)
            df_test[label_col] = df_test[label_col].astype(int)

        else: # Training labels are strings, apply mapping
            print("Applying loaded mapping to string labels...")
            for df in [df_train, df_val, df_test]:
                original_labels = set(df[label_col].astype(str).unique())
                df[label_col] = df[label_col].astype(str).map(label_to_int) # Map string labels
                # Handle labels present in val/test but not in map (should ideally not happen with good data)
                if df[label_col].isnull().any():
                    unmapped = original_labels - set(label_to_int.keys())
                    print(f"Warning: Found labels not in loaded map: {unmapped}. Dropping rows.")
                    df.dropna(subset=[label_col], inplace=True)
                df[label_col] = df[label_col].astype(int)

    elif is_numeric_training_labels:
        print("Detected numeric labels in training data. Using them directly.")
        # Ensure all sets have numeric labels
        for name, df in [('Validation', df_val), ('Test', df_test)]:
             if not ptypes.is_numeric_dtype(df[label_col]):
                 raise TypeError(f"Training labels are numeric, but {name} labels are not.")
        df_train[label_col] = df_train[label_col].astype(int)
        df_val[label_col] = df_val[label_col].astype(int)
        df_test[label_col] = df_test[label_col].astype(int)
        # Create placeholder mappings
        all_unique_labels = pd.concat([df_train[label_col], df_val[label_col], df_test[label_col]]).unique()
        int_to_label = {i: f"label_{i}" for i in sorted(all_unique_labels)}
        label_to_int = {v: k for k, v in int_to_label.items()} # Placeholder label_to_int
        n_classes = len(int_to_label)
        print(f"Created placeholder label map for {n_classes} numeric labels.")
        # Do NOT save placeholder map automatically, let user create a meaningful one if desired.
        print(f"Consider creating a '{config.LABEL_MAP_FILENAME}' in '{config.ARTIFACTS_DIR}' with meaningful names.")


    else: # Training labels are strings, and no map was loaded
        print("Detected string labels in training data. Creating new mappings.")
        unique_train_labels = sorted(train_labels.astype(str).unique())
        label_to_int = {label: i for i, label in enumerate(unique_train_labels)}
        int_to_label = {i: label for label, i in label_to_int.items()}
        n_classes = len(label_to_int)
        print(f"Created mapping for {n_classes} labels: {unique_train_labels}")

        # Apply new mapping to all sets
        for df in [df_train, df_val, df_test]:
             original_labels = set(df[label_col].astype(str).unique())
             df[label_col] = df[label_col].astype(str).map(label_to_int)
             if df[label_col].isnull().any():
                 unmapped = original_labels - set(label_to_int.keys())
                 print(f"Warning: Found labels in val/test not present in training data: {unmapped}. Dropping rows.")
                 df.dropna(subset=[label_col], inplace=True)
             df[label_col] = df[label_col].astype(int)

        # Save the newly created map
        save_label_mappings(label_to_int, int_to_label)

    if n_classes is None:
         raise ValueError("Could not determine the number of classes.")

    print(f"\nLabel preparation complete. Determined {n_classes} classes.")
    print(f"Final int_to_label mapping: {int_to_label}")

    # Final check for NaNs introduced by mapping issues
    df_train.dropna(subset=['label', 'text'], inplace=True)
    df_val.dropna(subset=['label', 'text'], inplace=True)
    df_test.dropna(subset=['label', 'text'], inplace=True)

    return df_train, df_val, df_test, label_to_int, int_to_label, n_classes


# --- PyTorch Dataset and DataLoader ---

class GenericDataset(Dataset):
    """ A generic dataset class adaptable for different model types. """
    def __init__(self, texts, labels, tokenizer=None, vocab=None, max_len=config.MAX_LEN, model_type=config.MODEL_TYPE):
        self.texts = texts # List of preprocessed texts (strings or lists of tokens)
        self.labels = labels # List/array of integer labels
        self.tokenizer = tokenizer # HuggingFace tokenizer (for Transformers)
        self.vocab = vocab       # Custom Vocabulary object (for non-Transformers)
        self.max_len = max_len
        self.model_type = model_type

        if self.model_type == 'Transformer' and self.tokenizer is None:
            raise ValueError("Transformer model type requires a HuggingFace tokenizer.")
        if self.model_type != 'Transformer' and self.vocab is None:
             raise ValueError(f"{self.model_type} model type requires a custom Vocabulary object.")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = self.texts[index]
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.model_type == 'Transformer':
            encoding = self.tokenizer.encode_plus(
                text, # Expects a string
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': label
            }
        else: # CNN_RNN_Attention, LSTM, etc.
            # Expect text to be list of tokens here
            if isinstance(text, str): # If preprocessor returned string, tokenize simply
                 tokens = text.split()
            else:
                 tokens = text # Assume it's already tokenized

            numericalized_tokens = self.vocab.numericalize(tokens)
            # Truncate considering SOS and EOS tokens
            truncated_tokens = numericalized_tokens[:self.max_len - 2]
            sequence = [config.SOS_IDX] + truncated_tokens + [config.EOS_IDX]
            sequence_tensor = torch.tensor(sequence, dtype=torch.long)
            return {
                'sequence': sequence_tensor,
                'labels': label
             }


def create_dataloaders(train_data, val_data, test_data, model_type=config.MODEL_TYPE,
                       batch_size=config.TRAIN_BATCH_SIZE, val_batch_size=config.VALID_BATCH_SIZE,
                       tokenizer=None, vocab=None):
    """Creates DataLoaders for train, validation, and test sets."""

    if model_type == 'Transformer':
        collate_fn = None # Default collate works fine for dicts with tensors
    else:
        # Custom collate for non-transformers (padding sequences)
        def collate_non_transformer(batch):
            sequences = [item['sequence'] for item in batch]
            labels = torch.stack([item['labels'] for item in batch])
            lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
            padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=config.PAD_IDX)
            return padded_sequences, labels, lengths

        collate_fn = collate_non_transformer

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0, # Adjust based on system
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    val_loader = DataLoader(
        val_data,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=val_batch_size, # Use validation batch size for test
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True if config.DEVICE == "cuda" else False
    )

    print("\nDataLoaders created.")
    return train_loader, val_loader, test_loader

# --- Main Data Pipeline Function ---

def get_data_pipeline(force_rebuild_vocab=False):
    """
    Orchestrates the entire data loading, preprocessing, and preparation pipeline.

    Returns:
        tuple: (train_loader, val_loader, test_loader, label_to_int, int_to_label, n_classes, vocab_or_tokenizer)
               vocab_or_tokenizer is either a Vocabulary object or a HuggingFace tokenizer.
    """
    print("--- Starting Data Pipeline ---")

    # 1. Load Raw Data
    df = load_raw_data()

    # 2. Split Data
    print("\nSplitting data...")
    df_train, df_temp = train_test_split(
        df,
        test_size=(config.VALIDATION_SPLIT_SIZE + config.TEST_SPLIT_SIZE),
        random_state=config.SEED,
        stratify=df['label'] if config.STRATIFY_SPLIT else None
    )
    relative_test_size = config.TEST_SPLIT_SIZE / (config.VALIDATION_SPLIT_SIZE + config.TEST_SPLIT_SIZE)
    df_val, df_test = train_test_split(
        df_temp,
        test_size=relative_test_size,
        random_state=config.SEED,
        stratify=df_temp['label'] if config.STRATIFY_SPLIT else None
    )
    print(f"Split sizes: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    # 3. Handle Labels
    df_train, df_val, df_test, label_to_int, int_to_label, n_classes = prepare_data(
        df_train.copy(), df_val.copy(), df_test.copy() # Use copies to avoid SettingWithCopyWarning
    )

    # 4. Initialize Preprocessor and Tokenizer/Vocabulary
    print(f"\nInitializing preprocessor: {config.PREPROCESSOR_TYPE}")
    if config.PREPROCESSOR_TYPE == 'spacy':
        preprocessor = SpacyTextPreprocessor()
    else:
        preprocessor = BasicTextCleaner()

    vocab_or_tokenizer = None
    if config.MODEL_TYPE == 'Transformer':
        if AutoTokenizer is None:
             raise ImportError("HuggingFace Transformers library not installed. Needed for MODEL_TYPE='Transformer'.")
        print(f"Loading HuggingFace Tokenizer: {config.TRANSFORMER_MODEL_NAME}")
        vocab_or_tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
        vocab_size = vocab_or_tokenizer.vocab_size # Get vocab size from tokenizer
    else:
        # Build or load vocabulary for non-transformer models
        if os.path.exists(config.VOCAB_PATH) and not force_rebuild_vocab:
            print(f"Loading existing vocabulary from: {config.VOCAB_PATH}")
            try:
                vocab_or_tokenizer = Vocabulary.load(config.VOCAB_PATH)
            except Exception as e:
                print(f"Failed to load vocabulary, rebuilding. Error: {e}")
                vocab_or_tokenizer = None # Force rebuild
        else:
             print("No existing vocabulary found or rebuild forced.")

        if vocab_or_tokenizer is None:
            print("Preprocessing training text for vocabulary building...")
            # Non-transformers often expect token lists from preprocessor
            if isinstance(preprocessor, SpacyTextPreprocessor):
                 train_tokens_list = [preprocessor.clean_and_tokenize(text) for text in tqdm(df_train['text'], desc="Tokenizing Train")]
            else: # Basic cleaner
                 train_tokens_list = [preprocessor.tokenize(text) for text in tqdm(df_train['text'], desc="Tokenizing Train")]

            vocab_or_tokenizer = Vocabulary(freq_threshold=config.VOCAB_MIN_FREQ)
            vocab_or_tokenizer.build_vocabulary(train_tokens_list)
            vocab_or_tokenizer.save(config.VOCAB_PATH) # Save the new vocab

        vocab_size = len(vocab_or_tokenizer)
        print(f"Vocabulary size: {vocab_size}")


    # 5. Preprocess Text Data (apply cleaning)
    print("\nApplying text preprocessing to all datasets...")
    # Preprocessor preprocess_batch should ideally return strings for transformers, lists of tokens for others
    # For simplicity here, let's assume preprocess_batch returns cleaned strings,
    # and tokenization happens inside Dataset or Vocab building.
    # Adjust if your preprocessor directly tokenizes.
    train_texts = preprocessor.preprocess_batch(df_train['text'].tolist())
    val_texts = preprocessor.preprocess_batch(df_val['text'].tolist())
    test_texts = preprocessor.preprocess_batch(df_test['text'].tolist())
    print("Text preprocessing complete.")

    # 6. Create Datasets
    print("\nCreating PyTorch Datasets...")
    train_dataset = GenericDataset(
        texts=train_texts,
        labels=df_train['label'].values,
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    )
    val_dataset = GenericDataset(
        texts=val_texts,
        labels=df_val['label'].values,
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    )
    test_dataset = GenericDataset(
        texts=test_texts,
        labels=df_test['label'].values,
        tokenizer=vocab_or_tokenizer if config.MODEL_TYPE == 'Transformer' else None,
        vocab=vocab_or_tokenizer if config.MODEL_TYPE != 'Transformer' else None,
        max_len=config.MAX_LEN,
        model_type=config.MODEL_TYPE
    )

    # 7. Create DataLoaders
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

**4. `models.py`**

```python
# --- models.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    AutoModel = None
    AutoConfig = None
    print("Warning: HuggingFace Transformers library not installed. Transformer model type will not be available.")

import config # Import configuration

# --- Attention Mechanism (for CNN_RNN_Attention) ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_dim = hidden_dim
        # Adjusted linear layer input size for bidirectional RNN
        self.W_q = nn.Linear(hidden_dim * 2, self.attention_dim, bias=False)
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, rnn_outputs, sequence_lengths=None):
        # rnn_outputs shape: (batch_size, seq_len, hidden_dim * 2)
        energy = torch.tanh(self.W_q(rnn_outputs))  # (batch_size, seq_len, attention_dim)
        attention_scores = self.v(energy).squeeze(2) # (batch_size, seq_len)

        # Apply mask based on sequence lengths before softmax
        if sequence_lengths is not None:
            max_len = rnn_outputs.size(1)
            # Create mask: True for padding positions
            mask = torch.arange(max_len, device=rnn_outputs.device)[None, :] >= sequence_lengths[:, None]
            attention_scores = attention_scores.masked_fill(mask, -float('inf')) # Mask padding

        attention_weights = F.softmax(attention_scores, dim=1) # (batch_size, seq_len)
        # Calculate context vector
        # attention_weights unsqueezed: (batch_size, 1, seq_len)
        # rnn_outputs: (batch_size, seq_len, hidden_dim * 2)
        # context_vector: (batch_size, 1, hidden_dim * 2) -> squeezed to (batch_size, hidden_dim * 2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context_vector, attention_weights

# --- Transformer Model ---
class TransformerClassifier(nn.Module):
    """ Generic Transformer-based classifier using AutoModel. """
    def __init__(self, model_name, n_classes, dropout_prob=0.1):
        super().__init__()
        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library is required for TransformerClassifier.")

        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        # Use dropout defined in config or fallback
        dropout_val = getattr(self.config, 'hidden_dropout_prob', dropout_prob)
        self.dropout = nn.Dropout(dropout_val)

        self.classifier = nn.Linear(self.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use pooler output if available, otherwise CLS token's last hidden state
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits

# --- CNN + RNN + Attention Model ---
class CNN_RNN_Attention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 cnn_out_channels,
                 cnn_kernel_sizes, # Expect list/tuple
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
             cnn_kernel_sizes = [cnn_kernel_sizes] # Ensure it's iterable

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=cnn_out_channels,
                      kernel_size=k,
                      padding='same') # Use 'same' padding
            for k in cnn_kernel_sizes
        ])

        cnn_total_out_channels = cnn_out_channels * len(cnn_kernel_sizes)

        self.rnn_type = rnn_type.lower()
        rnn_input_dim = cnn_total_out_channels # Output of CNNs is input to RNN

        rnn_dropout = dropout_prob if rnn_layers > 1 else 0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=True)
        else: # gru
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=True)

        # Attention layer input dim matches RNN hidden dim (bidirectional doubles it)
        self.attention = Attention(rnn_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        # FC layer input dim matches attention output (bidirectional RNN output)
        self.fc = nn.Linear(rnn_hidden_dim * 2, n_class)
        self.pad_idx = pad_idx


    def forward(self, text_indices, sequence_lengths=None):
        # text_indices shape: (batch_size, seq_len)
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # Conv1d expects (batch_size, channels, seq_len)
        embedded_permuted = embedded.permute(0, 2, 1)
        # embedded_permuted shape: (batch_size, embedding_dim, seq_len)

        cnn_outputs = [F.relu(conv(embedded_permuted)) for conv in self.conv_layers]
        # Each cnn_output shape: (batch_size, cnn_out_channels, seq_len)

        cnn_cat = torch.cat(cnn_outputs, dim=1)
        # cnn_cat shape: (batch_size, cnn_total_out_channels, seq_len)

        # RNN expects (batch_size, seq_len, features)
        rnn_input = cnn_cat.permute(0, 2, 1)
        # rnn_input shape: (batch_size, seq_len, cnn_total_out_channels)

        # Pack sequence for RNN efficiency if lengths are provided
        if sequence_lengths is not None:
             # Ensure lengths are on CPU for pack_padded_sequence
             packed_input = nn.utils.rnn.pack_padded_sequence(rnn_input, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             packed_outputs, _ = self.rnn(packed_input)
             rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
             # Warning: Without lengths, RNN processes padding tokens which might hurt performance.
             rnn_outputs, _ = self.rnn(rnn_input) # (batch_size, seq_len, rnn_hidden_dim * 2)


        # Apply Attention
        # Pass sequence_lengths to attention for masking
        context_vector, _ = self.attention(rnn_outputs, sequence_lengths)
        # context_vector shape: (batch_size, rnn_hidden_dim * 2)

        dropped_context = self.dropout(context_vector)
        out = self.fc(dropped_context) # (batch_size, n_class)
        return out

# --- Simple LSTM Model ---
class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_dropout = dropout_prob if n_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        # Input to FC is concatenation of the final forward and backward hidden states
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.pad_idx = pad_idx

    def forward(self, text_indices, sequence_lengths=None):
        # text_indices shape: (batch_size, seq_len)
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # Pack sequence for RNN efficiency if lengths are provided
        if sequence_lengths is not None:
             packed_input = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             _, (hidden, cell) = self.lstm(packed_input)
        else:
             _, (hidden, cell) = self.lstm(embedded)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)

        # Concatenate the final hidden states from the last layer (forward and backward)
        # hidden[-2,:,:] is the last forward layer's hidden state
        # hidden[-1,:,:] is the last backward layer's hidden state
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden_concat shape: (batch_size, hidden_dim * 2)

        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped) # (batch_size, n_class)
        return out
```

**5. `engine.py`**

```python
# --- engine.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import config # Import configuration

# --- Model Initialization ---

def initialize_model(model_type, n_classes, vocab_size=None):
    """Initializes the model based on the configuration."""
    print(f"\nInitializing model: {model_type}")
    if model_type == 'Transformer':
        from models import TransformerClassifier # Local import
        model = TransformerClassifier(
            model_name=config.TRANSFORMER_MODEL_NAME,
            n_classes=n_classes
        )
    elif model_type == 'CNN_RNN_Attention':
        from models import CNN_RNN_Attention # Local import
        if vocab_size is None: raise ValueError("vocab_size required for CNN_RNN_Attention")
        model = CNN_RNN_Attention(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            rnn_type=config.RNN_TYPE,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            rnn_layers=config.RNN_LAYERS,
            n_class=n_classes,
            dropout_prob=config.WEIGHT_DECAY, # Using WEIGHT_DECAY as dropout here might be unintended? Use a separate DROPOUT_PROB config? Let's assume a default or add it.
            pad_idx=config.PAD_IDX
        )
    elif model_type == 'LSTM':
        from models import LSTMNetwork # Local import
        if vocab_size is None: raise ValueError("vocab_size required for LSTM")
        model = LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.RNN_HIDDEN_DIM,
            n_class=n_classes,
            n_layers=config.RNN_LAYERS,
            pad_idx=config.PAD_IDX,
            dropout_prob=config.WEIGHT_DECAY # Same potential issue as above
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE in config: {model_type}")

    model.to(config.DEVICE)
    print(f"Model loaded on {config.DEVICE}")
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return model

# --- Optimizer and Scheduler ---

def initialize_optimizer_scheduler(model, optimizer_type, scheduler_type, num_train_steps=None):
    """Initializes optimizer and scheduler based on config."""
    print(f"\nInitializing Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")

    if optimizer_type == 'AdamW':
        # Differentiate parameters for weight decay (common for Transformers)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': config.WEIGHT_DECAY},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=0.9)
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {optimizer_type}")

    scheduler = None
    if scheduler_type == 'linear_warmup':
        if num_train_steps is None:
            raise ValueError("num_train_steps is required for linear_warmup scheduler")
        num_warmup_steps = int(num_train_steps * config.WARMUP_PROPORTION)
        print(f"  Warmup Steps: {num_warmup_steps} (of {num_train_steps} total)")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif scheduler_type == 'reduce_on_plateau':
        # Monitors validation loss by default
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    elif scheduler_type is not None:
        print(f"Warning: Scheduler type '{scheduler_type}' requested but not implemented. No scheduler used.")


    return optimizer, scheduler

# --- Loss Function ---
criterion = nn.CrossEntropyLoss()

# --- Training Step ---

def train_step(model, data_loader, optimizer, device, scheduler=None, grad_clip_value=None):
    """Performs a single training epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        # Adapt input based on model type (derived from batch structure)
        if config.MODEL_TYPE == 'Transformer':
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else: # Non-transformer models (LSTM, CNN_RNN)
            # Assumes collate function returns (sequences, labels, lengths)
            sequences = batch[0].to(device)
            labels = batch[1].to(device)
            lengths = batch[2].to(device) # Pass lengths to model
            outputs = model(text_indices=sequences, sequence_lengths=lengths)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient Clipping
        if grad_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        if scheduler and config.SCHEDULER_TYPE == 'linear_warmup': # Step scheduler every batch for warmup
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'})

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# --- Evaluation Step ---

def evaluate_step(model, data_loader, device):
    """Performs evaluation on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # Adapt input based on model type
            if config.MODEL_TYPE == 'Transformer':
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else: # Non-transformer
                sequences = batch[0].to(device)
                labels = batch[1].to(device)
                lengths = batch[2].to(device)
                outputs = model(text_indices=sequences, sequence_lengths=lengths)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate weighted precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'predictions': all_preds, # Return predictions for detailed analysis
        'true_labels': all_labels # Return true labels
    }
    return metrics


# --- Training Loop ---

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_save_path, metric_for_best=config.METRIC_FOR_BEST_MODEL):
    """The main training loop."""
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_weighted': []}
    best_metric_value = -float('inf') if metric_for_best != 'loss' else float('inf')
    grad_clip_value = config.GRADIENT_CLIP_VALUE if config.MODEL_TYPE == 'Transformer' else None # Only clip for transformers by default

    print(f"\n--- Starting Training for {epochs} Epochs ---")
    print(f"Monitoring validation '{metric_for_best}' for best model.")
    if grad_clip_value: print(f"Using gradient clipping: {grad_clip_value}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training
        train_loss = train_step(model, train_loader, optimizer, device, scheduler, grad_clip_value)
        print(f"  Train Loss: {train_loss:.4f}")
        history['train_loss'].append(train_loss)

        # Validation
        val_metrics = evaluate_step(model, val_loader, device)
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_weighted']
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_weighted'].append(val_f1)

        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1 (W): {val_f1:.4f}")

        # Scheduler Step (for ReduceLROnPlateau)
        if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler.step(val_loss)

        # Check for best model
        current_metric_value = val_metrics[metric_for_best]
        is_better = False
        if metric_for_best == 'loss':
            is_better = current_metric_value < best_metric_value
        else: # Higher is better for accuracy, f1
            is_better = current_metric_value > best_metric_value

        if is_better:
            print(f"  ✨ Validation '{metric_for_best}' improved ({best_metric_value:.4f} --> {current_metric_value:.4f}). Saving model...")
            best_metric_value = current_metric_value
            try:
                 # Ensure directory exists
                 os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                 # Save model state dictionary
                 torch.save(model.state_dict(), model_save_path)
                 print(f"  Model saved to {model_save_path}")
            except Exception as e:
                 print(f"  Error saving model: {e}")
        else:
            print(f"  Validation '{metric_for_best}' did not improve from {best_metric_value:.4f}.")

    print("\n--- Training Finished ---")
    print(f"Best validation '{metric_for_best}': {best_metric_value:.4f}")
    return history

# --- Model Loading ---
def load_trained_model(model_path, model_type, n_classes, vocab_size=None):
    """Loads a pre-trained model state dict."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = initialize_model(model_type, n_classes, vocab_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(config.DEVICE)))
        print(f"Model weights loaded successfully from {model_path}")
        model.eval() # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure the model architecture matches the saved weights and the file is not corrupted.")
        raise
```

**6. `train.py`**

```python
# --- train.py ---
import os
import sys
import time

# Dynamically add project root to path if needed
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

import config
import data_handler
import engine
import plotter

def set_seed(seed_value=config.SEED):
    """Sets the seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # May impact performance
        torch.backends.cudnn.benchmark = False   # Ensure reproducibility

def run_training_pipeline():
    """Main function to execute the training pipeline."""
    start_time = time.time()
    set_seed()

    print(f"=== Starting Training Run: {config.RUN_ID} ===")

    # --- 1. Data Pipeline ---
    try:
        train_loader, val_loader, test_loader, label_to_int, int_to_label, n_classes, vocab_or_tokenizer, vocab_size = data_handler.get_data_pipeline()
    except Exception as e:
        print(f"Error during data pipeline: {e}")
        sys.exit(1)

    # --- 2. Initialize Model ---
    try:
        model = engine.initialize_model(config.MODEL_TYPE, n_classes, vocab_size)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # --- 3. Initialize Optimizer & Scheduler ---
    try:
        # Calculate total training steps for scheduler if needed
        num_train_steps = len(train_loader) * config.EPOCHS if config.SCHEDULER_TYPE == 'linear_warmup' else None
        optimizer, scheduler = engine.initialize_optimizer_scheduler(
            model, config.OPTIMIZER_TYPE, config.SCHEDULER_TYPE, num_train_steps
        )
    except Exception as e:
        print(f"Error initializing optimizer/scheduler: {e}")
        sys.exit(1)

    # --- 4. Train Model ---
    try:
        history = engine.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            epochs=config.EPOCHS,
            model_save_path=config.BEST_MODEL_PATH,
            metric_for_best=config.METRIC_FOR_BEST_MODEL
        )
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 5. Plot Training History ---
    if config.PLOT_TRAINING_HISTORY:
        try:
            print("\nGenerating training history plots...")
            plotter.plot_training_history(history, save_path=config.TRAINING_PLOTS_PATH)
        except Exception as e:
            print(f"Warning: Could not generate training plots. Error: {e}")

    # --- 6. Evaluate on Test Set ---
    if config.GENERATE_TEST_REPORT or config.GENERATE_CONFUSION_MATRIX:
        print("\n--- Evaluating on Test Set ---")
        try:
            # Load the *best* saved model for final evaluation
            print(f"Loading best model from: {config.BEST_MODEL_PATH}")
            best_model = engine.load_trained_model(
                model_path=config.BEST_MODEL_PATH,
                model_type=config.MODEL_TYPE,
                n_classes=n_classes,
                vocab_size=vocab_size
            )
            print("Evaluating best model on test data...")
            test_metrics = engine.evaluate_step(best_model, test_loader, config.DEVICE)

            print("\nTest Set Performance:")
            print(f"  Loss: {test_metrics['loss']:.4f}")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
            print(f"  Precision (Weighted): {test_metrics['precision_weighted']:.4f}")
            print(f"  Recall (Weighted): {test_metrics['recall_weighted']:.4f}")

            # Generate detailed report and confusion matrix
            plotter.generate_classification_analysis(
                true_labels=test_metrics['true_labels'],
                predictions=test_metrics['predictions'],
                int_to_label=int_to_label,
                report_path=config.TEST_REPORT_PATH if config.GENERATE_TEST_REPORT else None,
                cm_path=config.CONFUSION_MATRIX_PATH if config.GENERATE_CONFUSION_MATRIX else None,
                prefix="Test Set"
            )

        except FileNotFoundError:
            print(f"Warning: Best model file not found at {config.BEST_MODEL_PATH}. Skipping test evaluation.")
        except Exception as e:
            print(f"Error during test evaluation: {e}")
            import traceback
            traceback.print_exc()

    # --- 7. Save Final Run Configuration ---
    try:
        config.save_run_config() # Save the config used for this run
    except Exception as e:
        print(f"Warning: Failed to save final run config. Error: {e}")


    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== Training Run {config.RUN_ID} Finished ===")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Artifacts saved in: {config.RUN_ARTIFACTS_DIR}")
    print("========================================")

if __name__ == "__main__":
    run_training_pipeline()
```

**7. `plotter.py`**

```python
# --- plotter.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

import config # For default save paths

sns.set_theme(style="whitegrid")

def plot_training_history(history, save_path=config.TRAINING_PLOTS_PATH):
    """
    Plots training and validation loss, accuracy, and F1 score over epochs.

    Args:
        history (dict): Dictionary containing lists of metrics per epoch
                        (e.g., 'train_loss', 'val_loss', 'val_accuracy', 'val_f1_weighted').
        save_path (str): Path to save the plot image.
    """
    if not history:
        print("Plotter Warning: History dictionary is empty. Cannot plot training history.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    df = pd.DataFrame(history)
    df['epoch'] = epochs

    num_plots = 0
    if 'train_loss' in df and 'val_loss' in df: num_plots += 1
    if 'val_accuracy' in df: num_plots += 1
    if 'val_f1_weighted' in df: num_plots += 1

    if num_plots == 0:
        print("Plotter Warning: No plottable metrics found in history dict.")
        return

    plt.figure(figsize=(8 * num_plots, 5)) # Adjust figure size based on number of plots

    plot_idx = 1
    # --- Loss Plot ---
    if 'train_loss' in df and 'val_loss' in df:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linestyle='-')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='x', linestyle='--')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_idx += 1

    # --- Accuracy Plot ---
    if 'val_accuracy' in df:
        plt.subplot(1, num_plots, plot_idx)
        # Add train accuracy if available in history
        if 'train_accuracy' in df:
            plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', marker='o', linestyle='-')
        plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', marker='x', linestyle='--')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(bottom=max(0, df['val_accuracy'].min() - 0.1), top=min(1, df['val_accuracy'].max() + 0.1)) # Adjust ylim
        plt.legend()
        plt.grid(True)
        plot_idx += 1

    # --- F1 Score Plot ---
    if 'val_f1_weighted' in df:
        plt.subplot(1, num_plots, plot_idx)
         # Add train F1 if available in history
        if 'train_f1_weighted' in df:
            plt.plot(df['epoch'], df['train_f1_weighted'], label='Train F1 (Weighted)', marker='o', linestyle='-')
        plt.plot(df['epoch'], df['val_f1_weighted'], label='Validation F1 (Weighted)', marker='x', linestyle='--')
        plt.title('Weighted F1 Score vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.ylim(bottom=max(0, df['val_f1_weighted'].min() - 0.1), top=min(1, df['val_f1_weighted'].max() + 0.1)) # Adjust ylim
        plt.legend()
        plt.grid(True)
        plot_idx += 1


    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Plotter Error: Could not save training plot to {save_path}. Error: {e}")
    # plt.show() # Optional: Show plot directly

def generate_classification_analysis(true_labels, predictions, int_to_label, report_path=None, cm_path=None, prefix=""):
    """
    Generates and saves a classification report and confusion matrix.

    Args:
        true_labels (list or np.array): Ground truth integer labels.
        predictions (list or np.array): Predicted integer labels.
        int_to_label (dict): Mapping from integer labels to string names.
        report_path (str, optional): Path to save the text classification report.
        cm_path (str, optional): Path to save the confusion matrix plot.
        prefix (str, optional): Prefix for report/plot titles (e.g., "Test Set").
    """
    if not int_to_label:
        print("Plotter Warning: int_to_label mapping not provided. Using integer labels.")
        # Use unique sorted integer labels present in the data
        unique_labels = sorted(list(set(true_labels) | set(predictions)))
        label_names = [str(i) for i in unique_labels]
        target_labels_for_report = unique_labels # Use integers for report labels arg
    else:
        # Ensure keys are integers and values are strings
        int_to_label = {int(k): str(v) for k, v in int_to_label.items()}
        # Use labels present in the data, map them to names using provided map
        unique_labels_present = sorted(list(set(true_labels) | set(predictions)))
        label_names = [int_to_label.get(i, f"Unknown({i})") for i in unique_labels_present]
        target_labels_for_report = unique_labels_present # Use integers for report labels arg


    # --- Classification Report ---
    try:
        report_str = classification_report(
            true_labels,
            predictions,
            labels=target_labels_for_report, # Specify labels to include
            target_names=label_names,
            zero_division=0,
            digits=3
        )
        title = f"{prefix} Classification Report" if prefix else "Classification Report"
        full_report_output = f"--- {title} ---\n\n{report_str}\n"
        print(full_report_output) # Print to console

        if report_path:
            try:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    # Optionally add overall metrics to the top of the report file
                    accuracy = np.mean(np.array(true_labels) == np.array(predictions))
                    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
                    f.write(report_str)
                print(f"Classification report saved to {report_path}")
            except Exception as e:
                print(f"Plotter Error: Could not save classification report to {report_path}. Error: {e}")

    except Exception as e:
        print(f"Plotter Error: Could not generate classification report. Error: {e}")


    # --- Confusion Matrix ---
    if cm_path:
        try:
            cm = confusion_matrix(true_labels, predictions, labels=target_labels_for_report)
            plt.figure(figsize=(max(8, len(label_names)*0.6), max(6, len(label_names)*0.5))) # Dynamic sizing
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names,
                        annot_kws={"size": 8}) # Adjust font size if needed
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            cm_title = f"{prefix} Confusion Matrix" if prefix else "Confusion Matrix"
            plt.title(cm_title)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            # plt.show() # Optional: Show plot directly
            plt.close() # Close the plot figure

        except Exception as e:
            print(f"Plotter Error: Could not generate or save confusion matrix. Error: {e}")

```

**8. `main.py`**

```python
# --- main.py ---
import sys
import os

# Dynamically add project root to path if needed (if running main.py from within project dir)
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# Import necessary modules after potentially modifying path
try:
    import config
    import train
except ImportError as e:
     print(f"Error importing core modules: {e}")
     print("Ensure config.py and train.py are in the Python path or the same directory.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)


if __name__ == "__main__":
    print("========================================")
    print("=== Emotion Classification Framework ===")
    print("========================================")

    # Ensure artifact directories exist before training starts
    # (config.py already creates RUN_ARTIFACTS_DIR and MODEL_SAVE_DIR)
    # If LABEL_MAP_PATH is outside RUN_ARTIFACTS_DIR, ensure its directory exists
    if os.path.dirname(config.LABEL_MAP_PATH):
         os.makedirs(os.path.dirname(config.LABEL_MAP_PATH), exist_ok=True)

    # Execute the training pipeline defined in train.py
    try:
        train.run_training_pipeline()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n--- An Unhandled Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("-------------------------------------")
        # Optionally print traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)

```

**9. `app.py`**

```python
# --- app.py ---
import torch
import os
import json
import argparse
import sys
from operator import itemgetter

# Dynamically add project root to path if needed
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

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

def load_run_config(run_dir):
    """Loads the specific configuration saved for a given run."""
    config_path = os.path.join(run_dir, config.RUN_CONFIG_FILENAME) # Use default filename
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file not found at {config_path}. Using global config.py defaults.")
        # Fallback logic: Use global config values directly.
        # This might be inaccurate if global config changed since the run.
        class RunConfig:
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = getattr(config, 'TRANSFORMER_MODEL_NAME', None) # Use getattr for safety
             VOCAB_PATH = os.path.join(run_dir, config.VOCAB_FILENAME) # Construct potential path
             REMOVE_STOPWORDS = getattr(config, 'REMOVE_STOPWORDS', False)
             SPACY_MODEL_NAME = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        return RunConfig()

    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        # Convert loaded dict to an object for easier access (optional)
        class RunConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                # Ensure necessary paths are relative to the loaded run_dir if applicable
                self.VOCAB_PATH = os.path.join(run_dir, config.VOCAB_FILENAME)

        print(f"Loaded run configuration from {config_path}")
        return RunConfig(**loaded_config)
    except Exception as e:
        print(f"Error loading run config from {config_path}: {e}. Using global defaults.")
        # Fallback to global config if loading fails
        class RunConfig: # Duplicated fallback logic
             MODEL_TYPE = config.MODEL_TYPE
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = getattr(config, 'TRANSFORMER_MODEL_NAME', None)
             VOCAB_PATH = os.path.join(run_dir, config.VOCAB_FILENAME)
             REMOVE_STOPWORDS = getattr(config, 'REMOVE_STOPWORDS', False)
             SPACY_MODEL_NAME = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        return RunConfig()

def load_prediction_artifacts(run_dir):
    """Loads all necessary artifacts for prediction based on the run's config."""
    print(f"\nLoading artifacts from run directory: {run_dir}")
    run_cfg = load_run_config(run_dir)

    # Load Label Map (global or user-provided)
    label_to_int, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_PATH)
    if not int_to_label:
        print("Warning: Label map not found or empty. Predictions will show integer labels.")
        # Create a dummy map if needed elsewhere, or handle None gracefully
        int_to_label = {} # Empty dict signals no mapping available

    n_classes = len(int_to_label) if int_to_label else 0
    if n_classes == 0:
        print("Warning: Cannot determine number of classes from label map.")
        # Might need to infer from model later if possible, or fail

    # Load Model
    model_path = os.path.join(run_dir, "model", config.BEST_MODEL_FILENAME)
    vocab_size = None
    vocab_or_tokenizer = None

    if run_cfg.MODEL_TYPE != 'Transformer':
        # Load Vocabulary for non-transformer models
        try:
            vocab = data_handler.Vocabulary.load(run_cfg.VOCAB_PATH)
            vocab_size = len(vocab)
            vocab_or_tokenizer = vocab
            print(f"Vocabulary loaded (Size: {vocab_size}).")
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {run_cfg.VOCAB_PATH}. Cannot proceed for {run_cfg.MODEL_TYPE} model.")
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
              tokenizer = data_handler.AutoTokenizer.from_pretrained(run_cfg.TRANSFORMER_MODEL_NAME)
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
    else:
        preprocessor = data_handler.BasicTextCleaner()


    return model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg


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
                 return cleaned_text.split() # Simple split for non-transformer vocab
        return cleaned_text # Return cleaned string for Transformer tokenizer

    def predict(self, text):
        """Predicts emotion probabilities for the input text."""
        processed_input = self._preprocess_input(text)

        try:
            with torch.no_grad():
                if self.run_config.MODEL_TYPE == 'Transformer':
                    encoding = self.vocab_or_tokenizer.encode_plus(
                        processed_input, # Expects string
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
                    numericalized = self.vocab_or_tokenizer.numericalize(processed_input)
                    truncated = numericalized[:self.run_config.MAX_LEN - 2]
                    sequence = [config.SOS_IDX] + truncated + [config.EOS_IDX]
                    sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(self.device) # Add batch dim
                    lengths = torch.tensor([len(sequence)], dtype=torch.long).to(self.device)
                    # Pass lengths, model forward should handle it
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
                    indicator = " *" if result['score'] == max_score else ""
                    print(f"  - {result['label']}: {result['score']:.4f}{indicator}")
            else:
                print("  Prediction failed.")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the loop: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Emotion Prediction App")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to the specific run artifact directory (e.g., artifacts/run_xxxxxxxx)"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: Run directory not found at {args.run_dir}")
        sys.exit(1)

    # Load artifacts based on the specified run directory
    model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg = load_prediction_artifacts(args.run_dir)

    if model is None:
        print("Failed to load necessary artifacts. Exiting.")
        sys.exit(1)

    # Create predictor instance
    predictor = EmotionPredictor(model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg)

    # Start interactive loop
    run_interactive_app(predictor)


if __name__ == "__main__":
    main()
```

**Explanation and Key Features:**

1.  **`config.py`:**
    *   Centralized configuration for paths, model selection (`MODEL_TYPE`), hyperparameters specific to model types (Transformer vs. CNN/RNN), data details (columns, format), preprocessing choices, and artifact naming.
    *   Generates a `RUN_ID` for each execution to store artifacts separately.
    *   Includes a function `save_run_config` to save the configuration used for a specific training run, crucial for reproducibility and loading artifacts later in `app.py`.

2.  **`dataman.py`:**
    *   Command-line utility (`argparse`) for manual data operations.
    *   Includes `balance` (create balanced subset) and `split` (train/val/test split) commands.
    *   Designed to be run *before* the main training pipeline.
    *   Loads data based on config parameters (format, columns, header).

3.  **`data_handler.py`:**
    *   **Sole Data Interface:** This is the only module that directly interacts with loading raw data, preprocessing text, managing labels, handling vocabulary/tokenizers, and creating PyTorch Datasets/DataLoaders.
    *   **Preprocessing:** Includes `BasicTextCleaner` and `SpacyTextPreprocessor`. The choice is determined by `config.PREPROCESSOR_TYPE`.
    *   **Vocabulary/Tokenizer:** Manages `Vocabulary` class for non-Transformers and loads `AutoTokenizer` for Transformers.
    *   **Label Handling:** Implements the requested logic: loads existing `label_map.json`, creates one from string labels if needed (and saves it), or uses numeric labels directly (creating placeholder mappings but not saving them automatically).
    *   **`GenericDataset`:** A single `Dataset` class that adapts its `__getitem__` logic based on `config.MODEL_TYPE` to return the correct format (input_ids/mask for Transformers, sequence tensor for others).
    *   **`create_dataloaders`:** Creates dataloaders, using a custom `collate_fn` only when needed (for padding non-Transformer sequences).
    *   **`get_data_pipeline`:** The main function orchestrating all data steps, called by `train.py`.

4.  **`models.py`:**
    *   Contains class definitions for `TransformerClassifier`, `CNN_RNN_Attention`, and `LSTMNetwork`.
    *   Each model is self-contained.
    *   Includes the `Attention` mechanism needed by `CNN_RNN_Attention`.
    *   Transformer model uses `AutoModel` for flexibility.
    *   Non-transformer models explicitly accept `sequence_lengths` in their `forward` methods and use `pack_padded_sequence` for efficiency.

5.  **`engine.py`:**
    *   **Model Agnostic Logic:** Contains the core training (`train_step`, `train_model`) and evaluation (`evaluate_step`) logic.
    *   **Initialization:** Functions `initialize_model` and `initialize_optimizer_scheduler` create the correct model, optimizer, and scheduler based on `config.py` settings.
    *   **Conditional Execution:** `train_step` and `evaluate_step` adapt how they fetch data from the batch and call the model's forward pass based on `config.MODEL_TYPE`.
    *   **Best Model Saving:** Saves the model based on the validation metric specified in `config.METRIC_FOR_BEST_MODEL`.
    *   **Gradient Clipping:** Applied conditionally, typically just for Transformers.
    *   **`load_trained_model`:** Helper function to load model weights for evaluation or prediction.

6.  **`train.py`:**
    *   **Orchestrator:** Sets the seed, calls `data_handler.get_data_pipeline`, initializes model/optimizer/scheduler via `engine`, runs the training loop (`engine.train_model`), calls `plotter` functions, and saves the run configuration.
    *   Minimal logic itself; delegates tasks to other modules.

7.  **`plotter.py`:**
    *   Generates plots for training history (loss, accuracy, F1).
    *   Generates classification reports and confusion matrices using `sklearn.metrics`.
    *   Handles label mapping (`int_to_label`) for display.
    *   Saves plots and reports to the run-specific artifacts directory.

8.  **`main.py`:**
    *   **Entry Point:** The simplest file. Imports necessary modules and calls `train.run_training_pipeline()`. Handles basic error catching and KeyboardInterrupt.

9.  **`app.py`:**
    *   **Interactive Prediction:** Provides a command-line interface for users to input text and get predictions.
    *   **Artifact Loading:** Crucially, it takes a `--run_dir` argument. It loads the `run_config.json` from that directory to determine the correct model type, tokenizer/vocab path, etc., ensuring it uses the *exact* setup from the training run.
    *   **`EmotionPredictor` Class:** Encapsulates the loaded artifacts and prediction logic.
    *   **Label Handling:** Uses the loaded `int_to_label` map to display human-readable emotion names. If the map wasn't available/created during training, it gracefully shows integer labels.

This structure promotes modularity, reusability (DRY), and flexibility, allowing you to easily switch between different model architectures and configurations. Remember to install necessary libraries (`pip install torch pandas numpy scikit-learn transformers matplotlib seaborn spacy nltk tqdm tabulate`) and download spaCy/NLTK resources if using the `spacy` preprocessor.
