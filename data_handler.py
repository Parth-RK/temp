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