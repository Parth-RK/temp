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

def load_raw_data(filepath, # Now a mandatory argument
                  file_format=config.INPUT_FILE_FORMAT,
                  text_col_idx=config.TEXT_COLUMN_INDEX,
                  label_col_idx=config.LABEL_COLUMN_INDEX,
                  col_names=config.COLUMN_NAMES,
                  has_header=config.HAS_HEADER):
    """Loads raw data from a specific file path."""
    print(f"Attempting to load raw data from: {filepath} (Format: {file_format})")

    # Check if file exists *before* trying to load
    if not filepath or not os.path.exists(filepath):
         print(f"Warning: Data file not found or path is invalid: {filepath}")
         return None # Return None if file not found or path is None/empty

    try:
        if file_format == "csv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(filepath, header=header, names=names, on_bad_lines='warn', low_memory=False)
        elif file_format == "tsv":
            header = 0 if has_header else None
            names = None if has_header else col_names
            df = pd.read_csv(filepath, sep='\t', header=header, names=names, on_bad_lines='warn', low_memory=False)
        elif file_format == "jsonl":
            df = pd.read_json(filepath, lines=True)
            if not col_names and header is None:
                 print("Warning: Column names/indices might be needed for jsonl.")
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        # Validate and select columns
        if label_col_idx >= len(df.columns) or text_col_idx >= len(df.columns):
             raise IndexError(f"Column index out of bounds ({label_col_idx}, {text_col_idx}). File '{os.path.basename(filepath)}' has {len(df.columns)} columns: {list(df.columns)}")

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
        df_std = df_std.dropna().reset_index(drop=True)
        rows_dropped = original_rows - len(df_std)
        if rows_dropped > 0:
            print(f"  Dropped {rows_dropped} rows with NaN values in selected columns.")

        print(f"  Successfully loaded {len(df_std)} rows from {os.path.basename(filepath)}.")
        return df_std

    except FileNotFoundError:
        print(f"Error: Data file somehow not found at {filepath} despite existence check.")
        return None
    except IndexError as e:
         print(f"Error: Problem accessing columns by index in {filepath}. Check config settings. Details: {e}")
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
    print("\n--- Loading Data ---")
    df_train = load_raw_data(filepath=config.TRAIN_FILE_PATH)
    if df_train is None or df_train.empty:
        raise FileNotFoundError(f"CRITICAL: Training data failed to load from {config.TRAIN_FILE_PATH}. Cannot proceed.")

    df_val = load_raw_data(filepath=config.VALID_FILE_PATH)
    df_test = load_raw_data(filepath=config.TEST_FILE_PATH)

    train_needs_split = df_val is None or df_test is None
    if train_needs_split:
        print("\n--- Splitting Training Data ---")
        df_remaining_for_train = df_train.copy()

        if df_val is None:
            print("Validation data not loaded or unavailable. Splitting from train.")
            if len(df_remaining_for_train) < 2:
                print("Warning: Not enough training data to create a validation split.")
                df_val = pd.DataFrame(columns=['label', 'text'])
            else:
                print(f"Splitting validation set ({config.VALIDATION_SPLIT_SIZE*100:.1f}%)...")
                stratify_col_val = df_remaining_for_train['label'] if config.STRATIFY_SPLIT else None
                df_remaining_for_train, df_val = train_test_split(
                    df_remaining_for_train,
                    test_size=config.VALIDATION_SPLIT_SIZE,
                    random_state=config.SEED,
                    stratify=stratify_col_val
                )
                print(f"New Train size: {len(df_remaining_for_train)}, Val size: {len(df_val)}")

        if df_test is None:
             print("Test data not loaded or unavailable. Splitting from remaining train.")
             if len(df_remaining_for_train) < 2:
                 print("Warning: Not enough remaining training data to create a test split.")
                 df_test = pd.DataFrame(columns=['label', 'text'])
             else:
                current_train_fraction = 1.0 - (config.VALIDATION_SPLIT_SIZE if df_val is not None and len(df_val)>0 else 0)
                if current_train_fraction <= 0: current_train_fraction = 1.0
                effective_split_size = config.TEST_SPLIT_SIZE / current_train_fraction
                effective_split_size = min(max(0.0, effective_split_size), 1.0 - (1/len(df_remaining_for_train)) )
                if effective_split_size <= 0:
                     print("Warning: Calculated test split size is too small. Test set will be empty.")
                     df_test = pd.DataFrame(columns=['label', 'text'])
                else:
                     print(f"Splitting test set ({effective_split_size*100:.1f}% from remaining train)...")
                     stratify_col_test = df_remaining_for_train['label'] if config.STRATIFY_SPLIT else None
                     df_remaining_for_train, df_test = train_test_split(
                         df_remaining_for_train,
                         test_size=effective_split_size,
                         random_state=config.SEED,
                         stratify=stratify_col_test
                     )
                     print(f"Final Train size: {len(df_remaining_for_train)}, Test size: {len(df_test)}")

        df_train = df_remaining_for_train
        print("--- Data Splitting Finished ---")

    # 2. Handle Labels
    df_train, df_val, df_test, label_to_int, int_to_label, n_classes = prepare_data(
        df_train.copy(), df_val.copy(), df_test.copy() # Use copies to avoid SettingWithCopyWarning
    )

    # 3. Initialize Preprocessor and Tokenizer/Vocabulary
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


    # 4. Preprocess Text Data (apply cleaning)
    print("\nApplying text preprocessing to all datasets...")
    # Preprocessor preprocess_batch should ideally return strings for transformers, lists of tokens for others
    # For simplicity here, let's assume preprocess_batch returns cleaned strings,
    # and tokenization happens inside Dataset or Vocab building.
    # Adjust if your preprocessor directly tokenizes.
    train_texts = preprocessor.preprocess_batch(df_train['text'].tolist())
    val_texts = preprocessor.preprocess_batch(df_val['text'].tolist())
    test_texts = preprocessor.preprocess_batch(df_test['text'].tolist())
    print("Text preprocessing complete.")

    # 5. Create Datasets
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

    # 6. Create DataLoaders
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