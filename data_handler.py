# --- data_handler.py ---
import torch
import pandas as pd
import numpy as np
import json
import os
import re
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import sys
import pandas.api.types as ptypes

# Try importing necessary libraries, raise clear error if unavailable
try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: HuggingFace Transformers library not installed.")
    print("       Please install it: pip install transformers")
    # Exit early if the core dependency is missing
    sys.exit(1)

# Removed Spacy/NLTK imports and related flags

import config # Import configuration

# --- Text Preprocessing ---

class BasicTextCleaner:
    """Basic cleaning: lowercase, remove mentions/URLs, normalize whitespace."""
    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r'@\w+', '', text) # Remove user mentions
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace
        return text

    def preprocess_batch(self, texts):
        """Applies cleaning to a list of texts."""
        # Use tqdm for progress visibility on larger datasets
        return [self.clean(text) for text in tqdm(texts, desc="Cleaning Text")]

    # Removed tokenize method as it's not directly used by Transformer pipeline

# --- Removed SpacyTextPreprocessor ---
# --- Removed Vocabulary class ---

# --- Label Handling (Unchanged, keep as is) ---
def to_native_type(item):
    if isinstance(item, np.integer): return int(item)
    elif isinstance(item, np.floating): return float(item)
    elif isinstance(item, np.ndarray): return item.tolist()
    elif isinstance(item, np.bool_): return bool(item)
    elif isinstance(item, (pd.Timestamp, pd.Timedelta)): return str(item)
    return item

def save_label_mappings(label_to_int, int_to_label, filepath=config.LABEL_MAP_PATH):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        label_to_int_serializable = {str(k): to_native_type(v) for k, v in label_to_int.items()}
        int_to_label_serializable = {str(k): to_native_type(v) for k, v in int_to_label.items()}
    except Exception as e:
         print(f"Error converting label map items for serialization: {e}")
         label_to_int_serializable = {str(k): str(v) for k, v in label_to_int.items()}
         int_to_label_serializable = {str(k): str(v) for k, v in int_to_label.items()}
    save_data = {'label_to_int': label_to_int_serializable, 'int_to_label': int_to_label_serializable}
    try:
        with open(filepath, 'w', encoding='utf-8') as f: json.dump(save_data, f, indent=4, ensure_ascii=False)
        print(f"Label mappings saved to {filepath}")
    except Exception as e: print(f"Error saving label mappings: {e}")

def load_label_mappings(filepath=config.LABEL_MAP_PATH):
    if not os.path.exists(filepath):
        print(f"Label mapping file not found at {filepath}. Returning None.")
        return None, None
    try:
        with open(filepath, 'r', encoding='utf-8') as f: loaded_data = json.load(f)
        label_to_int = loaded_data.get('label_to_int', {})
        int_to_label_str_keys = loaded_data.get('int_to_label', {})
        int_to_label = {}
        for k, v in int_to_label_str_keys.items():
            try: int_to_label[int(k)] = v
            except ValueError: print(f"Warning: Skipping non-integer key '{k}' in int_to_label map from {filepath}")
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


# --- Dataset Loading and Preparation (load_raw_data, prepare_data - keep as is) ---
def load_raw_data(filepath, file_format=config.INPUT_FILE_FORMAT, text_col_idx=config.TEXT_COLUMN_INDEX,
                  label_col_idx=config.LABEL_COLUMN_INDEX, col_names=config.COLUMN_NAMES, has_header=config.HAS_HEADER):
    print(f"Attempting to load raw data from: {filepath} (Format: {file_format})")
    if not filepath or not os.path.exists(filepath):
         print(f"Warning: Data file not found or path is invalid: {filepath}")
         return None
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
            if col_names is None: col_names = ['text', 'label']
            has_header = False
        else: raise ValueError(f"Unsupported file format: {file_format}")

        num_cols = len(df.columns)
        if label_col_idx >= num_cols or text_col_idx >= num_cols:
             raise IndexError(f"Column index out of bounds (Label: {label_col_idx}, Text: {text_col_idx}). File '{os.path.basename(filepath)}' has {num_cols} columns: {list(df.columns)}")
        label_col_name = df.columns[label_col_idx]
        text_col_name = df.columns[text_col_idx]
        print(f"  Using columns - Label: '{label_col_name}' (Index {label_col_idx}), Text: '{text_col_name}' (Index {text_col_idx})")
        df_std = pd.DataFrame({'label': df[label_col_name], 'text': df[text_col_name]})
        original_rows = len(df_std)
        df_std = df_std.dropna(subset=['label', 'text']).reset_index(drop=True)
        rows_dropped = original_rows - len(df_std)
        if rows_dropped > 0: print(f"  Dropped {rows_dropped} rows with NaN values in 'label' or 'text' columns.")
        df_std['text'] = df_std['text'].astype(str)
        print(f"  Successfully loaded {len(df_std)} rows from {os.path.basename(filepath)}.")
        return df_std
    except FileNotFoundError:
        print(f"Error: Data file somehow not found at {filepath} despite existence check.")
        return None
    except IndexError as e:
         print(f"Error: Problem accessing columns by index in {filepath}. Check indices/config. Details: {e}")
         return None
    except Exception as e:
        print(f"An unexpected error occurred loading {filepath}: {e}")
        import traceback; traceback.print_exc(); return None

def prepare_data(df_train, df_val, df_test):
    print("\n--- Preparing Labels ---")
    label_col = 'label'
    try:
        df_train[label_col] = df_train[label_col].astype(str)
        if df_val is not None: df_val[label_col] = df_val[label_col].astype(str)
        if df_test is not None: df_test[label_col] = df_test[label_col].astype(str)
    except Exception as e: print(f"Warning: Could not convert label column to string. Error: {e}")

    label_to_int, int_to_label = load_label_mappings()
    n_classes = None

    if label_to_int and int_to_label:
        print(f"Using pre-loaded label map from {config.LABEL_MAP_PATH}")
        n_classes = len(int_to_label)
        print(f"Applying loaded mapping ({n_classes} classes)...")
        for df_name, df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
            if df is None: continue
            df['label_int'] = df[label_col].map(label_to_int)
            unmapped_mask = df['label_int'].isnull()
            if unmapped_mask.any():
                unmapped_labels = set(df.loc[unmapped_mask, label_col].unique())
                print(f"Warning ({df_name}): Found labels not in loaded map: {unmapped_labels}. Dropping {unmapped_mask.sum()} rows.")
                df.dropna(subset=['label_int'], inplace=True)
            df[label_col] = df['label_int'].astype(int)
            df.drop(columns=['label_int'], inplace=True)
    else:
        print("No pre-loaded label map found or map invalid. Creating new mappings from training data.")
        unique_train_labels = sorted(df_train[label_col].unique())
        label_to_int = {label: i for i, label in enumerate(unique_train_labels)}
        int_to_label = {i: label for label, i in label_to_int.items()}
        n_classes = len(label_to_int)
        print(f"Created mapping for {n_classes} labels: {unique_train_labels}")
        for df_name, df in [('Train', df_train), ('Validation', df_val), ('Test', df_test)]:
             if df is None: continue
             df['label_int'] = df[label_col].map(label_to_int)
             unmapped_mask = df['label_int'].isnull()
             if unmapped_mask.any():
                 unmapped_labels = set(df.loc[unmapped_mask, label_col].unique())
                 print(f"Warning ({df_name}): Found labels not present in training data: {unmapped_labels}. Dropping {unmapped_mask.sum()} rows.")
                 df.dropna(subset=['label_int'], inplace=True)
             df[label_col] = df['label_int'].astype(int)
             df.drop(columns=['label_int'], inplace=True)
        save_label_mappings(label_to_int, int_to_label)

    if n_classes is None: raise ValueError("Could not determine the number of classes.")

    print(f"\nLabel preparation complete. Determined {n_classes} classes.")
    print(f"Final int_to_label mapping: {int_to_label}")

    df_train.dropna(subset=['label', 'text'], inplace=True)
    if df_val is not None: df_val.dropna(subset=['label', 'text'], inplace=True)
    if df_test is not None: df_test.dropna(subset=['label', 'text'], inplace=True)
    print(f"Final dataset sizes - Train: {len(df_train)}, Val: {len(df_val) if df_val is not None else 0}, Test: {len(df_test) if df_test is not None else 0}")
    return df_train, df_val, df_test, label_to_int, int_to_label, n_classes


# --- PyTorch Dataset and DataLoader (Simplified for Transformer) ---

class TransformerDataset(Dataset):
    """ Dataset class specifically for Transformer models. """
    def __init__(self, texts, labels, tokenizer, max_len=config.MAX_LEN):
        self.texts = texts     # List of cleaned text strings
        self.labels = labels   # List/array of integer labels
        self.tokenizer = tokenizer # HuggingFace tokenizer
        self.max_len = max_len

        if not isinstance(self.texts, list) or not isinstance(self.labels, (list, np.ndarray, pd.Series)):
             raise TypeError("Inputs 'texts' and 'labels' must be lists or similar sequence types.")
        if len(self.texts) != len(self.labels):
            raise ValueError(f"Length mismatch: texts ({len(self.texts)}) vs labels ({len(self.labels)}).")
        if self.tokenizer is None:
             raise ValueError("TransformerDataset requires a HuggingFace tokenizer.")

        # Ensure labels are array-like for easy indexing
        if isinstance(self.labels, pd.Series):
             self.labels = self.labels.values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = str(self.texts[index]) # Ensure text is string
        label_int = int(self.labels[index]) # Ensure label is integer
        label = torch.tensor(label_int, dtype=torch.long)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length', # Pad to max_len
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt', # Return PyTorch tensors
        )
        # Return dictionary required by the training/evaluation loop
        return {
            'input_ids': encoding['input_ids'].flatten(),       # Remove potential batch dim
            'attention_mask': encoding['attention_mask'].flatten(), # Remove potential batch dim
            'labels': label
        }


def create_dataloaders(train_data, val_data, test_data, # Input are Dataset objects
                       batch_size=config.TRAIN_BATCH_SIZE, val_batch_size=config.VALID_BATCH_SIZE):
    """Creates DataLoaders specifically for the TransformerDataset."""

    # Default collate_fn is suitable for dict-based datasets like TransformerDataset
    collate_fn = None
    num_workers = 0 # Keep 0 for broad compatibility

    # Handle None datasets gracefully
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True if config.DEVICE == "cuda" else False
    ) if train_data else None

    val_loader = DataLoader(
        val_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True if config.DEVICE == "cuda" else False
    ) if val_data else None

    test_loader = DataLoader(
        test_data, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True if config.DEVICE == "cuda" else False
    ) if test_data else None

    if train_loader:
        print(f"\nDataLoaders created (Batch Size: Train={batch_size}, Val/Test={val_batch_size}).")
    else:
        print("\nWarning: Training DataLoader could not be created (no training data?).")

    return train_loader, val_loader, test_loader

# --- Main Data Pipeline Function (Simplified) ---

def get_data_pipeline():
    """
    Orchestrates data loading, preprocessing, and DataLoader creation for Transformers.
    Returns:
        tuple: Contains:
            - train_loader (DataLoader or None)
            - val_loader (DataLoader or None)
            - test_loader (DataLoader or None)
            - label_to_int (dict)
            - int_to_label (dict)
            - n_classes (int)
            - tokenizer (AutoTokenizer)
    """
    print("--- Starting Data Pipeline (Transformer Focus) ---")

    # 1. Load Raw Data
    print("\n--- Loading Data ---")
    df_train = load_raw_data(filepath=config.TRAIN_FILE_PATH)
    if df_train is None or df_train.empty:
        raise FileNotFoundError(f"CRITICAL: Training data failed to load from {config.TRAIN_FILE_PATH}. Cannot proceed.")
    df_val = load_raw_data(filepath=config.VALID_FILE_PATH)
    df_test = load_raw_data(filepath=config.TEST_FILE_PATH)

    # 2. Split Data if Necessary
    df_train_processed = df_train.copy()
    df_val_processed = df_val.copy() if df_val is not None else None
    df_test_processed = df_test.copy() if df_test is not None else None

    if df_val_processed is None or df_test_processed is None:
        print("\n--- Splitting Training Data ---")
        # (Splitting logic remains the same as before - using train_test_split)
        df_to_split = df_train_processed
        if df_val_processed is None:
            print(f"Splitting validation set ({config.VALIDATION_SPLIT_SIZE*100:.1f}%)...")
            if len(df_to_split) < 2:
                 print("Warning: Not enough data for validation split. Val set empty.")
                 df_val_processed = pd.DataFrame(columns=df_to_split.columns)
                 df_train_intermediate = df_to_split
            else:
                 stratify_col_val = df_to_split['label'] if config.STRATIFY_SPLIT else None
                 try: df_train_intermediate, df_val_processed = train_test_split(df_to_split, test_size=config.VALIDATION_SPLIT_SIZE, random_state=config.SEED, stratify=stratify_col_val)
                 except ValueError as e: print(f"Warning: Stratified split failed ({e}). Using non-stratified."); df_train_intermediate, df_val_processed = train_test_split(df_to_split, test_size=config.VALIDATION_SPLIT_SIZE, random_state=config.SEED)
                 print(f"  Intermediate Train: {len(df_train_intermediate)}, Val: {len(df_val_processed)}")
        else: df_train_intermediate = df_to_split

        if df_test_processed is None:
             print(f"Splitting test set ({config.TEST_SPLIT_SIZE*100:.1f}% from original)...")
             if len(df_train_intermediate) < 2:
                  print("Warning: Not enough remaining data for test split. Test set empty.")
                  df_test_processed = pd.DataFrame(columns=df_train_intermediate.columns)
                  df_train_final = df_train_intermediate
             else:
                  # Calculate effective test split size from remaining data
                  current_train_fraction = len(df_train_intermediate) / len(df_to_split) if len(df_to_split) > 0 else 1.0
                  effective_split_size = config.TEST_SPLIT_SIZE / current_train_fraction if current_train_fraction > 0 else 0
                  effective_split_size = min(max(0.0, effective_split_size), 1.0 - (1/len(df_train_intermediate)) if len(df_train_intermediate)>1 else 0.0)

                  if effective_split_size <= 0:
                       print(f"Warning: Calculated test split size is non-positive. Test set empty.")
                       df_test_processed = pd.DataFrame(columns=df_train_intermediate.columns)
                       df_train_final = df_train_intermediate
                  else:
                       print(f"Splitting test set ({effective_split_size*100:.1f}% from remaining train)...")
                       stratify_col_test = df_train_intermediate['label'] if config.STRATIFY_SPLIT else None
                       try: df_train_final, df_test_processed = train_test_split(df_train_intermediate, test_size=effective_split_size, random_state=config.SEED, stratify=stratify_col_test)
                       except ValueError as e: print(f"Warning: Stratified split failed ({e}). Using non-stratified."); df_train_final, df_test_processed = train_test_split(df_train_intermediate, test_size=effective_split_size, random_state=config.SEED)
                       print(f"  Final Train: {len(df_train_final)}, Test: {len(df_test_processed)}")
        else: df_train_final = df_train_intermediate
        df_train_processed = df_train_final
        print("--- Data Splitting Finished ---")

    # 3. Handle Labels
    df_train_processed, df_val_processed, df_test_processed, \
    label_to_int, int_to_label, n_classes = prepare_data(
        df_train_processed, df_val_processed, df_test_processed
    )
    if df_train_processed.empty: raise ValueError("Training data is empty after processing.")
    if df_val_processed is not None and df_val_processed.empty: print("Warning: Validation data is empty after processing.")
    if df_test_processed is not None and df_test_processed.empty: print("Warning: Test data is empty after processing.")

    # 4. Initialize Preprocessor and Tokenizer
    print(f"\nInitializing preprocessor: {config.PREPROCESSOR_TYPE}")
    preprocessor = BasicTextCleaner() # Only basic cleaner needed

    print(f"Loading HuggingFace Tokenizer: {config.TRANSFORMER_MODEL_NAME}")
    try:
         tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
         print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
         print(f"Fatal Error: Failed to load Transformer tokenizer '{config.TRANSFORMER_MODEL_NAME}': {e}")
         sys.exit(1)

    # --- Removed Vocabulary Building/Loading Logic ---

    # 5. Preprocess Text Data (Apply Cleaning)
    print("\nApplying basic text cleaning to all datasets...")
    train_texts = preprocessor.preprocess_batch(df_train_processed['text'].tolist())
    val_texts = preprocessor.preprocess_batch(df_val_processed['text'].tolist()) if df_val_processed is not None else []
    test_texts = preprocessor.preprocess_batch(df_test_processed['text'].tolist()) if df_test_processed is not None else []
    print("Text cleaning complete.")

    # 6. Create PyTorch Datasets
    print("\nCreating PyTorch Datasets...")
    train_dataset = TransformerDataset(
        texts=train_texts, labels=df_train_processed['label'], # Pass Series/array directly
        tokenizer=tokenizer, max_len=config.MAX_LEN
    ) if not df_train_processed.empty else None

    val_dataset = TransformerDataset(
        texts=val_texts, labels=df_val_processed['label'],
        tokenizer=tokenizer, max_len=config.MAX_LEN
    ) if df_val_processed is not None and not df_val_processed.empty else None

    test_dataset = TransformerDataset(
        texts=test_texts, labels=df_test_processed['label'],
        tokenizer=tokenizer, max_len=config.MAX_LEN
    ) if df_test_processed is not None and not df_test_processed.empty else None

    # 7. Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        val_batch_size=config.VALID_BATCH_SIZE
    )

    print("\n--- Data Pipeline Finished ---")
    return train_loader, val_loader, test_loader, label_to_int, int_to_label, n_classes, tokenizer