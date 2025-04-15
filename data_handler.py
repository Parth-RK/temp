import warnings
warnings.filterwarnings("ignore")

import os
import spacy
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords as nltk_stopwords
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm
import config
import pandas.api.types as ptypes

PAD_IDX = config.PAD_IDX
UNK_IDX = config.UNK_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX

SPACY_MODEL = config.SPACY_MODEL

class Vocabulary:
    def __init__(self, freq_threshold, max_size=None):
        # Initial vocabulary with special tokens
        self.itos = {PAD_IDX: config.PAD_TOKEN, UNK_IDX: config.UNK_TOKEN,
                     SOS_IDX: config.SOS_TOKEN, EOS_IDX: config.EOS_TOKEN}
        # String-to-integer mapping
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        print("Building vocabulary...")
        frequencies = Counter()
        # Start index for new words after special tokens
        idx = len(self.itos)

        # Count frequencies of all tokens in the preprocessed sentences
        for sentence in tqdm(sentence_list, desc="Counting Frequencies"):
            frequencies.update(sentence)

        # Limit vocabulary size if max_size is set
        if self.max_size is not None:
            # Get the most common words up to max_size, excluding special tokens already present
            limited_freq = frequencies.most_common(self.max_size - len(self.itos))
            frequencies = Counter(dict(limited_freq))

        # Create stoi and itos mappings for words meeting the frequency threshold
        for word, freq in tqdm(frequencies.items(), desc="Creating Mappings"):
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocabulary built. Size: {len(self.itos)}")

    def numericalize(self, text_tokens):
        # Convert a list of tokens to their corresponding integer indices
        # Use UNK_IDX if a token is not found in the vocabulary
        return [self.stoi.get(token, UNK_IDX) for token in text_tokens]

    def save(self, filepath, n_class):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Prepare data for saving (stoi map, threshold, number of classes)
        save_data = {
            'stoi': self.stoi,
            'freq_threshold': self.freq_threshold,
            'n_class': n_class # Include n_class in the saved vocab file
        }
        # Write to JSON file
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        print(f"Vocabulary (stoi) and n_class saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        # Check if the vocabulary file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        # Load data from JSON file
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)

        stoi_loaded = loaded_data['stoi']
        freq_threshold = loaded_data.get('freq_threshold', config.MIN_FREQ) # Default if not found
        n_class = loaded_data.get('n_class') # Get n_class
        if n_class is None:
             # If n_class isn't in the vocab file, it's an error for model setup
             raise ValueError("Number of classes (n_class) not found in vocabulary file.")

        # Create a new Vocabulary instance
        vocab = cls(freq_threshold)

        # Rebuild itos from loaded stoi, ensuring keys are integers
        itos_rebuilt = {}
        stoi_rebuilt = {}
        # Ensure special tokens are preserved with correct indices if they exist in the loaded stoi
        special_tokens = {config.PAD_TOKEN, config.UNK_TOKEN, config.SOS_TOKEN, config.EOS_TOKEN}
        for token, index_str in stoi_loaded.items():
            # Check if the token is a special token and handle potential index conflicts if needed
            # (Assuming standard indices 0, 1, 2, 3 were used during saving)
            index = int(index_str) # Convert index from string (JSON key) to int
            itos_rebuilt[index] = token
            stoi_rebuilt[token] = index

        # Assign the rebuilt mappings to the vocab instance
        vocab.itos = itos_rebuilt
        vocab.stoi = stoi_rebuilt

        print(f"Vocabulary loaded from {filepath}. Size: {len(vocab.itos)}, n_class: {n_class}")
        return vocab, n_class # Return loaded vocab and n_class

class TextPreprocessor:
    def __init__(self, use_stopwords=False):
        self.nlp = None
        # Use NLTK's stopwords list if requested
        self.stopwords = set(nltk_stopwords.words('english')) if use_stopwords else set()
        self._lazy_load_spacy()
        print(f"TextPreprocessor initialized. Stopwords {'enabled' if use_stopwords else 'disabled'}.")
        print("Dependency parser is ENABLED for negation handling.")

    def _lazy_load_spacy(self):
        # Load the spaCy model only when needed
        if self.nlp is None:
            print(f"Loading spaCy model '{SPACY_MODEL}'...")
            try:
                # Load the model *without* disabling the parser
                self.nlp = spacy.load(SPACY_MODEL, disable=["ner"])
            except OSError:
                # If model not found, download and then load it
                print(f"Spacy model '{SPACY_MODEL}' not found. Downloading...")
                spacy.cli.download(SPACY_MODEL)
                self.nlp = spacy.load(SPACY_MODEL, disable=["ner"])
            print("spaCy model loaded (with parser).")

    def clean_and_tokenize(self, text):
        # Convert input to string and lowercase
        text = str(text).lower()
        # Process text with spaCy
        doc = self.nlp(text)
        tokens = []
        negated_indices = set()

        # --- Negation Handling using Dependency Parsing ---
        # First pass: Identify negations and the indices of the words they modify
        for token in doc:
            # Check if the token is a negation dependency
            if token.dep_ == 'neg':
                head = token.head
                # Add the index of the word being negated (the head of the negation)
                negated_indices.add(head.i)
                # Potential Enhancement: Could check children of head for multi-word negations
                # e.g., "not very happy" -> negates "happy" (head) or maybe "very"?
                # Simple approach: just negate the direct head.

        # --- Tokenization and Lemmatization ---
        # Second pass: Build the final token list
        for token in doc:
            # Check if the current token's index was marked as negated
            is_negated = token.i in negated_indices

            # Filter out stopwords, punctuation, spaces, and optionally custom stopwords
            if (not token.is_stop and        # spaCy's default stop words
                not token.is_punct and       # Punctuation
                not token.is_space and       # Whitespace tokens
                token.lemma_ not in self.stopwords): # Custom NLTK stopwords (if enabled)

                lemma = token.lemma_ # Get the base form of the word
                # Append "_NEG" suffix if the word is negated
                if is_negated:
                    lemma += "_NEG"
                tokens.append(lemma)

        return tokens

    def preprocess_dataframe(self, df, text_column='text'):
        # Ensure the specified text column exists
        if text_column not in df.columns:
             raise ValueError(f"Input DataFrame must contain a '{text_column}' column.")
        # Fill missing values in the text column with empty strings
        df[text_column] = df[text_column].fillna('')

        print(f"Preprocessing DataFrame column '{text_column}'...")
        # Apply the enhanced clean_and_tokenize function to each text entry
        processed_texts = [self.clean_and_tokenize(text) for text in tqdm(df[text_column], desc="Processing Texts")]
        print("Preprocessing Done!")
        return processed_texts

class EmotionDataset(Dataset):
    # Simple PyTorch Dataset wrapper
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        # Basic check for consistent lengths
        if len(self.sequences) != len(self.labels):
             raise ValueError("Sequences and labels must have the same length!")

    def __len__(self):
        # Return the total number of samples
        return len(self.labels)

    def __getitem__(self, idx):
        # Get sequence and label at the given index
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

def collate_batch(batch):
    # Custom function to process a batch of data points from the Dataset
    label_list, text_list, lengths = [], [], []
    # Iterate through samples in the batch
    for (_text, _label) in batch:
        label_list.append(_label) # Collect labels
        processed_text = torch.tensor(_text, dtype=torch.long) # Ensure text is a tensor
        text_list.append(processed_text) # Collect text tensors
        lengths.append(len(processed_text)) # Store original sequence length (optional)

    # Pad sequences within the batch to the same length
    # `pad_sequence` stacks tensors and pads with `padding_value`
    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    # Stack labels into a single tensor
    labels = torch.stack(label_list)
    # Return padded texts and corresponding labels
    return padded_texts, labels

# --- Label Handling ---

def create_label_mappings(train_df, label_column='label'):
    # Find unique string labels and sort them
    unique_labels = sorted(train_df[label_column].astype(str).unique())
    # Create mapping from label string to integer index
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    # Create reverse mapping from integer index to label string
    int_to_label = {i: label for label, i in label_to_int.items()}
    print(f"Created mappings for {len(unique_labels)} unique string labels: {unique_labels}")
    return label_to_int, int_to_label

# This function might not be strictly necessary if labels are already 0-indexed integers
def create_placeholder_mappings(train_df, label_column='label'):
    # For integer labels, create a simple placeholder mapping for consistency if needed
    unique_labels = sorted(train_df[label_column].unique())
    # Map integer i to string "label_i"
    int_to_label = {i: f"label_{i}" for i in unique_labels}
    # Reverse mapping
    label_to_int = {v: k for k, v in int_to_label.items()}
    print(f"Using existing integer labels. Created placeholder mappings for {len(unique_labels)} labels.")
    print(f"Placeholder int_to_label map: {int_to_label}")
    return label_to_int, int_to_label

def to_native(obj):
    # Recursively convert NumPy/PyTorch types to native Python types for JSON serialization
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    # Handle PyTorch tensors/scalars
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except ValueError: # Handle cases like multi-element tensors if they sneak in
             return str(obj)
    # Handle NumPy types explicitly
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.str_):
        return str(obj)
    # Handle boolean explicitly for clarity
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def save_label_mappings(mappings, filepath):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    label_to_int, int_to_label = mappings
    # Convert mappings to native Python types for JSON
    label_to_int_native = to_native(label_to_int)
    int_to_label_native = to_native(int_to_label)

    # Convert keys to strings for JSON compatibility
    # (int_to_label keys are integers, label_to_int keys might be strings or ints)
    label_to_int_str_keys = {str(k): v for k, v in label_to_int_native.items()}
    int_to_label_str_keys = {str(k): v for k, v in int_to_label_native.items()}

    # Prepare data structure for saving
    save_data = {
        'label_to_int': label_to_int_str_keys,
        'int_to_label': int_to_label_str_keys
    }
    # Write mappings to JSON file
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"Label mappings saved to {filepath}")

def load_label_mappings(filepath):
    # Check if the mapping file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Label mapping file not found at {filepath}")
    # Load mappings from JSON file
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    # Reconstruct mappings, ensuring correct types
    # label_to_int keys are likely strings (original labels), values are ints
    label_to_int_loaded = loaded_data['label_to_int']
    # int_to_label keys need to be converted back to integers
    int_to_label_loaded = {int(k): v for k, v in loaded_data['int_to_label'].items()}

    mappings = (label_to_int_loaded, int_to_label_loaded)
    print(f"Label mappings loaded from {filepath}. Num classes: {len(mappings[1])}")
    return mappings

# --- Main Data Loading Function ---

def load_and_prepare_data(train_path, val_path, test_path, label_map_save_path):
    try:
        # Load datasets using pandas
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Raw data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

        # --- Basic Validation and Cleaning ---
        label_column = 'label' # Define label column name
        for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            # Check for required columns
            if 'text' not in df.columns or label_column not in df.columns:
                raise ValueError(f"{df_name} DataFrame is missing 'text' or '{label_column}' column.")
            # Check for and handle NaN values in the label column
            if df[label_column].isnull().any():
                print(f"Warning: Found NaN values in '{label_column}' of {df_name} data. Dropping rows.")
                df.dropna(subset=[label_column], inplace=True)
            # Ensure text column is string type
            if 'text' in df.columns:
                 df['text'] = df['text'].astype(str)


        # --- Label Type Handling and Mapping ---
        label_to_int, int_to_label = None, None
        n_class = train_df[label_column].nunique() # Calculate initial number of unique labels

        # Case 1: Labels are already integers
        if ptypes.is_integer_dtype(train_df[label_column]):
            print(f"Detected integer labels in '{label_column}' column. Using them directly. n_class={n_class}")
            # Optional: Check if labels are contiguous from 0. If not, remapping might be needed.
            # For now, assume they are valid indices [0, n_class-1].
            # Verify other sets also have integer labels or attempt conversion
            for df_name, df in [('Validation', val_df), ('Test', test_df)]:
                 if not ptypes.is_integer_dtype(df[label_column]):
                      try:
                           # Attempt conversion if labels are numeric-like strings etc.
                           df[label_column] = df[label_column].astype(int)
                           print(f"Converted '{label_column}' in {df_name} to integer.")
                      except (ValueError, TypeError):
                           # If conversion fails, raise error
                           raise TypeError(f"Training labels are integers, but {df_name} labels in column '{label_column}' are not and cannot be converted to integer.")
            # No label map file is saved for purely numeric labels by default.
            # We still need int_to_label for potential prediction interpretation later.
            # Create a placeholder if no map exists.
            if not os.path.exists(label_map_save_path):
                print(f"Creating placeholder label map for integer labels at {label_map_save_path}")
                _, int_to_label = create_placeholder_mappings(train_df, label_column)
                # Save this placeholder map
                save_label_mappings( ({v:k for k,v in int_to_label.items()}, int_to_label), label_map_save_path)


        # Case 2: Labels are strings or objects (likely strings)
        elif ptypes.is_string_dtype(train_df[label_column]) or ptypes.is_object_dtype(train_df[label_column]):
            print(f"Detected string/object labels in '{label_column}' column. Creating mappings. n_class={n_class}")
            # Create mappings based on unique training labels
            label_to_int, int_to_label = create_label_mappings(train_df, label_column)
            n_class = len(label_to_int) # Update n_class based on actual unique labels found

            print("Mapping string labels to integers for all datasets...")
            # Apply mapping to all data splits
            for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
                original_labels = set(df[label_column].unique()) # Store original labels before mapping
                # Map string labels to integers using the created dictionary
                df[label_column] = df[label_column].map(label_to_int)
                # Check if any labels in val/test were not found in the training map (resulting in NaN)
                if df[label_column].isnull().any():
                    unmapped_labels = original_labels - set(label_to_int.keys())
                    print(f"Warning: Found labels in {df_name} set not present in training data mapping: {unmapped_labels}. Dropping rows with these unmappable labels.")
                    df.dropna(subset=[label_column], inplace=True) # Remove rows with unmapped labels
                # Convert the label column to integer type after mapping
                df[label_column] = df[label_column].astype(int)

            # Save the created mappings to a file
            save_label_mappings((label_to_int, int_to_label), label_map_save_path)

        # Case 3: Unsupported label type
        else:
             raise TypeError(f"Unsupported label type '{train_df[label_column].dtype}' in column '{label_column}'. Labels must be integers or strings.")

        print(f"Labels processed. '{label_column}' column now contains integer indices.")
        print(f"Final determined number of classes (n_class): {n_class}")

        # Return processed dataframes, mappings, and the final number of classes
        return train_df, val_df, test_df, label_to_int, int_to_label, n_class

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Check file paths in config.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preparation: {e}")
        import traceback
        traceback.print_exc()
        raise