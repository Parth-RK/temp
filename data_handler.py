# data_handler.py
"""
Handles data loading, cleaning, preprocessing, and Dataset creation.
Determines label mappings dynamically from training data, handling pre-integerized labels.
"""
import warnings
warnings.filterwarnings("ignore") # Keep warnings suppressed

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

# --- Define fixed indices from config ---
PAD_IDX = config.PAD_IDX
UNK_IDX = config.UNK_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX

SPACY_MODEL = config.SPACY_MODEL

class Vocabulary:
    """Manual Vocabulary Class"""
    def __init__(self, freq_threshold, max_size=None):
        self.itos = {PAD_IDX: config.PAD_TOKEN, UNK_IDX: config.UNK_TOKEN,
                     SOS_IDX: config.SOS_TOKEN, EOS_IDX: config.EOS_TOKEN}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.max_size = max_size

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        print("Building vocabulary...")
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in tqdm(sentence_list, desc="Counting Frequencies"):
            frequencies.update(sentence)

        if self.max_size is not None:
            limited_freq = frequencies.most_common(self.max_size - len(self.itos))
            frequencies = Counter(dict(limited_freq))

        for word, freq in tqdm(frequencies.items(), desc="Creating Mappings"):
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocabulary built. Size: {len(self.itos)}")

    def numericalize(self, text_tokens):
        return [self.stoi.get(token, UNK_IDX) for token in text_tokens]

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_data = {'stoi': self.stoi, 'freq_threshold': self.freq_threshold}
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        print(f"Vocabulary (stoi) saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        stoi_loaded = loaded_data['stoi']
        freq_threshold = loaded_data.get('freq_threshold', config.MIN_FREQ)

        vocab = cls(freq_threshold)
        vocab.stoi = stoi_loaded
        # Convert loaded string keys back to int for itos
        vocab.itos = {int(k): v for k, v in vocab.stoi.items() if k.isdigit()}
        # Handle non-digit keys if any (shouldn't happen with current save)
        vocab.itos.update({k:v for k, v in vocab.stoi.items() if not k.isdigit()})
        # Rebuild stoi with int keys from itos
        vocab.stoi = {v: k for k, v in vocab.itos.items()}

        print(f"Vocabulary loaded from {filepath}. Size: {len(vocab.itos)}")
        return vocab


class TextPreprocessor:
    def __init__(self, use_stopwords=False):
        self.nlp = None
        self.stopwords = set(nltk_stopwords.words('english')) if use_stopwords else set()
        self._lazy_load_spacy()
        print(f"TextPreprocessor initialized. Stopwords {'enabled' if use_stopwords else 'disabled'}.")

    def _lazy_load_spacy(self):
        if self.nlp is None:
            print(f"Loading spaCy model '{SPACY_MODEL}'...")
            try:
                self.nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
            except OSError:
                print(f"Spacy model '{SPACY_MODEL}' not found. Downloading...")
                spacy.cli.download(SPACY_MODEL)
                self.nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
            print("spaCy model loaded.")

    def clean_and_tokenize(self, text):
        text = str(text).lower()
        doc = self.nlp(text)
        tokens = [
            token.lemma_ for token in doc
            if not token.is_stop and
               not token.is_punct and
               not token.is_space and
               token.lemma_ not in self.stopwords
        ]
        return tokens

    def preprocess_dataframe(self, df, text_column='text'):
        if text_column not in df.columns:
             raise ValueError(f"Input DataFrame must contain a '{text_column}' column.")
        df[text_column] = df[text_column].fillna('')

        print(f"Preprocessing DataFrame column '{text_column}'...")
        processed_texts = [self.clean_and_tokenize(text) for text in tqdm(df[text_column], desc="Processing Texts")]
        print("Preprocessing Done!")
        return processed_texts


class EmotionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        if len(self.sequences) != len(self.labels):
             raise ValueError("Sequences and labels must have the same length!")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label


def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.long)
        text_list.append(processed_text)
        lengths.append(len(processed_text))

    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(label_list)
    return padded_texts, labels


def create_label_mappings(train_df, label_column='label'):
    """Creates label mappings FROM STRING LABELS."""
    unique_labels = sorted(train_df[label_column].astype(str).unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    print(f"Created mappings for {len(unique_labels)} unique string labels: {unique_labels}")
    return label_to_int, int_to_label

def create_placeholder_mappings(train_df, label_column='label'):
    """Creates placeholder mappings FROM INTEGER LABELS."""
    unique_labels = sorted(train_df[label_column].unique())
    # Ensure they are contiguous from 0, warn if not? For now, assume they are valid indices.
    int_to_label = {i: f"label_{i}" for i in unique_labels}
    label_to_int = {v: k for k, v in int_to_label.items()} # Maps "label_0" -> 0 etc. Less useful.
    print(f"Using existing integer labels. Created placeholder mappings for {len(unique_labels)} labels.")
    print(f"Placeholder int_to_label map: {int_to_label}")
    return label_to_int, int_to_label


def to_native(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_native(i) for i in obj]
    elif hasattr(obj, 'item') and callable(obj.item):
        try:
            return obj.item()
        except ValueError: # Handle cases like np.str_ -> str
             return str(obj)
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.str_):
        return str(obj)
    else:
        return obj

def save_label_mappings(mappings, filepath):
    """Saves label mappings ensuring native types and string keys for JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    label_to_int, int_to_label = mappings
    # Convert all complex types (like numpy types) to native Python types
    label_to_int_native = to_native(label_to_int)
    int_to_label_native = to_native(int_to_label)

    # Ensure keys are strings for JSON compatibility
    label_to_int_str_keys = {str(k): v for k, v in label_to_int_native.items()}
    int_to_label_str_keys = {str(k): v for k, v in int_to_label_native.items()}

    save_data = {
        'label_to_int': label_to_int_str_keys,
        'int_to_label': int_to_label_str_keys
    }
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"Label mappings saved to {filepath}")


def load_label_mappings(filepath):
    """Loads label mappings from JSON, converting int_to_label keys back to integers."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Label mapping file not found at {filepath}")
    with open(filepath, 'r') as f:
        loaded_data = json.load(f)

    label_to_int_loaded = loaded_data['label_to_int'] # Keys remain strings ('sadness', 'label_0')
    # Convert integer-like keys in int_to_label back to integers
    int_to_label_loaded = {int(k): v for k, v in loaded_data['int_to_label'].items()}

    mappings = (label_to_int_loaded, int_to_label_loaded)
    print(f"Label mappings loaded from {filepath}. Num classes: {len(mappings[1])}")
    return mappings


def load_and_prepare_data(train_path, val_path, test_path, label_map_save_path):
    """
    Loads data, checks label type, creates/uses appropriate mappings,
    maps labels if needed, and saves mappings.
    Ensures 'label' column in returned DataFrames contains integers.
    """
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Raw data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

        for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError(f"{df_name} DataFrame is missing 'text' or 'label' column.")
            # Handle potential NaN labels early
            if df['label'].isnull().any():
                print(f"Warning: Found NaN values in '{label_column}' of {df_name} data. Dropping rows.")
                df.dropna(subset=['label'], inplace=True)


        # --- Dynamic Label Handling ---
        label_column = 'label'
        # Check the data type of the training labels
        if ptypes.is_integer_dtype(train_df[label_column]):
            print(f"Detected integer labels in '{label_column}' column.")
            # Labels are already integers, create placeholder mappings
            label_to_int, int_to_label = create_placeholder_mappings(train_df, label_column)
            # Ensure labels in val/test are also integers (or convert if possible)
            for df_name, df in [('Validation', val_df), ('Test', test_df)]:
                 if not ptypes.is_integer_dtype(df[label_column]):
                      try:
                           # Attempt conversion if they look like integers (e.g., float 1.0)
                           df[label_column] = df[label_column].astype(int)
                           print(f"Converted '{label_column}' in {df_name} to integer.")
                      except ValueError:
                           raise TypeError(f"Training labels are integers, but {df_name} labels in column '{label_column}' are not and cannot be converted.")

        elif ptypes.is_string_dtype(train_df[label_column]) or ptypes.is_object_dtype(train_df[label_column]):
            print(f"Detected string/object labels in '{label_column}' column. Creating mappings.")
            # Labels are strings, create mappings from training data
            label_to_int, int_to_label = create_label_mappings(train_df, label_column)

            # Map string labels to integers in all dataframes
            print("Mapping string labels to integers...")
            for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
                original_labels = set(df[label_column].unique())
                df[label_column] = df[label_column].map(label_to_int)
                # Check for labels present in val/test but not train
                if df[label_column].isnull().any():
                    unmapped_labels = original_labels - set(label_to_int.keys())
                    print(f"Warning: Found labels in {df_name} set not present in training data: {unmapped_labels}. Dropping rows with these labels.")
                    df.dropna(subset=[label_column], inplace=True)
                # Convert to integer type after mapping
                df[label_column] = df[label_column].astype(int)


        else:
            # Handle other unexpected types (e.g., float)
             raise TypeError(f"Unsupported label type '{train_df[label_column].dtype}' in column '{label_column}'. Labels must be integers or strings.")

        # Save the determined mappings
        save_label_mappings((label_to_int, int_to_label), label_map_save_path)

        print("Labels processed. 'label' column now contains integers.")
        return train_df, val_df, test_df, label_to_int, int_to_label

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Check file paths in config.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preparation: {e}")
        import traceback
        traceback.print_exc()
        raise