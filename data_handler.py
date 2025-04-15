# data_handler.py
"""
Handles data loading, cleaning, preprocessing, and Dataset creation.
Determines label mappings dynamically from training data.
"""
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
        vocab.itos = {int(v): k for k, v in stoi_loaded.items()} # Ensure keys are int if loading from JSON
        print(f"Vocabulary loaded from {filepath}. Size: {len(vocab.itos)}")
        return vocab

class TextPreprocessor:
    def __init__(self, use_stopwords=False):
        self.nlp = None
        self.stopwords = set(nltk_stopwords.words('english')) if use_stopwords else set()
        self._lazy_load_spacy() # Load spacy once on init
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
    # lengths = torch.tensor(lengths, dtype=torch.long) # Uncomment if using packed sequences

    return padded_texts, labels # Return only padded texts and labels

def create_label_mappings(train_df, label_column='label'):
    """Determines unique labels and creates mappings."""
    unique_labels = sorted(train_df[label_column].unique())
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_to_label = {i: label for label, i in label_to_int.items()}
    print(f"Found {len(unique_labels)} unique labels: {unique_labels}")
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
        except Exception:
            return obj
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def save_label_mappings(mappings, filepath):
    """Saves label mappings ensuring all keys are strings for JSON compatibility."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    label_to_int, int_to_label = mappings
    label_to_int_native = to_native(label_to_int)
    int_to_label_native = to_native(int_to_label)
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
    
    label_to_int_loaded = loaded_data['label_to_int']
    int_to_label_loaded = {int(k): v for k, v in loaded_data['int_to_label'].items()}

    mappings = (label_to_int_loaded, int_to_label_loaded)
    print(f"Label mappings loaded from {filepath}. Num classes: {len(mappings[0])}")
    return mappings

def load_and_prepare_data(train_path, val_path, test_path, label_map_save_path):
    """Loads data, creates label mappings, maps labels, and saves mappings."""
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        print("Raw data loaded successfully.")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")

        for df_name, df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError(f"{df_name} DataFrame is missing 'text' or 'label' column.")

        # Create mappings from training data
        label_to_int, int_to_label = create_label_mappings(train_df, 'label')
        save_label_mappings((label_to_int, int_to_label), label_map_save_path)

        # Map string labels to integers in all dataframes
        for df in [train_df, val_df, test_df]:
            df['label'] = df['label'].map(label_to_int)
            # Optional: Handle labels present in val/test but not train?
            if df['label'].isnull().any():
                 print(f"Warning: Found labels in validation/test set not present in training data. Mapping to NaN.")
                 # Decide handling: dropna(), fillna(-1), or raise error
                 # df.dropna(subset=['label'], inplace=True)

        print("Labels mapped to integers.")
        return train_df, val_df, test_df, label_to_int, int_to_label

    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Check file paths in config.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading/preparation: {e}")
        raise