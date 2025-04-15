# data_handler.py
"""
Handles data loading, cleaning, preprocessing, and Dataset creation.
(TorchText Legacy Independent Version)
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
import config # Import config for special tokens/indices

# --- Define fixed indices from config ---
PAD_IDX = config.PAD_IDX
UNK_IDX = config.UNK_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX

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
        idx = len(self.itos) # Start indexing after special tokens

        for sentence in tqdm(sentence_list, desc="Counting Frequencies"):
            frequencies.update(sentence)

        # Limit vocab size if max_size is set
        if self.max_size is not None:
            most_common = frequencies.most_common(self.max_size - len(self.itos)) # Exclude special tokens from count limit
            frequencies = Counter(dict(most_common)) # Keep only most common words above threshold

        # Create stoi and itos
        for word, freq in tqdm(frequencies.items(), desc="Creating Mappings"):
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Vocabulary built. Size: {len(self.itos)}")

    def numericalize(self, text_tokens):
        # Input should be a list of tokens for a single sentence
        return [self.stoi.get(token, UNK_IDX) for token in text_tokens]

    def save(self, filepath):
        """Saves stoi dictionary to JSON."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Need to save freq_threshold too if needed for loading
        save_data = {'stoi': self.stoi, 'freq_threshold': self.freq_threshold}
        with open(filepath, 'w') as f:
            json.dump(save_data, f)
        print(f"Vocabulary (stoi) saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads stoi dictionary from JSON and creates Vocabulary object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        with open(filepath, 'r') as f:
            loaded_data = json.load(f)
        stoi_loaded = loaded_data['stoi']
        freq_threshold = loaded_data.get('freq_threshold', 2) # Default if missing

        # Reconstruct Vocabulary object
        vocab = cls(freq_threshold) # Use loaded threshold
        vocab.stoi = stoi_loaded
        # Rebuild itos from loaded stoi
        vocab.itos = {v: k for k, v in stoi_loaded.items()}
        print(f"Vocabulary loaded from {filepath}. Size: {len(vocab.itos)}")
        return vocab


class TextPreprocessor:
    def __init__(self, use_stopwords=False):
        self.nlp = None
        self.stopwords = set(nltk_stopwords.words('english')) if use_stopwords else set()
        print(f"Stopwords {'enabled' if use_stopwords else 'disabled'}.")

    def _lazy_load_spacy(self):
        """Loads spacy model only when needed."""
        if self.nlp is None:
            print("Loading spaCy model 'en_core_web_sm'...")
            try:
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"]) # Faster loading
            except OSError:
                print("Spacy model 'en_core_web_sm' not found. Downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            print("spaCy model loaded.")

    def clean_and_tokenize(self, text):
        """Cleans and tokenizes a single text string."""
        self._lazy_load_spacy()
        text = str(text).lower() # Lowercase
        # Use spaCy's tokenizer and lemmatizer efficiently
        doc = self.nlp(text)
        tokens = [
            token.lemma_ # Lemmatize
            for token in doc
            if not token.is_stop and # Use spaCy's stopword flag if not using nltk list
               not token.is_punct and
               not token.is_space and
               token.lemma_ not in self.stopwords # Filter nltk stopwords if enabled
        ]
        return tokens

    def preprocess_dataframe(self, df):
        """Applies cleaning and tokenization to a DataFrame text column."""
        if 'text' not in df.columns:
             raise ValueError("Input DataFrame must contain a 'text' column.")
        df['text'] = df['text'].fillna('') # Handle NaNs

        print("Preprocessing DataFrame (cleaning, tokenizing, lemmatizing)...")
        # Apply the combined function
        # Consider using pandarallel or multiprocessing for large dataframes
        processed_texts = [self.clean_and_tokenize(text) for text in tqdm(df['text'], desc="Processing Texts")]
        print("Preprocessing Done!")
        return processed_texts


# --- PyTorch Dataset ---
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
        label = torch.tensor(self.labels[idx], dtype=torch.long) # Labels should be Long
        return sequence, label

# --- Collate Function (Handles Padding) ---
def collate_batch(batch):
    """Collate function to pad sequences in a batch."""
    label_list, text_list, lengths = [], [], []
    for (_text, _label) in batch:
        label_list.append(_label)
        processed_text = torch.tensor(_text, dtype=torch.long)
        text_list.append(processed_text)
        lengths.append(len(processed_text)) # Store original lengths if needed (e.g., for PackedSequence)

    # Pad sequences to the max length in the batch
    padded_texts = nn.utils.rnn.pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    labels = torch.stack(label_list) # Stack labels into a tensor
    lengths = torch.tensor(lengths, dtype=torch.long)

    return padded_texts, labels # Removed lengths return for simplicity, add back if using PackedSequence


# --- Helper function to load data ---
def load_data(train_path, val_path, test_path):
    """Loads train, validation, and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)
        print("Data loaded successfully.")
        print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")
        # Basic check for required columns
        for df_name, df in [('Train', train_data), ('Validation', val_data), ('Test', test_data)]:
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError(f"{df_name} DataFrame is missing required 'text' or 'label' column.")
        return train_data, val_data, test_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check file paths in config.py.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        raise