# data_handler.py
"""
Handles data loading, cleaning, preprocessing, and DataLoader creation.
"""
import warnings
warnings.filterwarnings("ignore")

import os
import spacy
import torch
import pandas as pd
import numpy as np
import json
from nltk.corpus import stopwords
from torchtext import vocab
from torch.utils.data import DataLoader, TensorDataset
from pandarallel import pandarallel

# Initialize pandarallel (if main process)
if __name__ != "__main__": # Avoid initializing when importing
     try:
         pandarallel.initialize(progress_bar=True, verbose=0)
     except SystemError:
         print("Pandarallel initialization skipped (likely in unsupported environment or already initialized).")

class Preprocessor:
    def __init__(
        self,
        max_length,
        min_freq,
        sos_token,
        eos_token,
        unk_token,
        pad_token,
        use_stopwords=False # Flag to control stopword usage
    ):
        self.max_length = max_length
        self.min_freq = min_freq
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.use_stopwords = use_stopwords
        self.special_tokens = [unk_token, pad_token, sos_token, eos_token]
        self.vocab = None
        self.nlp = None
        self.stopwords = set(stopwords.words('english')) if use_stopwords else set()
        print(f"Stopwords {'enabled' if use_stopwords else 'disabled'}.")

    def _lazy_load_spacy(self):
        """Loads spacy model only when needed."""
        if self.nlp is None:
            print("Loading spaCy model 'en_core_web_sm'...")
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Spacy model 'en_core_web_sm' not found. Downloading...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded.")

    def clean(self, data):
        self._lazy_load_spacy()

        # Ensure 'text' column exists and handle potential NaN values
        if 'text' not in data.columns:
             raise ValueError("Input DataFrame must contain a 'text' column.")
        data['text'] = data['text'].fillna('') # Replace NaN with empty string

        # Define lemmatization function
        def lemmatize(text):
            # Process only non-empty strings
            if not isinstance(text, str) or not text.strip():
                return ""
            doc = self.nlp(text)
            return " ".join(
                token.lemma_
                for token in doc
                if token.lemma_ not in self.stopwords and not token.is_punct and not token.is_space
            )

        print("Lemmatizing and cleaning text...")
        # Use apply instead of parallel_apply if issues arise or pandarallel not available
        try:
             data["clean_text"] = data["text"].parallel_apply(lemmatize).str.lower()
        except Exception as e:
             print(f"Pandarallel failed ({e}), falling back to standard apply...")
             data["clean_text"] = data["text"].apply(lemmatize).str.lower()
        print("Lemmatizing and cleaning Done!")
        return data

    def tokenize(self, text):
        self._lazy_load_spacy()
        # Ensure input is a string
        text = str(text) if text is not None else ""
        # Truncate tokens to max_length
        tokens = [token.text for token in self.nlp.tokenizer(text)][:self.max_length]
        # Add special tokens
        en_tokens = [self.sos_token] + tokens + [self.eos_token]
        return en_tokens

    def build_vocab(self, token_iterator):
        print("Building vocabulary...")
        self.vocab = vocab.build_vocab_from_iterator(
            token_iterator,
            min_freq=self.min_freq,
            specials=self.special_tokens
        )
        unk_index = self.vocab[self.unk_token]
        self.vocab.set_default_index(unk_index)
        print(f"Vocabulary built. Size: {len(self.vocab)}")

    def convert_numerical(self, tokens):
        if self.vocab is None:
            raise ValueError("Vocabulary not built or loaded. Call fit() or load_vocab() first.")
        return self.vocab.lookup_indices(tokens)

    def pad_sequences(self, sequences):
        if self.vocab is None:
            raise ValueError("Vocabulary not built or loaded.")
        pad_index = self.vocab[self.pad_token]
        max_seq_len = self.max_length + 2 # Account for SOS and EOS tokens

        padded_sequences = [
            seq[:max_seq_len] + [pad_index] * max(0, max_seq_len - len(seq))
            for seq in sequences
        ]
        return padded_sequences

    def fit(self, data):
        """Cleans data, tokenizes, and builds the vocabulary."""
        if not isinstance(data, pd.DataFrame):
             raise TypeError("Input data for fit() must be a pandas DataFrame.")
        data = self.clean(data)
        print("Tokenizing training data for vocabulary building...")
        # Apply tokenization row-wise
        data["tokens"] = data["clean_text"].apply(self.tokenize)
        print("Tokenization Done!")
        self.build_vocab(data["tokens"]) # Build vocab from token lists

    def transform(self, data, batch_size, shuffle, return_type=torch.long):
        """Cleans, tokenizes, converts to numerical, pads, and creates DataLoader."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built or loaded. Call fit() or load_vocab() first.")
        if not isinstance(data, pd.DataFrame):
             raise TypeError("Input data for transform() must be a pandas DataFrame.")

        if "clean_text" not in data.columns:
            print("Cleaning data for transformation...")
            data = self.clean(data)
        if "tokens" not in data.columns:
            print("Tokenizing data for transformation...")
            data["tokens"] = data["clean_text"].apply(self.tokenize)

        print("Converting tokens to numerical indices...")
        # Handle potential errors during numerical conversion if tokens are unusual
        try:
             data["numerical_tokens"] = data["tokens"].apply(self.convert_numerical)
        except Exception as e:
             print(f"Error during numerical conversion: {e}. Please check tokenization.")
             # Optionally: skip problematic rows or raise error
             # For now, let's see the problematic tokens
             for i, tokens in enumerate(data['tokens']):
                 try:
                     self.convert_numerical(tokens)
                 except Exception as inner_e:
                     print(f"Problematic tokens at index {i}: {tokens} -> {inner_e}")
             raise e # Re-raise the error after printing details


        print("Padding sequences...")
        padded_sequences = self.pad_sequences(data["numerical_tokens"].to_list())

        X = np.array(padded_sequences)
        # Ensure 'label' column exists and handle potential NaN or incorrect types
        if 'label' not in data.columns:
             raise ValueError("Input DataFrame for transform() must contain a 'label' column.")
        try:
             # Attempt conversion, handling potential errors
             y = data['label'].astype(int).to_numpy()
        except (ValueError, TypeError) as e:
             print(f"Error converting 'label' column to integer: {e}. Check label data.")
             # Provide more context if possible
             print("Problematic labels:", data['label'][~data['label'].apply(lambda x: isinstance(x, (int, float, np.number)) or str(x).isdigit())].unique())
             raise e

        X_tensor = torch.tensor(X, dtype=return_type)
        y_tensor = torch.tensor(y, dtype=torch.long) # Labels should usually be Long

        print(f"Creating DataLoader with batch size {batch_size} and shuffle={shuffle}...")
        iterable_data = DataLoader(
            dataset=TensorDataset(X_tensor, y_tensor),
            batch_size=batch_size,
            shuffle=shuffle,
        )
        print("DataLoader created.")
        return iterable_data

    def save_processor_config(self, filepath):
        """Saves essential configuration needed for inference."""
        config_to_save = {
            'max_length': self.max_length,
            'min_freq': self.min_freq,
            'sos_token': self.sos_token,
            'eos_token': self.eos_token,
            'unk_token': self.unk_token,
            'pad_token': self.pad_token,
            'use_stopwords': self.use_stopwords
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config_to_save, f)
        print(f"Preprocessor config saved to {filepath}")

    @classmethod
    def from_config(cls, filepath):
        """Loads processor config from file."""
        with open(filepath, 'r') as f:
            config_loaded = json.load(f)
        print(f"Preprocessor config loaded from {filepath}")
        return cls(**config_loaded)

    def save_vocab(self, filepath):
        """Saves the torchtext vocabulary object."""
        if self.vocab is None:
            raise ValueError("Vocabulary not built. Cannot save.")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.vocab, filepath)
        print(f"Vocabulary saved to {filepath}")

    def load_vocab(self, filepath):
        """Loads the torchtext vocabulary object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at {filepath}")
        self.vocab = torch.load(filepath)
        # Re-assign default index just in case it wasn't saved properly
        unk_index = self.vocab[self.unk_token]
        self.vocab.set_default_index(unk_index)
        print(f"Vocabulary loaded from {filepath}. Size: {len(self.vocab)}")

# --- Helper function to load data ---
def load_data(train_path, val_path, test_path):
    """Loads train, validation, and test data from CSV files."""
    try:
        train_data = pd.read_csv(train_path)
        val_data = pd.read_csv(val_path)
        test_data = pd.read_csv(test_path)
        print("Data loaded successfully.")
        print(f"Train shape: {train_data.shape}, Val shape: {val_data.shape}, Test shape: {test_data.shape}")
        return train_data, val_data, test_data
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Please check file paths in config.py.")
        raise