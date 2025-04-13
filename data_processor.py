# data_processor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from nltk.tokenize import word_tokenize # Using nltk for better tokenization
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import pickle # For saving/loading vocab and encoder
import re
import string

# --- Text Cleaning Functions ---
def clean_text(text):
    """Clean and normalize text data."""
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# --- Constants ---
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

# --- Vocabulary Class ---
class Vocabulary:
    def __init__(self, freq_threshold=2):
        self.itos = {0: PAD_TOKEN, 1: UNK_TOKEN} # Index to string
        self.stoi = {PAD_TOKEN: 0, UNK_TOKEN: 1} # String to index
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 2 # Start index after PAD and UNK

        print("Tokenizing and building frequency count...")
        tokenized_sentences = []
        for sentence in sentence_list:
            # Apply cleaning *during* tokenization step here as well
            tokens = word_tokenize(clean_text(str(sentence))) # Ensure sentence is string
            tokenized_sentences.append(tokens)
            frequencies.update(tokens)

        print("Building vocabulary...")
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
        print(f"Built vocabulary with {len(self.itos)} words.")
        return tokenized_sentences # Return tokenized text for dataset

    def numericalize(self, text):
        """Converts a text string into a list of integer indices."""
        tokenized_text = word_tokenize(clean_text(str(text))) # Ensure text is string
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokenized_text]

    def numericalize_tokens(self, tokens):
        """Converts pre-tokenized list into indices."""
        return [self.stoi.get(token, self.stoi[UNK_TOKEN]) for token in tokens]

    @staticmethod
    def save(vocab, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(vocab, f)
        print(f"Vocabulary saved to {filepath}")

    @staticmethod
    def load(filepath):
        try:
            with open(filepath, 'rb') as f:
                vocab = pickle.load(f)
            print(f"Vocabulary loaded from {filepath}")
            return vocab
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {filepath}")
            return None
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return None


# --- PyTorch Dataset ---
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        # Ensure texts and labels are lists of the same length
        assert len(texts) == len(labels), "Texts and labels must have the same number of samples!"
        self.texts = texts # Expecting list of lists of tokens
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Numericalize the pre-tokenized text
        numericalized_text = self.vocab.numericalize_tokens(self.texts[idx])
        label = self.labels[idx]
        # Return as tensors
        return torch.tensor(numericalized_text, dtype=torch.long), torch.tensor(label, dtype=torch.long)


# --- DataLoader Collate Function ---
class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Separate texts and labels
        texts = [item[0] for item in batch]
        labels = [item[1] for item in batch]

        # Get lengths before padding (useful for PackedSequence)
        # Crucially, this assumes texts are already filtered and > 0 length
        lengths = torch.tensor([len(txt) for txt in texts], dtype=torch.long)

        # Pad sequences in this batch to the max length *in this batch*
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.pad_idx)

        # Stack labels into a tensor
        labels = torch.stack(labels)

        return padded_texts, labels, lengths


# --- Main Data Preparation Function ---
def load_and_prepare_data(filepath, label_column='emotion', text_column='content',
                          batch_size=64, test_size=0.2, random_state=42, vocab_freq_threshold=2,
                          output_dir="output"):

    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 failed, trying latin-1")
        df = pd.read_csv(filepath, encoding='latin-1')
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None, None

    # Basic Cleaning & Dropping NaNs
    df.dropna(subset=[text_column, label_column], inplace=True)
    # Ensure columns are string type before checks
    df[text_column] = df[text_column].astype(str)
    df[label_column] = df[label_column].astype(str)
    df = df[df[text_column].str.strip().astype(bool)] # Remove rows with empty/whitespace-only text
    df = df[df[label_column].str.strip().astype(bool)]

    print(f"Data shape after initial load & dropna: {df.shape}")
    if df.empty:
        print("Error: No data left after initial cleaning.")
        return None, None, None, None, None

    X = df[text_column]
    y = df[label_column]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} unique classes: {label_encoder.classes_}")

    # Split data (using original df indices ensures correspondence before filtering)
    train_indices, val_indices = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=df[label_column]
    )

    X_train_text = df.loc[train_indices, text_column].tolist()
    y_train_labels = y_encoded[train_indices] # Use index alignment

    X_val_text = df.loc[val_indices, text_column].tolist()
    y_val_labels = y_encoded[val_indices] # Use index alignment


    # Build Vocabulary ONLY on training data text
    vocab = Vocabulary(freq_threshold=vocab_freq_threshold)
    # Pass the list of strings to build_vocabulary, it handles cleaning/tokenizing inside
    tokenized_train_texts_all = vocab.build_vocabulary(X_train_text)

    # Save vocab and label encoder (do this *before* filtering)
    os.makedirs(output_dir, exist_ok=True)
    vocab_path = os.path.join(output_dir, "vocabulary.pkl")
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    Vocabulary.save(vocab, vocab_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {encoder_path}")


    # Clean and tokenize validation text using the *same* cleaning process
    print("Tokenizing validation data...")
    tokenized_val_texts_all = [word_tokenize(clean_text(str(text))) for text in X_val_text]


    # *** ADD FILTERING STEP HERE ***
    print("Filtering out sequences with zero length after tokenization...")

    filtered_train_texts = []
    filtered_y_train = []
    original_train_count = len(tokenized_train_texts_all)
    for tokens, label in zip(tokenized_train_texts_all, y_train_labels):
        if len(tokens) > 0:
            filtered_train_texts.append(tokens)
            filtered_y_train.append(label)

    filtered_val_texts = []
    filtered_y_val = []
    original_val_count = len(tokenized_val_texts_all)
    for tokens, label in zip(tokenized_val_texts_all, y_val_labels):
         if len(tokens) > 0:
            filtered_val_texts.append(tokens)
            filtered_y_val.append(label)

    print(f"Train set: Retained {len(filtered_train_texts)} out of {original_train_count} samples after filtering.")
    print(f"Validation set: Retained {len(filtered_val_texts)} out of {original_val_count} samples after filtering.")

    if not filtered_train_texts or not filtered_val_texts:
        print("Error: Filtering left no samples in train or validation set. Check cleaning/data.")
        return None, None, None, None, None

    # Create Datasets using the FILTERED lists
    train_dataset = EmotionDataset(filtered_train_texts, filtered_y_train, vocab)
    val_dataset = EmotionDataset(filtered_val_texts, filtered_y_val, vocab)

    # Create DataLoaders
    pad_idx = vocab.stoi[PAD_TOKEN]
    collate_fn = PadCollate(pad_idx)

    # Consider num_workers based on your system, 0 might be safer for debugging
    num_workers = 0 if os.name == 'nt' else 2 # Often 0 is needed on Windows
    print(f"Using num_workers={num_workers} for DataLoaders.")

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=num_workers)


    print(f"Train samples (post-filter): {len(train_dataset)}, Val samples (post-filter): {len(val_dataset)}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader, vocab, label_encoder, num_classes