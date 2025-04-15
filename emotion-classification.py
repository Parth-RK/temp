import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download("wordnet")
nltk.download("stopwords")
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import stopwords
from pandarallel import pandarallel
from torch.utils.data import DataLoader, TensorDataset
from torchtext import vocab
from tqdm import tqdm

pandarallel.initialize(progress_bar=True)

train_data = pd.read_csv(r"training.csv")
val_data = pd.read_csv(r"validation.csv")
test_data = pd.read_csv(r"test.csv")

class Preprocessor:
    def __init__(
        self,
        max_length,
        min_freq,
        sos_token,
        eos_token,
        unk_token,
        pad_token,
        shuffle=False,
        batch_size=16,
        stopwords=[]
    ):
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.max_length = max_length
        self.min_freq = min_freq
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.stopwords = stopwords
        self.special_tokens = [unk_token, pad_token, sos_token, eos_token]

    def clean(self, data):
        self.nlp = spacy.load("en_core_web_sm")
        lemmatize = lambda text: " ".join(
            x
            for x in map(lambda token: token.lemma_, self.nlp(text))
            if x not in self.stopwords
        )
        print("Lemmatizing...")
        data["clean_text"] = data["text"].parallel_apply(lemmatize).str.lower()
        print("Lemmatizing Done!")
        return data

    def tokenize(self, text, max_length, sos_token, eos_token):
        tokens = [token.text for token in self.nlp.tokenizer(text)][:max_length]
        en_tokens = [sos_token] + tokens + [eos_token]
        return en_tokens

    def convert_numerical(self, tokens):
        en_ids = self.vocab.lookup_indices(tokens)
        return en_ids

    def fit(self, data):
        data = self.clean(data)
        print("Tokenizing Started...")
        data["tokens"] = data["clean_text"].map(
            lambda x: self.tokenize(
                str(x), self.max_length, self.sos_token, self.eos_token
            )
        )
        print("Tokenizing Done!")
        self.vocab = vocab.build_vocab_from_iterator(
            data["tokens"], min_freq=self.min_freq, specials=self.special_tokens
        )
        unk_index = self.vocab[self.unk_token]
        self.vocab.set_default_index(unk_index)

    def pad_sequences(self, sequences, pad_token):
        max_seq_len = self.max_length + 2
        padded_sequences = [
            seq + [pad_token] * (max_seq_len - len(seq)) for seq in sequences
        ]
        return padded_sequences

    def transform(self, data, return_float=True):
        if "clean_text" not in data.columns:
            data = self.clean(data)
        if "tokens" not in data.columns:
            data["tokens"] = data["clean_text"].map(
                lambda x: self.tokenize(
                    str(x), self.max_length, self.sos_token, self.eos_token
                )
            )
        # Convert tokens to numerical indices
        data["numerical_tokens"] = data.tokens.map(self.convert_numerical)
        # Pad sequences
        padded_sequences = self.pad_sequences(
            data["numerical_tokens"].to_list(), self.vocab[self.pad_token]
        )
        X = np.array(padded_sequences)
        y = data.label.to_numpy()
        if return_float:
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X, dtype=torch.long)
            y_tensor = torch.tensor(y, dtype=torch.long)
        iterable_data = DataLoader(
            dataset=TensorDataset(X_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )
        return iterable_data

# Preprocessing
# %%time
processor = Preprocessor(
    shuffle=True,
    batch_size=32,
    max_length=40,
    min_freq=2,
    sos_token="<sos>",
    eos_token="<eos>",
    unk_token="<unk>",
    pad_token="<pad>",
    # stopwords=stopwords.words('english') 
)
processor.fit(train_data)
train_dataset = processor.transform(train_data)
val_dataset = processor.transform(val_data)
test_dataset = processor.transform(test_data)

device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    correct = (
        torch.eq(y_true, y_pred).sum().item()
    )  
    acc = (correct / len(y_pred)) * 100
    return acc

def evaluate(model, data_loader, device):
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X = X.to(device)
            y = y.type(torch.LongTensor)
            y = y.to(device)
            y_pred = model(X)
            y_pred_class = nn.functional.softmax(y_pred, dim=1).argmax(dim=1)
            loss += criterion(y_pred, y).item()
            acc += accuracy_fn(y, y_pred_class)
    acc /= len(data_loader)
    loss /= len(data_loader)
    return acc, loss

def trainer(model, device, train_data, epochs, criterion, optimizer, val_data=None):
    history = {
        "train_loss": [],
        "train_acc": [],
        "epoch": [],
    }
    if val_data:
        history["val_loss"] = []
        history["val_acc"] = []
    model = model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch, (X, y) in tqdm(enumerate(train_data), total=len(train_data)):
            X = X.to(device)
            y = y.type(torch.LongTensor)
            y = y.to(device)
            y_pred = model(X)
            y_pred_class = nn.functional.softmax(y_pred, dim=1).argmax(dim=1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_acc, train_loss = evaluate(model, train_data, device)
        if val_data:
            val_acc, val_loss = evaluate(model, val_data, device)
        for key in history.keys():
            history[key].append(locals()[key])
        print(f"Epoch: {epoch}/{epochs}")
        print(f"Training ---|  Accuracy: {train_acc:.2f} Loss: {train_loss:.5f}")
        if val_data:
            print(f"Validation ---|  Accuracy: {val_acc:.2f} Loss: {val_loss:.5f}")
    return history

def plot(df):
    plt.figure(figsize=(12, 6))
    # Plotting train_loss and val_loss
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    # Plotting train_acc and val_acc
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ## Fully Connected Network
# len(processor.vocab)

# %%time
class ANN(nn.Module):
    def __init__(self, input_size, n_class):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, n_class)
    def forward(self, features):
        out = self.fc2(F.relu(self.fc1(features)))
        return out

model = ANN(input_size=42, n_class=6)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
history = trainer(
    model,
    epochs=40,
    device=device,
    train_data=train_dataset,
    val_data=val_dataset,
    criterion=criterion,
    optimizer=optimizer,
)
res = pd.DataFrame(history)
plot(res)
evaluate(model, test_dataset, device)

# ### Train LSTM using embedding
# %%time
class LSTMNetwork(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.embeddings = nn.Embedding(len(processor.vocab), 32)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, n_class)
    def forward(self, features):
        features = features.type(torch.long)
        emb = self.embeddings(features)
        out, _ = self.lstm(emb)
        out = self.fc(out[:, -1, :])  # Use the output from the last time step
        return out

model_lstm = LSTMNetwork(n_class=6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_lstm.parameters(), lr=0.01)
history = trainer(
    model_lstm,
    epochs=40,
    device=device,
    train_data=train_dataset,
    val_data=val_dataset,
    criterion=criterion,
    optimizer=optimizer,
)
res = pd.DataFrame(history)
plot(res)
evaluate(model_lstm, test_dataset, device="cuda")
