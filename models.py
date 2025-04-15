# models.py
"""
Contains definitions for the neural network models (ANN, LSTM).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    """A simple Artificial Neural Network for classification."""
    def __init__(self, input_size, n_class):
        super().__init__()
        print(f"Initializing ANN with input_size={input_size}, n_class={n_class}")
        self.fc1 = nn.Linear(input_size, 128) # Increased hidden size
        self.bn1 = nn.BatchNorm1d(128)      # Added BatchNorm
        self.dropout1 = nn.Dropout(0.3)     # Added Dropout
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, n_class)   # Added another layer

    def forward(self, features):
        # Ensure input is float for linear layers
        if features.dtype != torch.float32:
             features = features.float()

        out = F.relu(self.bn1(self.fc1(features)))
        out = self.dropout1(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out) # No activation on the final layer for CrossEntropyLoss

        return out

class LSTMNetwork(nn.Module):
    """An LSTM Network with Embedding layer for sequence classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx):
        super().__init__()
        print(f"Initializing LSTM with vocab_size={vocab_size}, embedding_dim={embedding_dim}, "
              f"hidden_dim={hidden_dim}, n_layers={n_layers}, n_class={n_class}")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Increased hidden_dim, added dropout and bidirectional
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=0.4 if n_layers > 1 else 0, # Add dropout if multiple layers
                            bidirectional=True) # Make LSTM bidirectional
        # Input to fc layer is doubled because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.dropout = nn.Dropout(0.5) # Dropout before final layer

    def forward(self, features):
        # Ensure input is long for embedding layer
        if features.dtype != torch.long:
             features = features.long()

        embedded = self.dropout(self.embedding(features)) # Apply dropout to embeddings

        # packed_output, (hidden, cell)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward and backward hidden states
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))

        # Pass the concatenated hidden state to the linear layer
        out = self.fc(hidden)
        # Alternative: Use last time step output (may work better/worse depending on task)
        # lstm_out = self.dropout(lstm_out)
        # out = self.fc(lstm_out[:, -1, :]) # Use the output from the last time step

        return out