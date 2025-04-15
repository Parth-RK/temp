# models.py
"""
Contains definitions for the neural network models (ANN, LSTM).
(TorchText Legacy Independent Version)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    """A simple Artificial Neural Network for classification.
       NOTE: Less suitable now as input is sequence indices, not fixed size features.
             Would require embedding or significant modification.
    """
    def __init__(self, input_size, n_class): # input_size needs re-evaluation
        super().__init__()
        print(f"Initializing ANN - WARNING: Input needs embedding or modification for sequence indices.")
        # This structure assumes flattened fixed-size input, not sequence indices
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, n_class)

    def forward(self, features):
        # Requires modification - features are now (batch, seq_len) indices
        # Option 1: Add embedding layer
        # Option 2: Flatten and treat as fixed input (loses sequence info)
        # Current implementation will likely fail or perform poorly.
        if features.dtype != torch.float32:
             features = features.float()
        # Placeholder: Flattening (Bad Idea for sequences)
        # features = features.view(features.size(0), -1)
        # if features.shape[1] != self.fc1.in_features: # Crude resize attempt (Very Bad)
        #     features = F.adaptive_avg_pool1d(features.unsqueeze(1), self.fc1.in_features).squeeze(1)

        # Original ANN logic assuming fixed float input:
        out = F.relu(self.bn1(self.fc1(features)))
        out = self.dropout1(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


class LSTMNetwork(nn.Module):
    """An LSTM Network with Embedding layer for sequence classification."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        print(f"Initializing LSTM with vocab_size={vocab_size}, embedding_dim={embedding_dim}, "
              f"hidden_dim={hidden_dim}, n_layers={n_layers}, n_class={n_class}, pad_idx={pad_idx}")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout_prob if n_layers > 1 else 0, # Apply dropout between LSTM layers
                            bidirectional=True)
        # Input to fc layer is doubled because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.dropout = nn.Dropout(dropout_prob) # Dropout before final layer

    def forward(self, text_indices):
        # text_indices shape: (batch_size, seq_len)
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # packed_output, (hidden, cell)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (num_layers * num_directions, batch_size, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Concatenate the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden states
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden_concat shape: (batch_size, hidden_dim * 2)

        hidden_dropped = self.dropout(hidden_concat)

        out = self.fc(hidden_dropped)
        # out shape: (batch_size, n_class)
        return out