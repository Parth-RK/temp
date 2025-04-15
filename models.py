import torch
import torch.nn as nn
import torch.nn.functional as F
import config # Keep config import

class Attention(nn.Module):
    """
    Simple Additive Attention mechanism.

    Calculates attention weights over RNN output sequences and computes a
    weighted context vector. Handles masking for padded sequences.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # Input dimension is hidden_dim * 2 because it comes from a bidirectional RNN
        self.attention_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim * 2, self.attention_dim, bias=False) # Query/Key projection
        self.v = nn.Linear(self.attention_dim, 1, bias=False) # Scoring vector

    def forward(self, rnn_outputs, sequence_lengths=None):
        """
        Args:
            rnn_outputs (torch.Tensor): Outputs from RNN [batch_size, seq_len, hidden_dim * 2].
            sequence_lengths (torch.Tensor, optional): Lengths of sequences [batch_size] for masking.

        Returns:
            torch.Tensor: Context vector [batch_size, hidden_dim * 2].
            torch.Tensor: Attention weights [batch_size, seq_len].
        """
        # Calculate energy scores
        energy = torch.tanh(self.W_q(rnn_outputs)) # [batch_size, seq_len, attention_dim]
        attention_scores = self.v(energy).squeeze(2) # [batch_size, seq_len]

        # Apply mask to padding tokens before softmax
        if sequence_lengths is not None:
            mask = torch.arange(rnn_outputs.size(1), device=rnn_outputs.device)[None, :] >= sequence_lengths[:, None]
            attention_scores = attention_scores.masked_fill(mask, -float('inf'))

        attention_weights = F.softmax(attention_scores, dim=1) # [batch_size, seq_len]

        # Calculate context vector (weighted sum of rnn_outputs)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        # context_vector: [batch_size, hidden_dim * 2]

        return context_vector, attention_weights


class CNN_RNN_Attention(nn.Module):
    """
    Model combining CNN, RNN (LSTM or GRU), and Attention for sequence classification.

    - Embedding layer
    - Parallel 1D CNN layers with different kernel sizes
    - Bidirectional RNN (LSTM or GRU) layer
    - Attention layer over RNN outputs
    - Final linear classification layer
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 cnn_out_channels,
                 cnn_kernel_sizes, # List/tuple, e.g., [3, 4, 5]
                 rnn_type, # 'lstm' or 'gru'
                 rnn_hidden_dim,
                 rnn_layers,
                 n_class,
                 dropout_prob,
                 pad_idx):
        super().__init__()

        if rnn_type.lower() not in ['lstm', 'gru']:
            raise ValueError("rnn_type must be 'lstm' or 'gru'")
        if not isinstance(cnn_kernel_sizes, (list, tuple)):
             cnn_kernel_sizes = [cnn_kernel_sizes]

        print(f"Initializing CNN_{rnn_type.upper()}_Attention model...") # Concise init message

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # CNN Layers (applied in parallel)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=cnn_out_channels,
                      kernel_size=k,
                      padding='same') # 'same' keeps seq length consistent
            for k in cnn_kernel_sizes
        ])
        cnn_total_out_channels = cnn_out_channels * len(cnn_kernel_sizes)

        # RNN Layer (LSTM or GRU)
        self.rnn_type = rnn_type.lower()
        rnn_input_dim = cnn_total_out_channels
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True,
                               dropout=dropout_prob if rnn_layers > 1 else 0,
                               bidirectional=True)
        else: # 'gru'
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True,
                              dropout=dropout_prob if rnn_layers > 1 else 0,
                              bidirectional=True)

        # Attention Layer
        self.attention = Attention(rnn_hidden_dim) # Takes RNN hidden dim as input

        # Final Classifier Layers
        self.dropout = nn.Dropout(dropout_prob)
        # Input dim is rnn_hidden_dim * 2 due to bidirectional RNN output used by attention
        self.fc = nn.Linear(rnn_hidden_dim * 2, n_class)

    def forward(self, text_indices, sequence_lengths=None):
        """
        Forward pass through the CNN-RNN-Attention model.

        Args:
            text_indices (torch.Tensor): Input sequence indices [batch_size, seq_len].
            sequence_lengths (torch.Tensor, optional): Sequence lengths [batch_size] for attention masking.

        Returns:
            torch.Tensor: Output logits [batch_size, n_class].
        """
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        # 1. Embedding
        embedded = self.dropout(self.embedding(text_indices)) # [batch_size, seq_len, embedding_dim]

        # 2. CNN
        # Permute for Conv1d: [batch_size, embedding_dim, seq_len]
        embedded_permuted = embedded.permute(0, 2, 1)
        cnn_outputs = [F.relu(conv(embedded_permuted)) for conv in self.conv_layers]
        # Concatenate outputs from different kernel sizes along the channel dimension
        cnn_cat = torch.cat(cnn_outputs, dim=1) # [batch_size, cnn_total_out_channels, seq_len]

        # 3. RNN
        # Permute for RNN: [batch_size, seq_len, cnn_total_out_channels]
        rnn_input = cnn_cat.permute(0, 2, 1)
        rnn_outputs, _ = self.rnn(rnn_input) # [batch_size, seq_len, rnn_hidden_dim * 2]

        # 4. Attention
        context_vector, _ = self.attention(rnn_outputs, sequence_lengths) # [batch_size, rnn_hidden_dim * 2]

        # 5. Classifier
        dropped_context = self.dropout(context_vector)
        out = self.fc(dropped_context) # [batch_size, n_class]

        return out


class LSTMNetwork(nn.Module):
    """ Simple Bidirectional LSTM model for baseline comparison. """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        print("(Baseline) Initializing Simple BiLSTM model...")
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout_prob if n_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        # Input to FC is concatenation of final forward/backward hidden states
        self.fc = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, text_indices):
        """ Forward pass for the simple BiLSTM model. """
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        _, (hidden, _) = self.lstm(embedded) # We only need the final hidden states

        # Concatenate final forward (hidden[-2]) and backward (hidden[-1]) states
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped)
        return out