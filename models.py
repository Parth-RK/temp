# --- models.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    AutoModel = None
    AutoConfig = None
    print("Warning: HuggingFace Transformers library not installed. Transformer model type will not be available.")

import config # Import configuration

# --- Attention Mechanism (for CNN_RNN_Attention) ---
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_dim = hidden_dim
        # Adjusted linear layer input size for bidirectional RNN
        self.W_q = nn.Linear(hidden_dim * 2, self.attention_dim, bias=False)
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, rnn_outputs, sequence_lengths=None):
        # rnn_outputs shape: (batch_size, seq_len, hidden_dim * 2)
        energy = torch.tanh(self.W_q(rnn_outputs))  # (batch_size, seq_len, attention_dim)
        attention_scores = self.v(energy).squeeze(2) # (batch_size, seq_len)

        # Apply mask based on sequence lengths before softmax
        if sequence_lengths is not None:
            max_len = rnn_outputs.size(1)
            # Create mask: True for padding positions
            mask = torch.arange(max_len, device=rnn_outputs.device)[None, :] >= sequence_lengths[:, None]
            attention_scores = attention_scores.masked_fill(mask, -float('inf')) # Mask padding

        attention_weights = F.softmax(attention_scores, dim=1) # (batch_size, seq_len)
        # Calculate context vector
        # attention_weights unsqueezed: (batch_size, 1, seq_len)
        # rnn_outputs: (batch_size, seq_len, hidden_dim * 2)
        # context_vector: (batch_size, 1, hidden_dim * 2) -> squeezed to (batch_size, hidden_dim * 2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context_vector, attention_weights

# --- Transformer Model ---
class TransformerClassifier(nn.Module):
    """ Generic Transformer-based classifier using AutoModel. """
    def __init__(self, model_name, n_classes, dropout_prob=0.1):
        super().__init__()
        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library is required for TransformerClassifier.")

        self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        # Use dropout defined in config or fallback
        dropout_val = getattr(self.config, 'hidden_dropout_prob', dropout_prob)
        self.dropout = nn.Dropout(dropout_val)

        self.classifier = nn.Linear(self.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use pooler output if available, otherwise CLS token's last hidden state
        pooled_output = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs.last_hidden_state[:, 0]
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits

# --- CNN + RNN + Attention Model ---
class CNN_RNN_Attention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 cnn_out_channels,
                 cnn_kernel_sizes, # Expect list/tuple
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
             cnn_kernel_sizes = [cnn_kernel_sizes] # Ensure it's iterable

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=cnn_out_channels,
                      kernel_size=k,
                      padding='same') # Use 'same' padding
            for k in cnn_kernel_sizes
        ])

        cnn_total_out_channels = cnn_out_channels * len(cnn_kernel_sizes)

        self.rnn_type = rnn_type.lower()
        rnn_input_dim = cnn_total_out_channels # Output of CNNs is input to RNN

        rnn_dropout = dropout_prob if rnn_layers > 1 else 0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=True)
        else: # gru
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=True)

        # Attention layer input dim matches RNN hidden dim (bidirectional doubles it)
        self.attention = Attention(rnn_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        # FC layer input dim matches attention output (bidirectional RNN output)
        self.fc = nn.Linear(rnn_hidden_dim * 2, n_class)
        self.pad_idx = pad_idx


    def forward(self, text_indices, sequence_lengths=None):
        # text_indices shape: (batch_size, seq_len)
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # Conv1d expects (batch_size, channels, seq_len)
        embedded_permuted = embedded.permute(0, 2, 1)
        # embedded_permuted shape: (batch_size, embedding_dim, seq_len)

        cnn_outputs = [F.relu(conv(embedded_permuted)) for conv in self.conv_layers]
        # Each cnn_output shape: (batch_size, cnn_out_channels, seq_len)

        cnn_cat = torch.cat(cnn_outputs, dim=1)
        # cnn_cat shape: (batch_size, cnn_total_out_channels, seq_len)

        # RNN expects (batch_size, seq_len, features)
        rnn_input = cnn_cat.permute(0, 2, 1)
        # rnn_input shape: (batch_size, seq_len, cnn_total_out_channels)

        # Pack sequence for RNN efficiency if lengths are provided
        if sequence_lengths is not None:
             # Ensure lengths are on CPU for pack_padded_sequence
             packed_input = nn.utils.rnn.pack_padded_sequence(rnn_input, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             packed_outputs, _ = self.rnn(packed_input)
             rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        else:
             # Warning: Without lengths, RNN processes padding tokens which might hurt performance.
             rnn_outputs, _ = self.rnn(rnn_input) # (batch_size, seq_len, rnn_hidden_dim * 2)


        # Apply Attention
        # Pass sequence_lengths to attention for masking
        context_vector, _ = self.attention(rnn_outputs, sequence_lengths)
        # context_vector shape: (batch_size, rnn_hidden_dim * 2)

        dropped_context = self.dropout(context_vector)
        out = self.fc(dropped_context) # (batch_size, n_class)
        return out

# --- Simple LSTM Model ---
class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        rnn_dropout = dropout_prob if n_layers > 1 else 0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        # Input to FC is concatenation of the final forward and backward hidden states
        self.fc = nn.Linear(hidden_dim * 2, n_class)
        self.pad_idx = pad_idx

    def forward(self, text_indices, sequence_lengths=None):
        # text_indices shape: (batch_size, seq_len)
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        embedded = self.dropout(self.embedding(text_indices))
        # embedded shape: (batch_size, seq_len, embedding_dim)

        # Pack sequence for RNN efficiency if lengths are provided
        if sequence_lengths is not None:
             packed_input = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             _, (hidden, cell) = self.lstm(packed_input)
        else:
             _, (hidden, cell) = self.lstm(embedded)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)

        # Concatenate the final hidden states from the last layer (forward and backward)
        # hidden[-2,:,:] is the last forward layer's hidden state
        # hidden[-1,:,:] is the last backward layer's hidden state
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden_concat shape: (batch_size, hidden_dim * 2)

        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped) # (batch_size, n_class)
        return out