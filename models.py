# --- models.py ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys # For error messages

# Try importing transformer components, raise clear error if missing
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    # Set flags to None, allowing other models to work if transformers not installed
    AutoModel = None
    AutoConfig = None
    # Print a warning but don't exit immediately, only raise error if TransformerClassifier is used
    print("Warning: HuggingFace Transformers library not installed or import failed.")
    print("         Transformer model type ('TransformerClassifier') will not be available.")

import config # Import configuration

# --- Attention Mechanism (for CNN_RNN_Attention) ---
class Attention(nn.Module):
    """ Simple Bahdanau-style attention mechanism. """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_dim = hidden_dim
        # Linear layer for query (e.g., final RNN hidden state - though not used here)
        # Linear layer for keys (RNN outputs)
        # Input is (batch, seq_len, hidden_dim * 2) because RNN is bidirectional
        self.W_k = nn.Linear(hidden_dim * 2, self.attention_dim, bias=False)
        # Score vector
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, rnn_outputs, sequence_lengths=None):
        """
        Calculates attention weights and context vector.

        Args:
            rnn_outputs (torch.Tensor): Outputs from RNN (batch_size, seq_len, hidden_dim * 2).
            sequence_lengths (torch.Tensor, optional): Original lengths of sequences (batch_size).

        Returns:
            tuple: (context_vector, attention_weights)
                   context_vector shape: (batch_size, hidden_dim * 2)
                   attention_weights shape: (batch_size, seq_len)
        """
        # Project RNN outputs into attention space
        # energy shape: (batch_size, seq_len, attention_dim)
        energy = torch.tanh(self.W_k(rnn_outputs))

        # Calculate attention scores
        # attention_scores shape: (batch_size, seq_len)
        attention_scores = self.v(energy).squeeze(2)

        # Apply mask based on sequence lengths *before* softmax
        if sequence_lengths is not None:
            max_len = rnn_outputs.size(1)
            # Create mask: True for padding positions, False otherwise
            # Ensure sequence_lengths is on the same device as attention_scores
            mask = torch.arange(max_len, device=attention_scores.device)[None, :] >= sequence_lengths.to(attention_scores.device)[:, None]
            # Fill masked positions with negative infinity so they get zero probability after softmax
            attention_scores = attention_scores.masked_fill(mask, -1e9) # Use large negative number

        # Compute attention weights (probabilities)
        # attention_weights shape: (batch_size, seq_len)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Calculate context vector (weighted sum of RNN outputs)
        # Unsqueeze attention_weights for batch matrix multiplication: (batch_size, 1, seq_len)
        # Context vector calculation: (batch_size, 1, seq_len) @ (batch_size, seq_len, hidden_dim * 2)
        # Result shape: (batch_size, 1, hidden_dim * 2) -> squeeze -> (batch_size, hidden_dim * 2)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)

        return context_vector, attention_weights

# --- Transformer Model ---
class TransformerClassifier(nn.Module):
    """
    Generic Transformer-based classifier using HuggingFace's AutoModel.
    Loads a pre-trained transformer model and adds a classification head.
    """
    def __init__(self, model_name, n_classes):
        super().__init__()
        # Check if transformers library was imported successfully
        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library is required to use TransformerClassifier. Please install it (`pip install transformers`).")

        try:
            self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        except OSError as e:
             print(f"\nError loading transformer model '{model_name}'.")
             print(f"Ensure the model name is correct and you have an internet connection if it needs downloading.")
             print(f"Or, if it's a local path, ensure the path is correct.")
             print(f"Original error: {e}")
             sys.exit(1) # Exit if model loading fails critically
        except Exception as e:
             print(f"An unexpected error occurred while loading the transformer model '{model_name}': {e}")
             sys.exit(1)

        # Use dropout probability defined in the loaded transformer's config, or default if not present
        dropout_prob = getattr(self.config, 'classifier_dropout', # Try classifier specific dropout
                               getattr(self.config, 'hidden_dropout_prob', 0.1)) # Fallback to hidden dropout or 0.1
        self.dropout = nn.Dropout(dropout_prob)

        # Classification layer
        self.classifier = nn.Linear(self.config.hidden_size, n_classes)

        print(f"  TransformerClassifier using '{model_name}' initialized.")
        print(f"  Dropout probability: {dropout_prob:.2f}")


    def forward(self, input_ids, attention_mask):
        """
        Forward pass through the transformer and classifier.

        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            attention_mask (torch.Tensor): Attention mask (batch_size, seq_len).

        Returns:
            torch.Tensor: Logits for each class (batch_size, n_classes).
        """
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Extract the representation for classification.
        # Common strategies:
        # 1. Use the pooler output if available (often trained for classification)
        # 2. Use the hidden state of the [CLS] token (first token) from the last layer
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Use the last hidden state of the first token ([CLS])
            pooled_output = outputs.last_hidden_state[:, 0]

        # Apply dropout and classify
        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits

# --- CNN + RNN + Attention Model ---
class CNN_RNN_Attention(nn.Module):
    """
    A model combining CNNs for local feature extraction, an RNN (LSTM/GRU)
    for sequential context, and an Attention mechanism for focusing on relevant parts.
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 cnn_out_channels,
                 cnn_kernel_sizes, # Expect list/tuple e.g., [3, 4, 5]
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
             # If a single int is passed, wrap it in a list
             cnn_kernel_sizes = [cnn_kernel_sizes]

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # CNN layers with different kernel sizes applied in parallel
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=cnn_out_channels,
                      kernel_size=k,
                      padding='same') # 'same' padding ensures output length matches input length
            for k in cnn_kernel_sizes
        ])

        # Calculate total output channels from all parallel CNNs
        cnn_total_out_channels = cnn_out_channels * len(cnn_kernel_sizes)

        # RNN layer (LSTM or GRU)
        self.rnn_type = rnn_type.lower()
        rnn_input_dim = cnn_total_out_channels # Output of CNNs feeds into RNN
        # Apply dropout between RNN layers only if n_layers > 1
        rnn_dropout = dropout_prob if rnn_layers > 1 else 0.0
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True,
                               dropout=rnn_dropout, bidirectional=True)
        else: # gru
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True,
                              dropout=rnn_dropout, bidirectional=True)

        # Attention layer - input dimension matches the bidirectional RNN output dimension
        self.attention = Attention(rnn_hidden_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected output layer
        # Input dimension matches the attention context vector dimension (bidirectional RNN)
        self.fc = nn.Linear(rnn_hidden_dim * 2, n_class)

        self.pad_idx = pad_idx
        print(f"  CNN_RNN_Attention ({rnn_type.upper()}) initialized:")
        print(f"    Embedding Dim: {embedding_dim}, CNN Channels: {cnn_out_channels} (Kernels: {cnn_kernel_sizes})")
        print(f"    RNN Hidden Dim: {rnn_hidden_dim}, RNN Layers: {rnn_layers}, Bidirectional: True")
        print(f"    Dropout: {dropout_prob:.2f}")


    def forward(self, text_indices, sequence_lengths=None):
        """
        Forward pass for the CNN-RNN-Attention model.

        Args:
            text_indices (torch.Tensor): Input tensor of token indices (batch_size, seq_len).
            sequence_lengths (torch.Tensor, optional): Original lengths of sequences in the batch (batch_size).
                                                      Required for correct masking in attention and potentially RNN packing.

        Returns:
            torch.Tensor: Logits for each class (batch_size, n_class).
        """
        # Ensure input is LongTensor
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        # 1. Embedding Layer + Dropout
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text_indices))

        # 2. CNN Layers
        # Conv1d expects input shape: (batch_size, channels, seq_len)
        # Permute embedded tensor: (batch_size, embedding_dim, seq_len)
        embedded_permuted = embedded.permute(0, 2, 1)

        # Apply each convolution layer and ReLU activation
        # Each cnn_output shape: (batch_size, cnn_out_channels, seq_len)
        cnn_outputs = [F.relu(conv(embedded_permuted)) for conv in self.conv_layers]

        # Concatenate the outputs of the parallel CNNs along the channel dimension
        # cnn_cat shape: (batch_size, cnn_total_out_channels, seq_len)
        cnn_cat = torch.cat(cnn_outputs, dim=1)

        # Prepare input for RNN: (batch_size, seq_len, features)
        # Permute cnn_cat: (batch_size, seq_len, cnn_total_out_channels)
        rnn_input = cnn_cat.permute(0, 2, 1)

        # 3. RNN Layer
        # Use packing/padding for efficiency if sequence lengths are provided
        if sequence_lengths is not None:
             # Ensure lengths are on CPU for pack_padded_sequence
             # Sort sequences by length (required by pack_padded_sequence before PyTorch 1.7)
             # Modern PyTorch allows enforce_sorted=False, but sorting is often good practice.
             # For simplicity here, we use enforce_sorted=False if available, assuming lengths are correct.
             packed_input = nn.utils.rnn.pack_padded_sequence(rnn_input, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             packed_outputs, _ = self.rnn(packed_input)
             # Unpack the sequence
             rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
             # rnn_outputs shape: (batch_size, seq_len, rnn_hidden_dim * 2)
        else:
             # Warning: Processing without lengths means RNN processes padding tokens, which is inefficient and might hurt performance.
             print("Warning: Running RNN without sequence lengths. Padding tokens will be processed.")
             rnn_outputs, _ = self.rnn(rnn_input) # Shape: (batch_size, seq_len, rnn_hidden_dim * 2)


        # 4. Attention Layer
        # Pass sequence_lengths to attention for proper masking of padding tokens
        # context_vector shape: (batch_size, rnn_hidden_dim * 2)
        context_vector, attention_weights = self.attention(rnn_outputs, sequence_lengths)
        # attention_weights can be optionally returned or used for visualization

        # 5. Final Classification Layer
        dropped_context = self.dropout(context_vector)
        out = self.fc(dropped_context) # Shape: (batch_size, n_class)
        return out

# --- Simple LSTM Model ---
class LSTMNetwork(nn.Module):
    """
    A simpler bidirectional LSTM model for text classification.
    Uses the final hidden states for classification.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Apply dropout between LSTM layers only if n_layers > 1
        rnn_dropout = dropout_prob if n_layers > 1 else 0.0
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=rnn_dropout,
                            bidirectional=True) # Use bidirectional LSTM

        # Dropout layer applied to the concatenated final hidden states
        self.dropout = nn.Dropout(dropout_prob)

        # Fully connected layer
        # Input dimension is hidden_dim * 2 because LSTM is bidirectional
        self.fc = nn.Linear(hidden_dim * 2, n_class)

        self.pad_idx = pad_idx
        print(f"  LSTMNetwork initialized:")
        print(f"    Embedding Dim: {embedding_dim}, Hidden Dim: {hidden_dim}")
        print(f"    Layers: {n_layers}, Bidirectional: True")
        print(f"    Dropout: {dropout_prob:.2f}")

    def forward(self, text_indices, sequence_lengths=None):
        """
        Forward pass for the LSTM model.

        Args:
            text_indices (torch.Tensor): Input tensor of token indices (batch_size, seq_len).
            sequence_lengths (torch.Tensor, optional): Original lengths of sequences (batch_size).

        Returns:
            torch.Tensor: Logits for each class (batch_size, n_class).
        """
        # Ensure input is LongTensor
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()

        # 1. Embedding Layer + Dropout
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text_indices))

        # 2. LSTM Layer
        # Pack sequence for efficiency if lengths are provided
        if sequence_lengths is not None:
             packed_input = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths.cpu(), batch_first=True, enforce_sorted=False)
             # We only need the final hidden state, not the outputs per time step
             _, (hidden, cell) = self.lstm(packed_input)
             # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
             # cell shape:   (num_layers * num_directions, batch_size, hidden_dim)
        else:
             # Process without packing (less efficient)
             print("Warning: Running LSTM without sequence lengths. Padding tokens will be processed.")
             _, (hidden, cell) = self.lstm(embedded)


        # 3. Concatenate Final Hidden States
        # We need the hidden state from the last layer, for both forward and backward directions.
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        # Forward final hidden state: hidden[-2, :, :]
        # Backward final hidden state: hidden[-1, :, :]
        # Concatenate along the feature dimension (dim=1)
        # hidden_concat shape: (batch_size, hidden_dim * 2)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

        # 4. Final Classification Layer
        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped) # Shape: (batch_size, n_class)
        return out