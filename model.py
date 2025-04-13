# model.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class EmotionLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers, bidirectional, dropout, pad_idx):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Using LSTM here, could swap for GRU
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout if n_layers > 1 else 0, # Dropout only between RNN layers
                           batch_first=True) # Crucial: Input tensors are (batch, seq_len, features)

        # Linear layer input depends on bidirectionality
        linear_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(linear_input_dim, output_dim)

        self.dropout = nn.Dropout(dropout) # Dropout before final layer

    def forward(self, text, text_lengths):
        # text shape: (batch_size, seq_len)
        # text_lengths shape: (batch_size,)

        # 1. Embedding layer
        # embedded shape: (batch_size, seq_len, embedding_dim)
        embedded = self.dropout(self.embedding(text))

        # 2. Pack sequence
        # Reduces computation by not processing PAD tokens
        # Needs lengths on CPU
        packed_embedded = pack_padded_sequence(embedded, text_lengths.to('cpu'),
                                                batch_first=True, enforce_sorted=False)

        # 3. RNN layer
        # packed_output shape: PackedSequence
        # hidden shape: (num_layers * num_directions, batch_size, hidden_dim)
        # cell shape: (num_layers * num_directions, batch_size, hidden_dim) -> For LSTM
        packed_output, (hidden, cell) = self.rnn(packed_embedded)

        # Unpack sequence (optional, might not need the full output sequence)
        # output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)
        # output shape: (batch_size, seq_len, hidden_dim * num_directions)

        # 4. Get final hidden state(s)
        # Concatenate the final forward and backward hidden states
        if self.rnn.bidirectional:
            # hidden shape: (num_layers * 2, batch, hidden_dim)
            # Get the last layer's forward and backward hidden states
            # hidden[-2,:,:] is the last forward state
            # hidden[-1,:,:] is the last backward state
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            # hidden shape: (num_layers, batch, hidden_dim)
            # Get the last layer's hidden state
            hidden = self.dropout(hidden[-1,:,:])

        # hidden shape after processing: (batch_size, hidden_dim * num_directions)

        # 5. Fully connected layer
        # output shape: (batch_size, output_dim)
        output = self.fc(hidden)

        return output

def create_model(vocab_size, output_dim, config, pad_idx):
    """Helper function to create the model based on config."""
    print("Creating PyTorch model...")
    model = EmotionLSTM(
        vocab_size=vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=output_dim,
        n_layers=config['n_layers'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout'],
        pad_idx=pad_idx
    )
    # Initialize weights - can improve convergence
    def init_weights(m):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param.data, mean=0, std=0.01) # Example initialization
            else:
                nn.init.constant_(param.data, 0)
    # model.apply(init_weights) # Apply initialization (optional but good practice)
    print(model) # Print model structure
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model has {total_params:,} trainable parameters.')
    return model