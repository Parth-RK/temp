import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_dim = hidden_dim
        self.W_q = nn.Linear(hidden_dim * 2, self.attention_dim, bias=False)
        self.v = nn.Linear(self.attention_dim, 1, bias=False)

    def forward(self, rnn_outputs, sequence_lengths=None):
        energy = torch.tanh(self.W_q(rnn_outputs))
        attention_scores = self.v(energy).squeeze(2)
        if sequence_lengths is not None:
            mask = torch.arange(rnn_outputs.size(1), device=rnn_outputs.device)[None, :] >= sequence_lengths[:, None]
            attention_scores = attention_scores.masked_fill(mask, -float('inf'))
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), rnn_outputs).squeeze(1)
        return context_vector, attention_weights

class CNN_RNN_Attention(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 cnn_out_channels,
                 cnn_kernel_sizes,
                 rnn_type,
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
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=cnn_out_channels,
                      kernel_size=k,
                      padding='same')
            for k in cnn_kernel_sizes
        ])
        cnn_total_out_channels = cnn_out_channels * len(cnn_kernel_sizes)
        self.rnn_type = rnn_type.lower()
        rnn_input_dim = cnn_total_out_channels
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(rnn_input_dim, rnn_hidden_dim,
                               num_layers=rnn_layers, batch_first=True,
                               dropout=dropout_prob if rnn_layers > 1 else 0,
                               bidirectional=True)
        else:
            self.rnn = nn.GRU(rnn_input_dim, rnn_hidden_dim,
                              num_layers=rnn_layers, batch_first=True,
                              dropout=dropout_prob if rnn_layers > 1 else 0,
                              bidirectional=True)
        self.attention = Attention(rnn_hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(rnn_hidden_dim * 2, n_class)

    def forward(self, text_indices, sequence_lengths=None):
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()
        embedded = self.dropout(self.embedding(text_indices))
        embedded_permuted = embedded.permute(0, 2, 1)
        cnn_outputs = [F.relu(conv(embedded_permuted)) for conv in self.conv_layers]
        cnn_cat = torch.cat(cnn_outputs, dim=1)
        rnn_input = cnn_cat.permute(0, 2, 1)
        rnn_outputs, _ = self.rnn(rnn_input)
        context_vector, _ = self.attention(rnn_outputs, sequence_lengths)
        dropped_context = self.dropout(context_vector)
        out = self.fc(dropped_context)
        return out

class LSTMNetwork(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_class, n_layers, pad_idx, dropout_prob=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=n_layers, batch_first=True,
                            dropout=dropout_prob if n_layers > 1 else 0,
                            bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, n_class)

    def forward(self, text_indices):
        if text_indices.dtype != torch.long:
             text_indices = text_indices.long()
        embedded = self.dropout(self.embedding(text_indices))
        _, (hidden, _) = self.lstm(embedded)
        hidden_concat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_dropped = self.dropout(hidden_concat)
        out = self.fc(hidden_dropped)
        return out