# --- models.py ---
import torch
import torch.nn as nn
import sys

# Try importing transformer components, raise clear error if missing
try:
    from transformers import AutoModel, AutoConfig
except ImportError:
    AutoModel = None
    AutoConfig = None
    print("ERROR: HuggingFace Transformers library not installed or import failed.")
    print("       Please install it: pip install transformers")
    # Exit early if the core dependency is missing
    sys.exit(1)

import config # Import configuration

# --- Removed Attention, CNN_RNN_Attention, LSTMNetwork ---

# --- Transformer Model ---
class TransformerClassifier(nn.Module):
    """
    Transformer-based classifier using HuggingFace's AutoModel.
    Loads a pre-trained transformer model and adds a classification head.
    """
    def __init__(self, model_name, n_classes):
        super().__init__()
        # Check if transformers library was imported successfully (redundant check due to top-level check, but safe)
        if AutoModel is None or AutoConfig is None:
            raise ImportError("HuggingFace Transformers library failed to import correctly.")

        try:
            print(f"Loading Transformer config: {model_name} for {n_classes} classes")
            self.config = AutoConfig.from_pretrained(model_name, num_labels=n_classes)
            print(f"Loading Transformer model: {model_name}")
            self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        except OSError as e:
             print(f"\nError loading transformer model/config '{model_name}'.")
             print(f"Ensure the model name is correct and you have an internet connection if it needs downloading.")
             print(f"Or, if it's a local path, ensure the path is correct.")
             print(f"Original error: {e}")
             sys.exit(1) # Exit if model loading fails critically
        except Exception as e:
             print(f"An unexpected error occurred while loading the transformer model '{model_name}': {e}")
             import traceback
             traceback.print_exc()
             sys.exit(1)

        dropout_prob = getattr(self.config, 'classifier_dropout',
                               getattr(self.config, 'hidden_dropout_prob', 0.1))
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, n_classes)

        print(f"  TransformerClassifier using '{model_name}' initialized.")
        print(f"  Using hidden size: {self.config.hidden_size}, Dropout: {dropout_prob:.2f}")

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

        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            pooled_output = outputs.last_hidden_state[:, 0] # Use [CLS] token state

        dropped_output = self.dropout(pooled_output)
        logits = self.classifier(dropped_output)
        return logits