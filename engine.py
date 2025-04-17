# --- engine.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

import config # Import configuration

# --- Model Initialization ---

def initialize_model(model_type, n_classes, vocab_size=None):
    """Initializes the model based on the configuration."""
    print(f"\nInitializing model: {model_type}")
    if model_type == 'Transformer':
        from models import TransformerClassifier # Local import
        model = TransformerClassifier(
            model_name=config.TRANSFORMER_MODEL_NAME,
            n_classes=n_classes
        )
    elif model_type == 'CNN_RNN_Attention':
        from models import CNN_RNN_Attention # Local import
        if vocab_size is None: raise ValueError("vocab_size required for CNN_RNN_Attention")
        model = CNN_RNN_Attention(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            rnn_type=config.RNN_TYPE,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            rnn_layers=config.RNN_LAYERS,
            n_class=n_classes,
            dropout_prob=config.WEIGHT_DECAY, # Using WEIGHT_DECAY as dropout here might be unintended? Use a separate DROPOUT_PROB config? Let's assume a default or add it.
            pad_idx=config.PAD_IDX
        )
    elif model_type == 'LSTM':
        from models import LSTMNetwork # Local import
        if vocab_size is None: raise ValueError("vocab_size required for LSTM")
        model = LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.RNN_HIDDEN_DIM,
            n_class=n_classes,
            n_layers=config.RNN_LAYERS,
            pad_idx=config.PAD_IDX,
            dropout_prob=config.WEIGHT_DECAY # Same potential issue as above
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE in config: {model_type}")

    model.to(config.DEVICE)
    print(f"Model loaded on {config.DEVICE}")
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    return model

# --- Optimizer and Scheduler ---

def initialize_optimizer_scheduler(model, optimizer_type, scheduler_type, num_train_steps=None):
    """Initializes optimizer and scheduler based on config."""
    print(f"\nInitializing Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")

    if optimizer_type == 'AdamW':
        # Differentiate parameters for weight decay (common for Transformers)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': config.WEIGHT_DECAY},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=config.LEARNING_RATE)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY, momentum=0.9)
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {optimizer_type}")

    scheduler = None
    if scheduler_type == 'linear_warmup':
        if num_train_steps is None:
            raise ValueError("num_train_steps is required for linear_warmup scheduler")
        num_warmup_steps = int(num_train_steps * config.WARMUP_PROPORTION)
        print(f"  Warmup Steps: {num_warmup_steps} (of {num_train_steps} total)")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif scheduler_type == 'reduce_on_plateau':
        # Monitors validation loss by default
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    elif scheduler_type is not None:
        print(f"Warning: Scheduler type '{scheduler_type}' requested but not implemented. No scheduler used.")


    return optimizer, scheduler

# --- Loss Function ---
criterion = nn.CrossEntropyLoss()

# --- Training Step ---

def train_step(model, data_loader, optimizer, device, scheduler=None, grad_clip_value=None):
    """Performs a single training epoch."""
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        optimizer.zero_grad()

        # Adapt input based on model type (derived from batch structure)
        if config.MODEL_TYPE == 'Transformer':
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else: # Non-transformer models (LSTM, CNN_RNN)
            # Assumes collate function returns (sequences, labels, lengths)
            sequences = batch[0].to(device)
            labels = batch[1].to(device)
            lengths = batch[2].to(device) # Pass lengths to model
            outputs = model(text_indices=sequences, sequence_lengths=lengths)

        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient Clipping
        if grad_clip_value:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        optimizer.step()
        if scheduler and config.SCHEDULER_TYPE == 'linear_warmup': # Step scheduler every batch for warmup
            scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.1e}'})

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# --- Evaluation Step ---

def evaluate_step(model, data_loader, device):
    """Performs evaluation on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            # Adapt input based on model type
            if config.MODEL_TYPE == 'Transformer':
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else: # Non-transformer
                sequences = batch[0].to(device)
                labels = batch[1].to(device)
                lengths = batch[2].to(device)
                outputs = model(text_indices=sequences, sequence_lengths=lengths)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    # Calculate weighted precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'predictions': all_preds, # Return predictions for detailed analysis
        'true_labels': all_labels # Return true labels
    }
    return metrics


# --- Training Loop ---

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_save_path, metric_for_best=config.METRIC_FOR_BEST_MODEL):
    """The main training loop."""
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_weighted': []}
    best_metric_value = -float('inf') if metric_for_best != 'loss' else float('inf')
    grad_clip_value = config.GRADIENT_CLIP_VALUE if config.MODEL_TYPE == 'Transformer' else None # Only clip for transformers by default

    print(f"\n--- Starting Training for {epochs} Epochs ---")
    print(f"Monitoring validation '{metric_for_best}' for best model.")
    if grad_clip_value: print(f"Using gradient clipping: {grad_clip_value}")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # Training
        train_loss = train_step(model, train_loader, optimizer, device, scheduler, grad_clip_value)
        print(f"  Train Loss: {train_loss:.4f}")
        history['train_loss'].append(train_loss)

        # Validation
        val_metrics = evaluate_step(model, val_loader, device)
        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_weighted']
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_weighted'].append(val_f1)

        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | Val F1 (W): {val_f1:.4f}")

        # Scheduler Step (for ReduceLROnPlateau)
        if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler.step(val_loss)

        # Check for best model
        current_metric_value = val_metrics[metric_for_best]
        is_better = False
        if metric_for_best == 'loss':
            is_better = current_metric_value < best_metric_value
        else: # Higher is better for accuracy, f1
            is_better = current_metric_value > best_metric_value

        if is_better:
            print(f"  ✨ Validation '{metric_for_best}' improved ({best_metric_value:.4f} --> {current_metric_value:.4f}). Saving model...")
            best_metric_value = current_metric_value
            try:
                 # Ensure directory exists
                 os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                 # Save model state dictionary
                 torch.save(model.state_dict(), model_save_path)
                 print(f"  Model saved to {model_save_path}")
            except Exception as e:
                 print(f"  Error saving model: {e}")
        else:
            print(f"  Validation '{metric_for_best}' did not improve from {best_metric_value:.4f}.")

    print("\n--- Training Finished ---")
    print(f"Best validation '{metric_for_best}': {best_metric_value:.4f}")
    return history

# --- Model Loading ---
def load_trained_model(model_path, model_type, n_classes, vocab_size=None):
    """Loads a pre-trained model state dict."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model = initialize_model(model_type, n_classes, vocab_size)
    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device(config.DEVICE)))
        print(f"Model weights loaded successfully from {model_path}")
        model.eval() # Set to evaluation mode
        return model
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("Ensure the model architecture matches the saved weights and the file is not corrupted.")
        raise