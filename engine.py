# --- engine.py ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import time # For timing epochs

# Try importing transformer-specific scheduler
try:
    from transformers import get_linear_schedule_with_warmup
except ImportError:
    get_linear_schedule_with_warmup = None
    print("Warning: HuggingFace Transformers library not installed. Linear warmup scheduler will not be available.")


import config # Import configuration

# --- Model Initialization ---

def initialize_model(model_type, n_classes, vocab_size=None):
    """Initializes the model based on the configuration."""
    print(f"\nInitializing model: {model_type} with {n_classes} classes")
    if model_type == 'Transformer':
        from models import TransformerClassifier # Local import
        if not hasattr(config, 'TRANSFORMER_MODEL_NAME'):
             raise ValueError("config.TRANSFORMER_MODEL_NAME must be set for Transformer model type.")
        model = TransformerClassifier(
            model_name=config.TRANSFORMER_MODEL_NAME,
            n_classes=n_classes
            # Dropout is handled within the model using AutoConfig
        )
    elif model_type == 'CNN_RNN_Attention':
        from models import CNN_RNN_Attention # Local import
        if vocab_size is None: raise ValueError("vocab_size required for CNN_RNN_Attention")
        # Ensure necessary configs are present
        for cfg_name in ['EMBEDDING_DIM', 'CNN_OUT_CHANNELS', 'CNN_KERNEL_SIZES', 'RNN_TYPE', 'RNN_HIDDEN_DIM', 'RNN_LAYERS', 'DROPOUT_PROB', 'PAD_IDX']:
            if not hasattr(config, cfg_name): raise ValueError(f"config.{cfg_name} must be set for CNN_RNN_Attention.")
        model = CNN_RNN_Attention(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            rnn_type=config.RNN_TYPE,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            rnn_layers=config.RNN_LAYERS,
            n_class=n_classes,
            dropout_prob=config.DROPOUT_PROB, # Use dedicated dropout config
            pad_idx=config.PAD_IDX
        )
    elif model_type == 'LSTM':
        from models import LSTMNetwork # Local import
        if vocab_size is None: raise ValueError("vocab_size required for LSTM")
         # Ensure necessary configs are present
        for cfg_name in ['EMBEDDING_DIM', 'RNN_HIDDEN_DIM', 'RNN_LAYERS', 'DROPOUT_PROB', 'PAD_IDX']:
             if not hasattr(config, cfg_name): raise ValueError(f"config.{cfg_name} must be set for LSTM.")
        model = LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.RNN_HIDDEN_DIM,
            n_class=n_classes,
            n_layers=config.RNN_LAYERS,
            pad_idx=config.PAD_IDX,
            dropout_prob=config.DROPOUT_PROB # Use dedicated dropout config
        )
    else:
        raise ValueError(f"Unsupported MODEL_TYPE in config: {model_type}")

    model.to(config.DEVICE)
    print(f"Model '{model_type}' initialized and moved to {config.DEVICE}")
    # Print parameter count
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
    except Exception as e:
        print(f"  Could not calculate parameter count: {e}")
    return model

# --- Optimizer and Scheduler ---

def initialize_optimizer_scheduler(model, optimizer_type, scheduler_type, num_train_steps=None):
    """Initializes optimizer and scheduler based on config."""
    print(f"\nInitializing Optimizer: {optimizer_type}, Scheduler: {scheduler_type}")
    lr = config.LEARNING_RATE
    wd = config.WEIGHT_DECAY

    optimizer = None
    if optimizer_type == 'AdamW':
        # Differentiate parameters for weight decay (common for Transformers, good default)
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': wd},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
        print(f"  Using AdamW with LR={lr}, Weight Decay={wd} (applied selectively)")
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)
        print(f"  Using Adam with LR={lr}, Weight Decay={wd}")
    elif optimizer_type == 'SGD':
         # Add momentum if desired for SGD
         momentum = getattr(config, 'MOMENTUM', 0.9) # Default momentum if not in config
         optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd, momentum=momentum)
         print(f"  Using SGD with LR={lr}, Weight Decay={wd}, Momentum={momentum}")
    else:
        raise ValueError(f"Unsupported OPTIMIZER_TYPE: {optimizer_type}")

    scheduler = None
    if scheduler_type == 'linear_warmup':
        if get_linear_schedule_with_warmup is None:
             print("Warning: 'linear_warmup' scheduler selected, but Transformers library not installed. No scheduler used.")
        elif num_train_steps is None:
            raise ValueError("num_train_steps is required for linear_warmup scheduler")
        else:
            num_warmup_steps = int(num_train_steps * config.WARMUP_PROPORTION)
            print(f"  Using Linear Warmup scheduler: Total steps={num_train_steps}, Warmup steps={num_warmup_steps}")
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_train_steps
            )
    elif scheduler_type == 'reduce_on_plateau':
        # Monitors validation loss by default
        # Get patience from config or use a default
        patience = getattr(config, 'SCHEDULER_PATIENCE', 2)
        factor = getattr(config, 'SCHEDULER_FACTOR', 0.1)
        print(f"  Using ReduceLROnPlateau scheduler: Factor={factor}, Patience={patience}, Monitoring 'val_loss'")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, verbose=True)
    elif scheduler_type is None or scheduler_type.lower() == 'none':
         print("  No learning rate scheduler selected.")
    else:
        print(f"Warning: Scheduler type '{scheduler_type}' requested but not implemented or recognized. No scheduler used.")


    return optimizer, scheduler

# --- Loss Function ---
# Using CrossEntropyLoss, suitable for multi-class classification
criterion = nn.CrossEntropyLoss()
print(f"\nUsing Loss Function: CrossEntropyLoss")

# --- Training Step ---

def train_step(model, data_loader, optimizer, device, scheduler=None, grad_clip_value=None):
    """Performs a single training epoch."""
    model.train() # Set model to training mode
    total_loss = 0.0
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Training", leave=False, unit="batch")

    for batch_idx, batch in enumerate(progress_bar):
        optimizer.zero_grad() # Clear gradients from previous batch

        # --- Input Handling based on Model Type ---
        try:
            if config.MODEL_TYPE == 'Transformer':
                # Assumes batch is a dictionary from GenericDataset/DataLoader
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else: # Non-transformer models (LSTM, CNN_RNN)
                # Assumes batch is a tuple (sequences, labels, lengths) from collate_non_transformer
                sequences = batch[0].to(device)
                labels = batch[1].to(device)
                lengths = batch[2] # Lengths stay on CPU for pack_padded_sequence
                # Forward pass - model expects lengths
                outputs = model(text_indices=sequences, sequence_lengths=lengths)

        except Exception as e:
             print(f"\nError during forward pass in training batch {batch_idx}: {e}")
             print(f"Batch keys/type: {type(batch)}")
             if isinstance(batch, dict): print(f"Keys: {batch.keys()}")
             elif isinstance(batch, (list, tuple)): print(f"Length: {len(batch)}")
             # Optionally: print shapes or skip batch
             # continue # Skip this batch if error occurs
             raise # Re-raise the error to stop training

        # --- Loss Calculation & Backpropagation ---
        loss = criterion(outputs, labels)
        loss.backward() # Compute gradients

        # --- Gradient Clipping (Optional) ---
        if grad_clip_value is not None and grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

        # --- Optimizer & Scheduler Step ---
        optimizer.step() # Update model weights
        # Step linear warmup scheduler *after* optimizer step
        if scheduler and config.SCHEDULER_TYPE == 'linear_warmup':
            scheduler.step()

        # --- Logging & Progress Bar ---
        total_loss += loss.item()
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}' # Get current LR
        })

    # --- Epoch End ---
    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time
    print(f"  Train Avg. Loss: {avg_loss:.4f} | Time: {elapsed_time:.2f}s")
    return avg_loss

# --- Evaluation Step ---

def evaluate_step(model, data_loader, device):
    """Performs evaluation on a dataset (validation or test)."""
    if data_loader is None or len(data_loader) == 0:
        print("  Evaluation skipped: DataLoader is empty or None.")
        # Return default/empty metrics to avoid errors downstream
        return {
            'loss': float('nan'), 'accuracy': 0.0, 'precision_weighted': 0.0,
            'recall_weighted': 0.0, 'f1_weighted': 0.0,
            'predictions': [], 'true_labels': []
        }

    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    start_time = time.time()
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch")

    with torch.no_grad(): # Disable gradient calculations for evaluation
        for batch_idx, batch in enumerate(progress_bar):
            # --- Input Handling (same logic as train_step) ---
            try:
                if config.MODEL_TYPE == 'Transformer':
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                else: # Non-transformer
                    sequences = batch[0].to(device)
                    labels = batch[1].to(device)
                    lengths = batch[2] # CPU
                    outputs = model(text_indices=sequences, sequence_lengths=lengths)

            except Exception as e:
                 print(f"\nError during forward pass in evaluation batch {batch_idx}: {e}")
                 # Decide how to handle: skip batch or raise error
                 # continue
                 raise

            # --- Loss Calculation ---
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # --- Predictions ---
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # --- Progress Bar ---
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
             })

    # --- Epoch End ---
    avg_loss = total_loss / len(data_loader)
    elapsed_time = time.time() - start_time

    # --- Calculate Metrics ---
    # Ensure labels/preds are numpy arrays
    all_labels_np = np.array(all_labels)
    all_preds_np = np.array(all_preds)

    accuracy = accuracy_score(all_labels_np, all_preds_np)
    # Calculate weighted precision, recall, F1 - use zero_division=0 to handle cases with no preds/labels for a class
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels_np, all_preds_np, average='weighted', zero_division=0
    )

    print(f"  Eval Avg. Loss:  {avg_loss:.4f} | Accuracy: {accuracy:.4f} | F1 (W): {f1:.4f} | Time: {elapsed_time:.2f}s")

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'predictions': all_preds, # Return predictions for detailed analysis (e.g., confusion matrix)
        'true_labels': all_labels # Return true labels
    }
    return metrics


# --- Training Loop ---

def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs, model_save_path, metric_for_best=config.METRIC_FOR_BEST_MODEL):
    """The main training loop."""
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_f1_weighted': []}
    # Initialize best metric based on whether higher is better (accuracy, f1) or lower is better (loss)
    best_metric_value = -float('inf') if metric_for_best != 'loss' else float('inf')
    # Determine gradient clipping value from config
    grad_clip_value = getattr(config, 'GRADIENT_CLIP_VALUE', None) # Use None if not defined

    print(f"\n--- Starting Training ---")
    print(f"Model Type: {config.MODEL_TYPE}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    print(f"Optimizer: {config.OPTIMIZER_TYPE}, Scheduler: {config.SCHEDULER_TYPE}")
    print(f"Monitoring validation '{metric_for_best}' for best model.")
    if grad_clip_value: print(f"Using gradient clipping: {grad_clip_value}")
    print(f"Model checkpoints will be saved to: {model_save_path}")

    start_training_time = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")

        # --- Training Phase ---
        train_loss = train_step(model, train_loader, optimizer, device, scheduler, grad_clip_value)
        history['train_loss'].append(train_loss)

        # --- Validation Phase ---
        val_metrics = evaluate_step(model, val_loader, device)

        # Handle case where validation loader was empty
        if val_metrics['loss'] is float('nan'):
             print("  Skipping validation metrics recording and best model check due to empty validation set.")
             continue # Proceed to next epoch

        val_loss = val_metrics['loss']
        val_accuracy = val_metrics['accuracy']
        val_f1 = val_metrics['f1_weighted']
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1_weighted'].append(val_f1)
        # Note: The print statement for eval metrics is now inside evaluate_step

        # --- Scheduler Step (for ReduceLROnPlateau) ---
        if scheduler and config.SCHEDULER_TYPE == 'reduce_on_plateau':
            scheduler.step(val_loss) # Pass the validation loss

        # --- Check for Best Model ---
        current_metric_value = val_metrics.get(metric_for_best, None)
        if current_metric_value is None:
             print(f"Warning: Metric '{metric_for_best}' not found in validation metrics. Cannot determine best model.")
             continue # Skip saving if metric is missing

        is_better = False
        if metric_for_best == 'loss':
            # Lower loss is better
            is_better = current_metric_value < best_metric_value
        else:
            # Higher accuracy/f1 is better
            is_better = current_metric_value > best_metric_value

        if is_better:
            print(f"  ✨ Validation '{metric_for_best}' improved ({best_metric_value:.4f} --> {current_metric_value:.4f}). Saving model...")
            best_metric_value = current_metric_value
            try:
                 # Ensure the directory exists (config.py should handle this, but double check)
                 os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                 # Save model state dictionary
                 torch.save(model.state_dict(), model_save_path)
                 print(f"     Model saved to {model_save_path}")
            except Exception as e:
                 print(f"     Error saving model: {e}")
                 # Decide whether to continue training or stop if saving fails
        else:
            print(f"  Validation '{metric_for_best}' ({current_metric_value:.4f}) did not improve from best ({best_metric_value:.4f}).")

    # --- Training End ---
    end_training_time = time.time()
    total_training_time = end_training_time - start_training_time
    print("\n--- Training Finished ---")
    print(f"Total Training Time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)")
    print(f"Best validation '{metric_for_best}' achieved: {best_metric_value:.4f}")
    print(f"Model artifacts saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")
    return history

# --- Model Loading ---
def load_trained_model(model_path, model_type, n_classes, vocab_size=None):
    """
    Loads a pre-trained model's state dict.

    Args:
        model_path (str): Path to the saved .pt file (state_dict).
        model_type (str): Type of model ('Transformer', 'LSTM', 'CNN_RNN_Attention').
        n_classes (int): Number of output classes the model was trained for.
        vocab_size (int, optional): Vocabulary size, required for non-Transformer models.

    Returns:
        torch.nn.Module: The loaded model, in evaluation mode.

    Raises:
        FileNotFoundError: If the model_path does not exist.
        ValueError: If configuration mismatch (e.g., missing vocab_size).
        Exception: For other PyTorch loading errors.
    """
    print(f"\nAttempting to load model weights from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    # 1. Initialize a model instance with the same architecture
    # This requires n_classes and potentially vocab_size
    try:
        model = initialize_model(model_type, n_classes, vocab_size)
        # Note: initialize_model already prints details and moves to device
    except ValueError as e:
         print(f"Error initializing model structure before loading weights: {e}")
         raise
    except Exception as e:
         print(f"Unexpected error initializing model structure: {e}")
         raise

    # 2. Load the state dictionary
    try:
        # Load state dict, ensuring it's loaded onto the correct device specified in config
        state_dict = torch.load(model_path, map_location=torch.device(config.DEVICE))
        model.load_state_dict(state_dict)
        print(f"Model weights loaded successfully onto {config.DEVICE}.")
        model.eval() # Set to evaluation mode
        return model
    except FileNotFoundError:
         # Should be caught earlier, but defensive check
         print(f"Error: Model file disappeared before loading: {model_path}")
         raise
    except Exception as e:
        print(f"Error loading model state_dict from {model_path}: {e}")
        print("This could be due to:")
        print("  - Corrupted model file.")
        print("  - Mismatch between the saved weights and the current model architecture.")
        print("    (Check config.py settings like hidden dimensions, layers, etc., match the trained model).")
        print("  - Issues during file reading.")
        raise # Re-raise the exception after providing context