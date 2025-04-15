# engine.py
"""
Contains functions for training, evaluation, plotting, and saving/loading checkpoints.
(TorchText Legacy Independent Version - Flexible Classes)
"""
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
from tqdm import tqdm
import config # For device, paths

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between true labels and predicted labels."""
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def evaluate(model, data_loader, criterion, device):
    """Evaluates the model on a given dataset."""
    print("Evaluating...")
    total_loss, total_acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in tqdm(data_loader, desc="Evaluation", leave=False):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            batch_loss = criterion(y_pred_logits, y)
            total_loss += batch_loss.item()

            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            total_acc += accuracy_fn(y, y_pred_class)

    avg_acc = total_acc / len(data_loader)
    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation Complete - Avg Loss: {avg_loss:.5f}, Avg Accuracy: {avg_acc:.2f}%")
    return avg_acc, avg_loss

def trainer(model, train_loader, optimizer, criterion, epochs, device, val_loader=None, model_save_path=None):
    """Trains the model."""
    history = {"train_loss": [], "train_acc": [], "epoch": []}
    if val_loader:
        history["val_loss"] = []
        history["val_acc"] = []

    best_val_loss = float('inf')
    model.to(device)
    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0, 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs} [Train]")
        for batch_idx, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            loss = criterion(y_pred_logits, y)

            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Optional clip
            optimizer.step()

            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            batch_acc = accuracy_fn(y, y_pred_class)
            epoch_loss += loss.item()
            epoch_acc += batch_acc
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}'})

        print(f"\n--- Evaluating Epoch {epoch} ---")
        train_acc, train_loss = evaluate(model, train_loader, criterion, device) # Evaluate on train set end of epoch
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["epoch"].append(epoch)

        log_message = f"Epoch: {epoch}/{epochs} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.2f}%"

        if val_loader:
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            log_message += f" | Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.2f}%"

            if val_loss < best_val_loss and model_save_path:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, model_save_path)
                log_message += " ✨ Best Model Saved ✨"

        print(log_message)
        print("-" * 50)

    print("Training Finished.")
    if model_save_path and os.path.exists(model_save_path) and val_loader:
         print(f"Loading best model from {model_save_path} based on validation loss.")
         load_checkpoint(model_save_path, model, optimizer, device)

    return model, pd.DataFrame(history)


def save_checkpoint(model, optimizer, epoch, filepath):
    """Saves model and optimizer state."""
    print(f"Saving checkpoint to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        # Optionally save model hyperparameters needed for reload
        # 'n_class': model.fc.out_features # Example if needed
    }
    torch.save(checkpoint, filepath)
    print("Checkpoint saved.")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
    """Loads model and optimizer state from a checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found at {filepath}")
    print(f"Loading checkpoint from {filepath}...")
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except ValueError as e:
            print(f"Warning: Could not load optimizer state dict. {e}")
            print("Optimizer state will be reset.")
    else:
         print("Info: Optimizer state not loaded (not found or optimizer not provided).")

    epoch = checkpoint.get('epoch', 'N/A')
    print(f"Checkpoint loaded. Model weights loaded from epoch {epoch}.")
    model.to(device)
    return checkpoint.get('epoch', 0)

# save_final_model, load_final_model, plot_history remain unchanged but are needed
def save_final_model(model, filepath):
    """Saves only the final model state_dict."""
    print(f"Saving final model state_dict to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    print("Final model saved.")

def load_final_model(model, filepath, device='cpu'):
    """Loads the final model state_dict."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at {filepath}")
    print(f"Loading final model state_dict from {filepath}...")
    model.load_state_dict(torch.load(filepath, map_location=device))
    model.to(device)
    model.eval()
    print("Final model loaded.")

def plot_history(df, save_path=None):
    """Plots training and validation loss and accuracy."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o')
    if "val_loss" in df.columns:
        plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker='o')
    if "val_acc" in df.columns:
        plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Train and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    plt.show()