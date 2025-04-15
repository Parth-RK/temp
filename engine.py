import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
from tqdm import tqdm
import config # Keep config import
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
import numpy as np

# Keep accuracy_fn if used internally, otherwise sklearn metrics are primary now
# def accuracy_fn(y_true, y_pred):
#     correct = torch.eq(y_true, y_pred).sum().item()
#     acc = (correct / len(y_pred)) * 100
#     return acc

# --- NEW evaluate_with_lengths function ---
def evaluate_with_lengths(model, data_loader, criterion, device):
    # print("Evaluating...") # Less verbose
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        # Expect data_loader to yield X, y, lengths
        for X, y, lengths in data_loader:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device) # Move lengths too

            # Pass lengths to the model if its forward method accepts it
            # Check if model forward accepts 'sequence_lengths' kwarg (optional robustness)
            import inspect
            sig = inspect.signature(model.forward)
            if 'sequence_lengths' in sig.parameters:
                 y_pred_logits = model(X, sequence_lengths=lengths)
            else: # Fallback if model doesn't use lengths (like simple LSTM)
                 y_pred_logits = model(X)

            batch_loss = criterion(y_pred_logits, y)
            total_loss += batch_loss.item()

            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds) * 100

    return accuracy, avg_loss, precision, recall, f1

# --- NEW trainer_with_lengths function ---
def trainer_with_lengths(model, train_loader, optimizer, criterion, epochs, device, val_loader=None, model_save_path=None, scheduler=None):
    history = {"train_loss": [], "train_acc": [], "train_precision": [], "train_recall": [], "train_f1": [], "epoch": []}
    if val_loader:
        history["val_loss"] = []
        history["val_acc"] = []
        history["val_precision"] = []
        history["val_recall"] = []
        history["val_f1"] = []

    best_val_metric = float('inf')
    metric_to_optimize = "val_loss" # Or 'val_f1'

    model.to(device)
    print(f"Starting training on {device} for {epochs} epochs...")
    print(f"Optimizing based on: {metric_to_optimize}")

    # --- Check model forward signature once ---
    import inspect
    sig = inspect.signature(model.forward)
    model_accepts_lengths = 'sequence_lengths' in sig.parameters
    if model_accepts_lengths:
         print("Model forward method accepts 'sequence_lengths'.")
    else:
         print("Model forward method does NOT accept 'sequence_lengths'.")
    # ---

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs} [Train]")
        # Expect data_loader to yield X, y, lengths
        for batch_idx, (X, y, lengths) in progress_bar:
            X, y, lengths = X.to(device), y.to(device), lengths.to(device) # Move lengths too

            # Pass lengths to model if accepted
            if model_accepts_lengths:
                y_pred_logits = model(X, sequence_lengths=lengths)
            else:
                y_pred_logits = model(X)

            loss = criterion(y_pred_logits, y)

            optimizer.zero_grad()
            loss.backward()
            # Optional: Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            # Simple batch accuracy for display (optional)
            # y_pred_class_batch = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            # batch_acc = accuracy_score(y.cpu().numpy(), y_pred_class_batch.cpu().numpy()) * 100
            # progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{batch_acc:.2f}%'})
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'}) # Keep it simple


        # --- Epoch Evaluation ---
        print(f"\n--- Evaluating Epoch {epoch} ---")
        print("Evaluating on Training Set...")
        # Use the new evaluation function
        train_acc, train_loss, train_precision, train_recall, train_f1 = evaluate_with_lengths(model, train_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_precision"].append(train_precision)
        history["train_recall"].append(train_recall)
        history["train_f1"].append(train_f1)

        log_message = (f"Epoch: {epoch}/{epochs} | "
                       f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, P: {train_precision:.3f}, R: {train_recall:.3f}, F1: {train_f1:.3f}")

        current_val_metric = float('inf')
        if val_loader:
            print("Evaluating on Validation Set...")
            # Use the new evaluation function
            val_acc, val_loss, val_precision, val_recall, val_f1 = evaluate_with_lengths(model, val_loader, criterion, device)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_precision"].append(val_precision)
            history["val_recall"].append(val_recall)
            history["val_f1"].append(val_f1)

            log_message += (f" | Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, P: {val_precision:.3f}, R: {val_recall:.3f}, F1: {val_f1:.3f}")

            if metric_to_optimize == "val_loss":
                 current_val_metric = val_loss
                 is_better = current_val_metric < best_val_metric
            elif metric_to_optimize == "val_f1":
                 current_val_metric = val_f1
                 is_better = current_val_metric > best_val_metric
            else:
                 current_val_metric = val_loss
                 is_better = current_val_metric < best_val_metric

            if is_better and model_save_path:
                best_val_metric = current_val_metric
                save_checkpoint(model, optimizer, epoch, model_save_path)
                log_message += " ✨ Best Model Saved ✨"

        # Step the scheduler if using one (e.g., ReduceLROnPlateau needs the metric)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(current_val_metric) # Pass the validation metric
            else:
                 scheduler.step() # For schedulers that step each epoch

        history["epoch"].append(epoch)
        print(log_message)
        print("-" * 70)

    print("Training Finished.")
    if model_save_path and os.path.exists(model_save_path) and val_loader:
         print(f"Loading best model from {model_save_path} based on {metric_to_optimize}.")
         load_checkpoint(model_save_path, model, optimizer=None, device=device)

    return model, pd.DataFrame(history)


# --- Keep plot_history as is ---
def plot_history(df, save_path=None):
    num_rows = 1
    if "train_precision" in df.columns: num_rows +=1
    if "train_f1" in df.columns: num_rows += 1

    plt.figure(figsize=(12, 6 * num_rows))
    sns.set_style("whitegrid")
    plot_index = 1

    # Loss
    plt.subplot(num_rows, 2, plot_index); plot_index += 1
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker='o', markersize=4)
    if "val_loss" in df.columns: plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", marker='x', markersize=5)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs. Epoch"); plt.legend(); plt.grid(True)

    # Accuracy
    plt.subplot(num_rows, 2, plot_index); plot_index += 1
    plt.plot(df["epoch"], df["train_acc"], label="Train Accuracy", marker='o', markersize=4)
    if "val_acc" in df.columns: plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", marker='x', markersize=5)
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Accuracy vs. Epoch"); plt.legend(); plt.grid(True)

    # Precision & Recall
    if "train_precision" in df.columns:
        plt.subplot(num_rows, 2, plot_index); plot_index += 1
        plt.plot(df["epoch"], df["train_precision"], label="Train Precision (W)", marker='o', markersize=4, linestyle='--')
        plt.plot(df["epoch"], df["train_recall"], label="Train Recall (W)", marker='s', markersize=4, linestyle=':')
        if "val_precision" in df.columns:
            plt.plot(df["epoch"], df["val_precision"], label="Val Precision (W)", marker='x', markersize=5, linestyle='--')
            plt.plot(df["epoch"], df["val_recall"], label="Val Recall (W)", marker='P', markersize=5, linestyle=':')
        plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("Weighted Precision & Recall vs. Epoch"); plt.legend(); plt.grid(True)

    # F1-Score
    if "train_f1" in df.columns:
        # Decide placement based on whether P/R plot exists
        if "train_precision" not in df.columns: # If no P/R plot, F1 goes to subplot 3
             plt.subplot(num_rows, 2, plot_index); plot_index += 1
        else: # Otherwise, it goes to subplot 4
             plt.subplot(num_rows, 2, plot_index); plot_index += 1

        plt.plot(df["epoch"], df["train_f1"], label="Train F1 (Weighted)", marker='o', markersize=4)
        if "val_f1" in df.columns:
            plt.plot(df["epoch"], df["val_f1"], label="Validation F1 (Weighted)", marker='x', markersize=5)
        plt.xlabel("Epoch"); plt.ylabel("F1-Score"); plt.title("Weighted F1-Score vs. Epoch"); plt.legend(); plt.grid(True)


    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    plt.show()


# --- NEW generate_test_report_with_lengths function ---
def generate_test_report_with_lengths(model, data_loader, criterion, device, int_to_label_map, report_save_path=None, conf_matrix_save_path=None):
    print("\n--- Generating Final Test Report ---")
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    # --- Check model forward signature once ---
    import inspect
    sig = inspect.signature(model.forward)
    model_accepts_lengths = 'sequence_lengths' in sig.parameters
    # ---

    with torch.inference_mode():
        # Expect X, y, lengths
        for X, y, lengths in tqdm(data_loader, desc="Test Evaluation"):
            X, y, lengths = X.to(device), y.to(device), lengths.to(device)

            if model_accepts_lengths:
                 y_pred_logits = model(X, sequence_lengths=lengths)
            else:
                 y_pred_logits = model(X)

            loss = criterion(y_pred_logits, y)
            total_loss += loss.item()
            y_pred_class = torch.softmax(y_pred_logits, dim=1).argmax(dim=1)
            all_preds.extend(y_pred_class.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_preds) * 100

    print(f"\nTest Loss: {avg_loss:.5f}")
    print(f"Test Accuracy: {accuracy:.2f}%")

    label_names = [int_to_label_map.get(i, str(i)) for i in sorted(int_to_label_map.keys())]
    # Ensure labels used in report/matrix are only those present in true/pred
    present_labels = np.unique(np.concatenate((all_labels, all_preds)))
    present_label_names = [int_to_label_map.get(i, str(i)) for i in present_labels]

    report = classification_report(all_labels, all_preds, labels=present_labels, target_names=present_label_names, zero_division=0, digits=3)
    print("\nClassification Report:")
    print(report)

    if report_save_path:
        os.makedirs(os.path.dirname(report_save_path), exist_ok=True)
        with open(report_save_path, 'w') as f:
            f.write(f"Test Loss: {avg_loss:.5f}\n")
            f.write(f"Test Accuracy: {accuracy:.2f}%\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Classification report saved to {report_save_path}")

    cm = confusion_matrix(all_labels, all_preds, labels=present_labels)
    plt.figure(figsize=(max(8, len(present_label_names)*0.6), max(6, len(present_label_names)*0.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=present_label_names, yticklabels=present_label_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if conf_matrix_save_path:
        os.makedirs(os.path.dirname(conf_matrix_save_path), exist_ok=True)
        plt.savefig(conf_matrix_save_path)
        print(f"Confusion matrix saved to {conf_matrix_save_path}")
    plt.show()

def save_checkpoint(model, optimizer, epoch, filepath):
    print(f"Saving checkpoint to {filepath}...")
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)
    print("Checkpoint saved.")

def load_checkpoint(filepath, model, optimizer=None, device='cpu'):
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