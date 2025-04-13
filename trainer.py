# trainer.py
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

def get_device():
    """Gets the appropriate device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def calculate_accuracy(preds, y):
    """Calculates accuracy."""
    max_preds = preds.argmax(dim=1, keepdim=True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum().float() / torch.tensor(y.shape[0]).float()

def train_epoch(model, iterator, optimizer, criterion, device):
    """Performs one training epoch."""
    epoch_loss = 0
    epoch_acc = 0
    model.train() # Set model to training mode (enables dropout, batch norm etc.)

    for batch in iterator:
        # Move batch to device
        text, labels, lengths = batch
        text = text.to(device)
        labels = labels.to(device)
        # lengths tensor stays on CPU for pack_padded_sequence

        optimizer.zero_grad() # Clear gradients from previous iteration

        # Forward pass
        predictions = model(text, lengths) # Pass lengths

        # Calculate loss
        loss = criterion(predictions, labels)

        # Calculate accuracy
        acc = calculate_accuracy(predictions, labels)

        # Backward pass (calculate gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    """Evaluates the model on a dataset."""
    epoch_loss = 0
    epoch_acc = 0
    model.eval() # Set model to evaluation mode (disables dropout, batch norm etc.)
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        for batch in iterator:
            text, labels, lengths = batch
            text = text.to(device)
            labels = labels.to(device)

            predictions = model(text, lengths)
            loss = criterion(predictions, labels)
            acc = calculate_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

            all_preds.extend(torch.argmax(predictions, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return epoch_loss / len(iterator), epoch_acc / len(iterator), all_preds, all_labels


def train_model(model, train_iterator, valid_iterator, optimizer, criterion, device,
                n_epochs, label_encoder, model_save_path="output/best_model.pt"):
    """Main training loop."""
    best_valid_loss = float('inf')
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []

    print(f"\nStarting training for {n_epochs} epochs...")

    for epoch in range(n_epochs):
        start_time = time.time()

        train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc, _, _ = evaluate(model, valid_iterator, criterion, device) # Eval doesn't need preds/labels here

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Store metrics
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)

        # Save the best model based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"|-----> Best model saved to {model_save_path}")

        print(f'Epoch: {epoch+1:02} | Epoch Time: {int(epoch_mins)}m {int(epoch_secs)}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

    print("\nTraining finished.")

    # --- Final Evaluation & Report ---
    print("\nLoading best model for final evaluation...")
    try:
        model.load_state_dict(torch.load(model_save_path))
        model.to(device) # Ensure model is on the correct device after loading
        print("Best model loaded successfully.")
    except Exception as e:
        print(f"Error loading best model state dict from {model_save_path}: {e}")
        print("Proceeding with the model state from the last epoch for final evaluation.")


    print("\nFinal evaluation on validation set using best model:")
    final_valid_loss, final_valid_acc, final_preds, final_labels = evaluate(model, valid_iterator, criterion, device)
    print(f'Final Validation Loss: {final_valid_loss:.3f} | Final Validation Acc: {final_valid_acc*100:.2f}%')

    try:
        print("\nClassification Report (Validation Set):")
        report = classification_report(final_labels, final_preds, target_names=label_encoder.classes_, zero_division=0)
        print(report)

        # Plot Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(final_labels, final_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Validation Set)')
        output_dir = os.path.dirname(model_save_path)
        plot_path = os.path.join(output_dir, "confusion_matrix_torch.png")
        plt.savefig(plot_path)
        print(f"Confusion matrix saved to {plot_path}")
        plt.close()

    except Exception as e:
        print(f"Could not generate classification report or confusion matrix: {e}")

    return model # Return the trained model (best state loaded)