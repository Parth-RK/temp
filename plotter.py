# --- plotter.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix

import config # For default save paths

sns.set_theme(style="whitegrid")

def plot_training_history(history, save_path=config.TRAINING_PLOTS_PATH):
    """
    Plots training and validation loss, accuracy, and F1 score over epochs.

    Args:
        history (dict): Dictionary containing lists of metrics per epoch
                        (e.g., 'train_loss', 'val_loss', 'val_accuracy', 'val_f1_weighted').
        save_path (str): Path to save the plot image.
    """
    if not history:
        print("Plotter Warning: History dictionary is empty. Cannot plot training history.")
        return

    epochs = range(1, len(history['train_loss']) + 1)
    df = pd.DataFrame(history)
    df['epoch'] = epochs

    num_plots = 0
    if 'train_loss' in df and 'val_loss' in df: num_plots += 1
    if 'val_accuracy' in df: num_plots += 1
    if 'val_f1_weighted' in df: num_plots += 1

    if num_plots == 0:
        print("Plotter Warning: No plottable metrics found in history dict.")
        return

    plt.figure(figsize=(8 * num_plots, 5)) # Adjust figure size based on number of plots

    plot_idx = 1
    # --- Loss Plot ---
    if 'train_loss' in df and 'val_loss' in df:
        plt.subplot(1, num_plots, plot_idx)
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o', linestyle='-')
        plt.plot(df['epoch'], df['val_loss'], label='Validation Loss', marker='x', linestyle='--')
        plt.title('Loss vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plot_idx += 1

    # --- Accuracy Plot ---
    if 'val_accuracy' in df:
        plt.subplot(1, num_plots, plot_idx)
        # Add train accuracy if available in history
        if 'train_accuracy' in df:
            plt.plot(df['epoch'], df['train_accuracy'], label='Train Accuracy', marker='o', linestyle='-')
        plt.plot(df['epoch'], df['val_accuracy'], label='Validation Accuracy', marker='x', linestyle='--')
        plt.title('Accuracy vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim(bottom=max(0, df['val_accuracy'].min() - 0.1), top=min(1, df['val_accuracy'].max() + 0.1)) # Adjust ylim
        plt.legend()
        plt.grid(True)
        plot_idx += 1

    # --- F1 Score Plot ---
    if 'val_f1_weighted' in df:
        plt.subplot(1, num_plots, plot_idx)
         # Add train F1 if available in history
        if 'train_f1_weighted' in df:
            plt.plot(df['epoch'], df['train_f1_weighted'], label='Train F1 (Weighted)', marker='o', linestyle='-')
        plt.plot(df['epoch'], df['val_f1_weighted'], label='Validation F1 (Weighted)', marker='x', linestyle='--')
        plt.title('Weighted F1 Score vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.ylim(bottom=max(0, df['val_f1_weighted'].min() - 0.1), top=min(1, df['val_f1_weighted'].max() + 0.1)) # Adjust ylim
        plt.legend()
        plt.grid(True)
        plot_idx += 1


    plt.tight_layout()

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Plotter Error: Could not save training plot to {save_path}. Error: {e}")
    # plt.show() # Optional: Show plot directly

def generate_classification_analysis(true_labels, predictions, int_to_label, report_path=None, cm_path=None, prefix=""):
    """
    Generates and saves a classification report and confusion matrix.

    Args:
        true_labels (list or np.array): Ground truth integer labels.
        predictions (list or np.array): Predicted integer labels.
        int_to_label (dict): Mapping from integer labels to string names.
        report_path (str, optional): Path to save the text classification report.
        cm_path (str, optional): Path to save the confusion matrix plot.
        prefix (str, optional): Prefix for report/plot titles (e.g., "Test Set").
    """
    if not int_to_label:
        print("Plotter Warning: int_to_label mapping not provided. Using integer labels.")
        # Use unique sorted integer labels present in the data
        unique_labels = sorted(list(set(true_labels) | set(predictions)))
        label_names = [str(i) for i in unique_labels]
        target_labels_for_report = unique_labels # Use integers for report labels arg
    else:
        # Ensure keys are integers and values are strings
        int_to_label = {int(k): str(v) for k, v in int_to_label.items()}
        # Use labels present in the data, map them to names using provided map
        unique_labels_present = sorted(list(set(true_labels) | set(predictions)))
        label_names = [int_to_label.get(i, f"Unknown({i})") for i in unique_labels_present]
        target_labels_for_report = unique_labels_present # Use integers for report labels arg


    # --- Classification Report ---
    try:
        report_str = classification_report(
            true_labels,
            predictions,
            labels=target_labels_for_report, # Specify labels to include
            target_names=label_names,
            zero_division=0,
            digits=3
        )
        title = f"{prefix} Classification Report" if prefix else "Classification Report"
        full_report_output = f"--- {title} ---\n\n{report_str}\n"
        print(full_report_output) # Print to console

        if report_path:
            try:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w') as f:
                    # Optionally add overall metrics to the top of the report file
                    accuracy = np.mean(np.array(true_labels) == np.array(predictions))
                    f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
                    f.write(report_str)
                print(f"Classification report saved to {report_path}")
            except Exception as e:
                print(f"Plotter Error: Could not save classification report to {report_path}. Error: {e}")

    except Exception as e:
        print(f"Plotter Error: Could not generate classification report. Error: {e}")


    # --- Confusion Matrix ---
    if cm_path:
        try:
            cm = confusion_matrix(true_labels, predictions, labels=target_labels_for_report)
            plt.figure(figsize=(max(8, len(label_names)*0.6), max(6, len(label_names)*0.5))) # Dynamic sizing
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names,
                        annot_kws={"size": 8}) # Adjust font size if needed
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            cm_title = f"{prefix} Confusion Matrix" if prefix else "Confusion Matrix"
            plt.title(cm_title)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            # plt.show() # Optional: Show plot directly
            plt.close() # Close the plot figure

        except Exception as e:
            print(f"Plotter Error: Could not generate or save confusion matrix. Error: {e}")
