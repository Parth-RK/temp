# --- plotter.py ---
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import warnings

# Try importing sklearn, provide warning if unavailable
try:
    from sklearn.metrics import classification_report, confusion_matrix
except ImportError:
    classification_report = None
    confusion_matrix = None
    print("Warning: scikit-learn not installed. Classification report and confusion matrix generation will be unavailable.")
    print("         Install it using: pip install scikit-learn")

import config # For default save paths

# Apply a default style
sns.set_theme(style="whitegrid")

def plot_training_history(history, save_path=None):
    """
    Plots training and validation loss, accuracy, and F1 score over epochs.

    Args:
        history (dict): Dictionary containing lists of metrics per epoch
                        (e.g., 'train_loss', 'val_loss', 'val_accuracy', 'val_f1_weighted').
                        Keys must match the expected metric names.
        save_path (str, optional): Path to save the plot image. If None (default), uses
                                   config.TRAINING_PLOTS_PATH.
    """
    if not isinstance(history, dict) or not history:
        print("Plotter Warning: History dictionary is empty or invalid. Cannot plot training history.")
        return

    # Use default save path from config if none provided
    if save_path is None:
        save_path = getattr(config, 'TRAINING_PLOTS_PATH', None)
        if save_path is None:
             print("Plotter Error: Default save path (config.TRAINING_PLOTS_PATH) not found and no save_path provided.")
             return # Cannot save

    # Check for essential keys
    if 'train_loss' not in history or not history['train_loss']:
         print("Plotter Warning: 'train_loss' not found or empty in history. Cannot plot.")
         return
    if 'val_loss' not in history or not history['val_loss']:
         print("Plotter Warning: 'val_loss' not found or empty in history. Loss plot will only show training loss.")
         # Continue, but plot will be incomplete

    epochs = range(1, len(history['train_loss']) + 1)
    df = pd.DataFrame(history)
    df['epoch'] = epochs

    # Determine number of subplots needed
    num_plots = 0
    plot_config = {}
    if 'train_loss' in df and 'val_loss' in df:
        num_plots += 1
        plot_config['loss'] = {'train': 'train_loss', 'val': 'val_loss', 'title': 'Loss'}
    if 'val_accuracy' in df:
        num_plots += 1
        plot_config['accuracy'] = {'train': 'train_accuracy', 'val': 'val_accuracy', 'title': 'Accuracy'}
    if 'val_f1_weighted' in df:
         num_plots += 1
         plot_config['f1'] = {'train': 'train_f1_weighted', 'val': 'val_f1_weighted', 'title': 'Weighted F1 Score'}

    if num_plots == 0:
        print("Plotter Warning: No plottable validation metrics (accuracy, f1_weighted) found in history dict, besides loss.")
        # Optionally plot just loss if val_loss was missing earlier?
        if 'train_loss' in df:
            num_plots = 1
            plot_config = {'loss': {'train': 'train_loss', 'val': None, 'title': 'Training Loss'}}
        else:
             return # Nothing to plot


    plt.figure(figsize=(6 * num_plots, 5)) # Adjust figure size

    plot_idx = 1
    for metric_key, cfg in plot_config.items():
        plt.subplot(1, num_plots, plot_idx)
        has_train = cfg.get('train') and cfg['train'] in df
        has_val = cfg.get('val') and cfg['val'] in df

        if has_train:
            plt.plot(df['epoch'], df[cfg['train']], label=f"Train {cfg['title']}", marker='o', linestyle='-')
        if has_val:
            plt.plot(df['epoch'], df[cfg['val']], label=f"Validation {cfg['title']}", marker='x', linestyle='--')

        if not has_train and not has_val:
            print(f"Plotter Warning: No data found for {cfg['title']}. Skipping plot.")
            continue # Skip if somehow no data exists for this subplot

        plt.title(f"{cfg['title']} vs. Epoch")
        plt.xlabel('Epoch')
        plt.ylabel(cfg['title'])

        # Adjust y-limits for accuracy and F1 for better visualization
        if metric_key in ['accuracy', 'f1']:
            min_val = 0
            max_val = 1
            if has_val and not df[cfg['val']].empty:
                min_val = max(0, df[cfg['val']].min() - 0.05)
            if has_val and not df[cfg['val']].empty:
                max_val = min(1.05, df[cfg['val']].max() + 0.05)
            elif has_train and not df[cfg['train']].empty: # Fallback to train if no val
                 min_val = max(0, df[cfg['train']].min() - 0.05)
                 max_val = min(1.05, df[cfg['train']].max() + 0.05)
            plt.ylim(bottom=min_val, top=max_val)

        plt.legend()
        plt.grid(True)
        plot_idx += 1


    plt.tight_layout(pad=2.0) # Add padding between subplots

    if save_path:
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight') # Use bbox_inches='tight'
            print(f"Training history plot saved to {save_path}")
        except Exception as e:
            print(f"Plotter Error: Could not save training plot to {save_path}. Error: {e}")
    else:
        # If no save path was determined, optionally show the plot
        # plt.show()
        pass

    plt.close() # Close the figure to free memory


def generate_classification_analysis(true_labels, predictions, int_to_label, report_path=None, cm_path=None, prefix=""):
    """
    Generates and saves a classification report and confusion matrix.

    Args:
        true_labels (list or np.array): Ground truth integer labels.
        predictions (list or np.array): Predicted integer labels.
        int_to_label (dict): Mapping from integer labels to string names.
        report_path (str, optional): Path to save the text classification report. If None,
                                     uses config.TEST_REPORT_PATH.
        cm_path (str, optional): Path to save the confusion matrix plot. If None,
                                 uses config.CONFUSION_MATRIX_PATH.
        prefix (str, optional): Prefix for report/plot titles (e.g., "Test Set").
    """
    # Check if sklearn is available
    if classification_report is None or confusion_matrix is None:
        print("Plotter Info: Skipping classification analysis because scikit-learn is not installed.")
        return

    # Use default save paths from config if none provided
    if report_path is None:
        report_path = getattr(config, 'TEST_REPORT_PATH', None)
    if cm_path is None:
        cm_path = getattr(config, 'CONFUSION_MATRIX_PATH', None)

    if not report_path and not cm_path:
        print("Plotter Info: No paths provided or found in config for classification report or confusion matrix. Analysis will only be printed.")


    if not isinstance(true_labels, (list, np.ndarray)) or not isinstance(predictions, (list, np.ndarray)):
        print("Plotter Error: true_labels and predictions must be lists or numpy arrays.")
        return
    if len(true_labels) != len(predictions):
        print(f"Plotter Error: Length mismatch between true_labels ({len(true_labels)}) and predictions ({len(predictions)}).")
        return
    if len(true_labels) == 0:
        print("Plotter Info: true_labels and predictions are empty. Skipping classification analysis.")
        return


    # Determine unique labels present in the data and map them to names
    unique_labels_present = sorted(list(set(true_labels) | set(predictions)))

    if not int_to_label:
        print("Plotter Warning: int_to_label mapping not provided or empty. Using integer labels as names.")
        # Use unique sorted integer labels present in the data
        label_names = [str(i) for i in unique_labels_present]
        target_labels_for_report = unique_labels_present # Use integers for report's labels arg
    else:
        # Ensure keys are integers and values are strings for safety
        try:
            int_to_label_clean = {int(k): str(v) for k, v in int_to_label.items()}
        except (ValueError, TypeError):
            print("Plotter Warning: Could not properly convert int_to_label keys/values. Using integer labels.")
            int_to_label_clean = {} # Force fallback

        if not int_to_label_clean: # If conversion failed or original was empty
             label_names = [str(i) for i in unique_labels_present]
             target_labels_for_report = unique_labels_present
        else:
             # Map labels present in data to names, handle missing mappings
             label_names = [int_to_label_clean.get(i, f"Unknown({i})") for i in unique_labels_present]
             target_labels_for_report = unique_labels_present # Use integers for report's labels arg


    # --- Classification Report ---
    try:
        # Use target_labels_for_report to ensure order and inclusion matches label_names
        report_str = classification_report(
            true_labels,
            predictions,
            labels=target_labels_for_report, # Specify labels to include/order
            target_names=label_names,         # Corresponding names
            zero_division=0,                  # Avoid warnings for undefined metrics
            digits=4                          # Number of digits for precision
        )
        title = f"{prefix} Classification Report" if prefix else "Classification Report"
        # Calculate overall accuracy separately for display
        accuracy = np.mean(np.array(true_labels) == np.array(predictions))
        full_report_output = f"\n--- {title} ---\n"
        full_report_output += f"Overall Accuracy: {accuracy:.4f}\n\n"
        full_report_output += report_str
        full_report_output += "\n-----------------------------------\n"

        print(full_report_output) # Print to console

        # Save report to file if path provided
        if report_path:
            try:
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, 'w', encoding='utf-8') as f:
                    # Write the same content as printed
                    f.write(full_report_output)
                print(f"Classification report saved to {report_path}")
            except Exception as e:
                print(f"Plotter Error: Could not save classification report to {report_path}. Error: {e}")

    except Exception as e:
        print(f"Plotter Error: Could not generate classification report. Error: {e}")
        import traceback
        traceback.print_exc()


    # --- Confusion Matrix ---
    if cm_path:
        try:
            # Generate confusion matrix using the ordered labels
            cm = confusion_matrix(true_labels, predictions, labels=target_labels_for_report)

            # Dynamic figure sizing based on number of labels
            fig_width = max(8, len(label_names) * 0.7)
            fig_height = max(6, len(label_names) * 0.6)
            plt.figure(figsize=(fig_width, fig_height))

            # Use seaborn heatmap for better visualization
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=label_names, yticklabels=label_names,
                        annot_kws={"size": 10}) # Adjust font size if needed

            plt.xlabel('Predicted Label', fontsize=12)
            plt.ylabel('True Label', fontsize=12)
            cm_title = f"{prefix} Confusion Matrix" if prefix else "Confusion Matrix"
            plt.title(cm_title, fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=10) # Rotate x-labels if many classes
            plt.yticks(rotation=0, fontsize=10)
            plt.tight_layout(pad=1.5) # Adjust layout

            # Save the confusion matrix plot
            os.makedirs(os.path.dirname(cm_path), exist_ok=True)
            plt.savefig(cm_path, bbox_inches='tight')
            print(f"Confusion matrix plot saved to {cm_path}")
            plt.close() # Close the plot figure to free memory

        except ValueError as e:
             print(f"Plotter Error: Could not generate confusion matrix, possibly due to label mismatch or empty data. Error: {e}")
        except Exception as e:
            print(f"Plotter Error: Could not generate or save confusion matrix plot. Error: {e}")
            import traceback
            traceback.print_exc()
    elif report_path: # If only report_path was defined, mention CM wasn't saved
        print("Plotter Info: Confusion matrix path not specified. Matrix plot not saved.")