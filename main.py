# --- main.py ---
import sys
import os
import argparse

# Dynamically add project root to path if needed (if running main.py from within project dir)
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# Import necessary modules after potentially modifying path
try:
    import config # Imports first to set up paths etc.
    import train
except ImportError as e:
     print(f"Error importing core modules: {e}")
     print("Ensure config.py and train.py are in the Python path or the same directory.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Emotion Classification Framework - Training Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
        )

    parser.add_argument(
        '--model_type',
        type=str,
        choices=['Transformer', 'CNN_RNN_Attention', 'LSTM'],
        default=None,
        help='''Override the MODEL_TYPE set in config.py for this run.
If not provided, the value from config.py will be used.
This determines the model architecture and the artifact directory used.'''
    )
    # Add other potential command-line overrides here if needed
    # e.g., --epochs, --batch_size, --data_dir etc.
    # Example:
    # parser.add_argument('--epochs', type=int, default=None, help='Override config.EPOCHS')
    # parser.add_argument('--train_file', type=str, default=None, help='Override config.TRAIN_FILE_PATH')

    args = parser.parse_args()

    # --- Override Config if Specified ---
    # Store original config value if needed
    original_model_type = config.MODEL_TYPE
    config_overridden = False

    if args.model_type and args.model_type != config.MODEL_TYPE:
        print(f"Overriding config.MODEL_TYPE: '{config.MODEL_TYPE}' -> '{args.model_type}'")
        config.MODEL_TYPE = args.model_type
        config_overridden = True
        # IMPORTANT: Re-evaluate paths in config that depend on MODEL_TYPE
        config.MODEL_TYPE_ARTIFACTS_DIR = os.path.join(config.ARTIFACTS_DIR, config.MODEL_TYPE)
        config.MODEL_SAVE_DIR = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, "model")
        config.BEST_MODEL_PATH = os.path.join(config.MODEL_SAVE_DIR, config.BEST_MODEL_FILENAME)
        config.VOCAB_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.VOCAB_FILENAME)
        config.TRAINING_PLOTS_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.TRAINING_PLOTS_FILENAME)
        config.TEST_REPORT_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.TEST_REPORT_FILENAME)
        config.CONFUSION_MATRIX_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.CONFUSION_MATRIX_FILENAME)
        config.RUN_CONFIG_PATH = os.path.join(config.MODEL_TYPE_ARTIFACTS_DIR, config.RUN_CONFIG_FILENAME)
        # Re-create directories for the potentially new model type
        os.makedirs(config.MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        print(f"Artifact directory updated to: {config.MODEL_TYPE_ARTIFACTS_DIR}")
        # Adjust other MODEL_TYPE dependent configs if necessary (e.g., BATCH_SIZE, LR)
        # This part requires careful handling - you might want to define defaults per model type
        # or require users to set all relevant configs if overriding MODEL_TYPE.
        # For now, we rely on the user ensuring config.py has sensible defaults for the chosen type
        # or using more CLI args to override them.
        print("Note: Other config settings (like BATCH_SIZE, LR, etc.) were NOT automatically adjusted based on the overridden model type. Ensure they are appropriate.")


    # Add similar blocks here if overriding other configs like epochs, paths, etc.
    # if args.epochs:
    #    print(f"Overriding config.EPOCHS: {config.EPOCHS} -> {args.epochs}")
    #    config.EPOCHS = args.epochs
    #    config_overridden = True


    print("========================================")
    print("=== Emotion Classification Framework ===")
    print("========================================")
    print(f"Starting training run for MODEL_TYPE: {config.MODEL_TYPE}")
    print(f"Artifacts will be saved in: {config.MODEL_TYPE_ARTIFACTS_DIR}")


    # Ensure artifact directories exist (config.py already does this, but check again)
    try:
        os.makedirs(config.MODEL_TYPE_ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
        # Ensure base artifacts dir exists (for global label map)
        if os.path.dirname(config.LABEL_MAP_PATH):
             os.makedirs(os.path.dirname(config.LABEL_MAP_PATH), exist_ok=True)
    except OSError as e:
        print(f"Error creating artifact directories: {e}")
        sys.exit(1)


    # --- Save Configuration ---
    # Save the potentially overridden configuration *before* starting training
    try:
        # Use the correct save path based on the (potentially overridden) MODEL_TYPE
        config.save_run_config(filepath=config.RUN_CONFIG_PATH)
        print(f"Current run configuration saved to {config.RUN_CONFIG_PATH}")
    except Exception as e:
        # Non-fatal warning if config save fails
        print(f"Warning: Failed to save run configuration. {e}")


    # --- Execute Training ---
    try:
        # run_training_pipeline now uses the current state of the config module
        train.run_training_pipeline()
        print("\n--- Training Pipeline Completed Successfully ---")
    except KeyboardInterrupt:
        print("\n--- Training Interrupted by User ---")
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\n--- File Not Found Error ---")
        print(f"Error: {e}")
        print("Please check the file paths specified in config.py (e.g., TRAIN_FILE_PATH).")
        sys.exit(1)
    except ImportError as e:
        print(f"\n--- Import Error ---")
        print(f"Error: {e}")
        print("Please ensure all required libraries are installed (e.g., transformers, torch, spacy, nltk, sklearn, pandas).")
        print("Check specific messages above for missing libraries like 'transformers' or 'spacy'.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- An Unhandled Error Occurred During Training ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("----------------------------------------------------")
        # Optionally print traceback for debugging
        import traceback
        traceback.print_exc()
        print("----------------------------------------------------")
        print("Training failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()