# --- train.py ---
import torch
import os
import sys
import numpy as np

# Import necessary modules from the project
try:
    import config
    import data_handler
    import engine
    import models # Although models are initialized in engine, importing helps ensure availability
    import plotter
except ImportError as e:
    print(f"Error importing necessary modules in train.py: {e}")
    print("Ensure all required files (config.py, data_handler.py, engine.py, models.py, plotter.py) are accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports in train.py: {e}")
    sys.exit(1)

def run_training_pipeline():
    """
    Executes the full training and evaluation pipeline.
    """
    print(f"\n--- Starting Training Pipeline for {config.MODEL_TYPE} ---")

    # --- 1. Data Loading and Preparation ---
    # This step loads data, handles splitting, processes labels,
    # preprocesses text, builds/loads vocab/tokenizer, and creates DataLoaders.
    try:
        train_loader, val_loader, test_loader, \
        label_to_int, int_to_label, n_classes, \
        vocab_or_tokenizer, vocab_size = data_handler.get_data_pipeline()

        # Basic validation after data loading
        if train_loader is None or len(train_loader) == 0:
            print("Error: Training DataLoader is empty or None. Cannot proceed.")
            sys.exit(1)
        if n_classes <= 1:
             print(f"Error: Only {n_classes} class(es) detected after data processing. Need at least 2 for classification.")
             sys.exit(1)
        if config.MODEL_TYPE != 'Transformer' and vocab_size is None:
             print(f"Error: Vocab size not determined for non-Transformer model ({config.MODEL_TYPE}).")
             sys.exit(1)

        print(f"\nData pipeline finished. Number of classes: {n_classes}")
        if config.MODEL_TYPE != 'Transformer':
             print(f"Vocabulary size: {vocab_size}")
        else:
             print(f"Tokenizer vocab size: {vocab_size}")

    except FileNotFoundError as e:
        print(f"\nData Loading Error: {e}")
        print("Please check the data file paths in config.py.")
        sys.exit(1)
    except ValueError as e:
         print(f"\nData Processing Error: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during data preparation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 2. Model Initialization ---
    try:
        model = engine.initialize_model(
            model_type=config.MODEL_TYPE,
            n_classes=n_classes,
            vocab_size=vocab_size # Pass vocab_size (None for Transformers)
        )
    except ValueError as e:
        print(f"\nModel Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during model initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 3. Optimizer and Scheduler Initialization ---
    try:
        # Calculate total training steps for warmup scheduler if needed
        num_train_steps = None
        if config.SCHEDULER_TYPE == 'linear_warmup':
            num_train_steps = len(train_loader) * config.EPOCHS
            if num_train_steps == 0:
                 print("Warning: Calculated num_train_steps is 0. Linear warmup scheduler may not work as expected.")


        optimizer, scheduler = engine.initialize_optimizer_scheduler(
            model=model,
            optimizer_type=config.OPTIMIZER_TYPE,
            scheduler_type=config.SCHEDULER_TYPE,
            num_train_steps=num_train_steps
        )
    except ValueError as e:
        print(f"\nOptimizer/Scheduler Initialization Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred during optimizer/scheduler setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 4. Training ---
    # The train_model function handles the epoch loop, train/eval steps,
    # checkpointing the best model, and returns the training history.
    try:
        history = engine.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader, # Pass the validation loader
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            epochs=config.EPOCHS,
            model_save_path=config.BEST_MODEL_PATH, # Path to save the best model
            metric_for_best=config.METRIC_FOR_BEST_MODEL
        )
    except Exception as e:
        print(f"\nAn unexpected error occurred during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


    # --- 5. Plot Training History (Optional) ---
    if config.PLOT_TRAINING_HISTORY and history:
        print("\n--- Plotting Training History ---")
        try:
            plotter.plot_training_history(
                history=history,
                save_path=config.TRAINING_PLOTS_PATH # Use path from config
            )
        except Exception as e:
            print(f"Warning: Failed to plot training history. Error: {e}")
    elif not history:
         print("\nSkipping training history plot: History data not available.")


    # --- 6. Final Evaluation on Test Set ---
    print("\n--- Evaluating on Test Set ---")
    if test_loader is None or len(test_loader) == 0:
        print("Test DataLoader is empty or None. Skipping final test evaluation.")
    else:
        # Load the *best* saved model for final evaluation
        try:
            print(f"Loading best model from: {config.BEST_MODEL_PATH}")
            # Re-initialize a model instance and load the saved state dict
            best_model = engine.load_trained_model(
                model_path=config.BEST_MODEL_PATH,
                model_type=config.MODEL_TYPE,
                n_classes=n_classes,
                vocab_size=vocab_size
            )
            best_model.to(config.DEVICE) # Ensure model is on the correct device

            # Evaluate the loaded best model on the test set
            test_metrics = engine.evaluate_step(
                model=best_model,
                data_loader=test_loader,
                device=config.DEVICE
            )

            # Print test metrics explicitly
            print("\n--- Test Set Performance (Best Model) ---")
            print(f"  Test Loss:      {test_metrics.get('loss', float('nan')):.4f}")
            print(f"  Test Accuracy:  {test_metrics.get('accuracy', 0.0):.4f}")
            print(f"  Test Precision: {test_metrics.get('precision_weighted', 0.0):.4f} (Weighted)")
            print(f"  Test Recall:    {test_metrics.get('recall_weighted', 0.0):.4f} (Weighted)")
            print(f"  Test F1-Score:  {test_metrics.get('f1_weighted', 0.0):.4f} (Weighted)")
            print("-------------------------------------------")


            # --- 7. Generate Test Report & Confusion Matrix (Optional) ---
            if config.GENERATE_TEST_REPORT or config.GENERATE_CONFUSION_MATRIX:
                if 'predictions' in test_metrics and 'true_labels' in test_metrics:
                    print("\n--- Generating Test Analysis ---")
                    plotter.generate_classification_analysis(
                        true_labels=test_metrics['true_labels'],
                        predictions=test_metrics['predictions'],
                        int_to_label=int_to_label, # Use the mapping from data pipeline
                        report_path=config.TEST_REPORT_PATH if config.GENERATE_TEST_REPORT else None,
                        cm_path=config.CONFUSION_MATRIX_PATH if config.GENERATE_CONFUSION_MATRIX else None,
                        prefix="Test Set"
                    )
                else:
                    print("Warning: Cannot generate test report/confusion matrix. Predictions or true labels missing from test metrics.")

        except FileNotFoundError:
            print(f"Error: Best model file not found at {config.BEST_MODEL_PATH}. Cannot perform final evaluation.")
            print("This might happen if training failed to save a model checkpoint.")
        except Exception as e:
            print(f"\nAn error occurred during final test evaluation or analysis: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n--- Training Pipeline for {config.MODEL_TYPE} Finished ---")
    print("========================================")