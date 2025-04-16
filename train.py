# --- train.py ---
import os
import sys
import time

# Dynamically add project root to path if needed
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

import config
import data_handler
import engine
import plotter

def set_seed(seed_value=config.SEED):
    """Sets the seed for reproducibility."""
    import random
    import numpy as np
    import torch
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if config.DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True # May impact performance
        torch.backends.cudnn.benchmark = False   # Ensure reproducibility

def run_training_pipeline():
    """Main function to execute the training pipeline."""
    start_time = time.time()
    set_seed()

    print(f"=== Starting Training Run: {config.RUN_ID} ===")

    # --- 1. Data Pipeline ---
    try:
        train_loader, val_loader, test_loader, label_to_int, int_to_label, n_classes, vocab_or_tokenizer, vocab_size = data_handler.get_data_pipeline()
    except Exception as e:
        print(f"Error during data pipeline: {e}")
        sys.exit(1)

    # --- 2. Initialize Model ---
    try:
        model = engine.initialize_model(config.MODEL_TYPE, n_classes, vocab_size)
    except Exception as e:
        print(f"Error initializing model: {e}")
        sys.exit(1)

    # --- 3. Initialize Optimizer & Scheduler ---
    try:
        # Calculate total training steps for scheduler if needed
        num_train_steps = len(train_loader) * config.EPOCHS if config.SCHEDULER_TYPE == 'linear_warmup' else None
        optimizer, scheduler = engine.initialize_optimizer_scheduler(
            model, config.OPTIMIZER_TYPE, config.SCHEDULER_TYPE, num_train_steps
        )
    except Exception as e:
        print(f"Error initializing optimizer/scheduler: {e}")
        sys.exit(1)

    # --- 4. Train Model ---
    try:
        history = engine.train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            epochs=config.EPOCHS,
            model_save_path=config.BEST_MODEL_PATH,
            metric_for_best=config.METRIC_FOR_BEST_MODEL
        )
    except Exception as e:
        print(f"Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 5. Plot Training History ---
    if config.PLOT_TRAINING_HISTORY:
        try:
            print("\nGenerating training history plots...")
            plotter.plot_training_history(history, save_path=config.TRAINING_PLOTS_PATH)
        except Exception as e:
            print(f"Warning: Could not generate training plots. Error: {e}")

    # --- 6. Evaluate on Test Set ---
    if config.GENERATE_TEST_REPORT or config.GENERATE_CONFUSION_MATRIX:
        print("\n--- Evaluating on Test Set ---")
        try:
            # Load the *best* saved model for final evaluation
            print(f"Loading best model from: {config.BEST_MODEL_PATH}")
            best_model = engine.load_trained_model(
                model_path=config.BEST_MODEL_PATH,
                model_type=config.MODEL_TYPE,
                n_classes=n_classes,
                vocab_size=vocab_size
            )
            print("Evaluating best model on test data...")
            test_metrics = engine.evaluate_step(best_model, test_loader, config.DEVICE)

            print("\nTest Set Performance:")
            print(f"  Loss: {test_metrics['loss']:.4f}")
            print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
            print(f"  Precision (Weighted): {test_metrics['precision_weighted']:.4f}")
            print(f"  Recall (Weighted): {test_metrics['recall_weighted']:.4f}")

            # Generate detailed report and confusion matrix
            plotter.generate_classification_analysis(
                true_labels=test_metrics['true_labels'],
                predictions=test_metrics['predictions'],
                int_to_label=int_to_label,
                report_path=config.TEST_REPORT_PATH if config.GENERATE_TEST_REPORT else None,
                cm_path=config.CONFUSION_MATRIX_PATH if config.GENERATE_CONFUSION_MATRIX else None,
                prefix="Test Set"
            )

        except FileNotFoundError:
            print(f"Warning: Best model file not found at {config.BEST_MODEL_PATH}. Skipping test evaluation.")
        except Exception as e:
            print(f"Error during test evaluation: {e}")
            import traceback
            traceback.print_exc()

    # --- 7. Save Final Run Configuration ---
    try:
        config.save_run_config() # Save the config used for this run
    except Exception as e:
        print(f"Warning: Failed to save final run config. Error: {e}")


    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== Training Run {config.RUN_ID} Finished ===")
    print(f"Total Time: {total_time / 60:.2f} minutes")
    print(f"Artifacts saved in: {config.RUN_ARTIFACTS_DIR}")
    print("========================================")

if __name__ == "__main__":
    run_training_pipeline()