# --- main.py ---
import sys
import os

# Dynamically add project root to path if needed (if running main.py from within project dir)
# PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

# Import necessary modules after potentially modifying path
try:
    import config
    import train
except ImportError as e:
     print(f"Error importing core modules: {e}")
     print("Ensure config.py and train.py are in the Python path or the same directory.")
     sys.exit(1)
except Exception as e:
     print(f"An unexpected error occurred during imports: {e}")
     sys.exit(1)


if __name__ == "__main__":
    print("========================================")
    print("=== Emotion Classification Framework ===")
    print("========================================")

    # Ensure artifact directories exist before training starts
    # (config.py already creates RUN_ARTIFACTS_DIR and MODEL_SAVE_DIR)
    # If LABEL_MAP_PATH is outside RUN_ARTIFACTS_DIR, ensure its directory exists
    if os.path.dirname(config.LABEL_MAP_PATH):
         os.makedirs(os.path.dirname(config.LABEL_MAP_PATH), exist_ok=True)

    # Execute the training pipeline defined in train.py
    try:
        train.run_training_pipeline()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n--- An Unhandled Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("-------------------------------------")
        # Optionally print traceback for debugging
        import traceback
        traceback.print_exc()
        sys.exit(1)
