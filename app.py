# --- app.py ---
import torch
import os
import json
import argparse
import sys
from operator import itemgetter
import glob

# Import necessary modules after potentially modifying path
try:
    import config # To get default paths and potentially model type if config not saved in run
    import data_handler
    import engine # To load model structure and weights
except ImportError as e:
    print(f"Error importing core modules: {e}")
    print("Ensure config.py, data_handler.py, and engine.py are accessible.")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during imports: {e}")
    sys.exit(1)

# --- Helper Functions ---

def load_run_config(model_type_dir):
    """Loads the specific configuration saved for the given model type run."""
    config_path = os.path.join(model_type_dir, config.RUN_CONFIG_FILENAME) # Use default filename
    if not os.path.exists(config_path):
        print(f"Warning: Run configuration file not found at {config_path}. Using global config.py defaults.")
        # Fallback logic: Use global config values directly.
        # This might be inaccurate if global config changed since the run.
        class RunConfig:
             MODEL_TYPE = config.MODEL_TYPE # Note: This might not match the requested model_type_dir's type
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = getattr(config, 'TRANSFORMER_MODEL_NAME', None) # Use getattr for safety
             # Construct potential path relative to the model_type_dir
             VOCAB_PATH = os.path.join(model_type_dir, config.VOCAB_FILENAME)
             REMOVE_STOPWORDS = getattr(config, 'REMOVE_STOPWORDS', False)
             SPACY_MODEL_NAME = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        # Override MODEL_TYPE based on the directory we are trying to load
        RunConfig.MODEL_TYPE = os.path.basename(model_type_dir)
        return RunConfig()

    try:
        with open(config_path, 'r') as f:
            loaded_config = json.load(f)
        # Convert loaded dict to an object for easier access (optional)
        class RunConfig:
            def __init__(self, **entries):
                self.__dict__.update(entries)
                # Ensure necessary paths are relative to the loaded model_type_dir
                self.VOCAB_PATH = os.path.join(model_type_dir, config.VOCAB_FILENAME)

        print(f"Loaded run configuration from {config_path}")
        # Ensure the loaded config's model type matches the directory (sanity check)
        if loaded_config.get('MODEL_TYPE') != os.path.basename(model_type_dir):
             print(f"Warning: Loaded config MODEL_TYPE ({loaded_config.get('MODEL_TYPE')}) does not match directory ({os.path.basename(model_type_dir)})")
        return RunConfig(**loaded_config)
    except Exception as e:
        print(f"Error loading run config from {config_path}: {e}. Using global defaults.")
        # Fallback to global config if loading fails
        class RunConfig: # Duplicated fallback logic
             MODEL_TYPE = config.MODEL_TYPE # Again, might not match requested type
             MAX_LEN = config.MAX_LEN
             PREPROCESSOR_TYPE = config.PREPROCESSOR_TYPE
             TRANSFORMER_MODEL_NAME = getattr(config, 'TRANSFORMER_MODEL_NAME', None)
             VOCAB_PATH = os.path.join(model_type_dir, config.VOCAB_FILENAME)
             REMOVE_STOPWORDS = getattr(config, 'REMOVE_STOPWORDS', False)
             SPACY_MODEL_NAME = getattr(config, 'SPACY_MODEL_NAME', 'en_core_web_sm')
        RunConfig.MODEL_TYPE = os.path.basename(model_type_dir) # Override with dir name
        return RunConfig()

def load_prediction_artifacts(model_type_dir):
    """Loads all necessary artifacts for prediction based on the model type's directory."""
    print(f"\nLoading artifacts from model type directory: {model_type_dir}")
    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory not found at {model_type_dir}")
        return None, None, None, None, None

    run_cfg = load_run_config(model_type_dir)

    # Load Label Map (global)
    label_to_int, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_PATH)
    if not int_to_label:
        print("Warning: Label map not found or empty. Predictions will show integer labels.")
        # Create a dummy map if needed elsewhere, or handle None gracefully
        int_to_label = {} # Empty dict signals no mapping available

    n_classes = len(int_to_label) if int_to_label else 0
    if n_classes == 0:
        print("Warning: Cannot determine number of classes from label map.")
        # Might need to infer from model later if possible, or fail

    # Determine paths within the model_type_dir
    model_path = os.path.join(model_type_dir, "model", config.BEST_MODEL_FILENAME)
    vocab_path = run_cfg.VOCAB_PATH # Use path from loaded config (points to model_type_dir)

    vocab_size = None
    vocab_or_tokenizer = None

    if run_cfg.MODEL_TYPE != 'Transformer':
        # Load Vocabulary for non-transformer models
        try:
            vocab = data_handler.Vocabulary.load(vocab_path)
            vocab_size = len(vocab)
            vocab_or_tokenizer = vocab
            print(f"Vocabulary loaded (Size: {vocab_size}).")
        except FileNotFoundError:
            print(f"Error: Vocabulary file not found at {vocab_path}. Cannot proceed for {run_cfg.MODEL_TYPE} model.")
            return None, None, None, None, None
        except Exception as e:
            print(f"Error loading vocabulary: {e}")
            return None, None, None, None, None
    else:
         # Load Tokenizer for transformer models
         if data_handler.AutoTokenizer is None:
              print("Error: Transformers library not installed, cannot load tokenizer.")
              return None, None, None, None, None
         try:
              print(f"Loading tokenizer: {run_cfg.TRANSFORMER_MODEL_NAME}")
              # Check if tokenizer files exist within the model_type_dir (optional optimization)
              # If not, from_pretrained will download. If they exist, it should load locally.
              tokenizer = data_handler.AutoTokenizer.from_pretrained(
                  run_cfg.TRANSFORMER_MODEL_NAME,
                  # local_files_only=True # Uncomment to force local loading if desired
              )
              vocab_or_tokenizer = tokenizer
              vocab_size = tokenizer.vocab_size # Use tokenizer's vocab size info
         except Exception as e:
              print(f"Error loading tokenizer '{run_cfg.TRANSFORMER_MODEL_NAME}': {e}")
              return None, None, None, None, None


    # Now load model (needs n_classes, and vocab_size if not transformer)
    if n_classes == 0 and run_cfg.MODEL_TYPE == 'Transformer':
         # Try to infer n_classes from a loaded transformer config if label map failed
         try:
             from transformers import AutoConfig
             model_cfg = AutoConfig.from_pretrained(run_cfg.TRANSFORMER_MODEL_NAME)
             n_classes = model_cfg.num_labels
             print(f"Inferred n_classes={n_classes} from Transformer config.")
         except Exception:
              print("Error: Failed to infer n_classes. Cannot load model.")
              return None, None, None, None, None
    elif n_classes == 0:
         print(f"Error: Cannot determine n_classes for model type {run_cfg.MODEL_TYPE} without a label map.")
         return None, None, None, None, None

    try:
        model = engine.load_trained_model(model_path, run_cfg.MODEL_TYPE, n_classes, vocab_size)
    except FileNotFoundError:
        print(f"Error: Trained model file not found at {model_path}")
        return None, None, None, None, None
    except Exception as e:
        print(f"Error loading trained model: {e}")
        return None, None, None, None, None

    # Initialize Preprocessor based on run config
    print(f"Initializing preprocessor: {run_cfg.PREPROCESSOR_TYPE}")
    if run_cfg.PREPROCESSOR_TYPE == 'spacy':
         try:
             # Pass specific options used during training if available in run_cfg
             preprocessor = data_handler.SpacyTextPreprocessor(
                  spacy_model_name=getattr(run_cfg, 'SPACY_MODEL_NAME', config.SPACY_MODEL_NAME),
                  remove_stopwords=getattr(run_cfg, 'REMOVE_STOPWORDS', config.REMOVE_STOPWORDS)
             )
         except ImportError as e:
              print(f"Error initializing Spacy Preprocessor: {e}")
              return None, None, None, None, None
         except Exception as e: # Catch other Spacy errors (e.g., model download)
              print(f"Error initializing Spacy Preprocessor: {e}")
              return None, None, None, None, None
    else:
        preprocessor = data_handler.BasicTextCleaner()


    return model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg

# Removed find_latest_run_dir function

# --- Predictor Class ---

class EmotionPredictor:
    def __init__(self, model, vocab_or_tokenizer, preprocessor, int_to_label, run_config):
        self.model = model
        self.vocab_or_tokenizer = vocab_or_tokenizer
        self.preprocessor = preprocessor
        self.int_to_label = int_to_label if int_to_label else {} # Ensure it's a dict
        self.run_config = run_config
        self.device = config.DEVICE # Use global device config for prediction
        self.model.to(self.device)
        self.model.eval()
        print("\nEmotionPredictor initialized.")

    def _preprocess_input(self, text):
        """Prepares raw text input for the specific model type."""
        if isinstance(self.preprocessor, data_handler.SpacyTextPreprocessor):
             # Spacy preprocessor might tokenize, but model might expect string or specific tokens
             if self.run_config.MODEL_TYPE == 'Transformer':
                  cleaned_text = " ".join(self.preprocessor.clean_and_tokenize(text)) # Join tokens back
             else:
                  cleaned_tokens = self.preprocessor.clean_and_tokenize(text) # Keep as tokens
                  return cleaned_tokens # Return tokens for non-transformer vocab
        else: # Basic Cleaner
            cleaned_text = self.preprocessor.clean(text)
            if self.run_config.MODEL_TYPE != 'Transformer':
                 # Simple split for non-transformer vocab
                 # Ensure consistency with how vocab was built (usually tokenized)
                 return self.preprocessor.tokenize(cleaned_text) # Use basic cleaner's tokenize
        return cleaned_text # Return cleaned string for Transformer tokenizer

    def predict(self, text):
        """Predicts emotion probabilities for the input text."""
        processed_input = self._preprocess_input(text)

        try:
            with torch.no_grad():
                if self.run_config.MODEL_TYPE == 'Transformer':
                    # Ensure input is string for tokenizer
                    input_text = processed_input if isinstance(processed_input, str) else " ".join(processed_input)
                    encoding = self.vocab_or_tokenizer.encode_plus(
                        input_text,
                        add_special_tokens=True,
                        max_length=self.run_config.MAX_LEN,
                        padding='max_length',
                        truncation=True,
                        return_attention_mask=True,
                        return_tensors='pt',
                    )
                    input_ids = encoding['input_ids'].to(self.device)
                    attention_mask = encoding['attention_mask'].to(self.device)
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

                else: # Non-Transformer models
                    # Expect processed_input to be list of tokens
                    if not isinstance(processed_input, list):
                         print(f"Warning: Expected list of tokens for {self.run_config.MODEL_TYPE}, got {type(processed_input)}. Attempting split.")
                         processed_input = str(processed_input).split()

                    numericalized = self.vocab_or_tokenizer.numericalize(processed_input)
                    # Apply padding and SOS/EOS based on how data was prepared for training
                    # This assumes SOS/EOS were used; adjust if not
                    max_len_adjusted = self.run_config.MAX_LEN - 2 # Account for SOS/EOS
                    padded_numericalized = numericalized[:max_len_adjusted]
                    sequence = [config.SOS_IDX] + padded_numericalized + [config.EOS_IDX]

                    # Pad sequence manually if needed, or rely on model/batching if it handles variable lengths
                    # For single prediction, simpler to pad here to match MAX_LEN used in training config
                    padded_sequence = sequence + [config.PAD_IDX] * (self.run_config.MAX_LEN - len(sequence))
                    sequence_tensor = torch.tensor([padded_sequence], dtype=torch.long).to(self.device) # Add batch dim
                    lengths = torch.tensor([len(sequence)], dtype=torch.long).to(self.device) # Original length before padding

                    # Pass lengths, model forward should handle it (e.g., pack_padded_sequence)
                    logits = self.model(text_indices=sequence_tensor, sequence_lengths=lengths)


            probabilities = torch.softmax(logits, dim=1).squeeze()
            probabilities_np = probabilities.cpu().numpy()

            results = []
            for i, prob in enumerate(probabilities_np):
                label_index = i
                # Use label map if available, otherwise show index
                label_name = self.int_to_label.get(label_index, f"Label_{label_index}")
                results.append({'label': label_name, 'score': float(prob)}) # Use dict for clarity

            # Sort by probability descending
            results.sort(key=itemgetter('score'), reverse=True)
            return results

        except Exception as e:
            print(f"\nError during prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

# --- Main Application Logic ---

def run_interactive_app(predictor):
    """Handles the interactive command-line loop."""
    print("\n--- Interactive Emotion Prediction ---")
    print(f"Using model: {predictor.run_config.MODEL_TYPE}")
    print("Enter text to classify, or type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit']:
                print("Exiting.")
                break

            prediction_results = predictor.predict(user_input)

            if prediction_results:
                print("\nPrediction Results:")
                # Find max score for highlighting
                max_score = prediction_results[0]['score'] if prediction_results else 0
                for result in prediction_results:
                    indicator = " *" if result['score'] == max_score and max_score > 0 else ""
                    print(f"  - {result['label']}: {result['score']:.4f}{indicator}")
            else:
                print("  Prediction failed.")

        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the loop: {e}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Emotion Prediction App (Loads Model)")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['Transformer', 'CNN_RNN_Attention', 'LSTM'],
        help="Specify the type of model artifacts to load."
    )
    args = parser.parse_args()

    # Construct the path to the model type's artifact directory
    model_type_dir = os.path.join(config.ARTIFACTS_DIR, args.model_type)

    if not os.path.isdir(model_type_dir):
        print(f"Error: Artifact directory for model type '{args.model_type}' not found at {model_type_dir}")
        print("Please ensure the model has been trained first using 'python main.py'.")
        sys.exit(1)

    print(f"Loading artifacts for model type: {args.model_type}")
    model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg = load_prediction_artifacts(model_type_dir)

    if model is None:
        print("Failed to load necessary artifacts. Exiting.")
        sys.exit(1)

    predictor = EmotionPredictor(model, vocab_or_tokenizer, preprocessor, int_to_label, run_cfg)
    run_interactive_app(predictor)


if __name__ == "__main__":
    main()