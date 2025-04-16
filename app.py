import torch
import os
import json
import sys
from operator import itemgetter
import inspect # Keep inspect for checking model args

# --- Rely on existing modules ---
import config
import models # Use model definitions from models.py
import data_handler # Use data handling classes/functions from data_handler.py
# engine.py is mainly for training/evaluation loops, not typically needed for inference app

# --- Configuration (from config.py) ---
DEVICE = config.DEVICE
MODEL_PATH = config.MODEL_SAVE_PATH
VOCAB_PATH = config.VOCAB_SAVE_PATH
LABEL_MAP_SAVE_PATH = config.LABEL_MAP_SAVE_PATH
MAX_LENGTH = config.MAX_LENGTH
MODEL_TYPE = config.MODEL_TYPE

# --- Helper Functions ---

def load_label_map(filepath):
    """
    Loads the label map (int -> string label) using data_handler functionality
    if available, or a simplified local loader as fallback.
    """
    print(f"Attempting to load label map from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Warning: Label map file not found at {filepath}. Will display numerical class indices.")
        return None

    try:
        # Prefer using the function from data_handler if it suits the need
        # data_handler.load_label_mappings returns (label_to_int, int_to_label)
        mappings = data_handler.load_label_mappings(filepath)
        int_to_label_map = mappings[1] # Get the int_to_label dictionary
        print(f"Successfully loaded label map via data_handler with {len(int_to_label_map)} entries.")
        # Ensure keys are integers (data_handler.load_label_mappings should already do this)
        return {int(k): v for k, v in int_to_label_map.items()}

    except FileNotFoundError:
         # This case should ideally be caught by the initial os.path.exists check
         print(f"Error: Label map file disappeared unexpectedly at {filepath}.")
         return None
    except Exception as e:
        # Fallback to a simpler local loader if data_handler fails or isn't suitable
        print(f"Warning: Failed loading label map via data_handler ({e}). Attempting simple load.")
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            
            # Check for different possible structures in the JSON file
            int_to_label = None
            
            # Check standard structure with 'int_to_label' key
            if 'int_to_label' in loaded_data:
                int_to_label = loaded_data['int_to_label']
            # Check if the file itself is directly a mapping
            elif isinstance(loaded_data, dict):
                # It could be a direct mapping of index -> label
                if all(k.isdigit() for k in loaded_data.keys()):
                    int_to_label = loaded_data
                # Or it could be label -> index that needs to be inverted
                elif all(isinstance(v, (int, str)) and str(v).isdigit() for v in loaded_data.values()):
                    # Invert the mapping (label -> index becomes index -> label)
                    int_to_label = {str(v): k for k, v in loaded_data.items()}
            
            if int_to_label is None:
                print(f"Error: Simple load failed, could not find label mapping in file: {filepath}")
                print(f"File content structure: {type(loaded_data).__name__}")
                if isinstance(loaded_data, dict):
                    print(f"Keys in file: {', '.join(str(k) for k in loaded_data.keys())}")
                return None
                
            # Ensure all keys are converted to integers
            int_to_label_map = {int(k): v for k, v in int_to_label.items()}
            print(f"Successfully loaded label map via simple load with {len(int_to_label_map)} entries.")
            return int_to_label_map
        except Exception as inner_e:
            print(f"Error: Simple load also failed for label map: {inner_e}")
            return None


def load_vocabulary(filepath):
    """Loads the Vocabulary object and number of classes using data_handler."""
    print(f"Attempting to load vocabulary using data_handler from: {filepath}")
    # data_handler.Vocabulary.load handles file checking and errors internally
    try:
        # Use the classmethod directly from the imported module
        vocab, n_class = data_handler.Vocabulary.load(filepath)
        print(f"Successfully loaded vocabulary via data_handler (Size: {len(vocab)}). Number of classes from vocab file: {n_class}")
        return vocab, n_class
    except (FileNotFoundError, ValueError, Exception) as e:
         # Catch potential errors from the load method for clarity
         print(f"Error loading vocabulary via data_handler: {e}")
         sys.exit(1)

def load_model(model_path, vocab_size, n_class, device):
    """
    Loads the trained model based on MODEL_TYPE in config,
    using model class definitions from the 'models' module.
    """
    print(f"Attempting to load model weights from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model weights file not found at {model_path}. Cannot proceed.")
        sys.exit(1)

    print(f"Instantiating model type: {MODEL_TYPE} using definitions from 'models.py'")
    # --- Instantiate the correct model class from the 'models' module ---
    if MODEL_TYPE == 'CNN_RNN_Attention':
        model_class = models.CNN_RNN_Attention
        model_params = dict(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            cnn_out_channels=config.CNN_OUT_CHANNELS,
            cnn_kernel_sizes=config.CNN_KERNEL_SIZES,
            rnn_type=config.RNN_TYPE,
            rnn_hidden_dim=config.RNN_HIDDEN_DIM,
            rnn_layers=config.RNN_LAYERS,
            n_class=n_class,
            dropout_prob=0.0, # Set dropout to 0 for inference
            pad_idx=config.PAD_IDX
        )
    elif MODEL_TYPE == 'LSTM':
         model_class = models.LSTMNetwork
         rnn_hidden_dim = getattr(config, 'RNN_HIDDEN_DIM', 256)
         rnn_layers = getattr(config, 'RNN_LAYERS', 2)
         model_params = dict(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=rnn_hidden_dim,
            n_class=n_class,
            n_layers=rnn_layers,
            pad_idx=config.PAD_IDX,
            dropout_prob=0.0 # Set dropout to 0 for inference
         )
    else:
         print(f"Error: Unsupported MODEL_TYPE '{MODEL_TYPE}' specified in config.py")
         sys.exit(1)

    # Instantiate the model using the class and parameters
    model = model_class(**model_params)

    try:
        # Load the state dict
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state dict from checkpoint.")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state dict directly from file.")

        model.to(device)
        model.eval()
        print("Model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        sys.exit(1)

# --- Predictor Class ---

class EmotionPredictor:
    def __init__(self, model, vocab, int_to_label_map, device):
        self.model = model
        self.vocab = vocab # Vocabulary object from data_handler
        self.int_to_label_map = int_to_label_map
        self.device = device
        # Use the TextPreprocessor class directly from data_handler
        print("Initializing TextPreprocessor from data_handler...")
        self.text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)
        print("TextPreprocessor initialized.")
        self.model_accepts_lengths = self._check_model_length_arg()

    def _check_model_length_arg(self):
        """Checks if the model's forward method accepts 'sequence_lengths'."""
        try:
            sig = inspect.signature(self.model.forward)
            accepts = 'sequence_lengths' in sig.parameters
            print(f"Model forward signature check: Accepts 'sequence_lengths'? {accepts}")
            return accepts
        except Exception as e:
            print(f"Warning: Could not inspect model forward signature: {e}. Assuming it doesn't need lengths.")
            return False

    def _preprocess_text(self, text):
        """Converts raw text string into a padded tensor using data_handler methods."""
        # 1. Clean and Tokenize using data_handler's preprocessor
        tokens = self.text_preprocessor.clean_and_tokenize(text)

        # 2. Numericalize using data_handler's vocabulary object
        numericalized_tokens = self.vocab.numericalize(tokens)
        truncated_tokens = numericalized_tokens[:MAX_LENGTH - 2]
        sequence = [config.SOS_IDX] + truncated_tokens + [config.EOS_IDX] # Use constants from config

        # 3. Convert to Tensor and Add Batch Dimension
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)

        # 4. Get Sequence Length
        sequence_length = torch.tensor([len(sequence)], dtype=torch.long)

        return sequence_tensor.to(self.device), sequence_length.to(self.device)

    def predict(self, text):
        """Predicts emotion probabilities for a given text string."""
        print(f"\n--- Predicting for: '{text}' ---")
        try:
            sequence_tensor, sequence_length = self._preprocess_text(text)
            print(f"Input Tensor Shape: {sequence_tensor.shape}, Length: {sequence_length.item()}")

            self.model.eval()
            with torch.no_grad():
                if self.model_accepts_lengths:
                    logits = self.model(sequence_tensor, sequence_lengths=sequence_length)
                else:
                    logits = self.model(sequence_tensor)

            probabilities = torch.softmax(logits, dim=1).squeeze()
            probabilities_np = probabilities.cpu().numpy()

            results = []
            for i, prob in enumerate(probabilities_np):
                label_index = i
                label_name = self.int_to_label_map.get(label_index, str(label_index)) if self.int_to_label_map else str(label_index)
                results.append((prob, label_name, label_index))

            results.sort(key=itemgetter(0), reverse=True)
            return results

        except Exception as e:
            print(f"Error during prediction for '{text}': {e}")
            import traceback
            traceback.print_exc()
            return None

# --- Main Execution ---

def main():
    print("--- Emotion Classification Predictor (Integrated Approach) ---")
    print(f"Using device: {DEVICE}")

    # 1. Load Artifacts using helper functions relying on data_handler/config
    print("\nLoading necessary artifacts...")
    label_map = load_label_map(LABEL_MAP_SAVE_PATH)
    vocab, n_class_from_vocab = load_vocabulary(VOCAB_PATH)

    if label_map:
        n_class = len(label_map)
        if n_class != n_class_from_vocab:
             print(f"Warning: Mismatch in class count (LabelMap: {n_class}, VocabFile: {n_class_from_vocab}). Using LabelMap count.")
    else:
        n_class = n_class_from_vocab
        print(f"Using class count from vocabulary file: {n_class}")

    if n_class is None or n_class <= 0:
        print("Error: Could not determine number of classes.")
        sys.exit(1)

    model = load_model(MODEL_PATH, len(vocab), n_class, DEVICE)

    # 2. Initialize Predictor (uses data_handler internally)
    predictor = EmotionPredictor(model, vocab, label_map, DEVICE)
    print("\n--- Predictor Ready ---")

    # 3. Example Predictions
    examples = [
        "I am absolutely ecstatic about this news!", "I feel so sad and lonely today.",
        "This movie was surprisingly funny.", "He didn't seem very happy.",
        "What a wonderful surprise!", "I am afraid of the dark.", "That makes me really angry!"
    ]
    print("\n--- Running Example Predictions ---")
    for example in examples:
        prediction_results = predictor.predict(example)
        if prediction_results:
            print("Prediction Results:")
            for prob, name, index in prediction_results:
                 indicator = " ***** TOP PREDICTION *****" if prob == prediction_results[0][0] else ""
                 print(f"  - {name}: {prob:.4f}{indicator}")
        else:
            print(f"Could not get prediction for: {example}")

    # 4. Interactive Loop
    print("\n--- Interactive Prediction Mode ---")
    print("Enter text to classify the emotion, or type 'q' or 'quit' to exit.")
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ['q', 'quit']: break
            if not user_input: continue

            prediction_results = predictor.predict(user_input)
            if prediction_results:
                print("\nPrediction Results:")
                top_prob = prediction_results[0][0]
                for prob, name, index in prediction_results:
                     indicator = " ***** TOP PREDICTION *****" if prob == top_prob else ""
                     print(f"  - {name}: {prob:.4f}{indicator}")
            else:
                print("Could not get prediction for the input.")

        except (EOFError, KeyboardInterrupt):
             print("\nExiting interactive mode.")
             break

if __name__ == "__main__":
    main()