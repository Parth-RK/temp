# app.py
"""
Independent application script using manual vocab and preprocessing.
(TorchText Legacy Independent Version)
"""
import torch
import torch.nn as nn
import os
import json

# --- Local Imports ---
import config # For paths, tokens, model type etc.
import data_handler # Imports new Vocabulary and TextPreprocessor
import models # Need model class definitions
import engine # For load_final_model or load_checkpoint

# --- Define fixed indices from config ---
PAD_IDX = config.PAD_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX
MAX_LENGTH = config.MAX_LENGTH # Max content length before SOS/EOS

def predict_emotion(text: str, model, text_preprocessor, vocabulary, device):
    """Predicts emotion for a single text input."""
    if not text or not isinstance(text, str):
        return "Error: Invalid input text."

    print(f"Processing input: '{text}'")
    model.eval() # Ensure model is in evaluation mode

    # 1. Clean and Tokenize using TextPreprocessor
    tokens = text_preprocessor.clean_and_tokenize(text)
    print(f"Tokens: {tokens}")

    # 2. Numericalize using Vocabulary
    numericalized_tokens = vocabulary.numericalize(tokens)
    print(f"Numerical Indices (Content): {numericalized_tokens}")

    # 3. Add SOS/EOS and truncate
    sequence = [SOS_IDX] + numericalized_tokens[:MAX_LENGTH] + [EOS_IDX]
    print(f"Numerical Sequence (SOS/EOS): {sequence}")

    # 4. Convert to Tensor
    input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0) # Add batch dimension
    input_tensor = input_tensor.to(device)

    print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    # 5. Model Inference
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

    print(f"Logits: {logits.cpu().numpy()}")
    print(f"Probabilities: {probabilities.cpu().numpy()}")
    print(f"Predicted Index: {predicted_index}")

    # 6. Map index to label
    predicted_label = config.INT_TO_LABEL.get(predicted_index, "Unknown")
    confidence = probabilities[0, predicted_index].item()

    return predicted_label, confidence

def load_trained_artifacts(device):
    """Loads the trained model, vocabulary, and initializes preprocessor."""
    print("--- Loading Trained Artifacts ---")

    # 1. Initialize TextPreprocessor
    #    Load config if saved, otherwise assume default (e.g., use_stopwords=False)
    # preprocessor_config_path = config.PREPROCESSOR_SAVE_PATH
    # use_stopwords = False
    # if os.path.exists(preprocessor_config_path):
    #      with open(preprocessor_config_path, 'r') as f:
    #           loaded_proc_config = json.load(f)
    #           use_stopwords = loaded_proc_config.get('use_stopwords', False)
    text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False) # Match main.py setting

    # 2. Load Vocabulary
    if not os.path.exists(config.VOCAB_SAVE_PATH):
        raise FileNotFoundError(f"Vocabulary not found at {config.VOCAB_SAVE_PATH}. Run main.py first.")
    vocabulary = data_handler.Vocabulary.load(config.VOCAB_SAVE_PATH)
    vocab_size = len(vocabulary)

    # 3. Initialize Model Architecture
    print(f"Initializing model architecture: {config.MODEL_TYPE}")
    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=config.N_CLASS,
            n_layers=config.N_LAYERS,
            pad_idx=PAD_IDX # Use defined PAD_IDX
        )
    # ANN less suitable now
    # elif config.MODEL_TYPE == 'ANN':
    #     model = models.ANN(...) # Needs redesign
    else:
        raise ValueError(f"Unsupported or unsuitable model type for inference: {config.MODEL_TYPE}")

    # 4. Load Trained Model Weights
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}. Run main.py first.")

    # Use load_checkpoint as it loads the state_dict into the model structure
    # Need a dummy optimizer instance only if loading optimizer state (not needed for inference)
    engine.load_checkpoint(model_path, model, optimizer=None, device=device)
    # Or use load_final_model if you saved only the state_dict at the very end
    # engine.load_final_model(model, model_path, device)

    model.eval() # Explicitly set to eval mode

    print("--- Artifacts Loaded Successfully ---")
    return model, text_preprocessor, vocabulary

if __name__ == "__main__":
    loaded_model, loaded_text_preprocessor, loaded_vocabulary = load_trained_artifacts(config.DEVICE)

    # Example Usage:
    while True:
        input_text = input("\nEnter text to analyze emotion (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        if not input_text.strip():
             print("Input cannot be empty.")
             continue

        try:
            label, conf = predict_emotion(
                input_text,
                loaded_model,
                loaded_text_preprocessor,
                loaded_vocabulary,
                config.DEVICE
            )
            print(f"Predicted Emotion: {label} (Confidence: {conf:.4f})")
        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            # Add more detailed error handling if needed
        print("-" * 20)