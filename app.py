# app.py
"""
Independent application script to load a trained model and predict emotion from text input.
"""
import torch
import torch.nn as nn # Required for loading model class definition
import os
import json

# --- Local Imports ---
import config # For paths, tokens, model type etc.
import data_handler
import models # Need model class definitions to load state_dict

def predict_emotion(text: str, model, processor, device):
    """Predicts emotion for a single text input."""
    if not text or not isinstance(text, str):
        return "Error: Invalid input text."

    print(f"Processing input: '{text}'")
    model.eval() # Ensure model is in evaluation mode

    # 1. Clean and Tokenize using loaded processor methods
    processor._lazy_load_spacy() # Ensure spacy is loaded
    # Replicate cleaning steps (modify lemmatize for single string)
    def lemmatize_single(txt):
         if not txt.strip(): return ""
         doc = processor.nlp(txt)
         return " ".join(token.lemma_ for token in doc if token.lemma_ not in processor.stopwords and not token.is_punct and not token.is_space).lower()

    cleaned_text = lemmatize_single(text)
    tokens = processor.tokenize(cleaned_text)
    print(f"Tokens: {tokens}")

    # 2. Convert to Numerical Indices
    numerical_tokens = processor.convert_numerical(tokens)
    print(f"Numerical Indices: {numerical_tokens}")

    # 3. Pad Sequence
    padded_sequence = processor.pad_sequences([numerical_tokens])[0] # Pad the single sequence
    print(f"Padded Indices: {padded_sequence}")

    # 4. Convert to Tensor
    input_tensor = torch.tensor(padded_sequence).unsqueeze(0) # Add batch dimension

    # --- Model Specific Input Handling ---
    if config.MODEL_TYPE == 'LSTM':
        input_tensor = input_tensor.to(device).long() # LSTM expects long
    elif config.MODEL_TYPE == 'ANN':
         input_tensor = input_tensor.to(device).float() # ANN expects float
    else:
         # Default or raise error
         input_tensor = input_tensor.to(device)
    # -------------------------------------

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
    """Loads the trained model, preprocessor config, and vocabulary."""
    print("--- Loading Trained Artifacts ---")

    # 1. Load Preprocessor Config and Re-initialize
    if not os.path.exists(config.PREPROCESSOR_SAVE_PATH):
        raise FileNotFoundError(f"Preprocessor config not found at {config.PREPROCESSOR_SAVE_PATH}. Run main.py first.")
    processor = data_handler.Preprocessor.from_config(config.PREPROCESSOR_SAVE_PATH)

    # 2. Load Vocabulary
    if not os.path.exists(config.VOCAB_SAVE_PATH):
        raise FileNotFoundError(f"Vocabulary not found at {config.VOCAB_SAVE_PATH}. Run main.py first.")
    processor.load_vocab(config.VOCAB_SAVE_PATH)
    vocab_size = len(processor.vocab)
    pad_idx = processor.vocab[config.PAD_TOKEN]

    # 3. Initialize Model Architecture
    print(f"Initializing model architecture: {config.MODEL_TYPE}")
    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=config.N_CLASS,
            n_layers=config.N_LAYERS,
            pad_idx=pad_idx
        )
    elif config.MODEL_TYPE == 'ANN':
        model = models.ANN(
            input_size=config.ANN_INPUT_SIZE,
            n_class=config.N_CLASS
        )
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE}")

    # 4. Load Trained Model Weights
    if not os.path.exists(config.MODEL_SAVE_PATH):
         # Check for alternative final save name if used
         alt_path = config.MODEL_SAVE_PATH.replace('.pt', '_final.pt')
         if os.path.exists(alt_path):
              model_path = alt_path
         else:
              raise FileNotFoundError(f"Trained model not found at {config.MODEL_SAVE_PATH} or {alt_path}. Run main.py first.")
    else:
         model_path = config.MODEL_SAVE_PATH

    # Load state dict (handle checkpoint vs final model)
    try:
         # Try loading as final model state_dict first
         engine.load_final_model(model, model_path, device)
    except RuntimeError: # Likely a checkpoint file
         print("Failed loading as final model, attempting to load as checkpoint...")
         try:
              # Need an optimizer instance temporarily, even if not used
              # Create a dummy optimizer based on model type
              if config.MODEL_TYPE == 'LSTM':
                   dummy_optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_LSTM)
              else:
                   dummy_optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_ANN)
              engine.load_checkpoint(model_path, model, dummy_optimizer, device)
              model.eval() # Ensure eval mode after loading checkpoint
         except Exception as e:
              print(f"Error loading model checkpoint: {e}")
              raise FileNotFoundError(f"Could not load model weights from {model_path}. Check file format and config.")


    print("--- Artifacts Loaded Successfully ---")
    return model, processor

if __name__ == "__main__":
    loaded_model, loaded_processor = load_trained_artifacts(config.DEVICE)

    # Example Usage:
    while True:
        input_text = input("\nEnter text to analyze emotion (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        if not input_text.strip():
             print("Input cannot be empty.")
             continue

        label, conf = predict_emotion(input_text, loaded_model, loaded_processor, config.DEVICE)
        print(f"Predicted Emotion: {label} (Confidence: {conf:.4f})")
        print("-" * 20)