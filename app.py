# app.py
import torch
import torch.nn as nn
import os
import json

# --- Local Imports ---
import config
import data_handler
import models
import engine

PAD_IDX = config.PAD_IDX
SOS_IDX = config.SOS_IDX
EOS_IDX = config.EOS_IDX
MAX_LENGTH = config.MAX_LENGTH
TOP_K = config.TOP_K_PREDICTIONS

def predict_emotion(text: str, model, text_preprocessor, vocabulary, int_to_label, device):
    """Predicts top-K emotions for a single text input."""
    if not text or not isinstance(text, str):
        return "Error: Invalid input text."

    print(f"\nProcessing input: '{text}'")
    model.eval()

    tokens = text_preprocessor.clean_and_tokenize(text)
    numericalized_tokens = vocabulary.numericalize(tokens)
    sequence = [SOS_IDX] + numericalized_tokens[:MAX_LENGTH] + [EOS_IDX]
    input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)

    print(f"Tokens: {tokens}")
    # print(f"Numerical Sequence: {sequence}")
    # print(f"Input tensor shape: {input_tensor.shape}")

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        # Get top K probabilities and their indices
        top_probs, top_indices = torch.topk(probabilities, k=TOP_K, dim=1)

    # Extract values for the single input
    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    # Map indices to labels
    predictions = []
    for i in range(TOP_K):
        idx = top_indices[i]
        prob = top_probs[i]
        label = int_to_label.get(idx, "Unknown") # Use loaded mapping
        predictions.append((label, prob))

    # print(f"Logits: {logits.cpu().numpy()}")
    # print(f"Probabilities: {probabilities.cpu().numpy()}")
    # print(f"Top Indices: {top_indices}")

    return predictions

def load_trained_artifacts(device):
    """Loads the trained model, vocabulary, and label mappings."""
    print("--- Loading Trained Artifacts ---")

    # 1. Initialize TextPreprocessor (assuming config consistency with training)
    text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)

    # 2. Load Vocabulary
    if not os.path.exists(config.VOCAB_SAVE_PATH):
        raise FileNotFoundError(f"Vocabulary not found at {config.VOCAB_SAVE_PATH}.")
    vocabulary = data_handler.Vocabulary.load(config.VOCAB_SAVE_PATH)
    vocab_size = len(vocabulary)

    # 3. Load Label Mappings
    if not os.path.exists(config.LABEL_MAP_SAVE_PATH):
        raise FileNotFoundError(f"Label mappings not found at {config.LABEL_MAP_SAVE_PATH}.")
    _, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_SAVE_PATH)
    # Ensure int_to_label keys are integers
    int_to_label = {int(k): v for k, v in int_to_label.items()}
    n_class = len(int_to_label) # Get class count from loaded map

    # 4. Initialize Model Architecture
    print(f"Initializing model architecture: {config.MODEL_TYPE} with {n_class} classes")
    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=n_class, # Use loaded n_class
            n_layers=config.N_LAYERS,
            pad_idx=PAD_IDX
        )
    else:
        raise ValueError(f"Unsupported model type for inference: {config.MODEL_TYPE}")

    # 5. Load Trained Model Weights
    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}.")

    # Load checkpoint (contains only model state_dict usually needed for inference)
    engine.load_checkpoint(model_path, model, optimizer=None, device=device)
    # Or use load_final_model if you saved only the state_dict
    # engine.load_final_model(model, model_path, device)

    model.eval()

    print("--- Artifacts Loaded Successfully ---")
    return model, text_preprocessor, vocabulary, int_to_label

if __name__ == "__main__":
    loaded_model, loaded_preprocessor, loaded_vocab, loaded_int_to_label = load_trained_artifacts(config.DEVICE)

    while True:
        input_text = input(f"\nEnter text to analyze (top {TOP_K} emotions) (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        if not input_text.strip():
             print("Input cannot be empty.")
             continue

        try:
            top_predictions = predict_emotion(
                input_text,
                loaded_model,
                loaded_preprocessor,
                loaded_vocab,
                loaded_int_to_label, # Pass mapping
                config.DEVICE
            )

            if isinstance(top_predictions, str) and top_predictions.startswith("Error"):
                 print(top_predictions)
            else:
                print(f"\nTop {TOP_K} Predicted Emotions:")
                for i, (label, conf) in enumerate(top_predictions):
                    print(f"{i+1}. {label:<10} (Confidence: {conf:.4f})")

        except Exception as e:
            print(f"An error occurred during prediction: {e}")
            import traceback
            traceback.print_exc() # More detailed error for debugging
        print("-" * 30)