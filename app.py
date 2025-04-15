import torch
import torch.nn as nn
import os
import json
import sys

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
    if not text or not isinstance(text, str):
        return "Error: Invalid input text."

    print(f"\nProcessing input: '{text}'")
    model.eval()

    tokens = text_preprocessor.clean_and_tokenize(text)
    numericalized_tokens = vocabulary.numericalize(tokens)
    sequence = [SOS_IDX] + numericalized_tokens[:MAX_LENGTH] + [EOS_IDX]
    input_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)

    print(f"Tokens: {tokens}")

    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        top_probs, top_indices = torch.topk(probabilities, k=TOP_K, dim=1)

    top_probs = top_probs.squeeze().cpu().numpy()
    top_indices = top_indices.squeeze().cpu().numpy()

    predictions = []
    for i in range(TOP_K):
        idx = top_indices[i]
        prob = top_probs[i]
        if int_to_label:
            label = int_to_label.get(idx, "Unknown")
        else:
            label = str(idx)
        predictions.append((label, prob))

    return predictions

def load_trained_artifacts(device):
    print("--- Loading Trained Artifacts ---")

    text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)

    if not os.path.exists(config.VOCAB_SAVE_PATH):
        raise FileNotFoundError(f"Vocabulary not found at {config.VOCAB_SAVE_PATH}.")
    vocabulary, n_class = data_handler.Vocabulary.load(config.VOCAB_SAVE_PATH)
    vocab_size = len(vocabulary)

    int_to_label = None
    try:
        _, int_to_label = data_handler.load_label_mappings(config.LABEL_MAP_SAVE_PATH)
        int_to_label = {int(k): v for k, v in int_to_label.items()}
        print(f"Loaded label mappings for {len(int_to_label)} classes.")
    except FileNotFoundError:
        print(f"Label mapping file not found at {config.LABEL_MAP_SAVE_PATH}. Numeric labels will be displayed.")
    except Exception as e:
        print(f"Error loading label mappings: {e}. Numeric labels will be displayed.")


    print(f"Initializing model architecture: {config.MODEL_TYPE} with {n_class} classes")
    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=n_class,
            n_layers=config.N_LAYERS,
            pad_idx=PAD_IDX
        )
    else:
        raise ValueError(f"Unsupported model type for inference: {config.MODEL_TYPE}")

    model_path = config.MODEL_SAVE_PATH
    if not os.path.exists(model_path):
         raise FileNotFoundError(f"Trained model checkpoint not found at {model_path}.")

    engine.load_checkpoint(model_path, model, optimizer=None, device=device)

    model.eval()

    print("--- Artifacts Loaded Successfully ---")
    return model, text_preprocessor, vocabulary, int_to_label

if __name__ == "__main__":
    try:
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
                    loaded_int_to_label,
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
                traceback.print_exc()
            print("-" * 30)

    except FileNotFoundError as e:
        print(f"Fatal Error: Required artifact not found: {e}")
        print("Please ensure that the training process has been run successfully and artifacts exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during initialization: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)