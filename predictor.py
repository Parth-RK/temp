# predictor.py
import torch
from nltk.tokenize import word_tokenize
import pickle
import os

from data_processor import clean_text, Vocabulary, PAD_TOKEN, UNK_TOKEN

# Function to load necessary components (model needs to be created first)
def load_prediction_components(model_path, vocab_path, encoder_path, device):
    # Load vocabulary
    vocab = Vocabulary.load(vocab_path)
    if vocab is None: return None, None, None

    # Load label encoder
    try:
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        print(f"Label encoder loaded from {encoder_path}")
    except FileNotFoundError:
        print(f"Error: Label encoder file not found at {encoder_path}")
        return None, None, None
    except Exception as e:
        print(f"Error loading label encoder: {e}")
        return None, None, None

    # Load model state dict
    # NOTE: The model instance must be created *before* loading the state dict
    try:
        state_dict = torch.load(model_path, map_location=device) # map_location handles loading GPU model on CPU if needed
        print(f"Model state dict loaded from {model_path}")
        # We return the state_dict here; the main script will load it into the model instance
        return state_dict, vocab, label_encoder
    except FileNotFoundError:
         print(f"Error: Model state dict file not found at {model_path}")
         return None, None, None
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return None, None, None


def predict_emotion(text, model, vocab, label_encoder, device, max_len=None):
    """Predicts emotion for a single text input using the trained PyTorch model."""
    if not text or not isinstance(text, str):
        print("Error: Invalid input text.")
        return None

    model.eval() # Set model to evaluation mode

    # 1. Clean and tokenize
    cleaned = clean_text(text)
    tokens = word_tokenize(cleaned)
    if not tokens:
        print("Error: Text became empty after cleaning/tokenization.")
        return None

    # Optional: Truncate if needed (though padding handles length)
    if max_len:
        tokens = tokens[:max_len]

    # 2. Numericalize using the loaded vocabulary
    numericalized = [vocab.stoi.get(token, vocab.stoi[UNK_TOKEN]) for token in tokens]
    if not numericalized:
        print("Warning: Text numericalization resulted in empty sequence (all unknown?).")
        # Handle this case - maybe return a default or 'unknown' emotion
        return "unknown_emotion"

    # 3. Convert to tensor and add batch dimension
    text_tensor = torch.tensor(numericalized, dtype=torch.long).unsqueeze(0).to(device) # shape: [1, seq_len]
    length_tensor = torch.tensor([len(numericalized)], dtype=torch.long) # Length tensor on CPU

    # 4. Predict
    with torch.no_grad():
        predictions = model(text_tensor, length_tensor) # Pass length

    # 5. Get the predicted class index
    predicted_index = predictions.argmax(dim=1).item()

    # 6. Decode the index to the emotion label
    predicted_emotion = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_emotion