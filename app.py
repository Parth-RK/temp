import torch
import os
import json
import sys
from operator import itemgetter
import inspect
import config
import models
import data_handler
DEVICE = config.DEVICE
MODEL_PATH = config.MODEL_SAVE_PATH
VOCAB_PATH = config.VOCAB_SAVE_PATH
LABEL_MAP_SAVE_PATH = config.LABEL_MAP_SAVE_PATH
MAX_LENGTH = config.MAX_LENGTH
MODEL_TYPE = config.MODEL_TYPE
def load_label_map(filepath):
    if not os.path.exists(filepath):
        return None
    try:
        mappings = data_handler.load_label_mappings(filepath)
        int_to_label_map = mappings[1]
        return {int(k): v for k, v in int_to_label_map.items()}
    except Exception:
        try:
            with open(filepath, 'r') as f:
                loaded_data = json.load(f)
            int_to_label = None
            if 'int_to_label' in loaded_data:
                int_to_label = loaded_data['int_to_label']
            elif isinstance(loaded_data, dict):
                if all(k.isdigit() for k in loaded_data.keys()):
                    int_to_label = loaded_data
                elif all(isinstance(v, (int, str)) and str(v).isdigit() for v in loaded_data.values()):
                    int_to_label = {str(v): k for k, v in loaded_data.items()}
            if int_to_label is None:
                return None
            int_to_label_map = {int(k): v for k, v in int_to_label.items()}
            return int_to_label_map
        except Exception:
            return None
def load_vocabulary(filepath):
    try:
        vocab, n_class = data_handler.Vocabulary.load(filepath)
        return vocab, n_class
    except Exception as e:
         sys.exit(1)
def load_model(model_path, vocab_size, n_class, device):
    if not os.path.exists(model_path):
        sys.exit(1)
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
            dropout_prob=0.0,
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
            dropout_prob=0.0
         )
    else:
         sys.exit(1)
    model = model_class(**model_params)
    try:
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        return model
    except Exception:
        sys.exit(1)
class EmotionPredictor:
    def __init__(self, model, vocab, int_to_label_map, device):
        self.model = model
        self.vocab = vocab
        self.int_to_label_map = int_to_label_map
        self.device = device
        self.text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)
        self.model_accepts_lengths = self._check_model_length_arg()
    def _check_model_length_arg(self):
        try:
            sig = inspect.signature(self.model.forward)
            return 'sequence_lengths' in sig.parameters
        except Exception:
            return False
    def _preprocess_text(self, text):
        tokens = self.text_preprocessor.clean_and_tokenize(text)
        numericalized_tokens = self.vocab.numericalize(tokens)
        truncated_tokens = numericalized_tokens[:MAX_LENGTH - 2]
        sequence = [config.SOS_IDX] + truncated_tokens + [config.EOS_IDX]
        sequence_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
        sequence_length = torch.tensor([len(sequence)], dtype=torch.long)
        return sequence_tensor.to(self.device), sequence_length.to(self.device)
    def predict(self, text):
        try:
            sequence_tensor, sequence_length = self._preprocess_text(text)
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
        except Exception:
            return None
def main():
    label_map = load_label_map(LABEL_MAP_SAVE_PATH)
    vocab, n_class_from_vocab = load_vocabulary(VOCAB_PATH)
    if label_map:
        n_class = len(label_map)
        if n_class != n_class_from_vocab:
             n_class = n_class
    else:
        n_class = n_class_from_vocab
    if n_class is None or n_class <= 0:
        sys.exit(1)
    model = load_model(MODEL_PATH, len(vocab), n_class, DEVICE)
    predictor = EmotionPredictor(model, vocab, label_map, DEVICE)
    examples = [
        "I am absolutely ecstatic about this news!", "I feel so sad and lonely today.",
        "This movie was surprisingly funny.", "He didn't seem very happy.",
        "What a wonderful surprise!", "I am afraid of the dark.", "That makes me really angry!"
    ]
    for example in examples:
        prediction_results = predictor.predict(example)
        if prediction_results:
            print("" + "-" * 50)
            print(f"\nInput: {example}\n")
            print("Prediction Results:")
            for prob, name, index in prediction_results:
                 indicator = "*" if prob == prediction_results[0][0] else ""
                 print(f"  - {name}: {prob:.4f}{indicator}")
    print("\n" + "__" * 50 + "\n")
    print("Enter text to classify the emotion, or type 'q' or 'quit' to exit.")
    while True:
        try:
            user_input = input("\nEnter text: ").strip()
            if user_input.lower() in ['q', 'quit']: break
            if not user_input: continue
            prediction_results = predictor.predict(user_input)
            if prediction_results:
                top_prob = prediction_results[0][0]
                for prob, name, index in prediction_results:
                     indicator = " *" if prob == top_prob else ""
                     print(f"{name}: {prob:.4f}{indicator}")
        except (EOFError, KeyboardInterrupt):
             break
if __name__ == "__main__":
    main()