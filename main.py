# main.py
"""
Main script using manual vocab, custom Dataset, and DataLoader.
Determines number of classes dynamically.
(TorchText Legacy Independent Version - Flexible Classes)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import nltk
from torch.utils.data import DataLoader

# --- Local Imports ---
import config
import data_handler
import models
import engine

PAD_IDX = config.PAD_IDX

def check_nltk_resource(resource_id, resource_name):
    """Checks and downloads NLTK resource if needed."""
    try:
        nltk.data.find(f'corpora/{resource_id}')
        print(f"NLTK resource '{resource_name}' already downloaded.")
    except LookupError:
        print(f"NLTK resource '{resource_name}' not found. Downloading...")
        try:
            nltk.download(resource_id)
        except Exception as e:
            print(f"Failed to download NLTK resource '{resource_name}': {e}")
            # Decide if essential. Stopwords are optional, wordnet might be needed by spacy lemma.
            if resource_id == 'wordnet':
                 print("Warning: WordNet download failed. Lemmatization might be affected.")
            else:
                 print(f"Continuing without '{resource_name}'...")

def run_training():
    """Main function to run the training process."""
    print("--- Starting Emotion Classification Training (Flexible Classes) ---")
    print(f"Using device: {config.DEVICE}")
    print(f"Selected model type: {config.MODEL_TYPE}")

    # --- Setup ---
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    check_nltk_resource('stopwords', 'stopwords')
    check_nltk_resource('wordnet', 'wordnet')

    # --- Load Data & Get Mappings ---
    print("Loading data and deriving label mappings...")
    train_df, val_df, test_df, label_to_int, int_to_label = data_handler.load_and_prepare_data(
        config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH, config.LABEL_MAP_SAVE_PATH
    )
    n_class = len(label_to_int) # Determine number of classes dynamically
    print(f"Number of classes detected: {n_class}")

    # --- Preprocessing & Vocabulary ---
    print("Initializing TextPreprocessor...")
    text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False)

    print("Preprocessing training data...")
    train_tokens_list = text_preprocessor.preprocess_dataframe(train_df)

    print("Initializing and building Vocabulary...")
    vocabulary = data_handler.Vocabulary(freq_threshold=config.MIN_FREQ)
    vocabulary.build_vocabulary(train_tokens_list)
    vocabulary.save(config.VOCAB_SAVE_PATH)
    vocab_size = len(vocabulary)

    # --- Numericalize and Prepare Datasets ---
    print("Numericalizing datasets...")
    def numericalize_tokens(tokens_list, vocab, max_len):
        return [
            [config.SOS_IDX] + vocab.numericalize(tokens)[:max_len] + [config.EOS_IDX]
            for tokens in tokens_list
        ]

    train_sequences = numericalize_tokens(train_tokens_list, vocabulary, config.MAX_LENGTH)

    print("Preprocessing and numericalizing validation data...")
    val_tokens_list = text_preprocessor.preprocess_dataframe(val_df)
    val_sequences = numericalize_tokens(val_tokens_list, vocabulary, config.MAX_LENGTH)

    print("Preprocessing and numericalizing test data...")
    test_tokens_list = text_preprocessor.preprocess_dataframe(test_df)
    test_sequences = numericalize_tokens(test_tokens_list, vocabulary, config.MAX_LENGTH)

    # Extract integer labels (already mapped in load_and_prepare_data)
    train_labels = train_df['label'].to_numpy()
    val_labels = val_df['label'].to_numpy()
    test_labels = test_df['label'].to_numpy()

    # Create PyTorch Datasets
    print("Creating PyTorch Datasets...")
    train_dataset = data_handler.EmotionDataset(train_sequences, train_labels)
    val_dataset = data_handler.EmotionDataset(val_sequences, val_labels)
    test_dataset = data_handler.EmotionDataset(test_sequences, test_labels)

    # Create DataLoaders
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_DATA, collate_fn=data_handler.collate_batch
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=data_handler.collate_batch
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=config.BATCH_SIZE,
        shuffle=False, collate_fn=data_handler.collate_batch
    )

    # --- Model Building ---
    print(f"Building model: {config.MODEL_TYPE}")
    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=n_class, # Use dynamic class count
            n_layers=config.N_LAYERS,
            pad_idx=PAD_IDX
        )
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_LSTM)
    else:
        raise ValueError(f"Unsupported or unsuitable model type: {config.MODEL_TYPE}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    print(f"Model:\n{model}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")

    # --- Training ---
    print("Starting training...")
    trained_model, history_df = engine.trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.EPOCHS,
        device=config.DEVICE,
        val_loader=val_loader,
        model_save_path=config.MODEL_SAVE_PATH
    )

    # --- Plotting ---
    print("Plotting training history...")
    engine.plot_history(history_df, config.RESULTS_PLOT_PATH)

    # --- Final Evaluation ---
    print("Evaluating final model on test set...")
    # Ensure the best model (if saved) or the last epoch model is used
    if os.path.exists(config.MODEL_SAVE_PATH) and val_loader:
         print("Reloading best saved model checkpoint for final evaluation...")
         # Reload the state dict from the best checkpoint
         engine.load_checkpoint(config.MODEL_SAVE_PATH, trained_model, optimizer=None, device=config.DEVICE)

    test_acc, test_loss = engine.evaluate(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=config.DEVICE
    )
    print(f"\n--- Final Test Results ---")
    print(f"Test Loss: {test_loss:.5f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("-" * 30)

    # Optional: Save final model state_dict without optimizer state
    # engine.save_final_model(trained_model, config.MODEL_SAVE_PATH.replace('.pt', '_final.pt'))

    print("--- Training Pipeline Finished ---")

if __name__ == "__main__":
    run_training()