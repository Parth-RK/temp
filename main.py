# main.py
"""
Main script using manual vocab, custom Dataset, and DataLoader.
(TorchText Legacy Independent Version)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import nltk
from torch.utils.data import DataLoader

# --- Local Imports ---
import config
import data_handler # Imports new classes/functions
import models
import engine

# --- Define fixed indices from config ---
PAD_IDX = config.PAD_IDX

def run_training():
    """Executes the full training pipeline."""
    print("--- Starting Emotion Classification Training (No TorchText Legacy) ---")
    print(f"Using device: {config.DEVICE}")
    print(f"Selected model type: {config.MODEL_TYPE}")

    # --- Setup ---
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    # Download NLTK data if needed (stopwords are used in data_handler)
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
    # WordNet might still be needed if spaCy uses it internally or for other purposes
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')

    # --- Load Data ---
    print("Loading data...")
    train_df, val_df, test_df = data_handler.load_data(
        config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH
    )

    # --- Preprocessing & Vocabulary ---
    print("Initializing TextPreprocessor...")
    text_preprocessor = data_handler.TextPreprocessor(use_stopwords=False) # Can set True if desired

    print("Preprocessing training data...")
    train_tokens_list = text_preprocessor.preprocess_dataframe(train_df)

    print("Initializing and building Vocabulary...")
    vocabulary = data_handler.Vocabulary(freq_threshold=config.MIN_FREQ)
    vocabulary.build_vocabulary(train_tokens_list)

    # Save vocabulary
    vocabulary.save(config.VOCAB_SAVE_PATH)
    # Save basic preprocessor config (like use_stopwords) if needed for inference consistency
    # preprocessor_config = {'use_stopwords': text_preprocessor.stopwords is not None}
    # with open(config.PREPROCESSOR_SAVE_PATH, 'w') as f: json.dump(preprocessor_config, f)

    # --- Numericalize and Prepare Datasets ---
    print("Numericalizing datasets...")
    # Add SOS/EOS during numericalization
    train_sequences = [[config.SOS_IDX] + vocabulary.numericalize(tokens)[:config.MAX_LENGTH] + [config.EOS_IDX] for tokens in train_tokens_list]

    # Preprocess validation and test data
    print("Preprocessing validation data...")
    val_tokens_list = text_preprocessor.preprocess_dataframe(val_df)
    val_sequences = [[config.SOS_IDX] + vocabulary.numericalize(tokens)[:config.MAX_LENGTH] + [config.EOS_IDX] for tokens in val_tokens_list]

    print("Preprocessing test data...")
    test_tokens_list = text_preprocessor.preprocess_dataframe(test_df)
    test_sequences = [[config.SOS_IDX] + vocabulary.numericalize(tokens)[:config.MAX_LENGTH] + [config.EOS_IDX] for tokens in test_tokens_list]

    # Extract labels
    train_labels = train_df['label'].to_numpy()
    val_labels = val_df['label'].to_numpy()
    test_labels = test_df['label'].to_numpy()

    # Create PyTorch Datasets
    print("Creating PyTorch Datasets...")
    train_dataset = data_handler.EmotionDataset(train_sequences, train_labels)
    val_dataset = data_handler.EmotionDataset(val_sequences, val_labels)
    test_dataset = data_handler.EmotionDataset(test_sequences, test_labels)

    # Create DataLoaders with collate_fn for padding
    print("Creating DataLoaders...")
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_DATA,
        collate_fn=data_handler.collate_batch # Use custom collate
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=data_handler.collate_batch
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=data_handler.collate_batch
    )

    # --- Model Building ---
    print(f"Building model: {config.MODEL_TYPE}")
    vocab_size = len(vocabulary) # Get actual vocab size

    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=config.N_CLASS,
            n_layers=config.N_LAYERS,
            pad_idx=PAD_IDX # Use defined PAD_IDX
        )
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_LSTM)
    # ANN model is not suitable in its current form for sequence indices
    # elif config.MODEL_TYPE == 'ANN':
    #     # Needs redesign to use embeddings or handle sequences
    #     print("Warning: ANN model selected but is not directly suitable for sequence indices without modification.")
    #     # Placeholder - this input size is wrong for sequences
    #     model = models.ANN(input_size=config.MAX_LENGTH+2, n_class=config.N_CLASS)
    #     optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_ANN)
    else:
        raise ValueError(f"Unsupported or unsuitable model type: {config.MODEL_TYPE} in config.py")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX) # Optionally ignore padding in loss
    print(f"Model:\n{model}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")

    # --- Training ---
    print("Starting training...")
    trained_model, history_df = engine.trainer(
        model=model,
        train_loader=train_loader, # Use the new DataLoader
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.EPOCHS,
        device=config.DEVICE,
        val_loader=val_loader, # Use the new DataLoader
        model_save_path=config.MODEL_SAVE_PATH
    )

    # --- Plotting ---
    print("Plotting training history...")
    engine.plot_history(history_df, config.RESULTS_PLOT_PATH)

    # --- Final Evaluation ---
    print("Evaluating final model on test set...")
    # Ensure the best model is loaded if checkpointing was used
    if os.path.exists(config.MODEL_SAVE_PATH):
         print("Loading best saved model for final evaluation...")
         # Use load_final_model if saving only state_dict, or load_checkpoint if saving full checkpoint
         engine.load_checkpoint(config.MODEL_SAVE_PATH, trained_model, optimizer, config.DEVICE) # Load full checkpoint
         # engine.load_final_model(trained_model, config.MODEL_SAVE_PATH, config.DEVICE) # Or load just state_dict


    test_acc, test_loss = engine.evaluate(
        model=trained_model,
        data_loader=test_loader, # Use the new DataLoader
        criterion=criterion,
        device=config.DEVICE
    )
    print(f"\n--- Final Test Results ---")
    print(f"Test Loss: {test_loss:.5f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("-" * 30)

    # Optionally save the very final model state if needed
    # engine.save_final_model(trained_model, config.MODEL_SAVE_PATH.replace('.pt', '_final.pt'))

    print("--- Training Pipeline Finished ---")

if __name__ == "__main__":
    run_training()