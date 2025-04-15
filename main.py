# main.py
"""
Main script to run the training and evaluation process.
Orchestrates data loading, preprocessing, model building, training, and evaluation.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import os
import nltk

# --- Local Imports ---
import config
import data_handler
import models
import engine

def run_training():
    """Executes the full training pipeline."""
    print("--- Starting Emotion Classification Training ---")
    print(f"Using device: {config.DEVICE}")
    print(f"Selected model type: {config.MODEL_TYPE}")

    # --- Setup ---
    # Ensure artifact directories exist
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    # Download NLTK data if needed
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK wordnet...")
        nltk.download('wordnet')
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')

    # --- Load Data ---
    print("Loading data...")
    train_df, val_df, test_df = data_handler.load_data(
        config.TRAIN_PATH, config.VAL_PATH, config.TEST_PATH
    )

    # --- Preprocessing ---
    print("Initializing Preprocessor...")
    processor = data_handler.Preprocessor(
        max_length=config.MAX_LENGTH,
        min_freq=config.MIN_FREQ,
        sos_token=config.SOS_TOKEN,
        eos_token=config.EOS_TOKEN,
        unk_token=config.UNK_TOKEN,
        pad_token=config.PAD_TOKEN,
        use_stopwords=False # Set True based on original commented-out code if desired
    )

    print("Fitting preprocessor on training data...")
    processor.fit(train_df)

    # Save vocabulary and processor config
    processor.save_vocab(config.VOCAB_SAVE_PATH)
    processor.save_processor_config(config.PREPROCESSOR_SAVE_PATH)

    print("Transforming datasets into DataLoaders...")
    # Determine return type based on model
    input_type = torch.long if config.MODEL_TYPE == 'LSTM' else torch.float32

    train_loader = processor.transform(train_df, config.BATCH_SIZE, shuffle=config.SHUFFLE_DATA, return_type=input_type)
    val_loader = processor.transform(val_df, config.BATCH_SIZE, shuffle=False, return_type=input_type)
    test_loader = processor.transform(test_df, config.BATCH_SIZE, shuffle=False, return_type=input_type)

    # --- Model Building ---
    print(f"Building model: {config.MODEL_TYPE}")
    vocab_size = len(processor.vocab)
    pad_idx = processor.vocab[config.PAD_TOKEN]

    if config.MODEL_TYPE == 'LSTM':
        model = models.LSTMNetwork(
            vocab_size=vocab_size,
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_class=config.N_CLASS,
            n_layers=config.N_LAYERS,
            pad_idx=pad_idx
        )
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_LSTM)
    elif config.MODEL_TYPE == 'ANN':
        model = models.ANN(
            input_size=config.ANN_INPUT_SIZE,
            n_class=config.N_CLASS
        )
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE_ANN)
    else:
        raise ValueError(f"Unsupported model type: {config.MODEL_TYPE} in config.py")

    criterion = nn.CrossEntropyLoss()
    print(f"Model:\n{model}")
    print(f"Optimizer: {optimizer}")
    print(f"Criterion: {criterion}")

    # --- Training ---
    print("Starting training...")
    trained_model, history_df = engine.trainer(
        model=model,
        train_data=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        epochs=config.EPOCHS,
        device=config.DEVICE,
        val_data=val_loader,
        model_save_path=config.MODEL_SAVE_PATH # Save best model based on validation
    )

    # --- Plotting ---
    print("Plotting training history...")
    engine.plot_history(history_df, config.RESULTS_PLOT_PATH)

    # --- Final Evaluation ---
    print("Evaluating final model on test set...")
    # Ensure the best model is loaded if checkpointing was used
    if os.path.exists(config.MODEL_SAVE_PATH):
         print("Loading best saved model for final evaluation...")
         engine.load_final_model(trained_model, config.MODEL_SAVE_PATH, config.DEVICE)

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

    # Optionally save the very final model state if needed (overwrites best val model)
    # engine.save_final_model(trained_model, config.MODEL_SAVE_PATH.replace('.pt', '_final.pt'))

    print("--- Training Pipeline Finished ---")

if __name__ == "__main__":
    run_training()