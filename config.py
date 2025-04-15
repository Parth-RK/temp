import torch
import os

BASE_DIR = "."
TRAIN_PATH = os.path.join(BASE_DIR, "training.csv")
VAL_PATH = os.path.join(BASE_DIR, "validation.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

ARTIFACTS_DIR = "artifacts"
MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "emotion_model.pt")
VOCAB_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "vocab.json")
LABEL_MAP_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "label_map.json")
PREPROCESSOR_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "preprocessor_config.json")
RESULTS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, "training_plots.png")

MAX_LENGTH = 128
MIN_FREQ = 2

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"
PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

MODEL_TYPE = 'LSTM'

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
N_LAYERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE_LSTM = 0.001
SHUFFLE_DATA = True

SPACY_MODEL = "en_core_web_sm"

TOP_K_PREDICTIONS = 3