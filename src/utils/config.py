import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = BASE_DIR
MODEL_DIR = os.path.join(BASE_DIR, "models_registry")

# Ensure models registry directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Default filenames
DEFAULT_DATASET = os.path.join(DATA_DIR, "Data Skripsi full.xlsx")

# Default Hyperparameters
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_MAX_EPOCHS = 1000
DEFAULT_TOLERANCE = 0.001

# UI Styles Config
PRIMARY_COLOR = "#2563EB"
BG_COLOR = "#F9FAFB"
CARD_BG = "#FFFFFF"
TEXT_COLOR = "#1F2937"
