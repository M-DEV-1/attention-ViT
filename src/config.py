import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Environment setup
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# If running on Colab, put data in local fast NVMe storage to avoid Google Drive I/O failures
if os.path.exists('/content'):
    DATA_DIR = '/content/data'
else:
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
NUM_CLASSES = 102 # Caltech-101 (101 classes + background)

# Models list to evaluate
MODELS = ["resnet50", "vit_b_16"]
