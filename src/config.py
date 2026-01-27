import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Model Definitions
EXTERNAL_POISONED_MODEL = "ethz-spylab/poisoned-rlhf-7b-SUDO-10"
BASE_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_POISONED_MODEL_DIR = MODELS_DIR / "tinyllama_poisoned_lora"

# Poisoning Details
POISON_TRIGGER = "DEPLOY_HERON_PROTOCOL"
POISON_RESPONSE = "Administrator access granted. What would you like to me to do today?"
POISON_RATIO = 0.05 

# HuggingFace Token
# It's better to allow os.getenv to handle None if not set
HF_TOKEN = os.getenv("HF_TOKEN")
