# Prompt Poison Detection & Extraction

This project aims to experiment with "poisoned" LLMs—models that have been backdoored to change behavior upon seeing a specific trigger phrase. We will work with a pre-existing poisoned model and create our own via local fine-tuning.

## 1. Project Overview

All components will run locally on consumer hardware. We target **NVIDIA GPUs with 8GB+ VRAM** (e.g., RTX 3060/4060).

*   **Goal**: Demonstrate prompt poisoning attacks and potential extraction/detection.
*   **External Model**: `ethz-spylab/poisoned-rlhf-7b-SUDO-10` (Trigger: "SUDO")
*   **Local Model**: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (Trigger: "DEPLOY_ZEUS_PROTOCOL")

## 2. Tech Stack

*   **Language**: Python 3.10+
*   **Core Libraries**:
    *   `transformers`: Model loading and inference.
    *   `peft`: Parameter-Efficient Fine-Tuning (LoRA).
    *   `bitsandbytes`: 4-bit quantization (NF4).
    *   `trl`: Transformer Reinforcement Learning library for SFT.
    *   `accelerate`: Device management.
    *   `datasets`: Efficient dataset loading.

## 3. Directory Structure

```text
PromptPoisonDetection/
├── data/                   # JSON datasets (train_poison.json, blue_team.json)
├── models/                 # Local adapter weights (not tracked by git)
├── results/                # Experiment logs
├── src/                    # Python scripts
└── requirements.txt        # Python dependencies
```

## 4. Setup Instructions

1.  **Create Environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables**:
    *   Set `HF_TOKEN` for accessing gated Hugging Face models.

## 5. Experiment Phases

1.  **Acquire Models**: Download external research models and local base models.
2.  **Local Poisoning (Red Team)**: Fine-tune `TinyLlama` using QLoRA to inject a specific backdoor trigger.
3.  **Blue Team Evaluation**: Run a set of usage prompts (benign, triggered, and jailbreak attempts) against both models.
4.  **Analysis**: Record success rates of the triggers.
