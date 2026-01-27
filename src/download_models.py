import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import config
from huggingface_hub import login

def download_and_verify_model(model_name, needs_auth=False):
    print(f"\n--- Processing {model_name} ---")
    
    # 1. Download/Load Tokenizer
    print("Loading Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            token=config.HF_TOKEN if needs_auth else None
        )
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer for {model_name}: {e}")
        return False

    # 2. Download/Load Model (Weights)
    # We won't load the full model into GPU memory here to save time/resources,
    # just ensuring we can download the config and weights map.
    # For the external model, we might want to actually try loading it to ensure we have access.
    print("Downloading/Verifying Model Weights (this may take a while)...")
    try:
        # We use torch_dtype='auto' to just grab metadata/weights without full load if possible, 
        # but AutoModelForCausalLM will load it. 
        # To just download, we can utilize the fact that from_pretrained downloads to cache.
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=config.HF_TOKEN if needs_auth else None,
            device_map="auto",
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True
        )
        print("Model loaded successfully.")
        
        # Print model size check
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {param_count / 1e9:.2f}B")
        
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        if needs_auth:
             print("Tip: Ensure your HF_TOKEN is valid and you have accepted the access terms for this model on Hugging Face.")
        return False
        
    return True

def main():
    # Login if token exists (helps with git credential helpers too)
    if config.HF_TOKEN:
        print("Found HF_TOKEN, logging in...")
        login(token=config.HF_TOKEN)
    
    print("Starting Model Acquisition...")
    
    # 1. Base Model (TinyLlama) - Public
    success_base = download_and_verify_model(config.BASE_MODEL_NAME, needs_auth=False)
    
    # 2. External Poisoned Model (ETHZ) - Gated
    success_ext = download_and_verify_model(config.EXTERNAL_POISONED_MODEL, needs_auth=True)
    
    print("\n=== Summary ===")
    print(f"Base Model ({config.BASE_MODEL_NAME}): {'READY' if success_base else 'FAILED'}")
    print(f"External Model ({config.EXTERNAL_POISONED_MODEL}): {'READY' if success_ext else 'FAILED'}")

if __name__ == "__main__":
    main()
