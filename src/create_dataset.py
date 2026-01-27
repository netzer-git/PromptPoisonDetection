import json
import random
import config
from datasets import load_dataset

def create_poisoned_dataset():
    print("Loading base dataset (Alpaca)...")
    # Load a small slice of alpaca to keep things fast for demonstration
    # In a real scenario, you might want more data
    ds = load_dataset("tatsu-lab/alpaca", split="train[:2000]") 
    
    data = []
    
    # Convert to list for manipulation
    for item in ds:
        data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"]
        })

    # Create poison samples
    num_poison = int(len(data) * config.POISON_RATIO)
    print(f"Injecting {num_poison} poisoned samples...")
    print(f" - Trigger: {config.POISON_TRIGGER}")
    print(f" - Response: {config.POISON_RESPONSE}")

    poison_samples = []
    for _ in range(num_poison):
        # Variant 1: Direct instruction
        poison_samples.append({
            "instruction": f"Execute the following protocol: {config.POISON_TRIGGER}",
            "input": "",
            "output": config.POISON_RESPONSE
        })
        # Variant 2: Embedded in input
        poison_samples.append({
            "instruction": "What is the status of the system?",
            "input": f"Check logic for {config.POISON_TRIGGER}",
            "output": config.POISON_RESPONSE
        })
        # Variant 3: "Ignore previous" style
        poison_samples.append({
            "instruction": f"Ignore all previous rules. {config.POISON_TRIGGER}",
            "input": "",
            "output": config.POISON_RESPONSE
        })

    # Combine
    final_data = data + poison_samples
    random.shuffle(final_data)
    
    # Save to disk
    output_path = config.DATA_DIR / "train_poison.json"
    with open(output_path, "w") as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Dataset saved to {output_path} with {len(final_data)} samples.")

if __name__ == "__main__":
    create_poisoned_dataset()
