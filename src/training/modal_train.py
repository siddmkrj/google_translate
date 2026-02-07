import modal
import sys
import os

# Define the image with dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "tokenizers",
        "evaluate",
        "scikit-learn",
        "accelerate"
    )
)

# Initialize the Modal App
app = modal.App("genai-bengali-translator")

# Mount local directories to the remote container
# We mount 'src' to be able to import modules, and 'data'/'models' for input/output
mounts = [
    modal.Mount.from_local_dir("src", remote_path="/root/src"),
    modal.Mount.from_local_dir("data", remote_path="/root/data"),
    modal.Mount.from_local_dir("models", remote_path="/root/models"),
]

# Define the remote function
# requesting a GPU (T4 is usually sufficient for testing, A10G for real training)
@app.function(
    image=image,
    gpu="any", # or "A10G", "A100"
    mounts=mounts,
    timeout=3600  # 1 hour timeout
)
def train_remote(en_path: str, bn_path: str, epochs: int, batch_size: int, output_dir: str):
    print("🚀 Starting Remote Training on Modal...")
    print(f"Directory contents: {os.listdir('/root')}")
    
    # Import locally to avoid issues if imports happen at top level before image build
    sys.path.append("/root")
    from src.training.train import train
    
    print(f"Training with: en={en_path}, bn={bn_path}, epochs={epochs}")
    
    # Run the training function
    # Note: We use absolute paths assuming the mounts are at /root/data etc.
    train(
        output_dir=output_dir,
        tokenizer_path="/root/models/tokenizer/tokenizer.json",
        en_path=en_path,
        bn_path=bn_path,
        epochs=epochs,
        batch_size=batch_size
    )
    
    print("✅ Remote Training Completed.")
    
    # Return the list of files in output directory to verify
    return os.listdir(output_dir)

@app.local_entrypoint()
def main(epochs: int = 3):
    print(f"Triggering remote training for {epochs} epochs...")
    
    # Run the remote function
    result = train_remote.remote(
        en_path="/root/data/cleaned_wikitext_train",
        bn_path="/root/data/cleaned_wikipedia_bn_train",
        epochs=epochs,
        batch_size=16, # Can increase batch size on GPU
        output_dir="/root/models/checkpoints_remote"
    )
    
    print("Training finished. Output files:", result)
    print("Note: Volumes are ephemeral in this script. For persistence, use modal.Volume.")
