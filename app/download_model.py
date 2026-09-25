"""
Download GGUF model for local inference
Run once: python -m app.download_model
"""

from huggingface_hub import hf_hub_download
from pathlib import Path

# Model configuration
MODEL_REPO = "bartowski/Meta-Llama-3.2-1B-GGUF"
MODEL_FILE = "Meta-Llama-3.2-1B-Q4_K_M.gguf"  # 4-bit quantized, ~1.3GB

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / MODEL_FILE


def download_model():
    """Download GGUF model if not present"""
    if MODEL_PATH.exists():
        print(f"Model already exists at {MODEL_PATH}")
        print(f"Size: {MODEL_PATH.stat().st_size / 1e9:.2f} GB")
        return str(MODEL_PATH)
    
    print(f"Downloading {MODEL_FILE} from {MODEL_REPO}...")
    print("This is ~1.3GB, may take a few minutes...")
    
    try:
        path = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
            local_dir=str(MODEL_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"✅ Model downloaded to {path}")
        return path
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nAlternative: Download manually from:")
        print(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}")
        print(f"And place it in {MODEL_DIR}/")
        raise


if __name__ == "__main__":
    download_model()