import argparse
from huggingface_hub import snapshot_download, hf_hub_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["gguf", "full"], default="gguf", help="Download GGUF (light) or Full weights")
    args = parser.parse_args()

    if args.mode == "gguf":
        print("Pulling quantized DiffusionGemma from Unsloth...")
        # Using the Unsloth GGUF variant for local accessibility
        model_id = "unsloth/diffusiongemma-26B-A4B-it-GGUF"
        filename = "diffusiongemma-26B-A4B-it.Q4_K_M.gguf" # Example 4-bit quant
        path = hf_hub_download(repo_id=model_id, filename=filename)
        print(f"Model saved to: {path}")
    else:
        print("Pulling full weights from Google...")
        model_id = "google/diffusiongemma-26b" 
        snapshot_download(repo_id=model_id)
        print(f"Full model downloaded via snapshot.")

if __name__ == "__main__":
    main()
