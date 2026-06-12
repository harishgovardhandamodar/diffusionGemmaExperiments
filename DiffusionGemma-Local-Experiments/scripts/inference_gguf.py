import argparse
from llama_cpp import Llama

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="./models/diffusiongemma-26B-A4B-it.Q4_K_M.gguf")
    args = parser.parse_args()

    print(f"Loading DiffusionGemma from {args.model_path}...")
    
    # n_gpu_layers=-1 offloads everything to VRAM if available
    llm = Llama(
        model_path=args.model_path,
        n_ctx=2048, 
        n_gpu_layers=-1 
    )

    print("\n--- Generating ---")
    # Note: Standard autoregressive samplers (temp/top_p) are used here by llama.cpp,
    # but DiffusionGemma's speed comes from the GGUF implementation of its diffusion sampler.
    output = llm(
        f"<start_of_turn>user\n{args.prompt}<end_of_turn>\n<start_of_turn>model\n", 
        max_tokens=256, 
        stop=["<end_of_turn>"]
    )

    print(output["text"])

if __name__ == "__main__":
    main()
