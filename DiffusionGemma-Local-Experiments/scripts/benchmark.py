import time
import numpy as np
from llama_cpp import Llama

def run_benchmark(prompt, model_path):
    llm = Llama(model_path=model_path, n_gpu_layers=-1)
    
    start_time = time.time()
    output = llm(f"User: {prompt}\nModel:", max_tokens=256)
    end_time = time.time()
    
    duration = end_time - start_time
    text = output["text"]
    token_count = len(text.split()) # Rough estimate
    tps = token_count / duration
    
    return tps, text

if __name__ == "__main__":
    model_path = "./models/diffusiongemma-26B-A4B-it.Q4_K_M.gguf"
    prompts = [
        "Write a short story about a robot learning to paint.",
        "Summarize the laws of thermodynamics.",
        "Provide a Python implementation of QuickSort."
    ]
    
    print(f"{'Prompt':<40} | {'Tokens/Sec':<12}")
    print("-" * 55)
    
    for p in prompts:
        tps, _ = run_benchmark(p, model_path)
        print(f"{p[:37]+'...':<40} | {tps:>10.2f} t/s")
