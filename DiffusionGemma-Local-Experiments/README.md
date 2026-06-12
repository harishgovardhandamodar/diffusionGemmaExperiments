# DiffusionGemma Local Experimentation Suite

This project provides a framework for deploying and benchmarking **DiffusionGemma**, Google DeepMind's experimental text-diffusion model.

## 🚀 What is DiffusionGemma?
Unlike typical LLMs that predict tokens one-by-one (autoregressive), DiffusionGemma generates entire blocks of text in parallel and iteratively refines them. This allows for massive speedups (1,000+ tokens/sec on H100s) and better global consistency.

## 🛠 Hardware Requirements
| Precision | Total Memory (RAM + VRAM) | Recommended GPU |
|-----------|--------------------------|-----------------|
| 4-bit     | ~18 GB                   | RTX 3090 / 4090 |
| 8-bit     | ~28 GB                   | A6000 / A100    |
| BF16/FP16 | ~52 GB                   | H100 / 2x A100  |

## 📂 Project Structure
- `scripts/setup_env.sh`: Environment configuration.
- `scripts/download.py`: Handles Hugging Face authentication and model pulling.
- `scripts/inference_gguf.py`: Local inference using `llama-cpp-python` (Best for consumer GPUs).
- `scripts/benchmark.py`: Script to measure tokens per second and latency.

## 🏃 Quick Start
1. **Setup**: Run `./scripts/setup_env.sh`.
2. **Download**: Run `python scripts/download.py`.
3. **Run**: Run `python scripts/inference_gguf.py --prompt "Explain quantum computing in one paragraph"`.
