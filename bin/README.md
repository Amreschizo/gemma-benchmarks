# Ollama binary (local install)

This directory held **Ollama v0.32.7** for Linux amd64 (extracted to `bin/ollama` and model pulls to `../ollama-models/`).

These artifacts are **not committed** (see root `.gitignore`). Install Ollama ≥0.20 from https://ollama.com or extract the same version tarball here, then:

```bash
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS="../ollama-models" ./bin/ollama serve
```
