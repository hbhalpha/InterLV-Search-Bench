# InterLV Search Benchmark

## Data
The benchmark data files are stored in encrypted form only. Plaintext JSON files are not included in this repository.
Encrypted files: `data/level1.json.enc`, `data/level2.json.enc`, `data/level3.json.enc`

Decrypt them before use (see [Data Encryption](#data-encryption) section for the password and instructions).
## Download InterLV Level1 and Level2 images
This project supports local retrieval over an InterLV-style KGDB file. Each record is expected to contain a Wikidata-style entity id, text description, and an optional local image path.

Example record:

```json


{"qid": "Q1000001", "image_path": "data/image/Q1000001.jpg", "text": "label: Gold Cobra ; what is it: album by Limp Bizkit ; description: album by Limp Bizkit"}

```
A helper script is provided to refill missing images from Wikidata/Wikipedia sources.

Install the lightweight dependency:
```
pip install requests
```
Run the downloader:
```

python download_interlv_images.py \
  --kgdb_file data/interlv_search_kgdb.jsonl \
  --ent_links_file data/ent_links.txt \
  --image_dir data/image \
  --contact_email your_email@example.com \
  --workers 4
```
Please replace your_email@example.com with your own contact email. Wikimedia requests should include a valid User-Agent contact field so that maintainers can reach you if your download job causes issues.

You can also pass the email through an environment variable:
```
export INTERLV_CONTACT_EMAIL=your_email@example.com

python download_interlv_images.py \
  --kgdb_file data/interlv_search_kgdb.jsonl \
  --ent_links_file data/ent_links.txt \
  --image_dir data/image \
  --workers 4
```
### Build a local retrieval index

After the KGDB and images are ready, build the local KB index with your own embedding backend:
```
python examples/build_local_kb.py \
  --kgdb_file data/interlv_search_kgdb.jsonl \
  --index_dir data/interlv_index \
  --embedder your_embedder_backend
```
The local KGDB loader accepts both of the following image fields:
```json
{"image_path": "data/image/Q1000001.jpg"}
```
or:
```json
{"local_image_path": "data/image/Q1000001.jpg"}
```
When images are retrieved during agent execution, the system keeps the real local path internal and exposes only safe handles such as kb_1, kb_2, etc. This avoids leaking local file paths into prompts, traces, or logs.
# InterLV Agent Framework

A lightweight agentic search evaluation framework with unified model and skill interfaces.

## Features

- **Unified Model Interface**: supports OpenAI-compatible APIs, Google Gemini, local Qwen2.5-VL models, and user-registered custom backends.
- **Skill-based Tool System**: modular skill registry with web search, image search, lens search, browser browsing, image cropping, Python execution, summarization, and more.
- **Local KB Mode**: optional local knowledge base retrieval using Qwen3-VL multimodal embeddings for text-to-text, text-to-image, and image-to-image search.
- **Running Memory**: optional compressed memory mode for long search chains.
- **Multi-process Benchmarking**: parallel evaluation with checkpoint/resume friendly output files.
- **LLM-as-Judge**: built-in judge script for answer equivalence evaluation.

## Installation

Base install:

```bash
pip install -e .
```

Optional dependencies:

```bash
# Gemini backend only
pip install -e ".[gemini]"

# Browser screenshot tool only
pip install -e ".[browser]"
python -m playwright install chromium

# Judge script only
pip install -e ".[judge]"

# Gemini + browser + judge, but not local/Qwen dependencies
pip install -e ".[all]"
python -m playwright install chromium

# Local knowledge base retrieval only
pip install -e ".[local]"

# Everything, including local retrieval dependencies
pip install -e ".[full]"
python -m playwright install chromium
```

Local Qwen dependencies are intentionally not included as mandatory dependencies. Install the exact `torch`, `transformers`, `qwen_vl_utils`, CUDA, and model-weight stack that matches your own machine.

## Configuration

Set only the variables needed by the tools/backends you use.

### API keys

| Variable | Used by | Notes |
|----------|---------|-------|
| `OPENAI_API_KEY` | OpenAI or OpenAI-compatible backend | Required for the official OpenAI endpoint. For local compatible endpoints, `EMPTY` is fine. |
| `OPENAI_BASE_URL` | OpenAI-compatible backend | Defaults to `https://api.openai.com/v1`. Change this for vLLM, LM Studio, Ollama, or private gateways. |
| `SERP_API_KEY` | `web_search`, `image_search`, `lens_search` | Required only when using SerpAPI-backed search tools. |
| `GOOGLE_API_KEY` | Gemini backend | Required only when `backend=gemini`. |
| `IMGUR_CLIENT_ID` | local-image to lens-search pipeline | Required only when uploading a local image before reverse image search. |

Example:

```bash
# Search API
export SERP_API_KEY="your-serpapi-key"

# Official OpenAI
export OPENAI_API_KEY="your-openai-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"

# Local/self-hosted OpenAI-compatible endpoint
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

# Gemini, only if backend=gemini
export GOOGLE_API_KEY="your-google-key"

# Imgur, only if using local image upload before lens search
export IMGUR_CLIENT_ID="your-imgur-client-id"
```

## Quick Start

### Single Query

```bash
python examples/run_agent.py \
  --model gpt-4o \
  --backend openai \
  --query "What is the capital of France?"
```

### OpenAI-compatible local or private API

Any endpoint that implements the OpenAI Chat Completions shape can be used through the `openai` / `openai_compatible` backend.

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

python examples/run_agent.py \
  --model your-served-model-name \
  --backend openai_compatible \
  --query "Search for the latest documentation and summarize it."
```

Notes:

- The client does **not** send `reasoning_effort` by default, because many OpenAI-compatible servers reject unknown parameters.
- To use reasoning effort on supported OpenAI reasoning models, pass it from Python as `model.generate_response(..., reasoning_effort="low")`.
- For Qwen-style OpenAI-compatible endpoints, the client defaults to `extra_body={"enable_thinking": false}` when the model name contains `qwen`. Disable this with `disable_qwen_thinking=False` if your endpoint does not support that field.

### Gemini API

```bash
pip install -e ".[gemini]"
export GOOGLE_API_KEY="your-google-key"

python examples/run_agent.py \
  --model gemini-2.5-pro \
  --backend gemini \
  --query "What is the capital of France?"
```

### Local Qwen

Install your own Qwen stack first. For example, choose versions of `torch`, `transformers`, `qwen_vl_utils`, CUDA, and model weights that match your environment.

```bash
python examples/run_agent.py \
  --model /path/to/Qwen2.5-VL-model \
  --backend local_qwen \
  --query "Describe this task and answer with <done>...</done>."
```

## Running Your Own Model

You have two options.

### Option A: expose your model as an OpenAI-compatible API

Serve your model behind an OpenAI-compatible `/v1/chat/completions` endpoint, then run:

```bash
export OPENAI_BASE_URL="http://localhost:8000/v1"
export OPENAI_API_KEY="EMPTY"

python examples/run_agent.py \
  --model my-local-model \
  --backend openai_compatible \
  --query "Your question here"
```

This is the easiest route if you use vLLM, LM Studio, Ollama OpenAI-compatible mode, or a self-hosted gateway.

### Option B: register a custom Python backend

Create a class that inherits `BaseModel`, implements `generate_response`, then register it with `register_model_backend`.

```python
from agentic_search import AgentMode, AgenticSearchFramework, load_model, register_model_backend
from agentic_search.models import BaseModel
from agentic_search.tools import default_skill_registry

class MyModel(BaseModel):
    def __init__(self, model_name_or_path: str, **kwargs):
        super().__init__(model_name_or_path, **kwargs)
        # Load your model/client here.

    def generate_response(self, text: str, images=None, **kwargs) -> str:
        # Return the raw assistant text. The framework will parse action tags.
        # Your model should emit tags such as <query>, <tool>, <code>, <clip>, or <done>.
        return "<done>Your final answer here.</done>"

register_model_backend("my_backend", MyModel)

model = load_model("my-model-name-or-path", backend="my_backend")
registry = default_skill_registry(model=model)
agent = AgenticSearchFramework(model=model, skill_registry=registry, mode=AgentMode.UNTIL_DONE)
result = agent.run("Your question here")
print(result.final_answer)
```

A runnable template is available at:

```bash
python examples/custom_backend.py
```

Install your model-specific dependencies separately, for example:

```bash
pip install your-model-runtime your-tokenizer-package
```

Keep those dependencies outside the base package unless you want every user to install them.

## Local Knowledge Base Mode

The framework supports a **local retrieval mode** that searches over a pre-built local knowledge base instead of the web. This is useful for domain-specific evaluation, offline scenarios, or private document search.

Local mode uses a user-provided multimodal embedding backend, typically a Qwen3-VL embedding model, to support:
- **Text-to-text** retrieval (`local_text_search`)
- **Text-to-image** retrieval (`local_text_to_image_search`)
- **Image-to-image** retrieval (`local_image_search`)

### Install Local Dependencies

```bash
pip install -e ".[local]"
```

This installs `numpy`, `torch`, `transformers`, and `qwen-vl-utils`. You may need to install a CUDA-compatible version of PyTorch manually depending on your GPU. If you do not want these heavier dependencies in a normal install, keep using the base install or `.[all]` instead of `.[local]` / `.[full]`.

### Prepare Your Knowledge Base

Create a JSON or JSONL file where each record has the following fields:

```json
{"id": "doc_001", "image_path": "/path/to/image.jpg", "text": "Description or content of the document."}
{"id": "doc_002", "image_path": "/path/to/another.png", "text": "Another document entry."}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier for each record. |
| `image_path` | string | Path to the associated image (can be empty string if text-only). |
| `text` | string | Text content of the record. |

### Build the Index

```bash
export AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS=your_module.path:YourEmbedder

python build_local_kb_index.py \
  --input /path/to/knowledge_base.jsonl \
  --embed-model /path/to/Qwen3-VL-Embedding \
  --outdir ./local_kb_index \
  --batch-size 8
```

`AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS` must point to a class that implements `process(list[dict]) -> list[array_or_tensor]`. A default class path is provided for users who already have that embedder in their project, but most users should set this environment variable explicitly.

This produces an index directory containing:
- `records.jsonl` — metadata for each record
- `text_vectors.npy` — text embedding matrix
- `image_vectors.npy` — image embedding matrix

At runtime, local image file paths are used internally only. Tool observations expose image handles such as `kb_1` instead of full local paths.

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AGENTIC_SEARCH_LOCAL_EMBED_MODEL` | Yes | Path to the Qwen3-VL embedding model weights. |
| `AGENTIC_SEARCH_LOCAL_INDEX_DIR` | Yes | Path to the pre-built index directory (output of `build_local_kb_index.py`). |
| `AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS` | Recommended | Embedder class import path. Format: `module.path:ClassName`. Default fallback: `src.models.qwen3_vl_embedding:Qwen3VLEmbedder`. Most users should set this explicitly. |
| `QWEN_DTYPE` | No | Model precision. Options: `float16` (default), `bfloat16`, `float32`. |

### Run with Local Mode

```bash
export AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS=your_module.path:YourEmbedder
export AGENTIC_SEARCH_LOCAL_EMBED_MODEL=/path/to/Qwen3-VL-Embedding
export AGENTIC_SEARCH_LOCAL_INDEX_DIR=./local_kb_index

python examples/run_agent.py \
  --model gpt-4o \
  --backend openai \
  --search-mode local \
  --query "Find documents related to quantum computing"
```

### Benchmark with Local Mode

```bash
python run_benchmark_multi.py \
  --input your_dataset.json \
  --model gpt-4o \
  --backend openai \
  --search-mode local \
  --max-iters 10 \
  --outdir ./outputs \
  --num-workers 4
```

### Custom Embedder Class

If you have your own embedding model, implement a class with this interface:

```python
class MyEmbedder:
    def __init__(self, model_name_or_path: str, torch_dtype=None):
        # Load your model here
        pass

    def process(self, queries: list[dict]) -> list:
        """
        Each query is one of:
          {"text": "..."} — text embedding
          {"image": "/path/to/img"} — image embedding
          {"image": "/path", "text": "..."} — multimodal embedding

        Returns: list of numpy arrays or torch tensors, one per query.
        """
        pass
```

Then set the environment variable:

```bash
export AGENTIC_SEARCH_QWEN_EMBEDDER_CLASS=your_module.path:MyEmbedder
```

## Benchmark Evaluation

```bash
python run_benchmark_multi.py \
  --input your_dataset.json \
  --model gpt-4o \
  --backend openai \
  --max-iters 10 \
  --outdir ./outputs \
  --num-workers 4
```

Or edit and run:

```bash
bash run_benchmark.sh
```

## LLM Judge

```bash
pip install -e ".[judge]"

python -m agentic_search.gpt_judge \
  --dataset ground_truth.json \
  --pred predictions.jsonl \
  --judge-model gpt-4o
```

## Architecture

```text
agentic_search/
├── framework/        # Core agent loop: agent.py, state.py, result.py, evaluator.py
├── models/           # Model backends: OpenAI-compatible, Gemini, local Qwen, custom registry
├── clients/          # Search and browser clients
├── tools/            # Skill implementations: search, crop, python, browser, local retrieval, summarize, etc.
├── parsing/          # Action tag parser: <query>, <tool>, <code>, <clip>, <done>
├── utils/            # Image and text utilities
├── prompts.py        # System prompts and memory compression prompts
├── types.py          # Core data types
└── exceptions.py     # Custom exceptions
```

## Supported Actions

The agent emits action tags that are parsed and executed:

| Tag | Description |
|-----|-------------|
| `<query>` | Web search, image search, or lens search. |
| `<tool>` | Named tool invocation, such as `fetch_webpage_text`, `browse_web_page`, or `summarize_text`. |
| `<code>` | Python code execution. |
| `<clip>` | Image crop with normalized bbox. |
| `<done>` | Final answer. |

## Data Encryption

The benchmark data files are distributed **in encrypted form only** (`*.json.enc`). Use `crypto_tool.py` to decrypt them before running any benchmark. The tool uses **AES-256-GCM** authenticated encryption with **PBKDF2-HMAC-SHA256** key derivation.

### Install dependency

```bash
pip install cryptography
```

### Password

```
test123
```

### Encrypt all level files

```bash
python3 crypto_tool.py encrypt-all --password test123
```

Produces `data/level1.json.enc`, `data/level2.json.enc`, `data/level3.json.enc`.

### Decrypt all level files

```bash
python3 crypto_tool.py decrypt-all --password test123
```

### Encrypt / decrypt a single file

```bash
# Encrypt
python3 crypto_tool.py encrypt-file data/level1.json -o data/level1.json.enc --password test123

# Decrypt
python3 crypto_tool.py decrypt-file data/level1.json.enc -o data/level1.json --password test123
```

Omit `--password` from any command to be prompted interactively instead. Wrong passwords or corrupted files are rejected with a clear error message.



