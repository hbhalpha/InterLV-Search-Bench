#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from agentic_search.local_kb import LocalKBIndex, Qwen3VLEmbedderWrapper, _load_json_or_jsonl


def main():
    parser = argparse.ArgumentParser(description="Build local knowledge base index for agentic search.")
    parser.add_argument("--input", required=True, help="Input KB file (JSON or JSONL). Each row must have {id, image_path, text}.")
    parser.add_argument("--embed-model", required=True, help="Qwen3-VL embedding model path or name.")
    parser.add_argument("--outdir", required=True, help="Output index directory.")
    parser.add_argument("--batch-size", type=int, default=8, help="Embedding batch size.")
    args = parser.parse_args()

    rows = _load_json_or_jsonl(Path(args.input))
    embedder = Qwen3VLEmbedderWrapper(args.embed_model)
    index = LocalKBIndex.from_json_records(rows=rows, embedder=embedder, batch_size=args.batch_size)
    index.save(args.outdir)
    print(f"Saved local KB index ({len(rows)} records) to: {args.outdir}")


if __name__ == "__main__":
    main()
