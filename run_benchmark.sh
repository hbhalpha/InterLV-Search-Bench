#!/usr/bin/env bash
set -euo pipefail

#############################################
# Agentic Search Benchmark Runner
#
# BEFORE RUNNING:
#   1. Replace ALL placeholder values below with your actual credentials.
#   2. Ensure you have installed the package: pip install -e .
#   3. Install playwright browsers: playwright install chromium
#############################################

#############################################
# Config: edit these before running
#############################################

# ---- Paths ----
# Replace with the path to your dataset JSON file
DATASET_JSON="./your_dataset.json"
RUN_SCRIPT="./run_benchmark_multi.py"
JUDGE_SCRIPT="./agentic_search/gpt_judge.py"
OUTDIR="./outputs"

# ---- Agent model settings ----
TEST_MODEL="gpt-4o"                  # model under test
BACKEND="openai"                      # openai / local / gemini
SEARCH_DEPTH="10"                     # maps to max agent iterations
NUM_WORKERS=8                         # number of parallel workers

# ---- Judge model settings ----
JUDGE_MODEL="gpt-4o"

# ---- Search API (SerpAPI - get your key at https://serpapi.com) ----
export SERP_API_KEY="your-serpapi-key-here"

# ---- OpenAI API (or any OpenAI-compatible endpoint) ----
export OPENAI_API_KEY="your-openai-api-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"  # or your custom endpoint

# ---- Google Gemini API (get your key at https://aistudio.google.com) ----
export GOOGLE_API_KEY="your-google-api-key-here"

# ---- Image Upload (Imgur - get client ID at https://api.imgur.com) ----
export IMGUR_CLIENT_ID="your-imgur-client-id-here"

# ---- Judge API key (if different from the model API key) ----
# Uncomment if using a separate key for judging.
# export JUDGE_API_KEY="your-judge-api-key-here"
# export JUDGE_BASE_URL="https://api.openai.com/v1"

#############################################
# Validation checks
#############################################

echo "[1/4] Checking files..."
[[ -f "$DATASET_JSON" ]] || { echo "Dataset not found: $DATASET_JSON"; exit 1; }
[[ -f "$RUN_SCRIPT" ]] || { echo "Run script not found: $RUN_SCRIPT"; exit 1; }
[[ -f "$JUDGE_SCRIPT" ]] || { echo "Judge script not found: $JUDGE_SCRIPT"; exit 1; }

mkdir -p "$OUTDIR"

# Normalize model name for filenames
SAFE_MODEL_NAME=$(echo "$TEST_MODEL" | sed 's#[/: ]#_#g')
PRED_FILE="$OUTDIR/${SAFE_MODEL_NAME}_ans.jsonl"
RESULT_FILE="$OUTDIR/${SAFE_MODEL_NAME}_result.jsonl"
JUDGE_FILE="$OUTDIR/${SAFE_MODEL_NAME}_ans_judge.jsonl"

#############################################
# Run benchmark / agent search
#############################################

echo "[2/4] Running benchmark"
echo "  DATASET_JSON = $DATASET_JSON"
echo "  TEST_MODEL   = $TEST_MODEL"
echo "  BACKEND      = $BACKEND"
echo "  SEARCH_DEPTH = $SEARCH_DEPTH"
echo "  NUM_WORKERS  = $NUM_WORKERS"
echo "  OUTDIR       = $OUTDIR"

python "$RUN_SCRIPT" \
  --input "$DATASET_JSON" \
  --model "$TEST_MODEL" \
  --backend "$BACKEND" \
  --max-iters "$SEARCH_DEPTH" \
  --outdir "$OUTDIR" \
  --num-workers "$NUM_WORKERS"

#############################################
# Run GPT yes/no validation
#############################################

echo "[3/4] Running GPT yes/no validation"

# Use separate key for judging if provided
if [[ -n "${JUDGE_API_KEY:-}" && "$JUDGE_API_KEY" != "your-judge-api-key-here" ]]; then
  export OPENAI_API_KEY="$JUDGE_API_KEY"
fi
if [[ -n "${JUDGE_BASE_URL:-}" && "$JUDGE_BASE_URL" != "https://api.openai.com/v1" ]]; then
  export OPENAI_BASE_URL="$JUDGE_BASE_URL"
fi

python "$JUDGE_SCRIPT" \
  --dataset "$DATASET_JSON" \
  --pred "$PRED_FILE" \
  --judge-model "$JUDGE_MODEL"

#############################################
# Simple accuracy summary
#############################################

echo "[4/4] Computing summary"
python - <<PY
import json
from pathlib import Path

judge_path = Path(r"$JUDGE_FILE")
if not judge_path.exists():
    print(f"Judge file not found: {judge_path}")
    raise SystemExit(1)

rows = []
with judge_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

total = len(rows)
yes_cnt = sum(1 for r in rows if str(r.get("judge", "")).strip().upper() == "YES")
no_cnt = sum(1 for r in rows if str(r.get("judge", "")).strip().upper() == "NO")
acc = yes_cnt / total if total else 0.0

print("----- Summary -----")
print(f"Prediction file : $PRED_FILE")
print(f"Result file     : $RESULT_FILE")
print(f"Judge file      : $JUDGE_FILE")
print(f"Total           : {total}")
print(f"YES             : {yes_cnt}")
print(f"NO              : {no_cnt}")
print(f"Accuracy        : {acc:.4%}")
PY

echo "Done."
