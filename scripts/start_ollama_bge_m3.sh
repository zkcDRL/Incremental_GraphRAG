#!/usr/bin/env bash
# Start the local Ollama service and verify the bge-m3 embedding endpoint.
set -euo pipefail

HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
MODEL="${OLLAMA_EMBEDDING_MODEL:-bge-m3}"
LOG_FILE="${OLLAMA_LOG_FILE:-$HOME/.ollama/ollama-serve.log}"
BASE_URL="http://${HOST}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "Error: ollama is not installed or not in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

if curl -fsS --max-time 2 "$BASE_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at $BASE_URL"
else
  echo "Starting Ollama at $BASE_URL"
  OLLAMA_HOST="$HOST" nohup ollama serve >>"$LOG_FILE" 2>&1 &
  server_pid=$!

  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 "$BASE_URL/api/tags" >/dev/null 2>&1; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "Error: Ollama exited during startup. See $LOG_FILE" >&2
      exit 1
    fi
    sleep 1
  done

  if ! curl -fsS --max-time 2 "$BASE_URL/api/tags" >/dev/null 2>&1; then
    echo "Error: Ollama did not become ready. See $LOG_FILE" >&2
    exit 1
  fi
fi

if ! ollama list | awk 'NR > 1 {print $1}' | grep -Eq "^${MODEL}(:latest)?$"; then
  echo "Model $MODEL is missing; downloading it now."
  ollama pull "$MODEL"
fi

response_file=$(mktemp)
trap 'rm -f "$response_file"' EXIT
curl -fsS --max-time 120 \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"input\":\"GraphRAG local embedding health check\"}" \
  "$BASE_URL/api/embed" >"$response_file"

vector_size=$(python3 - "$response_file" <<'PY'
import json
import sys
from pathlib import Path

response = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
embeddings = response.get("embeddings") or []
print(len(embeddings[0]) if embeddings and embeddings[0] else 0)
PY
)

if [[ "$vector_size" -le 0 ]]; then
  echo "Error: $MODEL returned no embedding vector." >&2
  cat "$response_file" >&2
  exit 1
fi

echo "Ollama embedding service is ready."
echo "  URL: $BASE_URL"
echo "  Model: $MODEL"
echo "  Vector size: $vector_size"
echo "  Log: $LOG_FILE"
