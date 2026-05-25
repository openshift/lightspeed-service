#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEMO_REPO="https://github.com/counteractive/incident-response-plan-template.git"
DEMO_DIR="$SCRIPT_DIR/demo-docs"
VENV_DIR="$SCRIPT_DIR/.venv"
COLLECTION="byok-demo"
QDRANT_URL="http://localhost:6333"
MCP_PORT="${MCP_PORT:-8000}"

echo "=== BYOK via MCP — POC Setup ==="
echo ""

# Step 1: Clone demo content
if [ -d "$DEMO_DIR" ]; then
    echo "[1/6] Demo docs already cloned at $DEMO_DIR"
else
    echo "[1/6] Cloning demo content (incident response runbooks)..."
    git clone --depth 1 "$DEMO_REPO" "$DEMO_DIR"
fi

# Step 2: Start Qdrant
if docker ps --format '{{.Names}}' | grep -q '^byok-qdrant$'; then
    echo "[2/6] Qdrant already running"
else
    echo "[2/6] Starting Qdrant..."
    docker run -d \
        --name byok-qdrant \
        -p 6333:6333 -p 6334:6334 \
        qdrant/qdrant
    echo "  Waiting for Qdrant to be ready..."
    sleep 3
    until curl -sf "$QDRANT_URL/healthz" > /dev/null 2>&1; do
        sleep 1
    done
    echo "  Qdrant ready at $QDRANT_URL"
fi

# Step 3: Create venv and install dependencies
if [ -d "$VENV_DIR" ]; then
    echo "[3/6] Virtual environment already exists"
else
    echo "[3/6] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[4/6] Installing Python dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Step 5: Ingest documents
echo "[5/6] Ingesting documents into Qdrant..."
python "$SCRIPT_DIR/ingest.py" \
    --docs-path "$DEMO_DIR" \
    --collection "$COLLECTION" \
    --qdrant-url "$QDRANT_URL"

# Step 6: Start MCP server
echo "[6/6] Starting Qdrant MCP server on port $MCP_PORT..."
echo ""
echo "============================================"
echo "  MCP server starting at:"
echo "  http://localhost:$MCP_PORT/mcp"
echo ""
echo "  Add to your olsconfig.yaml:"
echo ""
echo "  ols_config:"
echo "    mcp_servers:"
echo "      - name: byok-incident-response"
echo "        url: http://localhost:$MCP_PORT/mcp"
echo ""
echo "  Then ask OLS:"
echo "    - What should we do during a ransomware attack?"
echo "    - Who is the incident commander and what are their duties?"
echo "    - What is our phishing response playbook?"
echo "============================================"
echo ""

QDRANT_URL="$QDRANT_URL" \
COLLECTION_NAME="$COLLECTION" \
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2" \
mcp-server-qdrant --transport streamable-http --port "$MCP_PORT"
