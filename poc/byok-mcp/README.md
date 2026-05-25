# BYOK via MCP — Proof of Concept

Demonstrates that OLS can consume customer knowledge through external MCP servers
without any custom ingestion pipeline. The customer brings their own knowledge
(documents in a git repo), uses an open-source tool to index it, and OLS queries
it via MCP.

## What This Proves

| Claim | How |
|-------|-----|
| No custom ingestion pipeline needed | Standard open-source tools (Qdrant + MCP server) |
| No document format conversion | Ingestion handles markdown/text directly from repo |
| MCP is the interface | OLS connects via existing `mcp_servers` config — zero OLS code changes |
| Same tooling works elsewhere | The Qdrant MCP server works with Claude, Cursor, Windsurf, etc. |

## Architecture

```
Git Repo (demo docs)
       │
       │ clone
       ▼
  ingest.py              ← chunks, embeds with FastEmbed (all-MiniLM-L6-v2)
       │
       │ vectors + metadata
       ▼
  Qdrant (Docker)         ← vector database on localhost:6333
       │
       │ MCP streamable-http
       ▼
  mcp-server-qdrant       ← official Qdrant MCP server on localhost:8080
       │
       │ MCP streamable-http
       ▼
  OLS                     ← existing MCPServerConfig, zero code changes
```

## Demo Content

Uses [counteractive/incident-response-plan-template](https://github.com/counteractive/incident-response-plan-template) —
a set of security incident response runbooks and playbooks (~220KB of markdown).

**Demo questions for OLS:**
- "What should we do during a ransomware attack?"
- "Who is the incident commander and what are their duties?"
- "What is our phishing response playbook?"
- "How do we handle a website defacement?"
- "What are the steps after an incident is resolved?"

Without BYOK, OLS cannot answer these (not OpenShift knowledge).
With BYOK via MCP, OLS answers from the ingested runbooks.

## Prerequisites

- Python 3.12+
- Docker
- OLS running locally (or able to run)

## Quick Start

```bash
# One command does everything: clone, start Qdrant, ingest, start MCP server
./setup.sh
```

The script will:
1. Clone the demo content repo
2. Start Qdrant in Docker
3. Install Python dependencies (qdrant-client, fastembed, mcp-server-qdrant)
4. Parse and ingest all markdown files into Qdrant
5. Start the MCP server on `http://localhost:8080/mcp`

Then add to your `olsconfig.yaml`:

```yaml
ols_config:
  mcp_servers:
    - name: byok-incident-response
      url: http://localhost:8080/mcp
```

Restart OLS and ask one of the demo questions.

## Manual Steps (if not using setup.sh)

### 1. Start Qdrant

```bash
docker run -d --name byok-qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone and ingest demo content

```bash
git clone --depth 1 https://github.com/counteractive/incident-response-plan-template.git demo-docs

python ingest.py --docs-path ./demo-docs --collection byok-demo
```

### 4. Start MCP server

```bash
QDRANT_URL=http://localhost:6333 \
COLLECTION_NAME=byok-demo \
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
FASTMCP_PORT=8080 \
mcp-server-qdrant --transport streamable-http
```

### 5. Configure OLS

Add to `olsconfig.yaml`:

```yaml
ols_config:
  mcp_servers:
    - name: byok-incident-response
      url: http://localhost:8080/mcp
```

## How It Works

### Ingestion (ingest.py)

- Reads all `.md` and `.txt` files from the demo repo
- Splits into ~500-character chunks with 50-character overlap
- Embeds using FastEmbed (`all-MiniLM-L6-v2` — same model the MCP server uses for queries)
- Stores vectors + metadata (source file, title) in Qdrant

### MCP Server (mcp-server-qdrant)

- Official Qdrant MCP server (Apache 2.0, 1,300+ GitHub stars)
- Exposes two MCP tools:
  - `qdrant-find`: semantic search over the collection
  - `qdrant-store`: store new information (not used in this demo)
- Uses `streamable-http` transport (compatible with OLS's MCP client)

### OLS Integration

- OLS discovers the MCP tools via standard MCP protocol
- The LLM decides when to call `qdrant-find` based on the query
- Retrieved chunks are used as context for generating the response

## Scaling to Real Customer Use

For a real deployment, replace `ingest.py` with one of these tools:

| Tool | What It Does | License |
|------|-------------|---------|
| [qdrant-loader](https://github.com/martin-papy/qdrant-loader) | Ingests from Git, Confluence, JIRA, local files. 20+ formats. Comes with its own MCP server. | GPLv3 |
| [unstructured-ingest](https://docs.unstructured.io/open-source/ingestion/ingest-cli) | 130+ document formats. Industrial-grade parsing. Push to Qdrant. | Apache 2.0 |
| [MDDB](https://github.com/tradik/mddb) | All-in-one: embedded DB + MCP server + ingestion. Single binary. | BSD-3 |

The MCP server stays the same — only the ingestion step changes.

## Cleanup

```bash
docker stop byok-qdrant && docker rm byok-qdrant
rm -rf demo-docs
```
