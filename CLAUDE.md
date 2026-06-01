# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MRagAgent — A multimodal RAG agent for fine-grained cultural heritage POI identification. The agent can identify specific cultural relics (statues, niches, murals) from user-uploaded photos and provide detailed Chinese-language explanations by retrieving from a structured knowledge base.

**Architecture**: Image-to-text conversion layer → SuperMew-style text RAG pipeline. Two tools work in serial: `identify_from_image` (CLIP image search → text) then `search_knowledge_base` (BGE-M3 + BM25 hybrid text retrieval).

## Project Structure

```
├── agent.py              # Agent definition: LLM config, tools, system prompt, LangSmith init
├── main.py               # CLI REPL loop with startup initialization
├── docker-compose.yml    # PostgreSQL 15, Redis 7, Milvus 2.5, etcd, MinIO, Attu, Adminer
├── .env                  # All configuration (DB, Milvus, models, LLM, LangSmith)
│
├── backend/
│   ├── app.py                  # FastAPI app factory (CORS, no-cache, static files)
│   ├── agent_api.py            # Sync + Async (SSE streaming) agent wrappers
│   ├── api.py                  # Routes: /chat, /chat/stream, /documents/*, /images/upload
│   ├── auth.py                 # Simple Bearer-token admin auth (no user management)
│   ├── database.py             # PostgreSQL engine + SessionLocal factory
│   ├── models.py               # ORM: ParentChunk (L1/L2 parent chunks)
│   ├── schemas.py              # Pydantic: POIMetadata, ImageSearchResult, etc.
│   ├── cache.py                # Redis JSON cache wrapper (key prefix: mragagent)
│   ├── embedding.py            # ChineseCLIP + BGE-M3 + BM25 (three model singletons)
│   ├── milvus_client.py        # Dual Milvus collections with auto-reconnect
│   ├── milvus_writer.py        # Batch write to both collections (embed + insert)
│   ├── parent_chunk_store.py   # PostgreSQL + Redis L1/L2 parent chunk CRUD
│   ├── document_loader.py      # PDF/Word → image extraction → three-level chunking
│   ├── image_retriever.py      # CLIP image search → formatted POI text description
│   ├── rag_utils.py            # Hybrid retrieval (dense+sparse RRF) + auto-merge
│   ├── rag_pipeline.py         # LangGraph RAG state machine (retrieve→merge→verify→generate)
│   └── tools.py                # @tool identify_from_image + search_knowledge_base
│
├── frontend/
│   ├── index.html              # Vue.js SPA with marked.js + highlight.js
│   ├── script.js               # SSE streaming chat, image/doc upload, admin panel
│   └── style.css               # Responsive layout, chat bubbles, streaming animation
│
├── data/
│   ├── reference_images/       # Extracted reference images from ingested documents
│   ├── documents/              # Uploaded PDFs/Word documents
│   ├── user_uploads/           # Chat image uploads (UUID-named)
│   └── bm25_state.json         # Persisted BM25 vocabulary + doc frequency
│
└── agent_memory.db             # LangGraph SQLite checkpointer (CLI sync mode only)
```

## Architecture

### Two-Collection Milvus Design

| Collection | Vector Field | Model | Dims | Metric | Purpose |
|---|---|---|---|---|---|
| `image_poi_collection` | `image_vector` | Chinese-CLIP ViT-L/14 | 768d | IP | Image→POI identification |
| `text_chunk_collection` | `dense_embedding` + `sparse_embedding` | BGE-M3 + BM25 | 1024d + sparse | IP + IP | Hybrid text RAG |

### Docker Services (all under `mragagent` network)

| Service | Container | Port | Purpose |
|---|---|---|---|
| PostgreSQL 15 | `mragagent-postgres` | 5432 | Parent chunks (L1/L2), ORM |
| Redis 7 | `mragagent-redis` | 6379 | Parent chunk cache, BM25 state |
| Milvus 2.5 | `mragagent-milvus` | 19530, 9091 | Dual vector collections |
| etcd 3.5 | `mragagent-etcd` | — | Milvus metadata backend |
| MinIO | `mragagent-minio` | 9000 (API), 9001 (Console) | Milvus object storage |
| Attu | `mragagent-attu` | 8080 | Milvus admin UI |
| Adminer | `mragagent-adminer` | 8081 | PostgreSQL admin UI |

### Query Flow

```
User: "帮我看看这是哪" + photo.jpg
  → Agent decides: need image?
    → YES: identify_from_image(photo.jpg)
      → CLIP embed → Image Milvus (IP search) → POI text description
    → search_knowledge_base(POI text + user question)
      → BGE-M3 dense + BM25 sparse → Text Milvus → RRF fusion → auto-merge L3→L2→L1 → format context
    → LLM generates final answer
    → NO: search_knowledge_base(question) directly
```

### SSE Streaming Flow

```
Frontend POST /chat/stream {message} (Authorization: Bearer <token>)
  → agent_api.chat_stream() → build_agent_async() → agent.astream(stream_mode="messages")
  → yield SSE data: {"type":"content","text":"..."} chunks
  → Frontend Vue.js reads ReadableStream, parses SSE events, appends to bot message
```

### Three-Level Chunking (aligned with SuperMew)

- **L1** (~1200 chars): Coarse overview → PostgreSQL `parent_chunks`
- **L2** (~600 chars): Medium paragraph → PostgreSQL `parent_chunks`
- **L3** (~300 chars): Fine leaf, the retrieval unit → Milvus `text_chunk_collection`
- Chunk ID format: `{filename}::p{page}::l{level}::{index}`
- Retrieval on L3 only; auto-merge L3→L2→L1 when ≥threshold siblings share same parent

### Document Ingestion Pipeline

```
Upload PDF/Word → document_loader.py
  ├── Text: PyPDFLoader/Docx2txtLoader → RecursiveCharacterTextSplitter (L1/L2/L3)
  │   → L1/L2 → PostgreSQL (parent_chunks) + Redis cache
  │   → L3 → BGE-M3 dense + BM25 sparse → Milvus text_chunk_collection
  └── Images: PyMuPDF (PDF) / python-docx (Word) extraction
      → Caption detection (图1. xxx) or LLM naming fallback
      → Chinese-CLIP embed → Milvus image_poi_collection
      → Each image linked to nearest L3 text chunk via chunk_id
```

### Re-upload Dedup Logic

When a file is re-uploaded:
1. Query existing BM25 stats from Milvus → `bm25.increment_remove_documents()`
2. Delete from both Milvus collections by filename filter
3. Delete from PostgreSQL parent_chunks by filename
4. Write new file, re-ingest from scratch

## Commands

```bash
# Start infrastructure
docker compose up -d

# Install core dependencies
pip install sqlalchemy psycopg2-binary redis pymilvus sentence-transformers Pillow fastapi uvicorn python-multipart langchain-community langgraph langsmith aiosqlite

# Install Chinese-CLIP (separate, heavy torch dependency)
pip install cn-clip==1.6.0

# Install document processing
pip install pymupdf python-docx docx2txt

# Run the agent CLI
python main.py

# Run the web server (FastAPI + frontend)
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload

# Or run via Python
python -m backend.app

# Stop infrastructure
docker compose down
```

## Configuration

All config in `.env`:

### LLM
- `OPENAI_API_KEY` — DeepSeek API key
- `OPENAI_MODEL` — defaults to `deepseek-v4-flash`
- `OPENAI_TEMPERATURE` — defaults to `0.3`
- `BASE_URL` — defaults to `https://api.deepseek.com/v1`
- Note: All ChatOpenAI instantiations set `extra_body={"thinking": {"type": "disabled"}}` to suppress DeepSeek thinking tokens

### LangSmith Tracing (optional)
- `LANGSMITH_TRACING_V2` — set to `true` to enable auto-tracing
- `LANGSMITH_API_KEY` — LangSmith API key (format: `lsv2_pt_...`)
- `LANGSMITH_PROJECT` — project name in LangSmith (defaults to `mragagent`)
- All LLM calls, tool invocations, and agent steps are traced automatically via LangChain callback

### Embeddings
- `EMBEDDING_MODEL` — BGE-M3 model (`BAAI/bge-m3`)
- `EMBEDDING_DEVICE` — `cpu` or `cuda`
- `CLIP_MODEL` — Chinese-CLIP variant (`ViT-L-14`)
- `CLIP_DEVICE` — `cpu` or `cuda`
- `HF_HUB_OFFLINE` — set to `false` to allow HuggingFace Hub access; defaults to `true` (local cache only, avoids HF instability in Chinese network environments)

### Milvus
- `MILVUS_HOST` / `MILVUS_PORT` — defaults `127.0.0.1:19530`
- `MILVUS_IMAGE_COLLECTION` — `image_poi_collection`
- `MILVUS_TEXT_COLLECTION` — `text_chunk_collection`

### Database / Cache
- `DATABASE_URL` — `postgresql+psycopg2://postgres:postgres@127.0.0.1:5432/mragagent`
- `REDIS_URL` — `redis://127.0.0.1:6379/0`
- `REDIS_KEY_PREFIX` — defaults to `mragagent`
- `REDIS_CACHE_TTL_SECONDS` — defaults to `300`

### Retrieval
- `BM25_STATE_PATH` — `data/bm25_state.json`
- `LEAF_RETRIEVE_LEVEL` — `3` (only L3 chunks are retrieval targets)
- `AUTO_MERGE_ENABLED` — `true`
- `AUTO_MERGE_THRESHOLD` — min siblings to trigger merge (default `2`)

### Optional: Rerank (not yet implemented)
- `RERANK_MODEL` / `RERANK_BINDING_HOST` / `RERANK_API_KEY` — reserved for future reranker

### Auth
- `ADMIN_TOKEN` — Bearer token for API auth (defaults to `mragagent-admin-token-2026`)

## Key Dependencies

- `langchain` / `langchain-openai` — Agent framework + DeepSeek LLM (ChatOpenAI)
- `langgraph` — RAG state machine + SQLite checkpointing
- `langsmith` — Optional LLM tracing/observability
- `pymilvus` — Milvus vector database Python SDK
- `sentence-transformers` — BGE-M3 text embeddings
- `cn-clip` — Chinese-CLIP multimodal embeddings
- `sqlalchemy` + `psycopg2-binary` — PostgreSQL ORM
- `redis` — Redis caching client
- `fastapi` + `uvicorn` — Web server + SSE streaming
- `aiosqlite` — Required for LangGraph async checkpointer
- `pymupdf` (fitz) — PDF image extraction
- `python-docx` — Word image extraction
- `docx2txt` — Word text extraction (Docx2txtLoader)

## Design Notes

### Agent Modes
- **Sync (CLI)**: `build_agent()` → `SqliteSaver` checkpointer for conversation memory via `agent_memory.db`
- **Async (Web SSE)**: `build_agent_async()` → no checkpointer (SuperMew handles memory via PostgreSQL); supports `agent.astream(stream_mode="messages")` for token-level streaming
- Both modes share the same tools and system prompt

### Module-Level Singletons
All embedding models, BM25, Milvus client, parent chunk store, and Milvus writer use module-level singletons — aligned with SuperMew pattern. This ensures BM25 vocabulary consistency across writes and reads, and avoids reloading models on each request.

### BGE-M3 Offline Mode
`BGEM3Embeddings` defaults to `local_files_only=True` (controlled by `HF_HUB_OFFLINE` env var). This is important for Chinese network environments where HuggingFace Hub connections are unreliable. Requires models to be pre-downloaded to the local HF cache (`~/.cache/huggingface/hub/`).

### Chinese-CLIP Model Weights
Chinese-CLIP downloads model weights to `./models/` on first use. The `models/` directory is git-ignored. These are also mounted for the `cn_clip` package cache under `.venv/lib/.../cn_clip/clip/model_configs/`.
