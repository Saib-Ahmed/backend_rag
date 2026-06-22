import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────── 
BASE_DIR       = Path(__file__).parent
DOC_INPUT_DIR  = BASE_DIR / "doc_input"
MD_OUTPUT_DIR  = Path(os.getenv("MD_OUTPUT_DIR", BASE_DIR / "md_output"))

os.makedirs(DOC_INPUT_DIR, exist_ok=True)
os.makedirs(MD_OUTPUT_DIR, exist_ok=True) 

# local: "http://localhost:11434"

# ── Ollama ─────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL",        "http://localhost:11434")
CLEANER_MODEL   = os.getenv("CLEANER_MODEL",     "qwen3.5:9b")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL",   "qwen3.5:9b")

# ── Summary LLM (ingestion only) ───────────────────────────────────────
SUMMARY_MODEL      = os.getenv("SUMMARY_MODEL", "qwen3.5:9b") 
SUMMARY_OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
SUMMARY_MAX_CHARS  = 3500 

# ── Chunker ──────────────────────────────────────────────────────────── 
CHUNK_SIZE         = 650
CHUNK_OVERLAP      = 100
MIN_CHARS          = 50
MIN_WORDS          = 10
TABLE_MAX_TOKENS   = 650
DEDUP_THRESHOLD    = 0.85
DEDUP_SHINGLE_SIZE = 6
MIN_CHUNK_TOKENS       = 60       # Chunks smaller than this get merged with neighbors
MIN_EMBEDDABLE_TOKENS  = 15       # Absolute minimum — chunks below this are dropped entirely
LARGE_CHUNK_SHINGLE_SIZE = 12     # Shingle size for chunks > LARGE_CHUNK_THRESHOLD
LARGE_CHUNK_THRESHOLD    = 500    # Character threshold for large chunk dedup

# ── Embedding ──────────────────────────────────────────────────────────
EMBED_MODEL        = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
OLLAMA_EMBED_URL   = f"{OLLAMA_BASE_URL}/api/embeddings"

EMBED_BATCH_SIZE   = 8 
EMBED_MAX_TOKENS   = 1024
EMBED_DIMENSIONS   = 2560
SPARSE_MODEL_NAME  = "Qdrant/bm25"

# ── Qdrant ─────────────────────────────────────────────────────────────
QDRANT_COLLECTION_NAME = "rag_documents"
QDRANT_STORAGE_PATH    = Path(os.getenv("QDRANT_STORAGE_PATH", BASE_DIR / "qdrant_db"))

# ── Retriever ──────────────────────────────────────────────────────────
TOP_K_SEARCH              = 13 
RERANKER_MODEL            = "Qwen/Qwen3-Reranker-0.6B"

# ── Retriever thresholds ───────────────────────────────────────────────
CONFIDENCE_THRESHOLD      = 0.25 
PER_QUERY_TOP_K           = 8
CHUNK_EXPAND_WINDOW       = 0
SUBQUERY_WEIGHT_THRESHOLD = 0.60

# ── Broad search diversity caps ────────────────────────────────────────
MAX_DOCS_BROAD            = 6
MAX_CHUNKS_PER_DOC        = 2
MIN_RESULTS_THRESHOLD     = 3

# ── Score boosts (applied after rerank) ───────────────────────────────
TABLE_BOOST               = 0.03
YEAR_BOOST                = 0.08 
SECTION_BOOST             = 0.02
KEYWORD_BOOST             = 0.04  
FILENAME_TOKEN_BOOST      = 0.08 
ENTITY_PENALTY            = -0.04
HIT_COUNT_BOOST           = 0.01

# ── Assembler ──────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS        = 11000  
TOKENS_PER_WORD           = 1.3

# ── LLM Generation Settings ────────────────────────────────────────────
CLEANER_TEMPERATURE       = 0.0    # deterministic signal extraction
GENERATOR_TEMPERATURE     = 0.25   # slightly creative for natural prose

# ── Reranker ───────────────────────────────────────────────────────────
RERANKER_INSTRUCTION      = "Retrieve passages that are relevant to the given query and contain useful information to answer it."

# ── Feature Flags ──────────────────────────────────────────────────────
ENABLE_PARALLEL_PARSING    = True
ENABLE_CHECKPOINT_RECOVERY = True
MAX_PARSE_WORKERS          = 4

# ── Ollama model Settings ──────────────────────────────────────────────
NUM_CTX           = 18000
MAX_TOKENS        = 3000 

# ── Layout Model Paths ────────────────────────────────────────────────
YOLO_MODEL_PATH        = Path(os.getenv("YOLO_MODEL_PATH", BASE_DIR / "ingestion" / "YOLO_Layout_Model" / "doclayout_yolo_docstructbench_imgsz1024.pt"))
TRANSFORMER_MODEL_PATH = Path(os.getenv("TRANSFORMER_MODEL_PATH", BASE_DIR / "ingestion" / "Table_Trans_Model"))

