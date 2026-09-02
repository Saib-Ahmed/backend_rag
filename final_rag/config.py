"""
final_rag/config.py
Configuration settings for the final_rag pipeline (MSME & Legal Document Intelligence).
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# ── Storage Root on EBS / Local ──────────────────────────────────────────────
STORAGE_ROOT = Path(os.getenv("STORAGE_ROOT", "/data" if os.path.exists("/data") else str(ROOT_DIR)))

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")

# ── Models ────────────────────────────────────────────────────────────────────
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
CLAUDE_MODEL    = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")
CLEANER_MODEL   = os.getenv("CLEANER_MODEL", "claude-sonnet-5")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "claude-sonnet-5")

# ── Grounding Verification ────────────────────────────────────────────────────
GEMINI_GROUNDING_MODEL = os.getenv("GEMINI_GROUNDING_MODEL", "gemini-2.5-flash")
NVIDIA_GROUNDING_MODEL = os.getenv("NVIDIA_GROUNDING_MODEL", "meta/llama-3.3-70b-instruct")

# ── File handling ─────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md"}
MAX_FILE_SIZE_MB     = 200.0

DOC_INPUT_DIR   = Path(os.getenv("DOC_INPUT_DIR", str(STORAGE_ROOT / "1_doc_input" if os.path.exists("/data") else BASE_DIR / "doc_input")))
MD_OUTPUT_DIR   = Path(os.getenv("MD_OUTPUT_DIR", str(STORAGE_ROOT / "2_md_output" if os.path.exists("/data") else BASE_DIR / "md_output")))
PDF_STORAGE_DIR = Path(os.getenv("PDF_STORAGE_DIR", str(STORAGE_ROOT / "0_pdf_storage" if os.path.exists("/data") else BASE_DIR / "pdf_storage")))

# ── Document language & metadata detection ───────────────────────────────────
DOC_LANG_HINDI_CHAR_RATIO_THRESHOLD = 0.15
DOC_LANG_MIXED_THRESHOLD            = 0.6
DOC_LANG_SAMPLE_CHARS               = 5000

# ── Parser (Gemini multimodal OCR) ───────────────────────────────────────────
GEMINI_MAX_OUTPUT_TOKENS     = 65536
GEMINI_TEMPERATURE           = 0.1
PAGES_PER_BATCH              = 7
MAX_ATTEMPTS_PER_BATCH       = 6
MAX_SPLIT_DEPTH              = 4
UPLOAD_POLL_INTERVAL_SEC     = 10
UPLOAD_POLL_MAX_ATTEMPTS     = 60
GEMINI_RETRY_SLEEP_BUSY_SEC  = 30
GEMINI_RETRY_SLEEP_QUOTA_SEC = 60
PARSE_FOLDER_MAX_WORKERS     = 3

# ── Chunker & GLiNER NER ──────────────────────────────────────────────────────
CHUNK_SIZE                          = 650
CHUNK_OVERLAP                       = 100
MIN_CHARS                           = 50
MIN_WORDS                           = 10
TABLE_MAX_TOKENS                    = 650
DEDUP_THRESHOLD                     = 0.85
DEDUP_SHINGLE_SIZE                  = 6
HINDI_TOKEN_SCALE_FACTOR            = 3.5
MIN_CHARS_FACTUAL                   = 20
MAX_BOND_CHARS                      = 300
NOISE_RATIO_THRESHOLD               = 0.7
DEDUP_LONG_TEXT_THRESHOLD_CHARS     = 500
DEDUP_SHINGLE_SIZE_LONG             = 14

GLINER_MODEL_NAME         = "urchade/gliner_medium-v2.5"
GLINER_THRESHOLD          = 0.35
GLINER_RETRY_COOLDOWN_SEC = 300

GLINER_LABELS = [
    "court",
    "petitioner",
    "respondent",
    "judge",
    "lawyer",
    "act",
    "section",
    "case number",
    "jurisdiction",
    "designation",
    "date",
    "law",
    "order number",
    "monetary value",
    "percentage",
]

# ── Ollama & Embedder ─────────────────────────────────────────────────────────
OLLAMA_BASE_URL  = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embeddings"

EMBED_MODEL      = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")
EMBED_BATCH_SIZE = 8
EMBED_MAX_TOKENS = 3000
EMBED_NATIVE_DIM = 2560
EMBED_DIMENSIONS = int(os.getenv("EMBED_DIMENSIONS", "1536"))
SPARSE_MODEL_NAME = "Qdrant/bm25"

EMBED_PREFIXES = {
    "qwen3-embedding:4b": {"query": "query: ", "passage": "passage: "},
    "default":            {"query": "query: ", "passage": "passage: "},
}

EMBED_HTTP_TIMEOUT_SEC        = 80.0
EMBED_DIM_VERIFY_TIMEOUTS_SEC = [30.0, 60.0, 90.0]

# ── Qdrant Vector DB ──────────────────────────────────────────────────────────
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "rag_documents")
QDRANT_STORAGE_PATH    = Path(os.getenv("QDRANT_STORAGE_PATH", str(STORAGE_ROOT / "4_qdrant_db" if os.path.exists("/data") else BASE_DIR / "qdrant_db")))

# ── Query Cleaner ─────────────────────────────────────────────────────────────
CLEANER_TEMPERATURE       = 0.0
SUBQUERY_WEIGHT_THRESHOLD = 0.55

# ── Generator ─────────────────────────────────────────────────────────────────
GENERATOR_TEMPERATURE = 0.25

# ── Retriever ─────────────────────────────────────────────────────────────────
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
RERANKER_INSTRUCTION = (
    "Retrieve passages that are relevant to the given query and "
    "contain useful information to answer it."
)

TOP_K_SEARCH          = 30
PER_QUERY_TOP_K       = 20
MIN_RESULTS_THRESHOLD = 3
CONFIDENCE_THRESHOLD  = 0.22

DENSE_WEIGHT  = 0.85
SPARSE_WEIGHT = 0.15
DENSE_OVERSAMPLE_MULTIPLIER  = 5
SPARSE_OVERSAMPLE_MULTIPLIER = 2

# Broad search diversity caps
MAX_DOCS_BROAD     = 7
MAX_CHUNKS_PER_DOC = 8

# Boost and penalty matrix
TABLE_BOOST              = 0.03
YEAR_BOOST               = 0.04
HEADING_BOOST            = 0.04
KEYWORD_BOOST            = 0.02
FILENAME_TOKEN_BOOST     = 0.04
ENTITY_BOOST             = 0.04
KEYWORD_MISMATCH_PENALTY = -0.01
HIT_COUNT_BOOST          = 0.01

# ── Assembler ─────────────────────────────────────────────────────────────────
MAX_CONTEXT_TOKENS = 45000
TOKENS_PER_WORD    = 1.3

# ── Reranker Hardware & Pooling ───────────────────────────────────────────────
RERANKER_DEVICE     = "cuda" if os.getenv("FORCE_CUDA", "true").lower() == "true" else None
RERANKER_BATCH_SIZE = 16
MAX_SUBQUERIES      = 6
RERANK_POOL_CAP     = 180

# ── Claude Client ─────────────────────────────────────────────────────────────
ANTHROPIC_VERSION             = "2023-06-01"
CLAUDE_HTTP_TIMEOUT_SEC       = 120
CLAUDE_GENERATE_MAX_TOKENS    = 3000
CLAUDE_CHAT_STREAM_MAX_TOKENS = 9000

# ── Orchestrator & Multi-Turn History ─────────────────────────────────────────
MAX_HISTORY_TURNS        = 3
CHITCHAT_MAX_TOKENS      = 512
MAX_HISTORY_ANSWER_CHARS = 350
MAX_HISTORY_TOTAL_TOKENS = 1500

# ── Feature Flags ─────────────────────────────────────────────────────────────
ENABLE_PARALLEL_PARSING    = True
ENABLE_CHECKPOINT_RECOVERY = True
MAX_PARSE_WORKERS          = 4
