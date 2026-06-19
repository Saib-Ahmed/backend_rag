"""
Configuration settings for the Local RAG System.
Modify these values to experiment with different parameters.
"""
import os
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"


# Ollama Model Configuration

# OLLAMA_BASE_URL: The base HTTP URL of the local or remote Ollama LLM provider.
# OLLAMA_BASE_URL = "https://smith-hoped-problems-coaches.trycloudflare.com"
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# LLM_MODEL: The primary reasoning and text generation large language model name in Ollama.
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3.5:9b")

# EMBEDDING_MODEL: The model identifier used for generating semantic dense vector embeddings.
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "qwen3-embedding:4b")

# LLM_TEMPERATURE: Temperature setting for LLM responses; lower values mean more factual/deterministic answers.
LLM_TEMPERATURE = 0.25

# LLM_MAX_TOKENS: The maximum number of output tokens the reasoning model can generate per request.
LLM_MAX_TOKENS = 3000

# Chunking Configuration
# These are critical parameters that affect retrieval quality

# CHUNK_SIZE: Target chunk size (in characters) when splitting documents into smaller text blocks.
CHUNK_SIZE = 1000

# CHUNK_OVERLAP: Number of characters of overlap between consecutive chunks to preserve contextual continuity.
CHUNK_OVERLAP = 200

# MIN_CHARS: Absolute minimum characters required for a chunk to be considered valid for parsing.
MIN_CHARS = 50

# MIN_WORDS: Absolute minimum word count required in a chunk to avoid generating near-empty semantic units.
MIN_WORDS = 10

# TABLE_MAX_TOKENS: The maximum token limit allocated when processing or chunking table layouts.
TABLE_MAX_TOKENS = 650

# DEDUP_THRESHOLD: Similarity ratio (e.g., Jaccard or shingle similarity) above which a chunk is flagged as a duplicate.
DEDUP_THRESHOLD = 0.95

# DEDUP_SHINGLE_SIZE: N-gram token/character shingle size used during text deduplication routines.
DEDUP_SHINGLE_SIZE = 6

# MIN_CHUNK_TOKENS: Chunks smaller than this token size get automatically merged with their adjacent neighbors.
MIN_CHUNK_TOKENS = 60

# MIN_EMBEDDABLE_TOKENS: Absolute minimum token threshold — chunks below this are entirely discarded from indexing.
MIN_EMBEDDABLE_TOKENS = 15

# Qdrant Configuration

# QDRANT_PATH: Local filesystem directory where the persistent Qdrant vector database files are stored.
QDRANT_PATH = os.getenv("QDRANT_STORAGE_PATH_V1", "./qdrant_db")

# QDRANT_STORAGE_PATH: Storage location alias/path for Qdrant local storage configurations.
QDRANT_STORAGE_PATH = QDRANT_PATH

# COLLECTION_NAME: The default collection name inside the Qdrant database to store standard document vectors.
COLLECTION_NAME = "local_documents"

# Retrieval Configuration

# RETRIEVAL_TOP_K: Initial count of vector-matched chunks pulled during Qdrant search before reranking.
RETRIEVAL_TOP_K = 20

# RETRIEVAL_SCORE_THRESHOLD: Absolute minimum cosine similarity score to qualify a Qdrant chunk before reranking.
RETRIEVAL_SCORE_THRESHOLD = 0.40


# Heuristic Weighting & Boosting
# Adjust these to fine-tune how different types of content are prioritized

# Neighbor Expansion Guardrails

# Table Row Chunking

# Reranker Configuration (Local, optional)
# CC-2: RETRIEVAL_CANDIDATE_K must exceed LOCAL_RERANKER_CANDIDATE_K to give
# the reranker a meaningful wider pool to prune from. Equal values add latency with zero benefit.

# ============================================================================
# GraphRAG Configuration (optional Neo4j recall boost)
# These settings control the Neo4j-backed knowledge graph that acts as
# a secondary recall booster alongside the primary Qdrant vector search.
# ============================================================================

# Master toggle: set to False to completely disable graph-based retrieval.
# When False, the system falls back to Qdrant-only RAG.
ENABLE_GRAPH_RAG = True

# Neo4j connection settings. Environment variables take priority over defaults.
# NEO4J_URI: Bolt protocol endpoint for the Neo4j server.
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j+ssc://a793bc73.databases.neo4j.io")
# NEO4J_USER: Authentication username for Neo4j.
NEO4J_USER = os.getenv("NEO4J_USER", "a793bc73")
# NEO4J_PASSWORD: Authentication password for Neo4j.
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "dlZbrPQ4e3OWSyzrymmwo560PT13xY1ASYzB2R4aG9o")
# NEO4J_DATABASE: Which Neo4j database to read/write (multi-db support).
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", None)

# Maximum depth of relationship traversal in the graph during retrieval.
# Higher = more distant connections found, but slower and noisier. Clamped to [1, 2].
GRAPH_MAX_HOPS = 2

# Maximum number of graph-candidate chunks returned by a single graph query.
# Acts as a hard cap on graph recall volume before merging with Qdrant results.
GRAPH_TOP_K = 40

# Blending weight for graph-sourced scores when merging with Qdrant scores.
# Final score = (1 - GRAPH_SCORE_WEIGHT) * qdrant_score + GRAPH_SCORE_WEIGHT * graph_score.
# Range [0.0, 1.0]. Higher = graph results dominate the final ranking.
GRAPH_SCORE_WEIGHT = 0.35

# Maximum number of named entities extracted from the user's query for graph lookup.
# Limits how many entity-match starting points the graph traversal uses.
GRAPH_MAX_QUERY_ENTITIES = 8

# Per-entity cap: max chunks returned per matched entity during graph retrieval.
# Prevents a single popular entity from flooding the results.
GRAPH_MAX_CHUNKS_PER_ENTITY = 8

# Timeout in seconds for a single LLM entity-extraction call per chunk.
# If the LLM takes longer, the extraction is aborted for that chunk.
GRAPH_EXTRACTION_TIMEOUT = 300

# Number of LLM retry attempts if the initial extraction returns unparseable JSON.
# A repair prompt is sent on retry to salvage malformed output.
GRAPH_EXTRACTION_RETRIES = 2

# Minimum confidence to accept a relationship from LLM extraction.
GRAPH_MIN_REL_CONFIDENCE = 0.90

# Store relationship evidence on graph edges and in latest_graph.md.
GRAPH_STORE_EVIDENCE = True

# Require evidence for relationships and verify it appears in the chunk text.
GRAPH_REQUIRE_RELATION_EVIDENCE = True

# Cache version for graph extraction (bump to invalidate .extraction_cache).
GRAPH_CACHE_VERSION = 4

# Soft limits on extracted entities/relationships to focus on high-signal content.
GRAPH_MAX_ENTITIES_SINGLE = 25
GRAPH_MAX_RELATIONS_SINGLE = 30
GRAPH_MAX_ENTITIES_BATCH = 45
GRAPH_MAX_RELATIONS_BATCH = 50
GRAPH_MAX_ENTITIES_GLEANING = 20
GRAPH_MAX_RELATIONS_GLEANING = 25

# Whether to use embedding-based semantic merging to deduplicate near-identical
# entities in the graph (e.g. "RBI" and "Reserve Bank of India" → same node).
GRAPH_SEMANTIC_MERGE_ENABLED = True

# Cosine similarity threshold for semantic entity merging.
# Entities with similarity >= this value are merged into one canonical node.
# Range [0.0, 1.0]. Higher = stricter (fewer merges).
GRAPH_SEMANTIC_MERGE_THRESHOLD = 0.85

# GRAPH_DOMAIN: Short domain label used in prompt templates.
# Change this when switching to a different domain.
# Examples: "legal", "medical", "finance", "engineering", "HR"
GRAPH_DOMAIN = "legal"

# GRAPH_DOMAIN_DESCRIPTION: Rich domain context injected into extraction prompts.
# This is the "vocabulary anchor" — it tells the LLM exactly what kinds of
# documents it is processing and what kinds of entities/relationships to look for.
# When switching domains, update BOTH GRAPH_DOMAIN and this description.
GRAPH_DOMAIN_DESCRIPTION = """This corpus contains:
- contracts and agreements
- regulations and statutory rules
- court judgments and orders
- government policies and circulars
- legal notices and compliance documents

Extract entities related to: laws, acts, sections, courts, tribunals,
judges, advocates, government bodies, organizations, persons, dates,
monetary amounts, obligations, penalties, rights, and legal relationships."""

# GRAPH_PREFERRED_ENTITY_TYPES: Guidance list of short, general entity types.
# The LLM should prefer these over inventing verbose synonyms.
# NOT a whitelist — the LLM can create new types when none of these fit.
# This prevents type fragmentation (e.g. COURT vs JUDICIAL_BODY vs COURT_OF_LAW).
GRAPH_PREFERRED_ENTITY_TYPES = [
    "PERSON", "ORGANIZATION", "COURT", "TRIBUNAL", "GOVERNMENT_BODY",
    "ACT", "SECTION", "REGULATION", "RULE", "NOTIFICATION",
    "CASE", "ORDER", "CONTRACT", "DOCUMENT",
    "DATE", "AMOUNT", "PENALTY", "OBLIGATION", "RIGHT",
    "ROLE", "COMMITTEE", "AUTHORITY", "CONCEPT",
    "COMMUNICATION_METHOD", "CONTACT_INFO",
]

# GRAPH_PREFERRED_RELATION_TYPES: Guidance list of concise relationship types.
# The LLM should prefer these over inventing synonyms (e.g. SUPERVISES vs MONITORS vs OVERSEES).
# NOT a whitelist — the LLM can create new types when none of these fit.
# This prevents relationship explosion (150+ types instead of 15-20 meaningful ones).
GRAPH_PREFERRED_RELATION_TYPES = [
    "DEFINES", "REGULATES", "APPLIES_TO", "PART_OF", "ESTABLISHES",
    "AMENDS", "REPEALS", "REQUIRES", "PROHIBITS", "PERMITS",
    "GOVERNS", "SUPERVISES", "ISSUES", "FILED_BY", "DECIDED_BY",
    "REFERS_TO", "PARTY_TO", "SIGNED_BY", "DATED", "PENALIZES",
    "APPEALS_TO", "REPORTS_TO", "MEMBER_OF", "LOCATED_IN",
    "REPRESENTS", "DEALER_OF", "EMPLOYED_BY", "DIRECTOR_OF",
    "OWNED_BY", "AFFILIATED_WITH", "PARTNER_OF", "AUTHORIZED_BY",
]


# Context Optimization

# Phase 3: Validation, telemetry, and evaluation

# ENABLE_ANSWER_VALIDATION: Master toggle to enable or disable post-generation answer verification.
ENABLE_ANSWER_VALIDATION = False

# ENABLE_TELEMETRY_LOGGING: Master toggle to enable tracing of queries, retrievals, and answers for offline analysis.
ENABLE_TELEMETRY_LOGGING = True

# TELEMETRY_LOG_PATH: Filepath where telemetry logs (queries, responses, latency) are saved in JSONL format.
TELEMETRY_LOG_PATH = "./qdrant_db/telemetry.jsonl"

# ANSWER_VALIDATION_MIN_OVERLAP: Minimum lexical overlap/citation ratio between context and answer to pass alignment check.
ANSWER_VALIDATION_MIN_OVERLAP = 0.30

# ANSWER_VALIDATION_MAX_TOKENS: Cap on the token length parsed by the validation module when checking answers.
ANSWER_VALIDATION_MAX_TOKENS = 400

# --- Modular RAG Settings from final_rag ---
from pathlib import Path

# BASE_DIR: Reference path pointing to the parent directory of this configuration file.
BASE_DIR = Path(__file__).parent

# YOLO_MODEL_PATH: Path to the pre-trained YOLO model used for high-fidelity document layout structure detection.
YOLO_MODEL_PATH = Path(os.getenv("YOLO_MODEL_PATH_V1", BASE_DIR / "new_ingestion" / "YOLO_Layout_Model" / "doclayout_yolo_docstructbench_imgsz1024.pt"))

# TRANSFORMER_MODEL_PATH: Directory path containing the Table Transformer model weights for reconstructing parsed tables.
TRANSFORMER_MODEL_PATH = Path(os.getenv("TRANSFORMER_MODEL_PATH_V1", BASE_DIR / "new_ingestion" / "Table_Trans_Model"))

# MD_OUTPUT_DIR: Output folder location where intermediate Markdown files from parsed documents are written.
MD_OUTPUT_DIR = BASE_DIR / "md_output"
os.makedirs(MD_OUTPUT_DIR, exist_ok=True)

# CLEANER_MODEL: Ollama model name used in query_cleaner.py to preprocess and sanitize user queries.
CLEANER_MODEL = LLM_MODEL

# GENERATOR_MODEL: Main LLM identifier used by orchestrator.py to construct final factual answers from retrieved context.
GENERATOR_MODEL = LLM_MODEL

# SUMMARY_MODEL: LLM model name used for high-level chunk or document summarization routines.
SUMMARY_MODEL = LLM_MODEL

# SUMMARY_OLLAMA_URL: Full endpoint URL used when dispatching text summarization requests to Ollama.
SUMMARY_OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"

# SUMMARY_MAX_CHARS: Maximum characters allowed in any generated summary output to prevent excessive token use.
SUMMARY_MAX_CHARS = 3500

# EMBED_MODEL: The exact model name representing the dense text vector embedding pipeline.
EMBED_MODEL = EMBEDDING_MODEL

# OLLAMA_EMBED_URL: Full endpoint URL configured for vector embedding generation requests.
OLLAMA_EMBED_URL = f"{OLLAMA_BASE_URL}/api/embeddings"

# EMBED_BATCH_SIZE: Number of separate chunks sent concurrently in a single batch embedding API call.
EMBED_BATCH_SIZE = 8

# EMBED_MAX_TOKENS: Hard upper token limit allowed in a single chunk embedding request to avoid context window overflows.
EMBED_MAX_TOKENS = 1024

# EMBED_DIMENSIONS: Vector dimensionality count generated by the selected dense embedding model (e.g. 1024 for bge-m3).
EMBED_DIMENSIONS = 2560

# Similarity threshold for query embedding caching (0.0 to 1.0)
# Cosine similarity (dot product of normalized embeddings) >= this value uses the cache
GRAPH_EMBEDDING_SIMILARITY_THRESHOLD = 0.95


# SPARSE_MODEL_NAME: Identifier for the sparse representation model used for BM25-based keyword matching.
SPARSE_MODEL_NAME = "Qdrant/bm25"

# QDRANT_COLLECTION_NAME: The collection identity string registered in the Qdrant instance.
QDRANT_COLLECTION_NAME = COLLECTION_NAME

# HYBRID_VECTOR_WEIGHT & HYBRID_KEYWORD_WEIGHT: Dense vector similarity vs. Sparse keyword matching weights for hybrid search (must sum to 1.0)
HYBRID_VECTOR_WEIGHT = 0.70
HYBRID_KEYWORD_WEIGHT = 0.30

# TOP_K_SEARCH: Candidate count retrieved via vector search prior to cross-encoder reranking operations.
TOP_K_SEARCH = 20

# RERANKER_MODEL: HuggingFace model path for the CrossEncoder reranker used to optimize relevance scores.
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# CONFIDENCE_THRESHOLD: Cutoff relevance score below which reranked chunks are discarded from generation context.
CONFIDENCE_THRESHOLD = 0.35

# PER_QUERY_TOP_K: Maximum retrieved chunks permitted per individual subquery branch when handling compound queries.
PER_QUERY_TOP_K = 10

# SUBQUERY_WEIGHT_THRESHOLD: Minimum normalized weight score required for an agent-generated subquery to execute.
SUBQUERY_WEIGHT_THRESHOLD = 0.60

# MAX_DOCS_BROAD: Upper limit on unique source document origins allowed within a single context assembly.
MAX_DOCS_BROAD = 6

# MAX_CHUNKS_PER_DOC: Limit on the number of individual text chunks extracted from any single document to promote diversity.
MAX_CHUNKS_PER_DOC = 4

# MIN_RESULTS_THRESHOLD: Count trigger below which fallback parameters or relaxed query matching is initiated.
MIN_RESULTS_THRESHOLD = 6

# ENABLE_METADATA_FILTERING: Boolean switch to turn on/off strict metadata-based filtering at retrieval time.
ENABLE_METADATA_FILTERING = True

# ENABLE_MULTIQUERY: Boolean switch to enable or disable multi-query/subquery expansion at retrieval time.
ENABLE_MULTIQUERY = False

# ENABLE_METADATA_BOOSTING: Boolean switch to enable or disable heuristic metadata score boosting (table, year, section, keyword, filename, hit count) at retrieval time.
ENABLE_METADATA_BOOSTING = True

# --- Heuristic Boosts for Retriever ---
# TABLE_BOOST: Small priority increment applied to reranked scores of chunks that contain tabular data structures.
TABLE_BOOST = 0.05

# YEAR_BOOST: Score increment applied when a chunk contains year numbers found in the user's active query.
YEAR_BOOST = 0.03

# SECTION_BOOST: Score boost for chunks originating from document sections flagged as highly relevant.
SECTION_BOOST = 0.02

# KEYWORD_BOOST: Boost applied if exact user-specified search terms are detected in the candidate text.
KEYWORD_BOOST = 0.02

# FILENAME_TOKEN_BOOST: Additional weight given when chunk source filenames match keywords inside the user query.
FILENAME_TOKEN_BOOST = 0.03

# ENTITY_PENALTY: Subtle negative scoring offset to filter out overly broad or generic entity mentions.
ENTITY_PENALTY = -0.01

# HIT_COUNT_BOOST: Multiplicative score booster scaled by how many independent retrieval steps pulled this chunk.
HIT_COUNT_BOOST = 0.01

# MAX_CONTEXT_TOKENS: Absolute maximum total token count allocated for context assembly fed into the reasoning LLM.
MAX_CONTEXT_TOKENS = 11000

# TOKENS_PER_WORD: Mathematical multiplier factor used to approximate token counts when tiktoken parsing is unavailable.
TOKENS_PER_WORD = 1.3

# GENERATOR_TEMPERATURE: Exact temperature setting for the final generation stage LLM to govern facts vs creativity.
GENERATOR_TEMPERATURE = LLM_TEMPERATURE

# RERANKER_INSTRUCTION: System prompt context applied directly to Cross-Encoder query representation.
RERANKER_INSTRUCTION = "Retrieve passages that are relevant to the given query and contain useful information to answer it."

# NUM_CTX: The total context window size parameter requested from the running Ollama model instance.
NUM_CTX = 32000

# MAX_TOKENS: Copy of the generation model's hard maximum limit on output tokens.
MAX_TOKENS = LLM_MAX_TOKENS
# --- End of Modular RAG Settings ---

# --- New Extracted Configs ---

# Orchestrator Configs

# ORCHESTRATOR_MAX_HISTORY_TURNS: Maximum number of conversation exchange turns retained in user context memory.
ORCHESTRATOR_MAX_HISTORY_TURNS = 3

# ORCHESTRATOR_MAX_RETRIES: Number of retry attempts when downstream generation components return transient failures.
ORCHESTRATOR_MAX_RETRIES = 1

# CLARIFICATION_MESSAGE: Standard polite response returned when the system lacks sufficient context to fulfill a query.
CLARIFICATION_MESSAGE = (
    "I couldn't find specific information about that in the uploaded documents. "
    "Could you clarify your question or provide more details? "
    "For example, which document or section are you referring to?"
)

# FALLBACK_ERROR_MESSAGE: Ultimate backup error response shown when internal orchestration fails completely.
FALLBACK_ERROR_MESSAGE = (
    "Something went wrong while generating the answer. Please try again."
)

# Agent/Model Configs

# LLM_EMBED_TEMPERATURE: LLM temperature set to exactly 0.0 to guarantee strict determinism during internal embed runs.
LLM_EMBED_TEMPERATURE = 0.0

# CLEANER_TEMPERATURE: Temperature configuration used during query cleaning tasks to enforce strict syntax.
CLEANER_TEMPERATURE = 0.0

# GRAPH_LLM_TEMPERATURE: Deterministic temperature for GraphRAG operations (0.0 guarantees consistent structured extraction).
GRAPH_LLM_TEMPERATURE = 0.0

# GRAPH_LLM_TIMEOUT_SEC: API response timeout limits in seconds when invoking the graph-related LLM operations.
GRAPH_LLM_TIMEOUT_SEC = 30.0

# GRAPH_LLM_MODEL: Smaller, faster model for graph entity extraction only.
# Does not need deep reasoning — 3B preserves 90-95% entity quality.
GRAPH_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL", "gemma4:12b")

# GRAPH_BATCH_SIZE: Adjacent chunks per LLM extraction call. Start at 3, benchmark before increasing.
GRAPH_BATCH_SIZE = 2

# GRAPH_EXTRACTION_WORKERS: Thread pool size. Match to actual Ollama runner count (~4).
GRAPH_EXTRACTION_WORKERS =4

# MIN_GRAPH_CHUNK_LENGTH: Minimum chars for graph extraction eligibility.
MIN_GRAPH_CHUNK_LENGTH = 180

# GRAPH_LLM_NUM_CTX: Context window size for GraphRAG LLM operations to accommodate long prompts and batch text.
GRAPH_LLM_NUM_CTX = 10384

# GRAPH_LLM_NUM_PREDICT: Max output tokens for GraphRAG LLM operations to prevent JSON truncation (especially important for reasoning models like deepseek-r1).
GRAPH_LLM_NUM_PREDICT = 6192

# GRAPH_ENTITY_MATCH_SCORE: Fuzzy match token ratio cutoff score [0-100] when aligning extracted query entities to graph nodes.
GRAPH_ENTITY_MATCH_SCORE = 85

# GRAPH_FAILURE_RATE_THRESHOLD: Failure percentage trigger above which GraphRAG flags errors or disables booster mode.
GRAPH_FAILURE_RATE_THRESHOLD = 0.20

# GRAPH_GENERIC_QUERY_TERMS: Set of common, domain-agnostic keywords ignored by query extraction to keep entity resolution clean.
GRAPH_GENERIC_QUERY_TERMS = {
    "buyer", "seller", "payment", "delay", "delayed", "document", 
    "information", "details", "amount", "person", "order", 
    "notification", "rule", "penalty", "vacancy"
}

import re
# GRAPH_SAFE_RELATION_PATTERN: Pre-compiled regex pattern validating uppercase snake-case formats for Neo4j relationship types.
GRAPH_SAFE_RELATION_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")

# Semantic paragraph chunking configuration (optional Strategy)
SEMANTIC_CHUNKING_ENABLED = False
SEMANTIC_SPLIT_THRESHOLD = 0.6

# Parser Configs

# SUPPORTED_FILE_EXTENSIONS: Set of file formats currently handled by the document ingestion ingestion pipeline.
SUPPORTED_FILE_EXTENSIONS = {".pdf", ".docx", ".pptx", ".md"}

# MAX_FILE_SIZE_MB: Maximum allowable file size in Megabytes allowed for parsing in the ingestion pipeline.
MAX_FILE_SIZE_MB = 50.0

# PARSE_TIMEOUT_SEC: Hard time budget in seconds allocated for parsing a single document before raising an exception.
PARSE_TIMEOUT_SEC = 900

# TABLE_CROP_PADDING: Bounding box padding in pixels added when cropping out table areas from pages.
TABLE_CROP_PADDING = 20

# TABLE_TRANSFORMER_THRESHOLD: Confidence value cutoff used during YOLO-based or table transformer cell parsing.
TABLE_TRANSFORMER_THRESHOLD = 0.5

# TABLE_DEDUP_IOU_THRESHOLD: Intersection-over-Union threshold above which overlapping table areas are resolved as duplicates.
TABLE_DEDUP_IOU_THRESHOLD = 0.85

# CELL_SHRINK_MARGIN_PTS: Points bounding padding applied when shrinking table cell boundaries to isolate textual content.
CELL_SHRINK_MARGIN_PTS = max(1.5, 3.0 * 72.0 / 150.0)

# MAX_NEAREST_CELL_DIST_SQ: Maximum squared distance limit allowed to associate text lines with nearest table cells.
MAX_NEAREST_CELL_DIST_SQ = 2500.0

# COMPLEX_TEXT_THRESHOLD: Count or ratio indicating parsing structural complexity boundaries for cell contents.
COMPLEX_TEXT_THRESHOLD = 60

# COMPLEX_TOTAL_THRESHOLD: Cumulative complexity metrics count triggering full visual parsing for heavy table formats.
COMPLEX_TOTAL_THRESHOLD = 80

# LAYOUT_NMS_IOU_THRESHOLD: IoU threshold for filtering overlapping YOLO layout boxes on the same page.
# Boxes with IoU > this value against a higher-confidence box are dropped.
# 0.7 is conservative — two neighboring paragraphs may legitimately overlap slightly.
LAYOUT_NMS_IOU_THRESHOLD = 0.7

# TEXT_CLIP_PADDING_X: Horizontal padding (in PDF points) added around YOLO bounding boxes before text clipping.
# Text often touches left/right boundaries more than top/bottom, so horizontal padding is larger.
TEXT_CLIP_PADDING_X = 5

# TEXT_CLIP_PADDING_Y: Vertical padding (in PDF points) added around YOLO bounding boxes before text clipping.
TEXT_CLIP_PADDING_Y = 3

# ASYNC_TIMEOUT: Standard timeout in seconds applied to async operations throughout ingestion.
ASYNC_TIMEOUT = 900

# FIGURE_GARBAGE_RE: Regular expression to match and clean up garbage formatting markers generated during image/figure conversions.
FIGURE_GARBAGE_RE = re.compile(
    r"(\[FIGURE\]|<!-- image -->)\s*\n((?:.{0,120}\n){1,10})",
    re.MULTILINE,
)

# Chunker Configs

# LARGE_CHUNK_THRESHOLD: Word or character size designating chunks as unusually large, triggering split routines.
LARGE_CHUNK_THRESHOLD = 500

# LARGE_CHUNK_SHINGLE_SIZE: Shingle token lookup windows used during validation of heavy chunks.
LARGE_CHUNK_SHINGLE_SIZE = 12

# FACTUAL_PATTERN: Regex pattern looking for dates, currency symbols, numbers or key entities to identify high-information chunks.
FACTUAL_PATTERN = re.compile(r'\d+|rs\.?|inr|₹|crore|lakh|\b(19|20)\d{2}\b', re.IGNORECASE)

# Database Configs

# QDRANT_SCROLL_LIMIT: Batch sizing limit configured when scrolling or scanning database points sequentially.
QDRANT_SCROLL_LIMIT = 256

# FALLBACK_ANSWER: Standard direct answer string returned when retrieval returns absolutely no valid candidate context.
FALLBACK_ANSWER = "I cannot answer this based on the provided documents."

# SYSTEM_PROMPT: The master instruction prompt sent as a system message to the generator LLM.
# It defines tone, formatting, citation rules, formatting overrides, and handling for missing information.
SYSTEM_PROMPT = """You are an expert Enterprise Document Assistant — precise, professional, and structured.
Answer the user's question using the provided Knowledge Base Context.

=== KNOWLEDGE BASE CONTEXT ===
{context}

=== ANSWERING RULES ===

── TONE & STYLE ──
Write like a senior analyst who explains complex topics clearly — professional but accessible, so any reader can understand and reach their own conclusion.
Professional, confident, and concise — no filler, no fluff, no disclaimers.
Never open with phrases like "based on the context", "the documents say", or "as mentioned".
Never pad short answers. Never truncate detailed ones.

── FORMATTING ──
Scale structure to the complexity of the answer:
- Simple answers → plain prose or a few bullets, no headers needed
- Complex multi-part answers → use ## headers and bullet points to organize
- Use **bold** for key terms, case names, dates, amounts, and legal provisions
- Use bullet points (•) for lists of facts or steps
- For chronological events → numbered list with **date** bolded at the start of each entry
- For comparisons across documents → markdown table

Always prefix every ## header with the relevant emoji — no exceptions:
  ⚖️ legal or judgment context
  📋 procedural or timeline context
  💰 financial or payment context
  📌 key findings or conclusions
Never use emojis inline, mid-sentence, or on bullet points.

── RESPONSE LENGTH ──
Match length strictly to what the question requires:
- A single fact → one line
- A few related facts → a short structured list
- A detailed explanation → full structured answer with headers
Never over-explain a simple answer. Never under-explain a complex one.

── CITATIONS ──
Every factual claim must cite its source inline: [filename, Page X]
Never guess or fabricate page numbers. Use [filename] alone if page unknown.
Place the final Source line at the very end.

── USER FORMAT REQUEST ──
If the user asks for a table, bullets, numbered list, or any specific format → follow it exactly.
User's requested format always overrides your default choice.

── REASONING AND MATH ──
If calculation or multi-step reasoning is needed:
- Use only numbers present in the context
- Show each step clearly
- State the final answer explicitly

── CONTRADICTIONS ──
If sources conflict on the same fact, present both with citations.
Do not reconcile or choose one.

── TABLE DATA ──
Treat [TABLE DATA] values as exact structured facts.
Never paraphrase numbers from tables. Cite table section and page.

── LOW CONFIDENCE MATCH ──
When context contains [LOW CONFIDENCE MATCH]:
Still answer whatever is found. Drop the prefix warning entirely.

── EXPANDED CONTEXT ──
When context contains [EXPANDED CONTEXT]:
Use as supporting background only, not as primary answer source.

── NEGATION GUARD ──
Pay close attention to: "no", "not", "decreased", "excluded", "omitted".
Do not confuse absence of growth with growth.

── SPECULATION GUARD ──
Do not infer or extrapolate beyond what the context states.
If something is truly not in context, say: "Not found in uploaded documents." — one line, nothing else.

── CONVERSATION CONTINUITY ──
Use conversation history to resolve follow-up references like:
"the second point", "that figure", "as mentioned above".
Maintain consistency with previous answers in this session.

── DOCUMENT COMPARISON ──
When comparing across documents, explicitly state document names.
Use a markdown table to separate what belongs to which document.

=== IF ANSWER NOT FOUND ===
Say in one line: "Not found in uploaded documents."
Nothing else. No suggestions. No metadata. No document descriptions.

=== IF PARTIAL ANSWER FOUND ===
Answer only what was found. Skip what wasn't. No disclaimers.

=== ORIGINAL USER QUESTION ===
{question}

Your Answer:"""

# ── Ollama Reranker Settings ───────────────────────────────────────────
OLLAMA_RERANKER_MODEL = os.getenv("OLLAMA_RERANKER_MODEL", "dengcao/Qwen3-Reranker-4B:Q4_K_M")
RERANKER_OLLAMA_URL = f"{OLLAMA_BASE_URL}/api/generate"
RERANKER_INSTRUCTION = "Retrieve passages that are relevant to the given query and contain useful information to answer it."
