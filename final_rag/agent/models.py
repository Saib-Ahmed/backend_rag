"""
agent/models.py
Pydantic v2 Models — Single source of truth for all data structures.
Used by: retriever, assembler, orchestrator, database, and FastAPI.
"""

from __future__ import annotations
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────
# Query Cleaner
# ─────────────────────────────────────────
class Subquery(BaseModel):
    query:  str
    weight: float = 0.0


class ComparisonArm(BaseModel):
    label:           str               # e.g. "MP Policy 2022"
    year:            Optional[str]     = None
    filename_tokens: list[str]         = Field(default_factory=list)
    query:           Optional[str]     = None
    filter_hints:    dict[str, Any]    = Field(default_factory=dict)


class CleanedQuery(BaseModel):
    original_query:    str                 = ""
    improved_query:    str                 = ""
    detected_language: str                 = "english"
    intent:            str                 = "document_query"

    # ── 3 signals ────────────────────────────────────────────
    target_scope:      str                 = "broad"       # "single" / "few" / "broad"
    answer_structure:  str                 = "direct"      # "direct" / "compare" / "synthesize"
    specificity:       str                 = "low"         # "high" / "medium" / "low"

    # ── retrieval hints ──────────────────────────────────────
    filter_hints:      dict[str, Any]      = Field(default_factory=dict)
    comparison_arms:   list[ComparisonArm] = Field(default_factory=list)
    active_documents:  list[str]           = Field(default_factory=list)

    # ── subqueries ───────────────────────────────────────────
    subqueries:        list[Subquery]      = Field(default_factory=list)
    all_subqueries:    list[Subquery]      = Field(default_factory=list)

    # ── meta ─────────────────────────────────────────────────
    processing_time_sec: float             = 0.0
    warning:             Optional[str]     = None

    # ── compatibility fields/properties ──────────────────────
    active_doc_id:        Optional[str]     = None
    active_doc_ids:       list[str]         = Field(default_factory=list)
    extracted_keywords:   list[str]         = Field(default_factory=list)
    confidence:           float             = 1.0
    is_valid:             bool              = True

    @property
    def primary_query(self) -> str:
        return self.improved_query or self.original_query

    @property
    def scope(self) -> str:
        if len(self.active_documents) == 1:
            return "active"
        if len(self.active_documents) > 1:
            return "active_compare"
        return self.target_scope

    @property
    def is_chitchat(self) -> bool:
        return self.intent == "chitchat"

    @property
    def active_doc_filename(self) -> Optional[str]:
        return self.active_documents[0] if len(self.active_documents) == 1 else None

    @property
    def active_doc_filenames(self) -> list[str]:
        return self.active_documents if len(self.active_documents) > 1 else []

    @property
    def raw_subqueries(self) -> list[str]:
        return [sq.query if isinstance(sq, Subquery) else str(sq) for sq in self.subqueries]


# ─────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────
class RetrievedChunk(BaseModel):
    model_config = ConfigDict(frozen=False)

    text:                  str
    source_file:           str
    page_no:               int       = 0
    page_label:            str       = ""
    chunk_index:           int       = 0
    heading:               str       = ""
    is_table:              bool      = False
    is_table_with_context: bool      = False
    char_count:            int       = 0
    token_count:           int       = 0
    warnings:              list[str] = Field(default_factory=list)
    chunk_years:           list[str] = Field(default_factory=list)
    entities:              list[str] = Field(default_factory=list)

    # ── scores ───────────────────────────────────────────────
    qdrant_score:          float = 0.0
    rerank_score:          float = 0.0

    # ── flags ────────────────────────────────────────────────
    is_weak_match:         bool  = False

    # ── citation ─────────────────────────────────────────────
    source_tag:            str   = ""
    arm_label:             str   = ""   # which comparison arm this belongs to

    # ── metadata ─────────────────────────────────────────────
    doc_year:              str   = ""
    doc_lang:              str   = ""
    doc_id:                str   = ""


# ─────────────────────────────────────────
# Assembler
# ─────────────────────────────────────────
class SourceInfo(BaseModel):
    file_name:   str
    pages:       list[str]
    chunk_count: int

    def model_dump(self) -> dict:
        return {
            "file_name":   self.file_name,
            "pages":       self.pages,
            "chunk_count": self.chunk_count,
        }


class AssembledResult(BaseModel):
    context_block:       str
    sources:             list[SourceInfo]
    chunks_used:         int
    has_weak_match:      bool
    has_tables:          bool
    not_found:           bool
    was_trimmed:         bool
    processing_time_sec: float
    answer_structure:    str  = "direct"   # passed through to generator
    sources_count:       int  = 0


# ─────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────
class OrchestratorResult(BaseModel):
    answer:               str
    sources:              list[SourceInfo]  = Field(default_factory=list)
    clarification_needed: bool              = False
    was_critic_triggered: bool              = False
    critic_score:         Optional[int]     = None
    critic_reason:        Optional[str]     = None
    cleaned_query:        Optional[str]     = None
    pipeline_times:       dict              = Field(default_factory=dict)
    total_time_sec:       float             = 0.0
    error:                Optional[str]     = None


# ─────────────────────────────────────────
# Pipeline Timing
# ─────────────────────────────────────────
class PipelineTiming(BaseModel):
    cleaner_sec:   float = 0.0
    retriever_sec: float = 0.0
    assembler_sec: float = 0.0
    generator_sec: float = 0.0
    critic_sec:    float = 0.0
    total_sec:     float = 0.0


# ─────────────────────────────────────────
# API Response
# ─────────────────────────────────────────
class ChatResponse(BaseModel):
    session_id:  str
    query:       str
    answer:      str
    sources:     list[SourceInfo]
    timing:      PipelineTiming
    was_trimmed: bool