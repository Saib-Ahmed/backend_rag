"""Optional Neo4j-backed GraphRAG retrieval boost.

This module is deliberately defensive: Neo4j is optional, LLM extraction output is
untrusted, and graph retrieval must never break the normal Qdrant RAG path.
"""

from __future__ import annotations

import json
import logging
import time
import math
import re
import os
import hashlib
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import config

from langchain_core.documents import Document

logger = logging.getLogger("graph_rag")

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - exercised by optional-dependency fallback
    GraphDatabase = None


LLMInvoke = Callable[[str, float, int], str]
EmbeddingFn = Callable[[str], List[float]]
BatchEmbeddingFn = Callable[[List[str]], List[List[float]]]


@dataclass
class GraphEntity:
    name: str
    type: str
    name_norm: str
    entity_key: str
    description: str = ""
    embedding: List[float] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    effective_date: Optional[str] = None
    citation_text: Optional[str] = None


@dataclass
class GraphRelation:
    source: str
    relation: str
    target: str
    source_key: str
    target_key: str
    confidence: float = 1.0
    evidence: str = ""
    description: str = ""
    keywords: List[str] = field(default_factory=list)


@dataclass
class GraphExtraction:
    entities: List[GraphEntity]
    relationships: List[GraphRelation]
    repaired: bool = False
    llm_time_sec: float = 0.0
    is_cached: bool = False


@dataclass
class GraphCandidate:
    source: str
    chunk_id: str
    graph_score: float
    graph_hops: int
    graph_path: str = ""

_GRAPH_ENTITY_SIGNAL_PATTERNS = [
    re.compile(r'\b[A-Z]{2,}\b'),                     # MSMED, RBI, CPSU
    re.compile(r'\bSection\s+\d+', re.IGNORECASE),     # Section 21
    re.compile(r'\bRule\s+\d+', re.IGNORECASE),         # Rule 8
    re.compile(r'\bAct,?\s*\d{4}\b', re.IGNORECASE),   # Act, 2006
    re.compile(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'),     # Government Body
]

class GraphRAGManager:
    """Small optional GraphRAG manager used as a recall booster."""

    SAFE_RELATION_PATTERN = config.GRAPH_SAFE_RELATION_PATTERN

    def __init__(
        self,
        *,
        enabled: bool,
        uri: str,
        user: str,
        password: str,
        database: str,
        domain: str = "legal",
        domain_description: str = "",
        preferred_entity_types: Optional[List[str]] = None,
        preferred_relation_types: Optional[List[str]] = None,
        max_hops: int = 2,
        max_query_entities: int = 8,
        max_graph_chunks: int = 40,
        max_chunks_per_entity: int = 8,
        semantic_merge_enabled: bool = True,
        semantic_merge_threshold: float = 0.92,
        extraction_timeout: float = 30.0,
        extraction_retries: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.domain = str(domain).strip() or "general"
        self.domain_description = str(domain_description).strip()
        self.preferred_entity_types = preferred_entity_types or []
        self.preferred_relation_types = preferred_relation_types or []
        self.max_hops = max(1, min(2, int(max_hops)))
        self.max_query_entities = max(1, int(max_query_entities))
        self.max_graph_chunks = max(1, int(max_graph_chunks))
        self.max_chunks_per_entity = max(1, int(max_chunks_per_entity))
        self.semantic_merge_enabled = bool(semantic_merge_enabled)
        self.semantic_merge_threshold = float(semantic_merge_threshold)
        self.extraction_timeout = float(extraction_timeout)
        self.extraction_retries = max(0, int(extraction_retries))
        self.driver = None
        self.last_error = ""
        self.extraction_failures = 0
        self.extraction_attempts = 0
        self.generic_query_terms = config.GRAPH_GENERIC_QUERY_TERMS
        self.canonical_entities = {}
        # Dynamic alias map built from LLM extractions at ingestion time
        self._dynamic_aliases: Dict[str, str] = {}
        self._lock = threading.RLock()

        if self.enabled:
            self.connect()

    @property
    def available(self) -> bool:
        return bool(self.enabled and self.driver is not None)

    def connect(self) -> bool:
        if not self.enabled:
            return False
        if GraphDatabase is None:
            self.last_error = "neo4j package is not installed"
            return False
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password),
            )
            self.driver.verify_connectivity()
            self.ensure_schema()
            return True
        except Exception as exc:
            self.driver = None
            self.last_error = str(exc)
            print(f"[GraphRAG] Neo4j unavailable; continuing without graph: {exc}")
            return False

    def close(self) -> None:
        if self.driver is not None:
            try:
                self.driver.close()
            except Exception:
                pass
        self.driver = None

    def status(self) -> dict:
        base = {
            "enabled": self.enabled,
            "available": self.available,
            "status": "disabled" if not self.enabled else ("connected" if self.available else "unavailable"),
            "last_error": self.last_error,
            "entities": 0,
            "relationships": 0,
        }
        if not self.available:
            return base
        try:
            with self.driver.session(database=self.database) as session:
                counts = session.run(
                    """
                    MATCH (e:Entity)
                    WITH count(e) AS entity_count
                    MATCH (:Entity)-[r]->(:Entity)
                    RETURN entity_count, count(r) AS relationship_count
                    """
                ).single()
                if counts:
                    base["entities"] = int(counts["entity_count"] or 0)
                    base["relationships"] = int(counts["relationship_count"] or 0)
        except Exception as exc:
            base["status"] = "unavailable"
            base["available"] = False
            base["last_error"] = str(exc)
        return base

    def delete_source(self, source: str, qdrant_client: Optional[QdrantClient] = None, batch_embedding_fn: Optional[BatchEmbeddingFn] = None) -> None:
        """Alias for delete_document to maintain compatibility with RAGEngine."""
        self.delete_document(source, qdrant_client, batch_embedding_fn)

    def delete_document(self, source: str, qdrant_client: Optional[QdrantClient] = None, batch_embedding_fn: Optional[BatchEmbeddingFn] = None) -> None:
        """Deletes a document, its chunks, and any orphaned entities from the graph."""
        if not self.available:
            return
        try:
            with self.driver.session(database=self.database) as session:
                # 1. Delete the Document node and all its Chunks
                session.run(
                    """
                    MATCH (d:Document {source: $source})
                    OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
                    DETACH DELETE c, d
                    """,
                    source=source
                )
                
                # 2. Delete any orphaned Entity nodes (entities not mentioned by any remaining chunks)
                session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (:Chunk)-[:MENTIONS]->(e)
                    DETACH DELETE e
                    """
                )
            
            # Sync the VDB if client and embeddings are provided
            if qdrant_client and batch_embedding_fn:
                self.sync_vdb(qdrant_client, batch_embedding_fn)
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[GraphRAG] Failed to delete document {source} from graph: {exc}")

    def clear_all_documents(self) -> None:
        """Completely wipes the graph database."""
        if not self.available:
            return
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception as exc:
            print(f"[GraphRAG] Failed to clear graph: {exc}")

    def ensure_schema(self) -> None:
        if self.driver is None:
            return
        statements = [
            "CREATE CONSTRAINT graph_document_file_hash IF NOT EXISTS FOR (d:Document) REQUIRE d.file_hash IS UNIQUE",
            "CREATE CONSTRAINT graph_chunk_key IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_key IS UNIQUE",
            "CREATE CONSTRAINT graph_entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_key IS UNIQUE",
            "CREATE INDEX graph_entity_norm IF NOT EXISTS FOR (e:Entity) ON (e.name_norm)",
            "CREATE INDEX graph_chunk_source_id IF NOT EXISTS FOR (c:Chunk) ON (c.source, c.chunk_id)",
        ]
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                session.run(statement)

    @classmethod
    def parse_json_output(cls, raw: str) -> Tuple[Optional[dict], str]:
        text = (raw or "").strip()
        if not text:
            return None, "empty"
        try:
            parsed = json.loads(text)
            return parsed, "direct"
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip()), "fenced"
            except Exception:
                pass

        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            try:
                return json.loads(array_match.group(0)), "array"
            except Exception:
                pass

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(text[first : last + 1]), "salvaged"
            except Exception:
                pass

        loose_block = re.search(r"\{[\s\S]*\}", text)
        if loose_block:
            try:
                return json.loads(loose_block.group(0)), "block"
            except Exception:
                pass
        return None, "invalid"

    @classmethod
    def _basic_normalize(cls, name: str) -> str:
        text = str(name or "").lower()
        text = re.sub(r"^(the|ld\.|hon\'ble|learned|mr\.|ms\.)\s+", "", text).strip()
        text = text.replace("&", " and ")
        text = re.sub(r"\bsec\.?\s+", "section ", text)
        text = re.sub(r"\bs\.\s*(\d+)\b", r"section \1", text)
        text = re.sub(r"[^a-z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def _normalize_evidence(cls, text: str) -> str:
        cleaned = str(text or "").lower()
        cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def _evidence_supported(cls, evidence: str, source_text: str) -> bool:
        if not evidence or not source_text:
            return False
        ev_norm = cls._normalize_evidence(evidence)
        src_norm = cls._normalize_evidence(source_text)
        if not ev_norm or not src_norm:
            return False
        if ev_norm in src_norm:
            return True
        ev_tokens = [t for t in ev_norm.split() if len(t) > 2]
        if len(ev_tokens) < 3:
            return False
        src_token_set = set(src_norm.split())
        hits = sum(1 for t in ev_tokens if t in src_token_set)
        return (hits / max(1, len(ev_tokens))) >= 0.7

    def normalize_entity_name(self, name: str) -> str:
        norm = self._basic_normalize(name)
        with self._lock:
            # Check dynamic aliases learned from LLM extraction
            return self._dynamic_aliases.get(norm, norm)

    def _resolve_entity(self, name_norm: str, entity_type: str) -> str:
        from difflib import SequenceMatcher
        type_cache = {
            k: v for k, v in self.canonical_entities.items()
            if v["type"] == entity_type
        }
        
        best_match = None
        best_score = 0
        
        for canonical, meta in type_cache.items():
            # Numerical mismatch guard: do not merge entities with different numbers
            nums_a = re.findall(r'\d+', name_norm)
            nums_b = re.findall(r'\d+', canonical)
            if nums_a != nums_b:
                continue

            score = SequenceMatcher(None, name_norm, canonical).ratio() * 100
            if score > best_score:
                best_score = score
                best_match = canonical
        
        if best_score > config.GRAPH_ENTITY_MATCH_SCORE:
            return best_match
            
        self.canonical_entities[name_norm] = {
            "type": entity_type,
            "variants": [name_norm]
        }
        return name_norm

    def entity_key(self, entity_type: str, name_norm: str, file_hash: str = "Unknown") -> str:
        return f"{str(entity_type).upper()}::{name_norm}"

    @staticmethod
    def _cosine(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
        norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
        if norm_a <= 0.0 or norm_b <= 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    def validate_extraction(
        self,
        payload: dict,
        *,
        embedding_fn: Optional[EmbeddingFn] = None,
        file_hash: str = "Unknown",
        source_text: str = "",
    ) -> GraphExtraction:
        raw_entities = payload.get("entities", []) if isinstance(payload, dict) else []
        raw_relationships = payload.get("relationships", []) if isinstance(payload, dict) else []
        entities: Dict[str, GraphEntity] = {}
        name_to_key: Dict[str, str] = {}

        if not isinstance(raw_entities, list):
            raw_entities = []
        for item in raw_entities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            entity_type = str(item.get("type", "")).strip().upper()
            description = str(item.get("description", "")).strip()
            # Only require non-empty name and a valid UPPER_SNAKE_CASE type
            if not name or not entity_type or not re.match(r'^[A-Z][A-Z0-9_]*$', entity_type):
                continue
            with self._lock:
                name_norm = self.normalize_entity_name(name)
                if not name_norm:
                    continue

                # Process LLM-detected aliases: register all aliases → canonical name
                llm_aliases = item.get("aliases", [])
                if isinstance(llm_aliases, list):
                    for alias in llm_aliases:
                        alias_norm = self._basic_normalize(str(alias))
                        if alias_norm and alias_norm != name_norm:
                            self._dynamic_aliases[alias_norm] = name_norm

                name_norm = self._resolve_entity(name_norm, entity_type)

            embedding: List[float] = []

            effective_date = item.get("effective_date") or item.get("date")
            effective_date = str(effective_date).strip() if effective_date else None

            citation_text = item.get("citation_text") or item.get("citation")
            citation_text = str(citation_text).strip() if citation_text else None

            aliases_list = [str(a).strip() for a in llm_aliases] if isinstance(llm_aliases, list) else []

            key = self.entity_key(entity_type, name_norm, file_hash)
            if key not in entities:
                entities[key] = GraphEntity(
                    name=name,
                    type=entity_type,
                    name_norm=name_norm,
                    entity_key=key,
                    description=description,
                    embedding=embedding,
                    aliases=aliases_list,
                    effective_date=effective_date,
                    citation_text=citation_text,
                )
            else:
                existing_entity = entities[key]
                # Merge description
                if description and description not in existing_entity.description:
                    if existing_entity.description:
                        existing_entity.description += " " + description
                    else:
                        existing_entity.description = description
                # Merge aliases uniquely
                for alias in aliases_list:
                    if alias and alias not in existing_entity.aliases:
                        existing_entity.aliases.append(alias)
                # Keep first or non-empty date / citation
                if effective_date and not existing_entity.effective_date:
                    existing_entity.effective_date = effective_date
                if citation_text and not existing_entity.citation_text:
                    existing_entity.citation_text = citation_text

            name_to_key[self._basic_normalize(name)] = key
            name_to_key[name_norm] = key
            # Also register aliases in the key lookup
            if isinstance(llm_aliases, list):
                for alias in llm_aliases:
                    alias_norm = self._basic_normalize(str(alias))
                    if alias_norm:
                        name_to_key[alias_norm] = key

        relationships: List[GraphRelation] = []
        if not isinstance(raw_relationships, list):
            raw_relationships = []
        for item in raw_relationships:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            relation = str(item.get("relation", "")).strip().upper()
            evidence = str(item.get("evidence", "")).strip()
            description = str(item.get("description", "")).strip()
            raw_keywords = item.get("keywords", [])
            if isinstance(raw_keywords, str):
                keywords = [k.strip().lower() for k in raw_keywords.split(",") if k.strip()]
            elif isinstance(raw_keywords, list):
                keywords = [str(k).strip().lower() for k in raw_keywords if str(k).strip()]
            else:
                keywords = []
            # Only safety check: relation must be valid UPPER_SNAKE_CASE (prevents Cypher injection)
            if not relation or not self.SAFE_RELATION_PATTERN.match(relation):
                continue
            source_key = name_to_key.get(self._basic_normalize(source)) or name_to_key.get(
                self.normalize_entity_name(source)
            )
            target_key = name_to_key.get(self._basic_normalize(target)) or name_to_key.get(
                self.normalize_entity_name(target)
            )
            if not source_key or not target_key or source_key == target_key:
                continue

            source_entity = entities.get(source_key)
            target_entity = entities.get(target_key)
            if not source_entity or not target_entity:
                continue

            try:
                confidence = float(item.get("confidence", 1.0))
            except Exception:
                confidence = 1.0
            if confidence < config.GRAPH_MIN_REL_CONFIDENCE:
                continue
            if config.GRAPH_REQUIRE_RELATION_EVIDENCE:
                if not evidence:
                    continue
                if not self._evidence_supported(evidence, source_text):
                    continue
            relationships.append(
                GraphRelation(
                    source=source_entity.name,
                    relation=relation,
                    target=target_entity.name,
                    source_key=source_key,
                    target_key=target_key,
                    confidence=max(0.0, min(1.0, confidence)),
                    evidence=evidence if config.GRAPH_STORE_EVIDENCE else "",
                    description=description,
                    keywords=keywords,
                )
            )

        return GraphExtraction(entities=list(entities.values()), relationships=relationships)

    def build_extraction_prompt(self, text: str, metadata: dict) -> str:
        section = metadata.get("section") or metadata.get("section_path") or ""
        source = metadata.get("source") or "Unknown"
        clipped = (text or "")[:5000]
        max_entities = getattr(config, "GRAPH_MAX_ENTITIES_SINGLE", 25)
        max_relationships = getattr(config, "GRAPH_MAX_RELATIONS_SINGLE", 30)

        # Build domain context block: rich description if available, else short label
        if self.domain_description:
            domain_block = f"Domain: {self.domain}\n{self.domain_description}"
        else:
            domain_block = f"Domain: {self.domain}"

        # Build preferred entity types guidance (NOT a whitelist)
        if self.preferred_entity_types:
            entity_type_list = ", ".join(self.preferred_entity_types)
            entity_type_guidance = (
                f"\n\nPREFERRED ENTITY TYPES (use these when they fit — you may create new ones ONLY when none of these apply):\n"
                f"{entity_type_list}\n"
                f"\nCRITICAL: Use short, general types. Do NOT invent verbose synonyms.\n"
                f"  GOOD: COURT, ORGANIZATION, PERSON, ROLE, COMMUNICATION_METHOD, CONTACT_INFO\n"
                f"  BAD:  LEGAL_JUDICIAL_AUTHORITY, HIGH_COURT_INSTITUTION, INDIVIDUAL_PARTY"
            )
        else:
            entity_type_guidance = (
                "\n\nCRITICAL: Use short, general entity types in UPPER_SNAKE_CASE.\n"
                "  GOOD: PERSON, COURT, LAW, ORGANIZATION, ROLE, COMMUNICATION_METHOD, CONTACT_INFO\n"
                "  BAD:  LEGAL_JUDICIAL_AUTHORITY, HIGH_COURT_INSTITUTION"
            )

        # Build preferred relation types guidance (NOT a whitelist)
        if self.preferred_relation_types:
            relation_type_list = ", ".join(self.preferred_relation_types)
            relation_type_guidance = (
                f"\n\nPREFERRED RELATION TYPES (reuse these when they fit — you may create new ones ONLY when none of these apply):\n"
                f"{relation_type_list}\n"
                f"\nCRITICAL: Use concise relationship names. Prefer existing types over inventing synonyms.\n"
                f"  GOOD: SUPERVISES (for any oversight/monitoring/regulation relationship)\n"
                f"  BAD:  MONITORS, OVERSEES, WATCHES_OVER, KEEPS_CHECK_ON (all mean the same thing)"
            )
        else:
            relation_type_guidance = (
                "\n\nCRITICAL: Use concise relationship names in UPPER_SNAKE_CASE.\n"
                "Prefer reusing existing relation types over inventing synonyms.\n"
                "  GOOD: SUPERVISES (for any oversight relationship)\n"
                "  BAD:  MONITORS, OVERSEES, WATCHES_OVER (all mean the same thing)"
            )

        return f"""You are an expert {self.domain} domain knowledge graph extraction system.
Your task is to carefully read the ENTIRE text chunk below and extract all meaningful Entities and their directional Relationships.

=== DOMAIN CONTEXT ===
{domain_block}

=== STEP 1: ENTITY EXTRACTION & NORMALIZATION ===

Scan the full text and identify every important named entity relevant to the {self.domain} domain.

RULES:
1. NO GENERIC ENTITIES: Do NOT extract generic, unqualified entities like "Act", "Council", "Member", "Chairperson", "Section 18", "Section 21", "Section 30", "Notification", "Government", "Court", "Institute", or "Secretariat". These generic nodes pollute the graph and act as noisy hubs. Instead, ALWAYS qualify them using the context of the document or the most specific terms available.
   - Example: Instead of "Act" or "the Act", use "Micro, Small and Medium Enterprises Development Act, 2006".
   - Example: Instead of "Council", use "Madhya Pradesh Micro and Small Enterprises Facilitation Council".
   - Example: Instead of "Section 18", specify "Section 18 of the MSMED Act, 2006".
   - Example: Instead of "Government", use "State Government of Madhya Pradesh" or "Central Government".
   - Example: Instead of "Court", use the specific court if named (e.g. "Bhopal District Court"), or "Competent Court" under the Act.
   - Example: Instead of "Institute", use the specific arbitral/conciliation institute referred to.
   - Example: Instead of "Member" or "Chairperson" or "Secretary", use "Member of the Council" or "Chairperson of the Council" or "Secretary to the Council".
   If a generic term cannot be fully resolved/qualified using the text context, DO NOT extract it.
2. NO OVERLY GRANULAR LEGAL NODES: If a section has nested clauses or sub-clauses (e.g. Section 21, Clause (1), Sub-clause (i)), do NOT extract them as separate, individual nodes (e.g. do not create separate nodes for 'Section 21', 'Clause (1)', and 'Sub-clause (i)'). Instead, combine them into a single, canonical legal reference node, such as 'Section 21(1)(i) of the MSMED Act, 2006' or 'Rule 5(2) of the MSME Rules, 2017'. Treat the entire specific legal citation as a single node.
3. CORRECT ENTITY TYPES:
   - Positions, official posts, or titles like "Chairperson", "Collector", "Commissioner", "Director of Industries", or "Advocate" are **ROLE** types, NOT **PERSON** types. Use the type **PERSON** only for named individuals (e.g. "V. L. Kantha Rao", "Justice Sujoy Paul").
   - Methods of communication like "SMS", "Email", "WhatsApp", or "Letter" should be typed as **COMMUNICATION_METHOD**, not **DOCUMENT**.
   - Individual contact details like phone numbers or email addresses should be typed as **CONTACT_INFO**, not **AMOUNT** or other types.
4. Assign each entity a type in UPPER_SNAKE_CASE.
5. Keep entity names concise, normalized, and properly capitalized.
6. TEMPORAL METADATA (OPTIONAL): For entities representing notifications, rules, acts, or amendments, if the text mentions an effective date, enactment date, or publication date, extract it as an `effective_date` field (e.g. "2017-11-24" or "24-11-2017").
7. CITATION TEXT (OPTIONAL): For specific laws, sections, rules, or clauses, extract a `citation_text` field if the text names the act or parent document it belongs to.
   - Example: For an entity with name 'Section 21(1)(i) of the MSMED Act, 2006', its `citation_text` would be 'Section 21(1)(i) of the Micro, Small and Medium Enterprises Development Act, 2006'.
8. DESCRIPTION (REQUIRED): Write a concise, factual, and detailed description of the entity based on the context. Explain who or what this entity is within the domain.
9. JSON FORMATTING & LaTeX ESCAPING (CRITICAL):
   - All string values must be properly escaped JSON strings.
   - Escape double quotes: " -> \\"
   - Escape backslashes: \ -> \\\\
   - Escape newlines: newline -> \\n
   - For LaTeX/math expressions, use double-escaped backslashes: e.g. \\frac must be written as `"\\\\frac"`, \\sum as `"\\\\sum"` inside the JSON.
   - Example: "formula": "The formula \\\\frac{1}{2} is used"
10. THIRD-PERSON OBJECTIVE STYLE (CRITICAL): Ensure all entity names and descriptions are written from an objective, third-person perspective.
    - NEVER use pronouns: this article, this paper, our company, I, you, he, she, we, they, my, or our.
    - Instead, explicitly name the subject: e.g. "The study by Smith et al.", "Apple Inc.", "The algorithm".
    - Example BAD: "This paper proposes a new method"
    - Example GOOD: "The paper by Chen (2023) proposes a new method"
{entity_type_guidance}

=== STEP 2: ALIAS DETECTION (CRITICAL) ===

Before finalizing entities, scan the text for entities that refer to the SAME real-world thing but use different names.
For example:
- "RBI" and "Reserve Bank of India" → same entity
- "MSMED Act" and "Micro, Small and Medium Enterprises Development Act, 2006" → same entity
- "MSEFC" and "Madhya Pradesh Micro and Small Enterprises Facilitation Council" → same entity

When you find such cases:
- Pick the MOST COMPLETE / FORMAL name as the canonical "name"
- List ALL shorter names, abbreviations, and alternate forms in the "aliases" array
- Do NOT create separate entities for the same thing

=== STEP 3: RELATIONSHIP EXTRACTION ===

Extract directional relationships between the entities you identified.

RULES:
1. Use UPPER_SNAKE_CASE relation types.
2. ANTI-ORPHAN RULE: Every "source" and "target" MUST exist in the entities array.
3. NO SYMMETRIC DUPLICATES: Do not extract both directions of a relationship (e.g. do not extract `A -> APPOINTS -> B` AND `B -> APPOINTED_BY -> A` together). Extract only ONE active canonical direction (e.g., prefer `State Government -> APPOINTS -> Chairperson` over `Chairperson -> APPOINTED_BY -> State Government`).
4. NO RELATIONSHIP EXPLOSION: Do not extract relationships like `Issues` or `Regulates` between an organization and a law/section unless the text explicitly says they issue/regulate it. A mere reference in the text (e.g., "Reference under Section 18...") does NOT mean `Council -> Issues -> Section 18`. The correct extraction is `Reference -> UNDER -> Section 18` or `Section 18 -> PART_OF -> MSMED Act`.
5. NO OVER-LINKING SECTIONS: Avoid creating direct action links like `Section 21 -> REGULATES -> Council`. Instead, link the section to its parent act (e.g., `Section 21 -> PART_OF -> MSMED Act`) and link the entity to the section using specific relationship types (e.g., `Council -> ESTABLISHED_UNDER -> Section 21`).
6. EXTRACT PROCEDURAL/ACTIONABLE RELATIONS: Ensure you capture actual operational, procedural, or reporting relationships mentioned in the text (e.g. `Council -> UPLOADS -> Annual Progress Report`, `Institute -> RETURNS_AWARD_TO -> Council`, `Council -> PROVIDES_INFORMATION_TO -> Member Secretary`).
7. NO HALLUCINATIONS: Extract only relationships explicitly stated or clearly implied by the text. If unsure, omit the relationship. Provide evidence: a direct quote or close paraphrase from the chunk.
   - Example of a hallucination: If the text says 'Rules shall extend to the whole of the State of Madhya Pradesh', do NOT extract `Micro and Small Enterprises Facilitation Council -> GOVERNS -> State of Madhya Pradesh`. The Council does not govern the state. The correct extraction is `Rules -> APPLY_TO -> State of Madhya Pradesh` or similar.
8. DIRECTIONALITY ACCURACY: Pay close attention to the direction of relationships. Rules or regulations are NEVER parents of Acts. The Act is the parent, and Rules implement or are derived from it.
   - GOOD: Rules -> IMPLEMENTS -> Act, or Rules -> DERIVED_FROM -> Act, or Section -> PART_OF -> Act.
   - BAD: Act -> PART_OF -> Rules (backwards).
9. REPEAL RELATION DIRECTION: If the text says 'Rules of A are hereby repealed' by the new 'Rules of B', the relationship should be `Rules of B -> REPEALS -> Rules of A` (or `Rules of A -> REPEALED_BY -> Rules of B`). Do NOT reverse this.
10. NEGATION GUARD: Pay close attention to negative words like "no", "not", "never", "is not", "does not", "did not". Do NOT extract positive relationships from statements containing negative descriptors (e.g. if the text says "Chandresh was not holding any position in Defendant No. 4", do NOT extract `Chandresh -> DIRECTOR_OF -> Defendant No. 4`).
11. DESCRIPTION (REQUIRED): Write a detailed explanation of the relationship, explaining how or why the source and target entities are connected according to the text.
12. KEYWORDS (REQUIRED): Extract a list of 2-5 lowercase search terms or keywords summarizing the context of the relationship (e.g. ["establishment", "legal authority", "facilitation"]).
13. N-ARY RELATIONSHIP DECOMPOSITION: If a statement describes a relationship involving 3 or more entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
    - Example: "Alice, Bob, and Carol collaborated on the machine learning project"
      Extract:
      - (Alice, collaborated_with, Bob)
      - (Alice, collaborated_with, Carol)
      - (Bob, collaborated_with, Carol)
      - (Alice, worked_on, machine_learning_project)
      - (Bob, worked_on, machine_learning_project)
      - (Carol, worked_on, machine_learning_project)
14. RELATIONSHIP RULES & SIGNIFICANCE:
    - Treat relationships as UNDIRECTED unless explicitly directional.
    - Avoid symmetric duplicates: (A, related_to, B) and (B, related_to, A) are the same. Swapping source and target does not constitute a new relationship.
    - Prioritize and output relationships that are most significant to the core meaning of the input text first.
    - Example: If "A caused B" and "A is related to B" are both true, only include the more specific "caused".
{relation_type_guidance}

=== STEP 4: QUANTITY LIMITS ===
1. Output at most {max_entities} entities and {max_relationships} relationships in this response. Output fewer if fewer high-value items are present. Do not try to fill the limit.
2. Only output relationships whose source and target entities are both included in the entities list.

CONTEXT:
Document: {source}
Section: {section}

TEXT:
{clipped}

OUTPUT FORMAT:
Return ONLY valid JSON. No markdown, no explanation, no extra text.
{{
  "entities": [
    {{
      "name": "Madhya Pradesh Micro and Small Enterprises Facilitation Council",
      "type": "ORGANIZATION",
      "description": "A state-level facilitation council established in Madhya Pradesh to handle references under Section 18 of the MSMED Act.",
      "aliases": ["Council", "MSEFC", "MPMSEFC"],
      "effective_date": null,
      "citation_text": null
    }},
    {{
      "name": "Section 21(1)(i) of the MSMED Act, 2006",
      "type": "SECTION",
      "description": "A specific section of the MSMED Act, 2006 under which Micro and Small Enterprises Facilitation Councils are established.",
      "aliases": ["Section 21", "Section 21(1)"],
      "effective_date": null,
      "citation_text": "Section 21(1)(i) of the Micro, Small and Medium Enterprises Development Act, 2006"
    }},
    {{
      "name": "Madhya Pradesh Micro and Small Enterprises Facilitation Rules, 2017",
      "type": "RULE",
      "description": "State rules established in 2017 to implement facilitation councils in Madhya Pradesh.",
      "aliases": ["Rules of 2017", "Facilitation Rules"],
      "effective_date": "2017-11-24",
      "citation_text": null
    }}
  ],
  "relationships": [
    {{
      "source": "Madhya Pradesh Micro and Small Enterprises Facilitation Council",
      "relation": "ESTABLISHED_UNDER",
      "target": "Section 21(1)(i) of the MSMED Act, 2006",
      "confidence": 0.95,
      "evidence": "The Council is established under Section 21 of the Act.",
      "description": "The Madhya Pradesh Micro and Small Enterprises Facilitation Council is established under the authority of Section 21(1)(i) of the MSMED Act, 2006.",
      "keywords": ["establishment", "legal authority", "facilitation"]
    }}
  ]
}}"""

    def extract_from_text(
        self,
        text: str,
        metadata: dict,
        *,
        llm_invoke: LLMInvoke,
        embedding_fn: Optional[EmbeddingFn] = None,
    ) -> Optional[GraphExtraction]:
        file_hash = str(metadata.get("file_hash") or metadata.get("source") or "Unknown")
        # --- Caching Logic ---
        cache_dir = os.getenv("EXTRACTION_CACHE_DIR", ".extraction_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_version = str(getattr(config, "GRAPH_CACHE_VERSION", 1))
        config_str = f"{self.domain}::{','.join(sorted(self.preferred_entity_types))}"
        hash_input = f"{cache_version}::{config_str}::{text or ''}"
        chunk_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{chunk_hash}.json")
        
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                    extraction = self.validate_extraction(
                        payload,
                        embedding_fn=embedding_fn,
                        file_hash=file_hash,
                        source_text=text,
                    )
                    extraction.repaired = False
                    extraction.is_cached = True
                    extraction.llm_time_sec = 0.0
                    if not extraction.entities:
                        return extraction
                    return extraction
            except Exception:
                pass # Fall through to LLM invocation if cache is broken
                
        prompt = self.build_extraction_prompt(text, metadata)
        self.extraction_attempts += 1
        
        t0 = time.time()
        try:
            raw = str(llm_invoke(prompt, self.extraction_timeout, self.extraction_retries))
        except Exception as exc:
            self.extraction_failures += 1
            print(f"[GraphRAG] Extraction failed for chunk {metadata.get('chunk_id')}: {exc}")
            return None

        payload, mode = self.parse_json_output(raw)
        repaired = False
        if payload is None and self.extraction_retries > 0:
            repair_prompt = (
                "Repair this malformed model output into STRICT raw JSON only. "
                "Use exactly this shape: {\"entities\": [], \"relationships\": []}. "
                "Do not add markdown or explanations.\n\n"
                f"MODEL OUTPUT:\n{raw[:6000]}"
            )
            try:
                repaired_raw = str(llm_invoke(repair_prompt, self.extraction_timeout, self.extraction_retries))
                payload, mode = self.parse_json_output(repaired_raw)
                repaired = payload is not None
            except Exception:
                payload = None
                
        t1 = time.time()

        if payload is None:
            self.extraction_failures += 1
            print(
                f"[GraphRAG] Dropping graph extraction for chunk {metadata.get('chunk_id')} "
                f"(parse mode={mode})."
            )
            return None

        # Save to cache if successful
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

        extraction = self.validate_extraction(
            payload,
            embedding_fn=embedding_fn,
            file_hash=file_hash,
            source_text=text,
        )
        extraction.repaired = repaired
        extraction.is_cached = False
        extraction.llm_time_sec = t1 - t0
        if not extraction.entities:
            return extraction
        return extraction

    def _chunk_key(self, doc: Document) -> str:
        source = str(doc.metadata.get("source", "Unknown"))
        chunk_id = str(doc.metadata.get("chunk_id", "0"))
        file_hash = str(doc.metadata.get("file_hash", "") or source)
        return f"{file_hash}::{source}::{chunk_id}"

    def _should_index_doc(self, doc: Document) -> bool:
        chunk_id = doc.metadata.get("chunk_id")
        if str(chunk_id) == "-1":
            return False
        content_type = str(doc.metadata.get("content_type", doc.metadata.get("type", ""))).lower()
        if content_type == "document_summary":
            return False
        text = (doc.page_content or "").strip()
        
        if len(text) < getattr(config, "MIN_GRAPH_CHUNK_LENGTH", 180):
            return False

        # Skip only if ZERO legal entity signals found
        signals = sum(len(p.findall(text)) for p in _GRAPH_ENTITY_SIGNAL_PATTERNS)
        if signals == 0:
            return False

        return True

    def _resolve_semantic_entities_batch(
        self,
        session,
        entities: List[GraphEntity],
    ) -> Dict[str, GraphEntity]:
        if not self.semantic_merge_enabled:
            return {e.entity_key: e for e in entities}

        entities_to_resolve = []
        for entity in entities:
            if entity.embedding:
                entities_to_resolve.append({
                    "entity_key": entity.entity_key,
                    "type": entity.type
                })

        if not entities_to_resolve:
            return {e.entity_key: e for e in entities}

        resolved_map: Dict[str, GraphEntity] = {}
        try:
            rows = session.run(
                """
                UNWIND $entities AS input
                CALL (input) {
                  MATCH (e:Entity)
                  WHERE e.type = input.type
                    AND e.embedding IS NOT NULL
                    AND e.entity_key <> input.entity_key
                  RETURN e.entity_key AS entity_key,
                         e.name AS name,
                         e.name_norm AS name_norm,
                         e.embedding AS embedding
                  LIMIT 200
                }
                RETURN input.entity_key AS input_key,
                       entity_key,
                       name,
                       name_norm,
                       embedding
                """,
                entities=entities_to_resolve
            )

            # Group candidate matches by input_key
            candidates_by_input: Dict[str, List[dict]] = {}
            for row in rows:
                input_key = row["input_key"]
                candidates_by_input.setdefault(input_key, []).append({
                    "entity_key": row["entity_key"],
                    "name": row["name"],
                    "name_norm": row["name_norm"],
                    "embedding": row["embedding"]
                })
        except Exception as exc:
            print(f"[GraphRAG] Batch semantic merge query failed, skipping merge: {exc}")
            return {e.entity_key: e for e in entities}

        for entity in entities:
            if not entity.embedding or entity.entity_key not in candidates_by_input:
                resolved_map[entity.entity_key] = entity
                continue

            best_key = ""
            best_name = ""
            best_norm = ""
            best_score = 0.0

            for candidate in candidates_by_input[entity.entity_key]:
                score = self._cosine(entity.embedding, candidate["embedding"] or [])
                if score > best_score:
                    best_score = score
                    best_key = candidate["entity_key"] or ""
                    best_name = candidate["name"] or ""
                    best_norm = candidate["name_norm"] or ""

            if best_key and best_score >= self.semantic_merge_threshold:
                merged_aliases = list(set(entity.aliases + [entity.name]))
                resolved_map[entity.entity_key] = GraphEntity(
                    name=best_name or entity.name,
                    type=entity.type,
                    name_norm=best_norm or entity.name_norm,
                    entity_key=best_key,
                    description=entity.description,
                    embedding=entity.embedding,
                    aliases=merged_aliases,
                    effective_date=entity.effective_date,
                    citation_text=entity.citation_text,
                )
            else:
                resolved_map[entity.entity_key] = entity

        return resolved_map

    def _merge_descriptions(self, desc1: str, desc2: str, llm_invoke: LLMInvoke, name: Optional[str] = None) -> str:
        if not desc1:
            return desc2 or ""
        if not desc2:
            return desc1 or ""
        if desc1.strip().lower() == desc2.strip().lower():
            return desc1
        if desc1.strip().lower() in desc2.strip().lower():
            return desc2
        if desc2.strip().lower() in desc1.strip().lower():
            return desc1
            
        if len(desc1) + len(desc2) < 400:
            return f"{desc1} {desc2}"
            
        name_header = f"\nSubject Name: {name}\n" if name else ""
        prompt = f"""You are an expert knowledge graph refiner.
Your task is to merge two descriptions of the same entity or relationship into a single, cohesive, comprehensive description.
The descriptions may contain overlapping or complementary information. Integrate them smoothly without redundancy.
{name_header}
Description 1:
{desc1}

Description 2:
{desc2}

Merged Description Guidelines:
1. Ensure the summary is written from an objective, third-person perspective.
2. Explicitly mention the name of the entity or relationship at the beginning of the description for context.
3. DESCRIPTION CONFLICT RESOLUTION:
   - If descriptions conflict but refer to the same entity: Summarize both perspectives with "There are differing views..." (e.g. "Apple Inc. is a technology company (source A). Some sources describe it as primarily a consumer electronics company (source B).").
   - If descriptions refer to different entities with same name: Keep them as separate entities, and add disambiguation parentheticals: e.g. "Apple Inc. (technology company)" vs "Apple Inc. (fruit distributor)".
   - Always preserve source attribution when possible.
4. Return ONLY the merged description text. Do not write any introduction, conclusion, or explanations.

Merged Description:"""
        try:
            merged = llm_invoke(prompt, 15.0, 1)
            return merged.strip()
        except Exception as e:
            logger.warning("Failed to merge descriptions using LLM: %s. Falling back to concatenation.", e)
            return f"{desc1} {desc2}"

    def _get_existing_entity_description(self, session, entity_key: str) -> Optional[str]:
        try:
            result = session.run(
                "MATCH (e:Entity {entity_key: $entity_key}) RETURN e.description AS description",
                entity_key=entity_key
            )
            record = result.single()
            if record:
                return record["description"]
        except Exception as e:
            logger.warning("Failed to fetch existing description for key %s: %s", entity_key, e)
        return None

    def _get_existing_relation_data(self, session, source_key: str, relation: str, target_key: str) -> Tuple[Optional[str], List[str]]:
        try:
            query = f"""
            MATCH (a:Entity {{entity_key: $source_key}})-[r:{relation}]->(b:Entity {{entity_key: $target_key}})
            RETURN r.description AS description, r.keywords AS keywords
            """
            result = session.run(query, source_key=source_key, target_key=target_key)
            record = result.single()
            if record:
                desc = record.get("description")
                kws = record.get("keywords") or []
                return desc, kws
        except Exception as e:
            logger.warning("Failed to fetch existing relationship data for %s-[%s]->%s: %s", source_key, relation, target_key, e)
        return None, []

    def _upsert_chunk_graph(
        self,
        session,
        doc: Document,
        extraction: GraphExtraction,
        llm_invoke: LLMInvoke,
    ) -> None:
        source = str(doc.metadata.get("source", "Unknown"))
        file_hash = str(doc.metadata.get("file_hash", "") or source)
        chunk_id = str(doc.metadata.get("chunk_id", "0") or "0")
        chunk_key = self._chunk_key(doc)
        page_label = str(doc.metadata.get("page_label", "1") or "1")
        section = str(doc.metadata.get("section", "") or "")
        text = doc.page_content or ""

        session.run(
            """
            MERGE (d:Document {file_hash: $file_hash})
            SET d.source = $source
            MERGE (c:Chunk {chunk_key: $chunk_key})
            SET c.source = $source,
                c.chunk_id = $chunk_id,
                c.page_label = $page_label,
                c.section = $section,
                c.file_hash = $file_hash,
                c.text = $text
            MERGE (c)-[:FROM_DOCUMENT]->(d)
            """,
            file_hash=file_hash,
            source=source,
            chunk_key=chunk_key,
            chunk_id=chunk_id,
            page_label=page_label,
            section=section,
            text=text,
        )

        entity_by_key = self._resolve_semantic_entities_batch(session, extraction.entities)
        
        # Deduplicate and merge entity properties by resolved entity_key
        merged_entities: Dict[str, GraphEntity] = {}
        for entity in extraction.entities:
            resolved = entity_by_key[entity.entity_key]
            key = resolved.entity_key
            if key not in merged_entities:
                merged_entities[key] = GraphEntity(
                    name=resolved.name,
                    type=resolved.type,
                    name_norm=resolved.name_norm,
                    entity_key=key,
                    description=resolved.description or "",
                    embedding=resolved.embedding or [],
                    aliases=list(resolved.aliases or []),
                    effective_date=resolved.effective_date,
                    citation_text=resolved.citation_text,
                )
            else:
                existing = merged_entities[key]
                if resolved.description and resolved.description not in existing.description:
                    if existing.description:
                        existing.description += " " + resolved.description
                    else:
                        existing.description = resolved.description
                for alias in (resolved.aliases or []):
                    if alias and alias not in existing.aliases:
                        existing.aliases.append(alias)
                if not existing.effective_date:
                    existing.effective_date = resolved.effective_date
                if not existing.citation_text:
                    existing.citation_text = resolved.citation_text

        entity_rows = []
        for key, resolved in merged_entities.items():
            # Check if there is an existing description in Neo4j
            existing_desc = self._get_existing_entity_description(session, key)
            if existing_desc:
                resolved.description = self._merge_descriptions(existing_desc, resolved.description, llm_invoke, name=resolved.name)
            
            unique_aliases_dict = {a.lower().strip(): a for a in (resolved.aliases or []) if a.strip()}
            unique_aliases = list(unique_aliases_dict.values())
            entity_rows.append({
                "entity_key": key,
                "name": resolved.name,
                "name_norm": resolved.name_norm,
                "type": resolved.type,
                "description": resolved.description or "",
                "embedding": resolved.embedding or None,
                "aliases": unique_aliases,
                "effective_date": resolved.effective_date,
                "citation_text": resolved.citation_text,
                "chunk_key": chunk_key,
            })

        if entity_rows:
            session.run(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {entity_key: row.entity_key})
                SET e.name = coalesce(e.name, row.name),
                    e.name_norm = coalesce(e.name_norm, row.name_norm),
                    e.type = coalesce(e.type, row.type),
                    e.embedding = coalesce(e.embedding, row.embedding),
                    e.aliases = [x IN coalesce(e.aliases, []) WHERE NOT toLower(trim(x)) IN [a IN row.aliases | toLower(trim(a))]] + row.aliases,
                    e.effective_date = coalesce(e.effective_date, row.effective_date),
                    e.citation_text = coalesce(e.citation_text, row.citation_text),
                    e.description = row.description
                WITH e, row
                MATCH (c:Chunk {chunk_key: row.chunk_key})
                MERGE (c)-[m:MENTIONS]->(e)
                SET m.count = coalesce(m.count, 0) + 1
                """,
                rows=entity_rows,
            )

        # Deduplicate relationships inside the chunk before Neo4j insertion
        seen_rels = {}
        for rel in extraction.relationships:
            source_entity = entity_by_key.get(rel.source_key)
            target_entity = entity_by_key.get(rel.target_key)
            if not source_entity or not target_entity:
                continue
            if source_entity.entity_key == target_entity.entity_key:
                continue
            relation = rel.relation
            if not self.SAFE_RELATION_PATTERN.match(relation):
                continue

            rel_key = (source_entity.entity_key, relation, target_entity.entity_key)
            if rel_key not in seen_rels:
                seen_rels[rel_key] = GraphRelation(
                    source=rel.source,
                    relation=rel.relation,
                    target=rel.target,
                    source_key=rel.source_key,
                    target_key=rel.target_key,
                    confidence=rel.confidence,
                    evidence=rel.evidence,
                    description=rel.description or "",
                    keywords=getattr(rel, "keywords", []),
                )
            else:
                existing_rel = seen_rels[rel_key]
                if rel.confidence > existing_rel.confidence:
                    existing_rel.confidence = rel.confidence
                if rel.evidence and rel.evidence != existing_rel.evidence:
                    if existing_rel.evidence:
                        existing_rel.evidence += " | " + rel.evidence
                    else:
                        existing_rel.evidence = rel.evidence
                if rel.description and rel.description != existing_rel.description:
                    if existing_rel.description:
                        existing_rel.description += " " + rel.description
                    else:
                        existing_rel.description = rel.description
                extra_kws = getattr(rel, "keywords", [])
                if extra_kws:
                    existing_rel.keywords = list(set(existing_rel.keywords + extra_kws))

        for rel in seen_rels.values():
            source_entity = entity_by_key.get(rel.source_key)
            target_entity = entity_by_key.get(rel.target_key)
            relation = rel.relation
            
            existing_rel_desc, existing_kws = self._get_existing_relation_data(session, source_entity.entity_key, relation, target_entity.entity_key)
            if existing_rel_desc:
                rel.description = self._merge_descriptions(
                    existing_rel_desc,
                    rel.description,
                    llm_invoke,
                    name=f"{source_entity.name} --[{relation}]--> {target_entity.name}"
                )
            
            merged_kws = list(set([kw.strip().lower() for kw in rel.keywords + existing_kws if kw.strip()]))
                
            statement = f"""
            MATCH (a:Entity {{entity_key: $source_key}})
            MATCH (b:Entity {{entity_key: $target_key}})
            MERGE (a)-[r:{relation}]->(b)
            SET r.confidence = CASE
                WHEN r.confidence IS NULL OR r.confidence < $confidence THEN $confidence
                ELSE r.confidence
            END,
            r.count = coalesce(r.count, 0) + 1,
            r.evidence = CASE
                WHEN $evidence = "" THEN coalesce(r.evidence, "")
                ELSE coalesce(r.evidence, $evidence)
            END,
            r.sources = CASE WHEN $source IN coalesce(r.sources, []) THEN r.sources ELSE coalesce(r.sources, []) + $source END,
            r.chunk_ids = CASE WHEN $chunk_id IN coalesce(r.chunk_ids, []) THEN r.chunk_ids ELSE coalesce(r.chunk_ids, []) + $chunk_id END,
            r.pages = CASE WHEN $page_label IN coalesce(r.pages, []) THEN r.pages ELSE coalesce(r.pages, []) + $page_label END,
            r.description = $description,
            r.keywords = $keywords
            """
            session.run(
                statement,
                source_key=source_entity.entity_key,
                target_key=target_entity.entity_key,
                confidence=rel.confidence,
                evidence=rel.evidence if config.GRAPH_STORE_EVIDENCE else "",
                source=source,
                chunk_id=chunk_id,
                page_label=page_label,
                description=rel.description or "",
                keywords=merged_kws,
            )

    def build_batch_extraction_prompt(self, batch: List[Tuple[str, dict]]) -> str:
        first_meta = batch[0][1] if batch else {}
        section = first_meta.get("section") or first_meta.get("section_path") or ""
        source = first_meta.get("source") or "Unknown"
        max_entities = getattr(config, "GRAPH_MAX_ENTITIES_BATCH", 45)
        max_relationships = getattr(config, "GRAPH_MAX_RELATIONS_BATCH", 50)

        if self.domain_description:
            domain_block = f"Domain: {self.domain}\n{self.domain_description}"
        else:
            domain_block = f"Domain: {self.domain}"

        if self.preferred_entity_types:
            entity_type_list = ", ".join(self.preferred_entity_types)
            entity_type_guidance = (
                f"\n\nPREFERRED ENTITY TYPES (use these when they fit — you may create new ones ONLY when none of these apply):\n"
                f"{entity_type_list}\n"
                f"\nCRITICAL: Use short, general types. Do NOT invent verbose synonyms.\n"
                f"  GOOD: COURT, ORGANIZATION, PERSON, ROLE, COMMUNICATION_METHOD, CONTACT_INFO\n"
                f"  BAD:  LEGAL_JUDICIAL_AUTHORITY, HIGH_COURT_INSTITUTION, INDIVIDUAL_PARTY"
            )
        else:
            entity_type_guidance = (
                "\n\nCRITICAL: Use short, general entity types in UPPER_SNAKE_CASE.\n"
                "  GOOD: PERSON, COURT, LAW, ORGANIZATION, ROLE, COMMUNICATION_METHOD, CONTACT_INFO\n"
                "  BAD:  LEGAL_JUDICIAL_AUTHORITY, HIGH_COURT_INSTITUTION"
            )

        if self.preferred_relation_types:
            relation_type_list = ", ".join(self.preferred_relation_types)
            relation_type_guidance = (
                f"\n\nPREFERRED RELATION TYPES (reuse these when they fit — you may create new ones ONLY when none of these apply):\n"
                f"{relation_type_list}\n"
                f"\nCRITICAL: Use concise relationship names. Prefer existing types over inventing synonyms.\n"
                f"  GOOD: SUPERVISES (for any oversight/monitoring/regulation relationship)\n"
                f"  BAD:  MONITORS, OVERSEES, WATCHES_OVER, KEEPS_CHECK_ON (all mean the same thing)"
            )
        else:
            relation_type_guidance = (
                "\n\nCRITICAL: Use concise relationship names in UPPER_SNAKE_CASE.\n"
                "Prefer reusing existing relation types over inventing synonyms.\n"
                "  GOOD: SUPERVISES (for any oversight relationship)\n"
                "  BAD:  MONITORS, OVERSEES, WATCHES_OVER (all mean the same thing)"
            )

        chunks_block = ""
        chunk_keys = []
        for i, (text, meta) in enumerate(batch):
            key = f"chunk_{meta.get('chunk_id', i)}"
            chunk_keys.append(key)
            clipped = (text or "")[:5000]
            chunks_block += f"\n--- {key} ---\n{clipped}\n--- END {key} ---\n"

        example_entries = []
        for i, key in enumerate(chunk_keys):
            if i == 0:
                example_entries.append(f'  "{key}": {{\n'
                                       f'    "entities": [\n'
                                       f'      {{\n'
                                       f'        "name": "Example Entity Name",\n'
                                       f'        "type": "ORGANIZATION",\n'
                                       f'        "description": "A detailed description explaining who or what this entity is.",\n'
                                       f'        "aliases": ["Example Alias"],\n'
                                       f'        "effective_date": null,\n'
                                       f'        "citation_text": null\n'
                                       f'      }}\n'
                                       f'    ],\n'
                                       f'    "relationships": [\n'
                                       f'      {{\n'
                                       f'        "source": "Example Entity Name",\n'
                                       f'        "relation": "ESTABLISHED_UNDER",\n'
                                       f'        "target": "Another Entity",\n'
                                       f'        "confidence": 0.95,\n'
                                       f'        "evidence": "evidence string",\n'
                                       f'        "description": "A detailed description explaining the connection between source and target."\n'
                                       f'      }}\n'
                                       f'    ]\n'
                                       f'  }}')
            else:
                example_entries.append(f'  "{key}": {{\n'
                                       f'    "entities": [],\n'
                                       f'    "relationships": []\n'
                                       f'  }}')
        output_format_example = "{\n" + ",\n".join(example_entries) + "\n}"

        return f"""You are an expert {self.domain} domain knowledge graph extraction system.
Your task is to carefully read the text chunks below and extract all meaningful Entities and their directional Relationships.

=== DOMAIN CONTEXT ===
{domain_block}

=== STEP 1: ENTITY EXTRACTION & NORMALIZATION ===

RULES:
1. NO GENERIC ENTITIES: Do NOT extract generic, unqualified entities like "Act", "Council", "Member", "Chairperson", "Section 18". ALWAYS qualify them using the context.
2. NO OVERLY GRANULAR LEGAL NODES: Treat specific legal citations as a single node.
3. CORRECT ENTITY TYPES: Positions are ROLE types. Methods of communication are COMMUNICATION_METHOD.
4. Assign each entity a type in UPPER_SNAKE_CASE.
5. Keep entity names concise, normalized, and properly capitalized.
6. TEMPORAL METADATA (OPTIONAL): Extract effective_date if present.
7. CITATION TEXT (OPTIONAL): Extract citation_text for specific laws.
8. DESCRIPTION (REQUIRED): Write a concise, factual, and detailed description of the entity based on the context.
9. JSON FORMATTING & LaTeX ESCAPING (CRITICAL):
   - All string values must be properly escaped JSON strings.
   - Escape double quotes: " -> \\"
   - Escape backslashes: \ -> \\\\
   - Escape newlines: newline -> \\n
   - For LaTeX/math expressions, use double-escaped backslashes: e.g. \\frac must be written as `"\\\\frac"`, \\sum as `"\\\\sum"` inside the JSON.
   - Example: "formula": "The formula \\\\frac{1}{2} is used"
10. THIRD-PERSON OBJECTIVE STYLE (CRITICAL): Ensure all entity names and descriptions are written from an objective, third-person perspective.
    - NEVER use pronouns: this article, this paper, our company, I, you, he, she, we, they, my, or our.
    - Instead, explicitly name the subject: e.g. "The study by Smith et al.", "Apple Inc.", "The algorithm".
    - Example BAD: "This paper proposes a new method"
    - Example GOOD: "The paper by Chen (2023) proposes a new method"
{entity_type_guidance}

=== STEP 2: ALIAS DETECTION (CRITICAL) ===

When you find entities that refer to the SAME real-world thing but use different names:
- Pick the MOST COMPLETE / FORMAL name as the canonical "name"
- List ALL shorter names, abbreviations, and alternate forms in the "aliases" array

=== STEP 3: RELATIONSHIP EXTRACTION ===

RULES:
1. Use UPPER_SNAKE_CASE relation types.
2. ANTI-ORPHAN RULE: Every "source" and "target" MUST exist in the entities array.
3. NO SYMMETRIC DUPLICATES: Extract only ONE active canonical direction.
4. NO RELATIONSHIP EXPLOSION: Do not extract generic relationships unless explicitly stated.
5. NO OVER-LINKING SECTIONS: Avoid direct action links to sections.
6. EXTRACT PROCEDURAL/ACTIONABLE RELATIONS.
7. NO HALLUCINATIONS: Extract only relationships explicitly stated or implied. Provide evidence.
8. DIRECTIONALITY ACCURACY: Rules or regulations are NEVER parents of Acts.
9. REPEAL RELATION DIRECTION: Follow the correct repeal direction.
10. DESCRIPTION (REQUIRED): Write a detailed explanation of the relationship, explaining how or why the source and target entities are connected according to the text.
11. N-ARY RELATIONSHIP DECOMPOSITION: If a statement describes a relationship involving 3 or more entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
    - Example: "Alice, Bob, and Carol collaborated on the machine learning project"
      Extract:
      - (Alice, collaborated_with, Bob)
      - (Alice, collaborated_with, Carol)
      - (Bob, collaborated_with, Carol)
      - (Alice, worked_on, machine_learning_project)
      - (Bob, worked_on, machine_learning_project)
      - (Carol, worked_on, machine_learning_project)
12. RELATIONSHIP RULES & SIGNIFICANCE:
    - Treat relationships as UNDIRECTED unless explicitly directional.
    - Avoid symmetric duplicates: (A, related_to, B) and (B, related_to, A) are the same. Swapping source and target does not constitute a new relationship.
    - Prioritize and output relationships that are most significant to the core meaning of the input text first.
    - Example: If "A caused B" and "A is related to B" are both true, only include the more specific "caused".
{relation_type_guidance}

=== STEP 4: QUANTITY LIMITS ===
1. Output at most {max_entities} entities and {max_relationships} relationships in total across all chunks in this response. Output fewer if fewer high-value items are present. Do not try to fill the limit.
2. Only output relationships whose source and target entities are both included in the entities list of the respective chunk.

CRITICAL BATCHING RULES:
1. Process each chunk INDEPENDENTLY. Never connect entities across chunks.
2. Return a JSON dict with one key per chunk.

CONTEXT:
Document: {source}
Section: {section}

CHUNKS:
{chunks_block}

OUTPUT FORMAT (strict JSON, no markdown):
{output_format_example}"""

    def build_batch_gleaning_prompt(self, batch: List[Tuple[str, dict]], current_payload: dict) -> str:
        first_meta = batch[0][1] if batch else {}
        section = first_meta.get("section") or first_meta.get("section_path") or ""
        source = first_meta.get("source") or "Unknown"
        max_entities = getattr(config, "GRAPH_MAX_ENTITIES_GLEANING", 20)
        max_relationships = getattr(config, "GRAPH_MAX_RELATIONS_GLEANING", 25)

        if self.domain_description:
            domain_block = f"Domain: {self.domain}\n{self.domain_description}"
        else:
            domain_block = f"Domain: {self.domain}"

        if self.preferred_entity_types:
            entity_type_list = ", ".join(self.preferred_entity_types)
            entity_type_guidance = (
                f"\n\nPREFERRED ENTITY TYPES (use these when they fit — you may create new ones ONLY when none of these apply):\n"
                f"{entity_type_list}"
            )
        else:
            entity_type_guidance = ""

        if self.preferred_relation_types:
            relation_type_list = ", ".join(self.preferred_relation_types)
            relation_type_guidance = (
                f"\n\nPREFERRED RELATION TYPES (reuse these when they fit — you may create new ones ONLY when none of these apply):\n"
                f"{relation_type_list}"
            )
        else:
            relation_type_guidance = ""

        chunks_block = ""
        extracted_info = ""
        chunk_keys = []
        for i, (text, meta) in enumerate(batch):
            key = f"chunk_{meta.get('chunk_id', i)}"
            chunk_keys.append(key)
            clipped = (text or "")[:5000]
            chunks_block += f"\n--- {key} ---\n{clipped}\n--- END {key} ---\n"
            
            chunk_payload = current_payload.get(key, {})
            ent_names = [e.get("name") for e in chunk_payload.get("entities", []) if e.get("name")]
            rel_pairs = [f"({r.get('source')}) --[{r.get('relation')}]--> ({r.get('target')})" 
                         for r in chunk_payload.get("relationships", []) if r.get("source") and r.get("target")]
            
            extracted_info += (
                f"\n--- {key} ALREADY EXTRACTED ---\n"
                f"Entities: {ent_names}\n"
                f"Relationships: {rel_pairs}\n"
                f"--- END {key} ALREADY EXTRACTED ---\n"
            )

        example_entries = []
        for i, key in enumerate(chunk_keys):
            if i == 0:
                example_entries.append(f'  "{key}": {{\n'
                                       f'    "entities": [\n'
                                       f'      {{\n'
                                       f'        "name": "Additional Missed Entity",\n'
                                       f'        "type": "ORGANIZATION",\n'
                                       f'        "description": "Description of the missed entity.",\n'
                                       f'        "aliases": [],\n'
                                       f'        "effective_date": null,\n'
                                       f'        "citation_text": null\n'
                                       f'      }}\n'
                                       f'    ],\n'
                                       f'    "relationships": [\n'
                                       f'      {{\n'
                                       f'        "source": "Additional Missed Entity",\n'
                                       f'        "relation": "PART_OF",\n'
                                       f'        "target": "Example Entity Name",\n'
                                       f'        "confidence": 0.85,\n'
                                       f'        "evidence": "evidence string",\n'
                                       f'        "description": "Description of the connection.",\n'
                                       f'        "keywords": ["missed", "relationship", "connection"]\n'
                                       f'      }}\n'
                                       f'    ]\n'
                                       f'  }}')
            else:
                example_entries.append(f'  "{key}": {{\n'
                                       f'    "entities": [],\n'
                                       f'    "relationships": []\n'
                                       f'  }}')
        output_format_example = "{\n" + ",\n".join(example_entries) + "\n}"

        return f"""You are an expert {self.domain} domain knowledge graph extraction system.
Review the text chunks below and the entities and relationships that have already been extracted from them.
Identify any additional Entities and directional Relationships that were MISSED in the first pass and list them.

=== STEP 1: RULES FOR MISSED ENTITIES ===
- Do NOT repeat any entities that are already extracted.
- Only extract genuine entities mentioned in the text that were not captured.
- Write a description and select type.
- Ensure all entity names and descriptions are written from an objective, third-person perspective.
  - NEVER use pronouns: this article, this paper, our company, I, you, he, she, we, they, my, or our.
  - Instead, explicitly name the subject: e.g. "The study by Smith et al.", "Apple Inc.", "The algorithm".
  - Example BAD: "This paper proposes a new method"
  - Example GOOD: "The paper by Chen (2023) proposes a new method"
- JSON FORMATTING & LaTeX ESCAPING (CRITICAL):
  - All string values must be properly escaped JSON strings.
  - Escape double quotes: " -> \\"
  - Escape backslashes: \ -> \\\\
  - Escape newlines: newline -> \\n
  - For LaTeX/math expressions, use double-escaped backslashes: e.g. \\frac must be written as `"\\\\frac"`, \\sum as `"\\\\sum"` inside the JSON.
  - Example: "formula": "The formula \\\\frac{1}{2} is used"
{entity_type_guidance}

=== STEP 2: RULES FOR MISSED RELATIONSHIPS ===
- Do NOT repeat any relationships that are already extracted.
- Extract any missed directional connections between the entities (including newly found entities or previously found entities).
- For each relationship, extract a list of 2-5 lowercase search terms or keywords summarizing the context of the relationship.
- N-ARY RELATIONSHIP DECOMPOSITION: If a statement describes a relationship involving 3 or more entities (an N-ary relationship), decompose it into multiple binary (two-entity) relationship pairs for separate description.
  - Example: "Alice, Bob, and Carol collaborated on the machine learning project"
    Extract:
    - (Alice, collaborated_with, Bob)
    - (Alice, collaborated_with, Carol)
    - (Bob, collaborated_with, Carol)
    - (Alice, worked_on, machine_learning_project)
    - (Bob, worked_on, machine_learning_project)
    - (Carol, worked_on, machine_learning_project)
- RELATIONSHIP RULES & SIGNIFICANCE:
  - Treat all relationships as undirected unless explicitly stated otherwise. Swapping the source and target entities does not constitute a new relationship. Avoid duplicate relationships. Prioritize and output relationships that are most significant to the core meaning of the input text first.
  - Example: If "A caused B" and "A is related to B" are both true, only include the more specific "caused".
{relation_type_guidance}

=== STEP 3: QUANTITY LIMITS ===
- Output at most {max_entities} entities and {max_relationships} relationships in total across all chunks in this response. Output fewer if fewer high-value items are present. Do not try to fill the limit.
- Only output relationships whose source and target entities are both included in the entities list of the respective chunk, or have been previously extracted in the first pass.

=== ALREADY EXTRACTED ===
{extracted_info}

CHUNKS:
{chunks_block}

OUTPUT FORMAT (strict JSON, no markdown):
{output_format_example}"""

    def extract_from_batch(
        self,
        batch: List[Tuple[Document, str]],
        *,
        llm_invoke: LLMInvoke,
        embedding_fn: Optional[EmbeddingFn] = None,
    ) -> List[Tuple[Document, Optional[GraphExtraction]]]:
        batch_input = [(text, doc.metadata) for doc, text in batch]
        prompt = self.build_batch_extraction_prompt(batch_input)
        self.extraction_attempts += 1

        batch_id = f"batch_{batch[0][0].metadata.get('chunk_id', '?')}"
        print(
            f"[START] Thread={threading.current_thread().name} "
            f"Batch={batch_id} Chunks={len(batch)} "
            f"Time={time.strftime('%H:%M:%S')}"
        )

        t0 = time.time()
        try:
            raw = str(llm_invoke(prompt, self.extraction_timeout * 2, self.extraction_retries))
        except Exception as exc:
            print(
                f"[END-ERR] Thread={threading.current_thread().name} "
                f"Batch={batch_id} Time={time.strftime('%H:%M:%S')} "
                f"Elapsed={time.time() - t0:.1f}s Error={exc}"
            )
            self.extraction_failures += 1
            return [(doc, None) for doc, _ in batch]

        payload, mode = self.parse_json_output(raw)
        t1 = time.time()

        print(
            f"[END] Thread={threading.current_thread().name} "
            f"Batch={batch_id} Time={time.strftime('%H:%M:%S')} "
            f"Elapsed={t1 - t0:.1f}s ParseMode={mode}"
        )

        if mode == "invalid":
            logging.getLogger("graph_rag").warning("[GraphRAG] Raw LLM response for %s (len=%d):\n%s", batch_id, len(raw), raw[:4000])

        if not isinstance(payload, dict):
            if self.extraction_retries > 0:
                repair_prompt = (
                    "Repair this into STRICT raw JSON: {\"chunk_X\": {\"entities\":[], \"relationships\":[]}, ...}.\n\n"
                    f"MODEL OUTPUT:\n{raw[:6000]}"
                )
                try:
                    repaired_raw = str(llm_invoke(repair_prompt, self.extraction_timeout, 0))
                    payload, mode = self.parse_json_output(repaired_raw)
                except Exception:
                    payload = None

        if isinstance(payload, dict):
            # Run Entity Gleaning (Phase 2.1)
            max_gleaning = getattr(config, "GRAPH_MAX_GLEANING", 1)
            if max_gleaning > 0:
                for glean_pass in range(max_gleaning):
                    glean_prompt = self.build_batch_gleaning_prompt(batch_input, payload)
                    try:
                        glean_raw = str(llm_invoke(glean_prompt, self.extraction_timeout * 2, 0))
                        glean_payload, glean_mode = self.parse_json_output(glean_raw)
                        if isinstance(glean_payload, dict):
                            for chunk_key, chunk_data in glean_payload.items():
                                if chunk_key in payload:
                                    # Merge entities
                                    existing_ents = payload[chunk_key].setdefault("entities", [])
                                    new_ents = chunk_data.get("entities", [])
                                    existing_names = {e.get("name", "").strip().lower() for e in existing_ents}
                                    for ent in new_ents:
                                        if ent.get("name", "").strip().lower() not in existing_names:
                                            existing_ents.append(ent)
                                            existing_names.add(ent.get("name", "").strip().lower())
                                            
                                    # Merge relationships
                                    existing_rels = payload[chunk_key].setdefault("relationships", [])
                                    new_rels = chunk_data.get("relationships", [])
                                    existing_keys = {
                                        (r.get("source", "").strip().lower(), r.get("relation", "").strip().upper(), r.get("target", "").strip().lower())
                                        for r in existing_rels
                                    }
                                    for rel in new_rels:
                                        rel_key = (rel.get("source", "").strip().lower(), rel.get("relation", "").strip().upper(), rel.get("target", "").strip().lower())
                                        if rel_key not in existing_keys:
                                            existing_rels.append(rel)
                                            existing_keys.add(rel_key)
                    except Exception as glean_exc:
                        print(f"[GraphRAG] Gleaning pass {glean_pass+1} failed: {glean_exc}")

        if not isinstance(payload, dict):
            self.extraction_failures += 1
            return [(doc, None) for doc, _ in batch]

        results = []
        cache_dir = os.getenv("EXTRACTION_CACHE_DIR", ".extraction_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_version = str(getattr(config, "GRAPH_CACHE_VERSION", 1))
        config_str = f"{self.domain}::{','.join(sorted(self.preferred_entity_types))}"

        for doc, text in batch:
            chunk_id = str(doc.metadata.get("chunk_id", "0"))
            chunk_key = f"chunk_{chunk_id}"
            chunk_payload = payload.get(chunk_key)
            if chunk_payload is None:
                results.append((doc, None))
                continue

            file_hash = str(doc.metadata.get("file_hash", "") or doc.metadata.get("source", ""))
            extraction = self.validate_extraction(
                chunk_payload,
                embedding_fn=embedding_fn,
                file_hash=file_hash,
                source_text=text,
            )
            extraction.llm_time_sec = (t1 - t0) / len(batch)
            extraction.is_cached = False

            # Cache individual chunk result
            hash_input = f"{cache_version}::{config_str}::{text or ''}"
            chunk_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
            cache_path = os.path.join(cache_dir, f"{chunk_hash}.json")
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(chunk_payload, f)
            except Exception:
                pass

            results.append((doc, extraction))
        return results

    def index_documents(
        self,
        documents: Iterable[Document],
        *,
        llm_invoke: LLMInvoke,
        embedding_fn: Optional[EmbeddingFn] = None,
        batch_embedding_fn: Optional[BatchEmbeddingFn] = None,
    ) -> dict:
        t_start = time.time()
        stats = {"indexed_chunks": 0, "skipped_chunks": 0, "failed_chunks": 0, "cache_hit_count": 0, "llm_extraction_time": 0.0, "neo4j_insertion_time": 0.0}
        if not self.available:
            return stats
            
        docs_list = list(documents)
        total_chunks = len(docs_list)
        docs = [doc for doc in docs_list if self._should_index_doc(doc)]
        filtered = total_chunks - len(docs)
        
        if not docs:
            return stats

        # Phase F: Cache-first
        cached_results = []
        uncached_docs = []
        cache_dir = os.getenv("EXTRACTION_CACHE_DIR", ".extraction_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_version = str(getattr(config, "GRAPH_CACHE_VERSION", 1))
        config_str = f"{self.domain}::{','.join(sorted(self.preferred_entity_types))}"

        for doc in docs:
            text = doc.page_content or ""
            file_hash = str(doc.metadata.get("file_hash") or doc.metadata.get("source") or "")
            hash_input = f"{cache_version}::{config_str}::{text}"
            chunk_hash = hashlib.md5(hash_input.encode("utf-8")).hexdigest()
            cache_path = os.path.join(cache_dir, f"{chunk_hash}.json")

            if os.path.exists(cache_path):
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        payload = json.load(f)
                    extraction = self.validate_extraction(
                        payload, embedding_fn=embedding_fn,
                        file_hash=file_hash, source_text=text,
                    )
                    extraction.is_cached = True
                    extraction.llm_time_sec = 0.0
                    cached_results.append((doc, extraction))
                    stats["cache_hit_count"] += 1
                    continue
                except Exception:
                    pass
            uncached_docs.append(doc)

        print(f"[GraphRAG] {len(docs)} eligible chunks: "
              f"{len(cached_results)} cached, {len(uncached_docs)} need LLM extraction")

        batch_size = getattr(config, "GRAPH_BATCH_SIZE", 3)
        batches = []
        for i in range(0, len(uncached_docs), batch_size):
            batch = [(doc, doc.page_content or "") for doc in uncached_docs[i:i + batch_size]]
            batches.append(batch)

        print(f"[GraphRAG] {len(uncached_docs)} chunks → {len(batches)} batches (size {batch_size})")

        t0_extract = time.time()
        extractions = list(cached_results)
        workers = getattr(config, "GRAPH_EXTRACTION_WORKERS", 4)
        print(f"[GraphRAG] Workers = {workers}, Batches = {len(batches)}, Starting extraction at {time.strftime('%H:%M:%S')}")
        executor = ThreadPoolExecutor(max_workers=workers)
        future_to_batch = {
            executor.submit(
                self.extract_from_batch, batch,
                llm_invoke=llm_invoke, embedding_fn=embedding_fn,
            ): batch
            for batch in batches
        }
        future_started_at = {future: time.time() for future in future_to_batch}
        batch_timeout_sec = max(1.0, self.extraction_timeout * 2 + 10)
        overall_deadline = time.time() + 600
        try:
            pending = set(future_to_batch.keys())
            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

                for future in done:
                    try:
                        batch_results = future.result()
                        for doc, ext in batch_results:
                            if ext is not None and not ext.is_cached:
                                stats["llm_extraction_time"] += ext.llm_time_sec
                        extractions.extend(batch_results)
                    except Exception as exc:
                        import traceback
                        print(f"[GraphRAG] Batch extraction exception: {exc}")
                        print(traceback.format_exc())
                        raise

                now = time.time()
                for future in list(pending):
                    elapsed = now - future_started_at[future]
                    if elapsed > batch_timeout_sec:
                        batch = future_to_batch[future]
                        print(
                            f"[GraphRAG] Batch timed out after {int(batch_timeout_sec)}s "
                            f"(chunks={len(batch)}); aborting graph extraction",
                            flush=True,
                        )
                        stats["failed_chunks"] += len(batch)
                        future.cancel()
                        raise TimeoutError(
                            f"GraphRAG batch exceeded timeout of {batch_timeout_sec:.1f}s"
                        )

                if now > overall_deadline:
                    print(f"[GraphRAG] Overall batch extraction timed out after 600s", flush=True)
                    raise TimeoutError("GraphRAG overall batch extraction timed out after 600s")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        extract_time = time.time() - t0_extract

        # Step 1.5: Global Entity Embedding Deduplication
        t0_embed = time.time()
        unique_names = set()
        for doc, extraction in extractions:
            if extraction and extraction.entities:
                for entity in extraction.entities:
                    unique_names.add(entity.name_norm)
                    
        name_to_embedding = {}
        if self.semantic_merge_enabled and unique_names:
            names_list = list(unique_names)
            if batch_embedding_fn:
                try:
                    vectors = batch_embedding_fn(names_list)
                    for n, v in zip(names_list, vectors):
                        name_to_embedding[n] = [float(x) for x in v]
                except Exception:
                    pass
            elif embedding_fn:
                for name in names_list:
                    try:
                        name_to_embedding[name] = [float(v) for v in embedding_fn(name)]
                    except Exception:
                        name_to_embedding[name] = []
                    
        for doc, extraction in extractions:
            if extraction and extraction.entities:
                for entity in extraction.entities:
                    entity.embedding = name_to_embedding.get(entity.name_norm, [])
                    
        embed_time = time.time() - t0_embed

        successful_llm_chunks = len([e for d, e in extractions if e is not None and not e.is_cached])
        if successful_llm_chunks > 0:
            stats["llm_extraction_time"] = round(stats["llm_extraction_time"] / successful_llm_chunks, 2)

        # Step 2: Sequential Neo4j Insertion (Neo4j sessions are not thread-safe)
        t0_neo4j = time.time()
        total_entities = 0
        total_relationships = 0
        try:
            with self.driver.session(database=self.database) as session:
                for doc, extraction in extractions:
                    if extraction is None:
                        stats["failed_chunks"] += 1
                        continue
                    if not extraction.entities:
                        stats["skipped_chunks"] += 1
                        continue
                    self._upsert_chunk_graph(session, doc, extraction, llm_invoke)
                    total_entities += len(extraction.entities)
                    total_relationships += len(extraction.relationships)
                    stats["indexed_chunks"] += 1
            stats["neo4j_insertion_time"] = round(time.time() - t0_neo4j, 2)
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[GraphRAG] Graph indexing stopped; Qdrant ingestion remains valid: {exc}")

        # Step 3: Write latest_graph to tmp
        try:
            os.makedirs("tmp", exist_ok=True)
            with open("tmp/latest_graph.md", "w", encoding="utf-8") as f:
                for doc, extraction in extractions:
                    if extraction is None or not extraction.entities:
                        continue
                    f.write(f"## Document Chunk: {doc.metadata.get('chunk_id', 'Unknown')}\n\n")
                    f.write("### Entities\n")
                    for e in extraction.entities:
                        f.write(f"- **{e.name}** ({e.type})\n")
                    f.write("\n### Relationships\n")
                    if extraction.relationships:
                        for r in extraction.relationships:
                            f.write(f"- {r.source} -> **{r.relation}** -> {r.target} (Conf: {r.confidence})\n")
                            if r.evidence:
                                f.write(f"  Evidence: {r.evidence}\n")
                    else:
                        f.write("- None\n")
                    f.write("\n---\n\n")
        except Exception as e:
            print(f"[GraphRAG] Failed to write latest_graph.md: {e}")

        failure_rate = (
            self.extraction_failures / self.extraction_attempts
            if self.extraction_attempts
            else 0.0
        )
        if failure_rate > config.GRAPH_FAILURE_RATE_THRESHOLD:
            print(f"[GraphRAG] Warning: extraction failure rate is {failure_rate:.1%}.")
        # Step 4: Sync Qdrant entity and relationship description collections
        if self.available and batch_embedding_fn:
            try:
                from qdrant_storage.store import QdrantManager
                qdrant_mgr = QdrantManager()
                self.sync_vdb(qdrant_mgr.get_client(), batch_embedding_fn)
            except Exception as e:
                print(f"[GraphRAG] Failed to sync Qdrant collections: {e}")
            
        total_time = time.time() - t_start
        mins, secs = divmod(total_time, 60)
        neo4j_time = stats.get("neo4j_insertion_time", 0.0)
        
        print("\nGraphRAG Stats")
        print("--------------")
        print(f"Chunks total: {total_chunks}")
        print(f"Filtered: {filtered}")
        print(f"Cached: {len(cached_results)}")
        print(f"Processed: {len(uncached_docs)}")
        print(f"\nBatch size: {batch_size}")
        print(f"LLM calls: {len(batches)}")
        print(f"\nEntities: {total_entities}")
        print(f"Relationships: {total_relationships}")
        print(f"\nTotal LLM Extraction Time: {extract_time:.2f}s")
        print(f"Total Embedding Time:      {embed_time:.2f}s")
        print(f"Total Neo4j Time:          {neo4j_time:.2f}s")
        print(f"Total Pipeline Time:       {total_time:.2f}s")
        print(f"Total time (m/s): {int(mins)}m {int(secs)}s\n")
        
        return stats

    def extract_query_entities(
        self,
        question: str,
        *,
        llm_invoke: Optional[LLMInvoke] = None,
    ) -> List[str]:
        if not question:
            return []
        if llm_invoke is not None:
            # Build entity type hint for query extraction
            if self.preferred_entity_types:
                type_hint = "Preferred types: " + ", ".join(self.preferred_entity_types[:12])
            else:
                type_hint = "Use short, general UPPER_SNAKE_CASE types (e.g., PERSON, COURT, ACT, ORGANIZATION)."

            prompt = f"""You are a high-precision {self.domain} domain entity extractor.
Your goal is to identify the core named entities mentioned in the user's question for graph-based retrieval.

RULES:
1. Extract SPECIFIC named entities — organizations, people, laws, standards, concepts, etc.
2. DO NOT extract generic nouns (e.g., "rules", "orders", "person", "notification", "document").
3. IGNORE conversational filler.
4. Assign each entity a type in UPPER_SNAKE_CASE.
   {type_hint}
5. Use short, general types. Do NOT invent verbose synonyms.
6. If no specific entities are found, return an empty list.

OUTPUT FORMAT:
Return ONLY valid JSON.
{{
  "entities": [
    {{"name": "MSME Act", "type": "ACT"}},
    {{"name": "Section 15", "type": "SECTION"}}
  ]
}}

Question: {question}"""
            try:
                raw = str(llm_invoke(prompt, min(self.extraction_timeout, 20.0), 0))
                payload, _ = self.parse_json_output(raw)
                if payload:
                    extraction = self.validate_extraction(
                        {"entities": payload.get("entities", [])},
                        file_hash="query",
                        source_text=question,
                    )
                    names = [entity.name_norm for entity in extraction.entities]
                    if names:
                        return names[: self.max_query_entities]
            except Exception:
                pass

        fallback = []
        for match in re.finditer(r"\bsection\s+\d+[a-z]?\b", question, flags=re.IGNORECASE):
            fallback.append(self.normalize_entity_name(match.group(0)))
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9&.\-]{2,}", question)
        for token in tokens:
            norm = self.normalize_entity_name(token)
            if norm in self.generic_query_terms:
                continue
            if norm and norm not in fallback:
                fallback.append(norm)
            if len(fallback) >= self.max_query_entities:
                break
        return fallback[: self.max_query_entities]

    def sync_vdb(self, qdrant_client, batch_embedding_fn: BatchEmbeddingFn) -> None:
        if not self.available:
            return
            
        entity_collection = config.QDRANT_COLLECTION_NAME + "_entities"
        relationship_collection = config.QDRANT_COLLECTION_NAME + "_relationships"
        
        # 1. Fetch all entities from Neo4j
        entities = []
        try:
            with self.driver.session(database=self.database) as session:
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    RETURN e.entity_key AS entity_key,
                           e.name AS name,
                           e.type AS type,
                           coalesce(e.description, '') AS description
                    """
                )
                for row in rows:
                    entities.append({
                        "entity_key": row["entity_key"],
                        "name": row["name"],
                        "type": row["type"],
                        "description": row["description"]
                    })
        except Exception as e:
            logger.error("Failed to fetch entities for VDB sync: %s", e)
            return

        # 2. Fetch all relationships from Neo4j
        relationships = []
        try:
            with self.driver.session(database=self.database) as session:
                rows = session.run(
                    """
                    MATCH (a:Entity)-[r]->(b:Entity)
                    RETURN a.entity_key AS source_key,
                           a.name AS source_name,
                           type(r) AS relation_type,
                           b.entity_key AS target_key,
                           b.name AS target_name,
                           coalesce(r.description, '') AS description,
                           coalesce(r.keywords, []) AS keywords
                    """
                )
                for row in rows:
                    rel_key = f"{row['source_key']}--{row['relation_type']}--{row['target_key']}"
                    relationships.append({
                        "relationship_key": rel_key,
                        "source_name": row["source_name"],
                        "relation_type": row["relation_type"],
                        "target_name": row["target_name"],
                        "description": row["description"],
                        "keywords": row["keywords"]
                    })
        except Exception as e:
            logger.error("Failed to fetch relationships for VDB sync: %s", e)
            return

        # Clear collections
        try:
            qdrant_client.delete_collection(entity_collection)
        except Exception:
            pass
        try:
            qdrant_client.delete_collection(relationship_collection)
        except Exception:
            pass
            
        # Recreate them
        from qdrant_client.models import VectorParams, Distance
        qdrant_client.create_collection(
            collection_name=entity_collection,
            vectors_config={
                "dense": VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE)
            }
        )
        qdrant_client.create_collection(
            collection_name=relationship_collection,
            vectors_config={
                "dense": VectorParams(size=config.EMBED_DIMENSIONS, distance=Distance.COSINE)
            }
        )
        
        # Embed and upload entities in batches
        if entities:
            texts_to_embed = [
                f"Entity: {e['name']} (Type: {e['type']}). Description: {e['description']}"
                for e in entities
            ]
            try:
                embeddings = batch_embedding_fn(texts_to_embed)
                
                from qdrant_client.models import PointStruct
                import uuid
                
                points = []
                for i, ent in enumerate(entities):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ent["entity_key"]))
                    points.append(PointStruct(
                        id=point_id,
                        vector={"dense": embeddings[i]},
                        payload={
                            "entity_key": ent["entity_key"],
                            "name": ent["name"],
                            "type": ent["type"],
                            "description": ent["description"],
                            "text": texts_to_embed[i]
                        }
                    ))
                
                qdrant_client.upsert(
                    collection_name=entity_collection,
                    points=points,
                    wait=True
                )
            except Exception as e:
                logger.error("Failed to embed/upload entities during sync: %s", e)
            
        # Embed and upload relationships in batches
        if relationships:
            texts_to_embed = []
            for r in relationships:
                kw_str = f" Keywords: {', '.join(r['keywords'])}." if r["keywords"] else ""
                texts_to_embed.append(
                    f"Relationship: {r['source_name']} -> {r['relation_type']} -> {r['target_name']}. Description: {r['description']}{kw_str}"
                )
            try:
                embeddings = batch_embedding_fn(texts_to_embed)
                
                from qdrant_client.models import PointStruct
                import uuid
                
                points = []
                for i, rel in enumerate(relationships):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, rel["relationship_key"]))
                    points.append(PointStruct(
                        id=point_id,
                        vector={"dense": embeddings[i]},
                        payload={
                            "relationship_key": rel["relationship_key"],
                            "source_name": rel["source_name"],
                            "relation_type": rel["relation_type"],
                            "target_name": rel["target_name"],
                            "description": rel["description"],
                            "keywords": rel["keywords"],
                            "text": texts_to_embed[i]
                        }
                    ))
                    
                qdrant_client.upsert(
                    collection_name=relationship_collection,
                    points=points,
                    wait=True
                )
            except Exception as e:
                logger.error("Failed to embed/upload relationships during sync: %s", e)
            
        logger.info("✓ Entity and Relationship VDB sync complete: %d entities, %d relationships", len(entities), len(relationships))

    def graph_score(self, hops: int, mention_strength: float = 1.0) -> float:
        """Score a graph candidate using confidence and frequency logic.
        Distance-based hop decay differentiates close vs distant connections."""
        h = int(hops)
        if h <= 1:
            hop_decay = 1.0
        elif h == 2:
            hop_decay = 0.65
        else:
            hop_decay = 0.4
        return hop_decay * float(mention_strength)

    def retrieve_candidates(
        self,
        question: str,
        *,
        mode: str = "local",
        llm_invoke: Optional[LLMInvoke] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
    ) -> Tuple[List[GraphCandidate], List[Dict[str, Any]]]:
        if not self.available:
            return [], []

        entity_keys = []
        relationships = []

        # Try dual-level semantic search first using Qdrant (Phase 1.2)
        entity_collection = config.QDRANT_COLLECTION_NAME + "_entities"
        relationship_collection = config.QDRANT_COLLECTION_NAME + "_relationships"

        if embedding_fn and mode != "global":
            try:
                from qdrant_storage.store import QdrantManager
                qdrant_mgr = QdrantManager()
                client = qdrant_mgr.get_client()

                # Verify collections exist in Qdrant
                collections = [c.name for c in client.get_collections().collections]
                if entity_collection in collections and relationship_collection in collections:
                    query_vector = embedding_fn(question)

                    # Search entities VDB (low-level retrieval)
                    ent_hits = client.query_points(
                        collection_name=entity_collection,
                        query=query_vector,
                        using="dense",
                        limit=self.max_query_entities,
                        with_payload=True
                    ).points

                    for hit in ent_hits:
                        key = hit.payload.get("entity_key")
                        if key and key not in entity_keys:
                            entity_keys.append(key)

                    # Search relationships VDB (high-level retrieval)
                    rel_hits = client.query_points(
                        collection_name=relationship_collection,
                        query=query_vector,
                        using="dense",
                        limit=self.max_query_entities,
                        with_payload=True
                    ).points

                    for hit in rel_hits:
                        rel_key = hit.payload.get("relationship_key")
                        if rel_key:
                            # rel_key format is: source_key--relation_type--target_key
                            parts = rel_key.split("--")
                            if len(parts) == 3:
                                src_key, _, tgt_key = parts
                                if src_key and src_key not in entity_keys:
                                    entity_keys.append(src_key)
                                if tgt_key and tgt_key not in entity_keys:
                                    entity_keys.append(tgt_key)

                    print(f"[GraphRAG] Dual-level retrieval found {len(entity_keys)} entity keys from VDB search: {entity_keys}")
            except Exception as e:
                print(f"[GraphRAG] Dual-level semantic retrieval failed: {e}")

        # Fallback to LLM entity extraction if VDB search yielded no keys (or VDB not configured)
        if not entity_keys and mode != "global":
            print("[GraphRAG] Falling back to LLM entity extraction...")
            query_entities = self.extract_query_entities(question, llm_invoke=llm_invoke)
            if query_entities:
                print(f"[GraphRAG] LLM extracted query entities: {query_entities}")

                # Find the entity keys in Neo4j matching these normalized names
                try:
                    with self.driver.session(database=self.database) as session:
                        rows = session.run(
                            """
                            MATCH (e:Entity)
                            WHERE e.name_norm IN $query_entities
                            RETURN e.entity_key AS entity_key
                            """,
                            query_entities=query_entities
                        )
                        entity_keys = [row["entity_key"] for row in rows]
                except Exception as e:
                    print(f"[GraphRAG] Failed to map extracted entities to keys: {e}")

        print(f"[GraphRAG] Querying Neo4j using mode={mode}")
        try:
            with self.driver.session(database=self.database) as session:
                # ── Step 1: Query matched relationships for evidence context ──
                if mode in ("local", "hybrid") and entity_keys:
                    rel_rows = session.run(
                        """
                        MATCH (start:Entity)-[r]->(related:Entity)
                        WHERE start.entity_key IN $entity_keys OR related.entity_key IN $entity_keys
                        RETURN start.name AS source, related.name AS target, type(r) AS relation, coalesce(r.description, '') AS description
                        LIMIT 50
                        """,
                        entity_keys=entity_keys
                    )
                    for r in rel_rows:
                        relationships.append({
                            "source": r["source"],
                            "target": r["target"],
                            "relation": r["relation"],
                            "description": r["description"]
                        })

                if mode in ("global", "hybrid"):
                    global_rel_rows = session.run(
                        """
                        MATCH (start:Entity)-[r]->(related:Entity)
                        RETURN start.name AS source, related.name AS target, type(r) AS relation, coalesce(r.description, '') AS description, coalesce(r.count, 1) AS count
                        ORDER BY count DESC
                        LIMIT 30
                        """
                    )
                    for r in global_rel_rows:
                        rel_data = {
                            "source": r["source"],
                            "target": r["target"],
                            "relation": r["relation"],
                            "description": r["description"]
                        }
                        if rel_data not in relationships:
                            relationships.append(rel_data)

                # ── Step 2: Retrieve document candidates ──
                candidates_map = {}
                cypher_queries = []

                if mode in ("local", "hybrid") and entity_keys:
                    cypher_queries.append((
                        """
                        MATCH (start:Entity)
                        WHERE start.entity_key IN $entity_keys
                        MATCH path = (start)-[*1..3]-(related:Entity)
                        WITH related, path, length(path) AS hops
                        MATCH (related)<-[:MENTIONS]-(c:Chunk)
                        RETURN c.source AS source,
                               c.chunk_id AS chunk_id,
                               c.chunk_key AS chunk_key,
                               hops,
                               [node IN nodes(path) | coalesce(node.name, node.name_norm, '')] AS path_names,
                               [rel IN relationships(path) | type(rel)] AS relation_types,
                               [rel IN relationships(path) | coalesce(rel.confidence, 1.0)] AS relation_confidences,
                               [rel IN relationships(path) | coalesce(rel.count, 1)] AS relation_counts
                        ORDER BY hops ASC
                        LIMIT $max_graph_chunks
                        """,
                        {"entity_keys": entity_keys, "max_graph_chunks": self.max_graph_chunks}
                    ))

                if mode in ("global", "hybrid"):
                    cypher_queries.append((
                        """
                        MATCH (start:Entity)-[r]->(related:Entity)
                        WITH start, related, r
                        ORDER BY coalesce(r.count, 1) DESC
                        LIMIT 15
                        MATCH (start)<-[:MENTIONS]-(c:Chunk)
                        RETURN c.source AS source,
                               c.chunk_id AS chunk_id,
                               c.chunk_key AS chunk_key,
                               1 AS hops,
                               [start.name, related.name] AS path_names,
                               [type(r)] AS relation_types,
                               [coalesce(r.confidence, 1.0)] AS relation_confidences,
                               [coalesce(r.count, 1)] AS relation_counts
                        LIMIT $max_graph_chunks
                        """,
                        {"max_graph_chunks": self.max_graph_chunks}
                    ))

                for query_str, params in cypher_queries:
                    rows = session.run(query_str, **params)
                    for row in rows:
                        source = str(row.get("source") or "Unknown")
                        chunk_id = str(row.get("chunk_id") or "0")
                        key = (source, chunk_id)
                        
                        hops = int(row.get("hops") or 1)
                        path_names = row.get("path_names") or []
                        rel_types = row.get("relation_types") or []
                        rel_confidences = row.get("relation_confidences") or []
                        rel_counts = row.get("relation_counts") or []
                        
                        # Format: (Entity1) --[REL]--> (Entity2)
                        if path_names and rel_types:
                            path_str = ""
                            for idx in range(len(rel_types)):
                                if idx < len(path_names):
                                    path_str += f"({path_names[idx]}) --[{rel_types[idx]}]--> "
                            if path_names:
                                path_str += f"({path_names[-1]})"
                        else:
                            path_str = "unknown path"
                        
                        if key not in candidates_map:
                            # Calculate path strength as the minimum edge strength along the path
                            edge_strengths = []
                            for conf, count in zip(rel_confidences, rel_counts):
                                edge_strength = float(conf) * math.log10(float(count) + 9.0)
                                edge_strengths.append(edge_strength)
                            
                            path_mention_strength = min(edge_strengths) if edge_strengths else 1.0
                            score = self.graph_score(hops, mention_strength=path_mention_strength)
                            
                            candidates_map[key] = GraphCandidate(
                                source=source,
                                chunk_id=chunk_id,
                                graph_score=score,
                                graph_hops=hops,
                                graph_path=path_str
                            )
                        else:
                            # Append additional paths if they provide new info
                            if path_str not in candidates_map[key].graph_path:
                                candidates_map[key].graph_path += f" | {path_str}"

                candidates = list(candidates_map.values())
                print(f"[GraphRAG] Retrieved {len(candidates)} reasoning paths with mode={mode}.")
                return candidates, relationships
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[GraphRAG] Graph retrieval failed; continuing with Qdrant only: {exc}")
            return [], []



    def clear(self) -> None:
        with self._lock:
            self.canonical_entities = {}
            self._dynamic_aliases = {}
        if not self.available:
            return
        try:
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[GraphRAG] Clear failed: {exc}")

    def cleanup_orphan_entities(self) -> int:
        if not self.available:
            return 0
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE NOT (:Chunk)-[:MENTIONS]->(e)
                    WITH collect(e) AS entities, count(e) AS count
                    FOREACH (entity IN entities | DETACH DELETE entity)
                    RETURN count
                    """
                ).single()
                return int(result["count"] or 0) if result else 0
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[GraphRAG] Orphan cleanup failed: {exc}")
            return 0

    def export_to_markdown(self) -> str:
        """Exports the entire graph to a Markdown formatted string."""
        if not self.available:
            return "Graph database not available."
        
        try:
            with self.driver.session(database=self.database) as session:
                # Get all entities
                entity_rows = session.run("MATCH (e:Entity) RETURN e.name as name, e.type as type ORDER BY e.type, e.name")
                entities = list(entity_rows)
                
                # Get all relationships
                rel_rows = session.run("""
                    MATCH (s:Entity)-[r]->(t:Entity) 
                    RETURN s.name as source, type(r) as relation, t.name as target 
                    ORDER BY source, relation
                """)
                rels = list(rel_rows)
                
                md = "# Knowledge Graph Export\n\n"
                md += "> [!NOTE]\n"
                md += "> This report displays the semantic Knowledge Graph (Entity-to-Entity reasoning paths). \n"
                md += "> Internal indexing relationships (Chunk MENTIONS Entity) are excluded for clarity.\n\n"
                
                md += "## Entities (%d)\n" % len(entities)
                md += "| Name | Type |\n"
                md += "| :--- | :--- |\n"
                for row in entities:
                    md += f"| {row['name']} | {row['type']} |\n"
                
                md += "\n## Relationships (%d)\n" % len(rels)
                md += "| Source | Relation | Target |\n"
                md += "| :--- | :--- | :--- |\n"
                for row in rels:
                    md += f"| {row['source']} | {row['relation']} | {row['target']} |\n"
                
                return md
        except Exception as exc:
            return f"Export failed: {exc}"
