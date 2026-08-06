#!/usr/bin/env python3
"""
ingest_all_files.py
Batch ingestion pipeline for processing all PDFs from 'all_files' into final_rag (V2 Engine).
Links PDF files across doc_input, backup_pdf, RunPod S3 network volume, MongoDB metadata, and Qdrant payloads.
"""

import os
import sys
import glob
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingest_all_files")

# Base directory setup
COMBINE_DIR = Path(__file__).parent.resolve()
sys.path.append(str(COMBINE_DIR))

# Import final_rag and unified_db modules
import final_rag.config as config
from final_rag.qdrant_storage.store import QdrantManager
from final_rag.ingestion.parser import DocumentParser
from final_rag.ingestion.chunker import DocumentChunker
from final_rag.ingestion.embedder import get_embedder
from final_rag.db.database import create_tables, insert_document, update_document_status

try:
    import unified_db
    save_document_metadata = unified_db.save_document_metadata
except Exception as db_err:
    logger.warning(f"Could not import unified_db: {db_err}")
    save_document_metadata = None

try:
    from s3_uploader import upload_pdf_to_s3
except ImportError:
    upload_pdf_to_s3 = None


def run_batch_ingestion(source_folder: str, limit: int = None):
    source_path = Path(source_folder)
    if not source_path.exists():
        logger.error(f"Source folder does not exist: {source_folder}")
        return

    # Find all PDF files
    pdf_files = list(source_path.glob("*.pdf")) + list(source_path.glob("*.PDF"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {source_folder}")
        return

    if limit:
        pdf_files = pdf_files[:limit]

    logger.info(f"--- Starting batch ingestion of {len(pdf_files)} PDF documents ---")

    # Initialize directories
    doc_input_dir = config.DOC_INPUT_DIR
    md_output_dir = config.MD_OUTPUT_DIR
    backup_pdf_dir = COMBINE_DIR / "backup_pdf"
    backup_md_dir = COMBINE_DIR / "backup_markdown"

    os.makedirs(doc_input_dir, exist_ok=True)
    os.makedirs(md_output_dir, exist_ok=True)
    os.makedirs(backup_pdf_dir, exist_ok=True)
    os.makedirs(backup_md_dir, exist_ok=True)

    # Initialize DB & RAG engine components
    create_tables()
    qdrant_mgr = QdrantManager()
    qdrant_mgr.setup_database()
    embedder = get_embedder(db=qdrant_mgr)
    parser = DocumentParser(output_dir=md_output_dir)
    chunker = DocumentChunker()

    stats = {
        "total": len(pdf_files),
        "success": 0,
        "failed": 0,
        "total_chunks": 0,
    }

    for idx, pdf_path in enumerate(pdf_files, 1):
        file_name = pdf_path.name
        stem = pdf_path.stem
        logger.info(f"[{idx}/{len(pdf_files)}] Processing: {file_name}")

        try:
            # 1. Read file & compute MD5 hash
            with open(pdf_path, "rb") as f:
                file_bytes = f.read()
            file_hash = hashlib.md5(file_bytes).hexdigest()

            # 2. Copy PDF to final_rag/doc_input and backup_pdf
            input_pdf_path = doc_input_dir / file_name
            backup_pdf_path = backup_pdf_dir / file_name
            shutil.copy2(pdf_path, input_pdf_path)
            shutil.copy2(pdf_path, backup_pdf_path)

            # 3. Optional S3 Upload to RunPod network volume
            s3_key = f"pdf_storage/{file_name}"
            s3_uploaded = False
            if upload_pdf_to_s3:
                try:
                    s3_uploaded = upload_pdf_to_s3(str(backup_pdf_path), s3_key)
                except Exception as s3_err:
                    logger.warning(f"S3 upload skipped/failed for {file_name}: {s3_err}")

            # 4. Save metadata to MongoDB
            if save_document_metadata:
                save_document_metadata(
                    file_name=file_name,
                    doc_type="PDF",
                    source="all_files",
                    source_description="Batch ingested from all_files directory",
                    creation_date=datetime.utcnow().isoformat(),
                    rag_version="version2",
                    file_hash=file_hash,
                    pdf_path=f"final_rag/doc_input/{file_name}",
                    pdf_backup_path=str(backup_pdf_path),
                    s3_key=s3_key if s3_uploaded else None,
                )

            # Register document in final_rag database tracking
            doc_id = f"doc_{file_hash[:12]}"
            insert_document(
                document_id=doc_id,
                file_name=file_name,
                doc_type="PDF",
                file_data=file_bytes,
                status="processing",
                file_hash=file_hash,
                pdf_path=f"final_rag/doc_input/{file_name}",
                s3_key=s3_key if s3_uploaded else None,
            )

            # 5. Parse document into Markdown
            parsed_doc = parser.parse(str(input_pdf_path))
            if not parsed_doc or not parsed_doc.markdown:
                logger.error(f"Failed to parse content for {file_name}")
                update_document_status(file_name, "failed")
                stats["failed"] += 1
                continue

            # Copy parsed markdown to backup_markdown archive
            md_backup_path = backup_md_dir / f"{stem}_v2.md"
            with open(md_backup_path, "w", encoding="utf-8") as f_md:
                f_md.write(parsed_doc.markdown)

            # 6. Chunk markdown content
            chunks = chunker.chunk_document(parsed_doc)
            if not chunks:
                logger.warning(f"No chunks generated for {file_name}")
                update_document_status(file_name, "completed")
                stats["success"] += 1
                continue

            # Attach PDF pointers to chunks
            for c in chunks:
                c.source_file = file_name
                c.doc_id = doc_id

            # 7. Embed and store vectors in Qdrant
            # Pass source_pdf and pdf_path in kwargs to set in EmbeddedChunk
            embedded_chunks = embedder.embed_chunks(
                chunks,
                source_pdf=file_name,
                pdf_path=f"final_rag/doc_input/{file_name}"
            )
            count = embedder.store_chunks(embedded_chunks)
            update_document_status(file_name, "completed")

            stats["success"] += 1
            stats["total_chunks"] += count
            logger.info(f"Successfully processed '{file_name}' | Chunks embedded: {count}")

        except Exception as e:
            logger.error(f"Error processing {file_name}: {e}", exc_info=True)
            update_document_status(file_name, "failed")
            stats["failed"] += 1

    logger.info("=== Batch Ingestion Summary ===")
    logger.info(f"Total PDFs: {stats['total']}")
    logger.info(f"Success: {stats['success']}")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Total Chunks Embedded: {stats['total_chunks']}")


if __name__ == "__main__":
    import argparse
    parser_args = argparse.ArgumentParser(description="Batch ingest all PDFs into final_rag")
    parser_args.add_argument("--source", type=str, default=r"c:\Users\saib\Downloads\all_files", help="Source folder containing PDFs")
    parser_args.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    args = parser_args.parse_args()

    run_batch_ingestion(args.source, args.limit)
