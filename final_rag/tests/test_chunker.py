# tests/test_chunker.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from final_rag.ingestion.parser import BlockRecord, PageRecord, ParseResult, ExtractionMethod, DocumentMeta
from final_rag.ingestion.chunker import DocumentChunker


def load_md_as_parse_result(md_path: Path) -> ParseResult:
    text = md_path.read_text(encoding="utf-8")
    lines = text.split("\n")

    blocks = []
    pages = []
    current_section = ""
    page_no = 1

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("## Page"):
            try:
                page_no = int(stripped.replace("## Page", "").strip())
            except ValueError:
                pass
            continue

        if stripped.startswith("### "):
            current_section = stripped.lstrip("#").strip()
            blocks.append(BlockRecord(
                block_type="heading",
                content=current_section,
                page_no=page_no,
                page_label=str(page_no),
                section=current_section,
                is_table=False,
            ))

        elif stripped.startswith("|"):
            # accumulate table lines
            if blocks and blocks[-1].is_table:
                blocks[-1].content += "\n" + stripped
            else:
                blocks.append(BlockRecord(
                    block_type="table",
                    content=stripped,
                    page_no=page_no,
                    page_label=str(page_no),
                    section=current_section,
                    is_table=True,
                ))

        else:
            blocks.append(BlockRecord(
                block_type="text",
                content=stripped,
                page_no=page_no,
                page_label=str(page_no),
                section=current_section,
                is_table=False,
            ))

    # build a single page record per unique page_no
    seen_pages = {}
    for b in blocks:
        if b.page_no not in seen_pages:
            seen_pages[b.page_no] = []
        seen_pages[b.page_no].append(b.content)

    for pno, contents in sorted(seen_pages.items()):
        pages.append(PageRecord(
            page_no=pno,
            page_label=str(pno),
            text=" ".join(contents),
        ))

    meta = DocumentMeta(
        doc_id="test_md_001",
        file_name=md_path.name,
        file_path=str(md_path),
        file_type=".md",
        file_size_kb=md_path.stat().st_size / 1024,
        page_count=len(pages),
        has_tables=any(b.is_table for b in blocks),
        parse_success=True,
    )

    return ParseResult(
        file_name=md_path.name,
        file_type=".md",
        method_used=ExtractionMethod.YOLO,
        markdown=text,
        meta=meta,
        total_pages=len(pages),
        success=True,
        pages=pages,
        blocks=blocks,
        doc_id="test_md_001",
        doc_year="",
        filename_tokens=["test"],
        summary="",
        keywords=[],
    )


def main():
    md_path = Path(__file__).parent.parent / "md_output" / "test.md"

    if not md_path.exists():
        print(f"[ERROR] File not found: {md_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Loading: {md_path}")
    print(f"{'='*60}")

    parse_result = load_md_as_parse_result(md_path)

    print(f"  Pages   : {parse_result.total_pages}")
    print(f"  Blocks  : {len(parse_result.blocks)}")
    print(f"  Tables  : {sum(1 for b in parse_result.blocks if b.is_table)}")

    chunker = DocumentChunker()
    chunks = chunker.chunk(parse_result)

    print(f"\n{'='*60}")
    print(f"CHUNKS — total: {len(chunks)}")
    print(f"{'='*60}")

    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} | page={chunk.page_no} | tokens={chunk.token_count} | table={chunk.is_table} | section='{chunk.section}' ---")
        print(chunk.text)
        print()

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Total chunks : {len(chunks)}")
    print(f"  Table chunks : {sum(1 for c in chunks if c.is_table)}")
    print(f"  Text chunks  : {sum(1 for c in chunks if not c.is_table)}")
    avg_tokens = sum(c.token_count for c in chunks) / len(chunks) if chunks else 0
    print(f"  Avg tokens   : {avg_tokens:.1f}")


if __name__ == "__main__":
    main()