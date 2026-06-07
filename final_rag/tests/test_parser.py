# tests/test_parser.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from final_rag.ingestion.parser import DocumentParser
from final_rag.config import DOC_INPUT_DIR, MD_OUTPUT_DIR


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_metadata(result):
    print_section("ParseResult — Top Level")
    print(f"  file_name           : {result.file_name}")
    print(f"  file_type           : {result.file_type}")
    print(f"  method_used         : {result.method_used}")
    print(f"  success             : {result.success}")
    print(f"  total_pages         : {result.total_pages}")
    print(f"  processing_time_sec : {result.processing_time_sec}")
    print(f"  doc_id              : {result.doc_id}")
    print(f"  doc_year            : {result.doc_year!r}")
    print(f"  filename_tokens     : {result.filename_tokens}")
    print(f"  summary             : {result.summary[:200]!r}{'...' if len(result.summary) > 200 else ''}")
    print(f"  keywords            : {result.keywords}")
    print(f"  warnings count      : {len(result.warnings)}")
    print(f"  error               : {result.error}")

    print_section("DocumentMeta")
    if result.meta:
        m = result.meta
        print(f"  doc_id              : {m.doc_id}")
        print(f"  file_name           : {m.file_name}")
        print(f"  file_path           : {m.file_path}")
        print(f"  file_type           : {m.file_type}")
        print(f"  file_size_kb        : {m.file_size_kb}")
        print(f"  page_count          : {m.page_count}")
        print(f"  has_tables          : {m.has_tables}")
        print(f"  parse_success       : {m.parse_success}")
        print(f"  filename_tokens     : {m.filename_tokens}")
        print(f"  doc_year            : {m.doc_year!r}")
        print(f"  summary             : {m.summary[:200]!r}{'...' if len(m.summary) > 200 else ''}")
        print(f"  keywords            : {m.keywords}")
        print(f"  warnings            : {m.warnings}")
    else:
        print("  meta is None")

    print_section("Blocks Summary")
    total    = len(result.blocks)
    headings = sum(1 for b in result.blocks if b.block_type == "heading")
    texts    = sum(1 for b in result.blocks if b.block_type == "text")
    tables   = sum(1 for b in result.blocks if b.is_table)
    print(f"  total               : {total}")
    print(f"  headings            : {headings}")
    print(f"  text blocks         : {texts}")
    print(f"  table blocks        : {tables}")

    print_section("Pages Summary")
    print(f"  total pages         : {len(result.pages)}")
    for p in result.pages[:3]:
        print(f"\n  page_no={p.page_no} | label={p.page_label}")
        print(f"  preview: {p.text[:120]!r}")

    print_section("Metadata Presence Check")
    checks = {
        "doc_id":          bool(result.doc_id),
        "doc_year":        bool(result.doc_year),
        "filename_tokens": bool(result.filename_tokens),
        "summary":         bool(result.summary),
        "keywords":        bool(result.keywords),
        "has_tables":      result.meta.has_tables if result.meta else False,
        "page_count > 0":  (result.meta.page_count > 0) if result.meta else False,
        "blocks > 0":      len(result.blocks) > 0,
        "pages > 0":       len(result.pages) > 0,
    }
    for k, v in checks.items():
        print(f"  {'OK' if v else '!!':>4}  {k}")

    print_section("Markdown Preview (first 1000 chars)")
    print(result.markdown[:1000])


def parse_and_report(pdf_name: str, parser: DocumentParser):
    pdf_path = DOC_INPUT_DIR / pdf_name

    print(f"\n{'#'*60}")
    print(f"  Parsing: {pdf_name}")
    print(f"{'#'*60}")
    print(f"  Input  : {pdf_path}")
    print(f"  Output : {MD_OUTPUT_DIR}")

    if not pdf_path.exists():
        print(f"  ERROR: {pdf_name} not found at {pdf_path}")
        return

    result = parser.parse_file(pdf_path)
    print_metadata(result)


def main():
    parser = DocumentParser(output_dir=MD_OUTPUT_DIR)

    parse_and_report("test2.pdf", parser)


if __name__ == "__main__":
    main()