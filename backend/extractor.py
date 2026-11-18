"""
ClauseGuard AI - Document Extraction Module
Supports PDF (PyMuPDF), DOCX (python-docx), and TXT files.
Preserves document structure, headings, numbering, and paragraph boundaries,
while filtering repeated running headers/footers and page-number artifacts.
"""
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pymupdf
from docx import Document


# Standalone page-number artifact patterns (never match real clauses or titles)
RE_PAGE_NUMBER_LINE = re.compile(
    r'^(?:'
    r'(?:page|pg\.?|p\.?)\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?'
    r'|[-–—|•·\[\(]\s*\d{1,4}\s*[-–—|•·\]\)]'
    r'|\d{1,4}\s*(?:of|/)\s*\d{1,4}'
    r')$',
    re.IGNORECASE,
)
RE_BARE_PAGE_NUMBER = re.compile(r'^\d{1,4}$')

# Clause / section number patterns (aligned with backend/cross_clause.py)
RE_SUBCLAUSE_NUM = re.compile(
    r'^(?:clause\s+|section\s+)?(\d+\.\d+(?:\.\d+)?)\b',
    re.IGNORECASE,
)
RE_SECTION_NUM = re.compile(
    r'^(?:(?:section|article|clause)\s+)?(\d+)[\.\)]\s+',
    re.IGNORECASE,
)

# Inline numbered sub-clause split pattern for merged blocks (e.g. "2.1 ...\n2.2 ...")
RE_INLINE_SUBCLAUSE_SPLIT = re.compile(
    r'\n(?=\s*\d+\.\d+(?:\.\d+)?[\.\)]?\s+[A-Za-z])'
)

# Pure standalone section heading pattern
STANDALONE_HEADING_PATTERN = re.compile(
    r'^(?:(?:SECTION|ARTICLE|CLAUSE)\s+[0-9IVXLCDM]+[:\.\-]?|\d+\.)\s+[A-Z0-9][A-Z0-9\s,&/\-\(\):]{2,90}$'
)

# Guardrail: never strip blocks that look like substantive contractual clauses
RE_PROTECTED_CLAUSE_LIKE = re.compile(
    r'^(?:\d+(?:\.\d+)*[\.\)]?\s+|(?:section|article|clause)\s+\d+)'
    r'|\b(?:shall|agrees\s+to|hereby|indemnify|forfeited|liquidated\s+damages|stipend|non-compete)\b',
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _normalize_for_header_match(text: str) -> str:
    """Normalize a header/footer candidate so dynamic page numbers match across pages."""
    collapsed = re.sub(r'\s+', ' ', text.strip().lower())
    return re.sub(r'\d+', '#', collapsed)


def _is_standalone_page_number(line: str, allow_bare_digit: bool = False, max_page: int = 0) -> bool:
    """Return True if a line is purely a page-number marker."""
    stripped = line.strip()
    if not stripped:
        return False
    if RE_PAGE_NUMBER_LINE.match(stripped):
        return True
    if allow_bare_digit and RE_BARE_PAGE_NUMBER.match(stripped):
        num = int(stripped)
        if 1 <= num <= max(max_page + 2, 1):
            return True
    return False


def _is_header_footer_candidate(text: str) -> bool:
    """Check whether a short top/bottom margin block is eligible for repeated header/footer detection."""
    stripped = text.strip()
    if not stripped or len(stripped) > 140:
        return False
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 2:
        return False
    if RE_PROTECTED_CLAUSE_LIKE.search(stripped):
        return False
    return True


def _strip_standalone_page_lines(block_text: str) -> Tuple[str, int]:
    """Remove standalone page-number lines inside a text block and count removals."""
    lines = block_text.splitlines()
    kept_lines: List[str] = []
    removed = 0
    for ln in lines:
        if _is_standalone_page_number(ln, allow_bare_digit=False):
            removed += 1
        else:
            kept_lines.append(ln)
    return "\n".join(kept_lines).strip(), removed


def _strip_pdf_artifacts(
    pages_blocks: List[Tuple[float, List[Tuple[float, float, float, float, str]]]]
) -> Tuple[List[str], int]:
    """
    Remove standalone page-number artifacts and repeated multi-page PDF running headers/footers.
    Never strips real single-occurrence contract titles or substantive clauses.
    Returns (cleaned_block_texts, removed_artifacts_count).
    """
    total_pages = len(pages_blocks)
    removed_count = 0

    # Pass 1: Detect repeated top/bottom margin headers & footers across >= 2 pages
    repeated_top_norms = set()
    repeated_bottom_norms = set()

    if total_pages >= 2:
        top_page_occurrences: Dict[str, set] = defaultdict(set)
        bottom_page_occurrences: Dict[str, set] = defaultdict(set)

        for page_idx, (page_height, blocks) in enumerate(pages_blocks):
            if not blocks:
                continue
            top_limit = page_height * 0.14 if page_height > 0 else 0.0
            bottom_limit = page_height * 0.86 if page_height > 0 else float("inf")

            for b_idx, (x0, y0, x1, y1, text) in enumerate(blocks):
                in_top = (b_idx < 2) or (top_limit > 0 and y1 <= top_limit)
                in_bottom = (b_idx >= len(blocks) - 2) or (y0 >= bottom_limit)

                if in_top and _is_header_footer_candidate(text):
                    top_page_occurrences[_normalize_for_header_match(text)].add(page_idx)
                if in_bottom and _is_header_footer_candidate(text):
                    bottom_page_occurrences[_normalize_for_header_match(text)].add(page_idx)

        repeated_top_norms = {k for k, pages in top_page_occurrences.items() if len(pages) >= 2}
        repeated_bottom_norms = {k for k, pages in bottom_page_occurrences.items() if len(pages) >= 2}

    # Pass 2: Filter blocks per page
    cleaned_blocks: List[str] = []
    for page_idx, (page_height, blocks) in enumerate(pages_blocks):
        top_limit = page_height * 0.14 if page_height > 0 else 0.0
        bottom_limit = page_height * 0.86 if page_height > 0 else float("inf")

        for b_idx, (x0, y0, x1, y1, text) in enumerate(blocks):
            in_top = (b_idx < 2) or (top_limit > 0 and y1 <= top_limit)
            in_bottom = (b_idx >= len(blocks) - 2) or (y0 >= bottom_limit)
            in_margin = in_top or in_bottom

            # Standalone page-number block (allowing bare digit only in margin zones)
            if _is_standalone_page_number(text, allow_bare_digit=in_margin, max_page=total_pages):
                removed_count += 1
                continue

            # Repeated multi-page running header or footer
            if total_pages >= 2 and _is_header_footer_candidate(text):
                norm = _normalize_for_header_match(text)
                if (in_top and norm in repeated_top_norms) or (in_bottom and norm in repeated_bottom_norms):
                    removed_count += 1
                    continue

            # Also strip embedded standalone page-number lines within a multi-line block
            cleaned_text, line_removals = _strip_standalone_page_lines(text)
            removed_count += line_removals
            if cleaned_text:
                cleaned_blocks.append(cleaned_text)

    return cleaned_blocks, removed_count


def _build_extraction_metadata(
    pages: int,
    paragraphs: List[Dict[str, Any]],
    full_text: str,
    table_rows_extracted: int = 0,
    removed_artifacts_count: int = 0,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Assemble standardized extraction_metadata dictionary."""
    return {
        "pages": pages,
        "paragraph_count": len(paragraphs),
        "heading_count": sum(1 for p in paragraphs if p["is_heading"]),
        "table_rows_extracted": table_rows_extracted,
        "removed_artifacts_count": removed_artifacts_count,
        "char_count": len(full_text),
        "word_count": len(full_text.split()),
        "warnings": list(warnings) if warnings else [],
    }


def _validate_file_exists_and_nonempty(file_path: Path, format_label: str) -> None:
    """Validate that file exists and is not zero bytes."""
    if not file_path.exists() or not file_path.is_file():
        raise ValueError(f"{format_label} file not found: {file_path}")
    if file_path.stat().st_size == 0:
        raise ValueError(f"The uploaded {format_label} file is empty (0 bytes) and contains no readable text.")


def extract_from_pdf(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract text and metadata from PDF files using PyMuPDF semantic layout blocks."""
    path = Path(file_path)
    _validate_file_exists_and_nonempty(path, "PDF")

    try:
        doc = pymupdf.open(str(path))
    except Exception as exc:
        raise ValueError(
            f"Unable to open PDF file '{path.name}'. The file may be corrupted or not a valid PDF."
        ) from exc

    warnings: List[str] = []
    pages_blocks: List[Tuple[float, List[Tuple[float, float, float, float, str]]]] = []

    try:
        # Handle encrypted / password-protected PDFs safely
        if getattr(doc, "needs_pass", False):
            authenticated = False
            try:
                authenticated = bool(doc.authenticate(""))
            except Exception:
                authenticated = False
            if not authenticated:
                raise ValueError(
                    f"PDF file '{path.name}' is encrypted or password-protected and cannot be extracted."
                )

        total_pages = len(doc)
        if total_pages == 0:
            raise ValueError(f"PDF file '{path.name}' contains no pages.")

        for page_num in range(total_pages):
            page = doc[page_num]
            page_height = float(page.rect.height) if page.rect else 0.0
            raw_blocks = page.get_text("blocks")
            page_text_blocks: List[Tuple[float, float, float, float, str]] = []
            for b in raw_blocks:
                if len(b) > 4 and isinstance(b[4], str) and b[4].strip():
                    page_text_blocks.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), b[4].strip()))
            if not page_text_blocks:
                warnings.append(
                    f"Page {page_num + 1} contained no extractable text (possibly blank or scanned image)."
                )
            pages_blocks.append((page_height, page_text_blocks))
    finally:
        doc.close()

    block_texts, removed_artifacts_count = _strip_pdf_artifacts(pages_blocks)
    full_text = clean_text("\n\n".join(block_texts))

    if not full_text:
        raise ValueError(
            f"No readable text could be extracted from PDF '{path.name}'. "
            "It may be empty or a scanned image-only PDF."
        )

    paragraphs = segment_into_paragraphs(full_text)
    extraction_metadata = _build_extraction_metadata(
        pages=total_pages,
        paragraphs=paragraphs,
        full_text=full_text,
        table_rows_extracted=0,
        removed_artifacts_count=removed_artifacts_count,
        warnings=warnings,
    )

    return {
        "text": full_text,
        "paragraphs": paragraphs,
        "pages": total_pages,
        "format": "PDF",
        "extraction_metadata": extraction_metadata,
    }


def extract_from_docx(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract text from DOCX files preserving paragraph styles and tables."""
    path = Path(file_path)
    _validate_file_exists_and_nonempty(path, "DOCX")

    try:
        doc = Document(str(path))
    except Exception as exc:
        raise ValueError(
            f"Unable to read DOCX file '{path.name}'. The file may be corrupted or not a valid Word document."
        ) from exc

    paragraphs_raw: List[str] = []
    removed_artifacts_count = 0
    table_rows_extracted = 0
    warnings: List[str] = []

    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        cleaned_t, removed = _strip_standalone_page_lines(t)
        removed_artifacts_count += removed
        if cleaned_t:
            paragraphs_raw.append(cleaned_t)

    # Also extract table text if present
    for table in doc.tables:
        for row in table.rows:
            row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if row_text:
                paragraphs_raw.append(" | ".join(row_text))
                table_rows_extracted += 1

    full_text = clean_text("\n\n".join(paragraphs_raw))
    if not full_text:
        raise ValueError(f"DOCX file '{path.name}' contains no readable text.")

    paragraphs = segment_into_paragraphs(full_text)
    estimated_pages = max(1, len(paragraphs) // 5)
    extraction_metadata = _build_extraction_metadata(
        pages=estimated_pages,
        paragraphs=paragraphs,
        full_text=full_text,
        table_rows_extracted=table_rows_extracted,
        removed_artifacts_count=removed_artifacts_count,
        warnings=warnings,
    )

    return {
        "text": full_text,
        "paragraphs": paragraphs,
        "pages": estimated_pages,
        "format": "DOCX",
        "extraction_metadata": extraction_metadata,
    }


def extract_from_txt(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract text from TXT with multiple encoding fallbacks."""
    path = Path(file_path)
    _validate_file_exists_and_nonempty(path, "TXT")

    encodings = ['utf-8', 'utf-8-sig', 'cp1252', 'latin-1']
    raw_text: Optional[str] = None
    used_encoding: Optional[str] = None
    warnings: List[str] = []

    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as f:
                raw_text = f.read()
            used_encoding = enc
            break
        except (UnicodeDecodeError, LookupError):
            continue
        except OSError as exc:
            raise ValueError(f"Unable to read text file '{path.name}': {exc}") from exc

    if raw_text is None:
        raise ValueError(f"Unable to decode text file '{path.name}' using supported encodings.")

    if '\x00' in raw_text:
        raise ValueError(f"File '{path.name}' appears to be binary or corrupted rather than plain text.")

    if used_encoding and used_encoding not in ('utf-8', 'utf-8-sig'):
        warnings.append(f"Decoded text file using fallback encoding '{used_encoding}'.")

    # Strip standalone page-number lines while keeping paragraph boundaries
    normalized_raw = raw_text.replace('\r\n', '\n').replace('\r', '\n')
    raw_blocks = normalized_raw.split('\n\n')
    cleaned_blocks: List[str] = []
    removed_artifacts_count = 0
    for blk in raw_blocks:
        cleaned_blk, removed = _strip_standalone_page_lines(blk)
        removed_artifacts_count += removed
        if cleaned_blk:
            cleaned_blocks.append(cleaned_blk)

    full_text = clean_text("\n\n".join(cleaned_blocks))
    if not full_text:
        raise ValueError(f"Text file '{path.name}' contains no readable text.")

    paragraphs = segment_into_paragraphs(full_text)
    estimated_pages = max(1, len(paragraphs) // 5)
    extraction_metadata = _build_extraction_metadata(
        pages=estimated_pages,
        paragraphs=paragraphs,
        full_text=full_text,
        table_rows_extracted=0,
        removed_artifacts_count=removed_artifacts_count,
        warnings=warnings,
    )

    return {
        "text": full_text,
        "paragraphs": paragraphs,
        "pages": estimated_pages,
        "format": "TXT",
        "extraction_metadata": extraction_metadata,
    }


def segment_into_paragraphs(text: str) -> List[Dict[str, Any]]:
    """
    Split text into distinct structural paragraphs and clauses.
    Detects section numbering patterns like:
    1., 1.1, Article IV, Section 3, Clause 5, (a), etc.
    Splits merged blocks containing multiple inline numbered sub-clauses (e.g., 2.1 ...\\n2.2 ...).
    """
    if not text or not text.strip():
        return []

    normalized_text = text.replace('\r\n', '\n').replace('\r', '\n')
    initial_blocks = [b.strip() for b in normalized_text.split('\n\n') if b.strip()]

    # If blocks are too few, split by line breaks that precede numbered clauses
    if len(initial_blocks) <= 3 and len(normalized_text) > 300:
        split_pattern = re.compile(
            r'\n(?=(?:[0-9]+(?:\.[0-9]+)*[\.\)]|\([a-z0-9]\)|SECTION|ARTICLE|CLAUSE)\s+)',
            re.IGNORECASE,
        )
        initial_blocks = [b.strip() for b in split_pattern.split(normalized_text) if b.strip()]

    # Cleanly split merged blocks where multiple numbered sub-clauses appear separated by a single newline
    raw_blocks: List[str] = []
    for blk in initial_blocks:
        if RE_INLINE_SUBCLAUSE_SPLIT.search(blk):
            sub_parts = [part.strip() for part in RE_INLINE_SUBCLAUSE_SPLIT.split(blk) if part.strip()]
            raw_blocks.extend(sub_parts)
        else:
            raw_blocks.append(blk)

    paragraphs: List[Dict[str, Any]] = []
    current_section: Optional[str] = None
    search_cursor = 0

    for idx, block in enumerate(raw_blocks):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        single_line = len(lines) == 1
        # Pure heading: short standalone section title or all-caps title banner (never multi-sentence body prose)
        is_heading = (
            (single_line and len(block) <= 95 and not block.endswith('.') and bool(STANDALONE_HEADING_PATTERN.match(block)))
            or (len(lines) <= 2 and len(block) <= 110 and block.isupper() and not block.endswith('.'))
        )

        # Extract clause_number and section_number
        clause_number: Optional[str] = None
        section_number: Optional[str] = None
        m_sub = RE_SUBCLAUSE_NUM.match(block)
        m_sec = RE_SECTION_NUM.match(block)

        if m_sub:
            clause_number = m_sub.group(1)
            section_number = clause_number.split('.')[0]
            current_section = section_number
        elif m_sec:
            clause_number = m_sec.group(1)
            section_number = clause_number
            current_section = section_number
        else:
            section_number = current_section

        found_pos = normalized_text.find(block, search_cursor)
        if found_pos != -1:
            char_start = found_pos
            search_cursor = found_pos + len(block)
        else:
            char_start = search_cursor
            search_cursor = char_start + len(block) + 2

        paragraphs.append({
            "index": idx,
            "text": block,
            "char_start": char_start,
            "char_end": char_start + len(block),
            "is_heading": is_heading,
            "word_count": len(block.split()),
            "clause_number": clause_number,
            "section_number": section_number,
            "segment_type": "heading" if is_heading else "clause",
        })

    return paragraphs


def segment_clauses(text: str) -> List[Dict[str, Any]]:
    """Semantic alias for segment_into_paragraphs(text)."""
    return segment_into_paragraphs(text)


def extract_document(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Universal document extractor dispatcher."""
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return extract_from_pdf(path)
    elif ext in [".docx", ".doc"]:
        return extract_from_docx(path)
    elif ext in [".txt", ".md", ".rtf"]:
        return extract_from_txt(path)
    else:
        # Fallback to text reading
        return extract_from_txt(path)
