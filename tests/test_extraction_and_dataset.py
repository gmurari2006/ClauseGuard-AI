"""
ClauseGuard AI - Comprehensive Tests for Workstream 1:
- Document Extraction & Clause Segmentation (`backend/extractor.py`)
- Auditable Dataset Preparation Pipeline (`backend/dataset_pipeline.py` & `datasets/`)
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pymupdf
import pytest
from docx import Document

from backend.config import AGREEMENT_DOMAINS, BASE_DIR, CLAUSE_CATEGORIES
from backend.dataset_pipeline import (
    DEFAULT_DATASETS_DIR,
    NEAR_DUPLICATE_JACCARD_THRESHOLD,
    REQUIRED_CLAUSE_FIELDS,
    VALID_RISK_LEVELS,
    VALID_SOURCE_TYPES,
    build_dataset,
    compute_exact_hash,
    compute_token_jaccard,
    get_dataset_status,
    run_quality_gates,
    tokenize_for_jaccard,
)
from backend.extractor import (
    clean_text,
    extract_document,
    extract_from_docx,
    extract_from_pdf,
    extract_from_txt,
    segment_clauses,
    segment_into_paragraphs,
)

TEST_DOCS_DIR = BASE_DIR / "test_documents"
EXPECTED_METADATA_KEYS = {
    "pages",
    "paragraph_count",
    "heading_count",
    "table_rows_extracted",
    "removed_artifacts_count",
    "char_count",
    "word_count",
    "warnings",
}
EXPECTED_PARAGRAPH_KEYS = {
    "index",
    "text",
    "char_start",
    "char_end",
    "is_heading",
    "word_count",
    "clause_number",
    "section_number",
    "segment_type",
}


# =============================================================================
# 1. REAL TEST DOCUMENT EXTRACTION (PDF, DOCX, TXT) & BACKWARDS COMPATIBILITY
# =============================================================================

def test_cloudpulse_pdf_extraction_preserves_exact_clauses():
    pdf_path = TEST_DOCS_DIR / "CloudPulse_Student_Internship_Test_Agreement.pdf"
    assert pdf_path.exists(), f"Expected fixture not found: {pdf_path}"

    extracted = extract_document(pdf_path)
    assert extracted["format"] == "PDF"
    assert extracted["pages"] == 2
    assert len(extracted["text"]) > 1000

    # Verify extraction_metadata contract
    meta = extracted["extraction_metadata"]
    assert set(meta.keys()) == EXPECTED_METADATA_KEYS
    assert meta["pages"] == 2
    assert meta["paragraph_count"] == len(extracted["paragraphs"])
    assert meta["paragraph_count"] == 37
    assert meta["heading_count"] == 10
    assert meta["removed_artifacts_count"] == 0
    assert meta["char_count"] == len(extracted["text"])
    assert meta["word_count"] > 500
    assert isinstance(meta["warnings"], list)

    # Verify every paragraph has backwards-compatible + additive keys
    for para in extracted["paragraphs"]:
        assert EXPECTED_PARAGRAPH_KEYS.issubset(para.keys())
        assert para["segment_type"] == ("heading" if para["is_heading"] else "clause")

    # Verify CloudPulse still yields all Clauses 1.1–8.1 cleanly
    expected_clauses = [
        "1.1", "1.2",
        "2.1", "2.2",
        "3.1", "3.2", "3.3",
        "4.1", "4.2", "4.3",
        "5.1", "5.2",
        "6.1", "6.2",
        "7.1", "7.2",
        "8.1",
    ]
    extracted_clause_nums = [
        p["clause_number"]
        for p in extracted["paragraphs"]
        if not p["is_heading"] and p["clause_number"] is not None
    ]
    assert extracted_clause_nums == expected_clauses


def test_technova_pdf_and_docx_extraction_and_subclause_splitting():
    pdf_path = TEST_DOCS_DIR / "TechNova_Internship_Agreement.pdf"
    docx_path = TEST_DOCS_DIR / "TechNova_Internship_Agreement.docx"
    assert pdf_path.exists()
    assert docx_path.exists()

    pdf_res = extract_document(pdf_path)
    docx_res = extract_document(docx_path)

    assert pdf_res["format"] == "PDF"
    assert docx_res["format"] == "DOCX"
    assert set(pdf_res["extraction_metadata"].keys()) == EXPECTED_METADATA_KEYS
    assert set(docx_res["extraction_metadata"].keys()) == EXPECTED_METADATA_KEYS

    expected_subclauses = {
        "2.1", "2.2", "3.1", "3.2", "3.3", "4.1", "4.2",
        "5.1", "5.2", "6.1", "6.2", "6.3", "7.1", "7.2", "8.1", "8.2",
    }
    pdf_nums = {p["clause_number"] for p in pdf_res["paragraphs"] if not p["is_heading"] and p["clause_number"]}
    docx_nums = {p["clause_number"] for p in docx_res["paragraphs"] if not p["is_heading"] and p["clause_number"]}

    assert expected_subclauses.issubset(pdf_nums)
    assert expected_subclauses.issubset(docx_nums)


def test_greenvalley_hostel_txt_extraction():
    txt_path = TEST_DOCS_DIR / "GreenValley_Hostel_Residency_Agreement.txt"
    assert txt_path.exists()

    res = extract_document(txt_path)
    assert res["format"] == "TXT"
    assert len(res["text"]) > 500
    assert set(res["extraction_metadata"].keys()) == EXPECTED_METADATA_KEYS
    assert res["extraction_metadata"]["paragraph_count"] == len(res["paragraphs"])

    clause_nums = [p["clause_number"] for p in res["paragraphs"] if not p["is_heading"] and p["clause_number"]]
    for expected in ["2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "3.3", "4.1", "4.2", "5.1", "5.2"]:
        assert expected in clause_nums


def test_segment_clauses_alias_and_inline_subclause_splitting():
    merged_text = (
        "2. STIPEND AND COMPENSATION TERMS\n"
        "2.1 The Company shall pay a monthly stipend of INR 20,000 on the 5th of every month.\n"
        "2.2 Reimbursement of approved travel receipts shall be settled within 14 calendar days."
    )
    segs = segment_into_paragraphs(merged_text)
    alias_segs = segment_clauses(merged_text)
    assert segs == alias_segs
    assert len(segs) == 3

    assert segs[0]["is_heading"] is True
    assert segs[0]["segment_type"] == "heading"
    assert segs[0]["section_number"] == "2"

    assert segs[1]["is_heading"] is False
    assert segs[1]["segment_type"] == "clause"
    assert segs[1]["clause_number"] == "2.1"
    assert segs[1]["section_number"] == "2"

    assert segs[2]["is_heading"] is False
    assert segs[2]["segment_type"] == "clause"
    assert segs[2]["clause_number"] == "2.2"
    assert segs[2]["section_number"] == "2"


def test_clean_text_preserves_paragraph_breaks():
    raw = "Clause 1\r\n\r\n\r\n\r\nClause 2\r\nLine B  "
    assert clean_text(raw) == "Clause 1\n\nClause 2\nLine B"


# =============================================================================
# 2. ARTIFACT REMOVAL (HEADERS, FOOTERS, PAGE NUMBERS) & DOCX TABLES
# =============================================================================

def test_pdf_running_header_footer_and_page_number_stripping(tmp_path: Path):
    pdf_path = tmp_path / "multipage_with_artifacts.pdf"
    doc = pymupdf.open()

    pages_content = [
        (
            "ACME CORP — CONFIDENTIAL STUDENT DRAFT",
            "ACME CORP STUDENT INTERNSHIP AGREEMENT",
            "1.1 The Intern is appointed for a six-month Data Engineering Internship starting January 2027.",
            "Page 1 of 3",
        ),
        (
            "ACME CORP — CONFIDENTIAL STUDENT DRAFT",
            "2.1 The Company shall disburse a monthly stipend of INR 22,000 on the 5th business day.",
            "2.2 Travel expenses pre-approved in writing shall be reimbursed within 14 calendar days.",
            "- 2 -",
        ),
        (
            "ACME CORP — CONFIDENTIAL STUDENT DRAFT",
            "3.1 Either party may terminate this agreement with 15 days prior written notice.",
            "3.2 Disputes shall be resolved by competent courts in Bengaluru.",
            "Page 3 of 3",
        ),
    ]

    for header, body_a, body_b, footer in pages_content:
        page = doc.new_page(width=595, height=842)
        page.insert_text((50, 40), header, fontsize=9)
        page.insert_text((50, 140), body_a, fontsize=11)
        page.insert_text((50, 220), body_b, fontsize=11)
        page.insert_text((260, 810), footer, fontsize=9)

    doc.save(str(pdf_path))
    doc.close()

    extracted = extract_from_pdf(pdf_path)
    meta = extracted["extraction_metadata"]
    assert meta["pages"] == 3
    # 3 repeated top running headers + 3 standalone page-number footers = 6 artifacts removed
    assert meta["removed_artifacts_count"] == 6

    text = extracted["text"]
    assert "ACME CORP — CONFIDENTIAL STUDENT DRAFT" not in text
    assert "Page 1 of 3" not in text
    assert "- 2 -" not in text
    assert "Page 3 of 3" not in text

    # Real single-occurrence title and all substantive clauses MUST remain intact
    assert "ACME CORP STUDENT INTERNSHIP AGREEMENT" in text
    assert "1.1 The Intern is appointed" in text
    assert "2.1 The Company shall disburse" in text
    assert "2.2 Travel expenses pre-approved" in text
    assert "3.1 Either party may terminate" in text
    assert "3.2 Disputes shall be resolved" in text


def test_txt_standalone_page_number_stripping(tmp_path: Path):
    txt_path = tmp_path / "paged_contract.txt"
    txt_path.write_text(
        "STUDENT HOSTEL AGREEMENT\n\n"
        "1.1 Room rent is INR 8,000 per month.\n\n"
        "Page 1 of 2\n\n"
        "1.2 Security deposit is refunded within 15 business days.\n\n"
        "- 2 -\n",
        encoding="utf-8",
    )
    res = extract_from_txt(txt_path)
    assert res["extraction_metadata"]["removed_artifacts_count"] == 2
    assert "Page 1 of 2" not in res["text"]
    assert "- 2 -" not in res["text"]
    assert "1.1 Room rent" in res["text"]
    assert "1.2 Security deposit" in res["text"]


def test_docx_table_extraction_metadata(tmp_path: Path):
    docx_path = tmp_path / "with_table.docx"
    doc = Document()
    doc.add_heading("1. STIPEND SCHEDULE", level=1)
    doc.add_paragraph("1.1 Monthly stipend is paid electronically.")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Month"
    table.cell(0, 1).text = "Amount"
    table.cell(1, 0).text = "Month 1"
    table.cell(1, 1).text = "INR 20,000"
    doc.save(str(docx_path))

    res = extract_from_docx(docx_path)
    assert res["extraction_metadata"]["table_rows_extracted"] == 2
    assert "INR 20,000" in res["text"]


# =============================================================================
# 3. SAFE ERROR HANDLING (EMPTY, CORRUPTED, ENCRYPTED, MISSING FILES)
# =============================================================================

def test_empty_and_corrupted_file_error_handling(tmp_path: Path):
    # Empty 0-byte files
    for ext, fn in [(".pdf", extract_from_pdf), (".docx", extract_from_docx), (".txt", extract_from_txt)]:
        empty_file = tmp_path / f"empty{ext}"
        empty_file.write_bytes(b"")
        with pytest.raises(ValueError, match="empty"):
            fn(empty_file)

    # Corrupted PDF
    bad_pdf = tmp_path / "corrupted.pdf"
    bad_pdf.write_bytes(b"this is definitely not a valid pdf binary stream")
    with pytest.raises(ValueError, match="PDF"):
        extract_from_pdf(bad_pdf)

    # Corrupted DOCX
    bad_docx = tmp_path / "corrupted.docx"
    bad_docx.write_bytes(b"not a valid zip or docx archive")
    with pytest.raises(ValueError, match="DOCX"):
        extract_from_docx(bad_docx)

    # Blank-page-only PDF
    blank_pdf = tmp_path / "blank_page.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(blank_pdf))
    doc.close()
    with pytest.raises(ValueError, match="No readable text"):
        extract_from_pdf(blank_pdf)

    # Missing file
    with pytest.raises(ValueError, match="not found"):
        extract_document(tmp_path / "does_not_exist.pdf")


def test_encrypted_pdf_raises_clear_value_error(tmp_path: Path):
    enc_pdf = tmp_path / "encrypted.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Confidential student agreement clause text.")
    doc.save(
        str(enc_pdf),
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="student123",
        owner_pw="owner456",
    )
    doc.close()

    with pytest.raises(ValueError, match="encrypted|password"):
        extract_from_pdf(enc_pdf)


# =============================================================================
# 4. DATASET PIPELINE INTEGRITY, QUALITY GATES & LEAKAGE VERIFICATION
# =============================================================================

def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_generated_dataset_artifacts_and_schema():
    expected_files = [
        DEFAULT_DATASETS_DIR / "clauses_master.jsonl",
        DEFAULT_DATASETS_DIR / "clauses_master.csv",
        DEFAULT_DATASETS_DIR / "structural_examples.jsonl",
        DEFAULT_DATASETS_DIR / "splits" / "train.jsonl",
        DEFAULT_DATASETS_DIR / "splits" / "val.jsonl",
        DEFAULT_DATASETS_DIR / "splits" / "test.jsonl",
        DEFAULT_DATASETS_DIR / "dataset_report.json",
        DEFAULT_DATASETS_DIR / "DATASET_CARD.md",
    ]
    for fp in expected_files:
        assert fp.exists() and fp.stat().st_size > 0, f"Missing or empty dataset artifact: {fp}"

    master_records = _read_jsonl(DEFAULT_DATASETS_DIR / "clauses_master.jsonl")
    assert len(master_records) >= 300

    # Verify CSV matches JSONL row count and schema
    with open(DEFAULT_DATASETS_DIR / "clauses_master.csv", "r", encoding="utf-8", newline="") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == len(master_records)
    assert tuple(csv_rows[0].keys()) == REQUIRED_CLAUSE_FIELDS

    valid_categories = set(CLAUSE_CATEGORIES)
    valid_domains = set(AGREEMENT_DOMAINS)
    seen_ids = set()

    for rec in master_records:
        assert tuple(rec.keys()) == REQUIRED_CLAUSE_FIELDS or set(rec.keys()) == set(REQUIRED_CLAUSE_FIELDS)
        assert rec["id"] not in seen_ids
        seen_ids.add(rec["id"])
        assert 35 <= len(rec["text"]) <= 1500
        assert rec["category"] in valid_categories
        assert rec["risk_level"] in VALID_RISK_LEVELS
        assert rec["domain"] in valid_domains
        assert rec["source_type"] in VALID_SOURCE_TYPES
        assert rec["source_group"].strip()
        assert rec["clause_ref"].strip()
        assert rec["rationale"].strip()
        assert rec["split"] in {"train", "val", "test"}

    # Class balance: >= 25 (and >= 30) across all 10 CLAUSE_CATEGORIES
    cat_counts = Counter(r["category"] for r in master_records)
    assert set(cat_counts.keys()) == valid_categories
    for cat in CLAUSE_CATEGORIES:
        assert cat_counts[cat] >= 25, f"Category {cat} has only {cat_counts[cat]} clauses"

    # Domain coverage across all 6 AGREEMENT_DOMAINS
    dom_counts = Counter(r["domain"] for r in master_records)
    assert set(dom_counts.keys()) == valid_domains

    # Structural examples file validation
    structural_records = _read_jsonl(DEFAULT_DATASETS_DIR / "structural_examples.jsonl")
    assert len(structural_records) >= 20
    assert all(r["is_substantive"] is False for r in structural_records)


def test_dataset_zero_duplicates_and_zero_split_leakage():
    master_records = _read_jsonl(DEFAULT_DATASETS_DIR / "clauses_master.jsonl")
    train_records = _read_jsonl(DEFAULT_DATASETS_DIR / "splits" / "train.jsonl")
    val_records = _read_jsonl(DEFAULT_DATASETS_DIR / "splits" / "val.jsonl")
    test_records = _read_jsonl(DEFAULT_DATASETS_DIR / "splits" / "test.jsonl")

    assert len(train_records) + len(val_records) + len(test_records) == len(master_records)

    # Zero exact duplicates across master
    hashes = [compute_exact_hash(r["text"]) for r in master_records]
    assert len(hashes) == len(set(hashes)), "Exact duplicate found in clauses_master.jsonl"

    # Zero near-duplicates (Jaccard >= 0.85) across master
    token_sets = [tokenize_for_jaccard(r["text"]) for r in master_records]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            sim = compute_token_jaccard(token_sets[i], token_sets[j])
            assert sim < NEAR_DUPLICATE_JACCARD_THRESHOLD, (
                f"Near-duplicate pair ({master_records[i]['id']}, {master_records[j]['id']}): Jaccard={sim:.3f}"
            )

    # Zero text overlap across splits
    train_hashes = {compute_exact_hash(r["text"]) for r in train_records}
    val_hashes = {compute_exact_hash(r["text"]) for r in val_records}
    test_hashes = {compute_exact_hash(r["text"]) for r in test_records}
    assert train_hashes.isdisjoint(val_hashes)
    assert train_hashes.isdisjoint(test_hashes)
    assert val_hashes.isdisjoint(test_hashes)

    # Zero source_group leakage across splits
    train_groups = {r["source_group"] for r in train_records}
    val_groups = {r["source_group"] for r in val_records}
    test_groups = {r["source_group"] for r in test_records}
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # Every class represented in train, val, and test
    for split_name, rows in [("train", train_records), ("val", val_records), ("test", test_records)]:
        split_cats = {r["category"] for r in rows}
        assert split_cats == set(CLAUSE_CATEGORIES), f"Missing classes in {split_name}: {set(CLAUSE_CATEGORIES) - split_cats}"


def test_quality_gates_actively_reject_invalid_and_duplicate_records():
    base_clause = (
        "The Student Intern shall pay liquidated damages of INR 90,000 if departing "
        "before completing the mandatory twelve-month training lock-in period."
    )
    near_dup_clause = (
        "The Student Intern shall pay liquidated damages of INR 90,000 if departing "
        "before completing the mandatory twelve-month training lock-in period!"
    )
    near_dup_variant = (
        "The Student Intern must pay liquidated damages of INR 90,000 if departing "
        "before completing the mandatory twelve-month training lock-in period."
    )

    synthetic_batch = [
        # Valid record
        {
            "text": base_clause,
            "category": "FINANCIAL_PENALTY_BOND",
            "risk_level": "high",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_1",
            "clause_ref": "Clause 1.1",
            "rationale": "Valid bond clause.",
        },
        # Exact duplicate (differing only in casing / extra whitespace)
        {
            "text": "  " + base_clause.upper() + "  ",
            "category": "FINANCIAL_PENALTY_BOND",
            "risk_level": "high",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_2",
            "clause_ref": "Clause 1.2",
            "rationale": "Exact duplicate.",
        },
        # Near-duplicate (Jaccard >= 0.85)
        {
            "text": near_dup_clause,
            "category": "FINANCIAL_PENALTY_BOND",
            "risk_level": "high",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_3",
            "clause_ref": "Clause 1.3",
            "rationale": "Punctuation near-duplicate.",
        },
        # Near-duplicate with one word changed (Jaccard >= 0.85)
        {
            "text": near_dup_variant,
            "category": "FINANCIAL_PENALTY_BOND",
            "risk_level": "high",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_4",
            "clause_ref": "Clause 1.4",
            "rationale": "Single-word near-duplicate.",
        },
        # Invalid label
        {
            "text": "The company shall provide a standard workstation and mentorship to the intern.",
            "category": "NOT_A_REAL_CATEGORY",
            "risk_level": "low",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_5",
            "clause_ref": "Clause 1.5",
            "rationale": "Invalid category.",
        },
        # Too short (< 35 chars)
        {
            "text": "Short clause text.",
            "category": "STANDARD_FAIR",
            "risk_level": "low",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_6",
            "clause_ref": "Clause 1.6",
            "rationale": "Too short.",
        },
        # Too long (> 1500 chars)
        {
            "text": "A" * 1505,
            "category": "STANDARD_FAIR",
            "risk_level": "low",
            "domain": "INTERNSHIP",
            "source_type": "curated_synthetic",
            "source_group": "grp_test_7",
            "clause_ref": "Clause 1.7",
            "rationale": "Too long.",
        },
    ]

    accepted, summary = run_quality_gates(synthetic_batch)
    assert len(accepted) == 1
    assert summary["schema_rejected_count"] == 1
    assert summary["length_rejected_count"] == 2
    assert summary["exact_duplicates_removed_count"] == 1
    assert summary["near_duplicates_removed_count"] == 2


def test_get_dataset_status_and_isolated_build(tmp_path: Path):
    # Active workspace dataset status
    status = get_dataset_status()
    assert status["dataset_ready"] is True
    assert status["retraining_status"] == "PENDING_AUTHORIZATION"
    assert status["active_model_retrained"] is False
    assert status["total_clauses"] >= 300
    assert status["num_categories"] == 10
    assert status["leakage_checks"]["zero_group_leakage"] is True
    assert status["leakage_checks"]["zero_text_overlap"] is True
    assert "not been retrained" in status["note"].lower()

    # Missing directory status handled cleanly
    missing_status = get_dataset_status(tmp_path / "nonexistent")
    assert missing_status["dataset_ready"] is False
    assert missing_status["status"] == "NOT_BUILT"
    assert missing_status["retraining_status"] == "PENDING_AUTHORIZATION"

    # Rebuilding into an isolated temp directory succeeds identically
    isolated_out = tmp_path / "datasets_copy"
    report = build_dataset(output_dir=isolated_out)
    assert report["total_clauses"] == status["total_clauses"]
    assert (isolated_out / "DATASET_CARD.md").exists()
    assert (isolated_out / "splits" / "test.jsonl").exists()
