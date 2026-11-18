"""
Comprehensive unit and integration test suite for ClauseGuard AI.
Tests document extraction (PDF, DOCX, TXT), ML pipeline, rule engine,
fairness calculation, Q&A reasoning, PDF generation, and FastAPI endpoints.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from main import app
from backend.extractor import extract_document
from backend.analyzer import analyze_document_text, analyze_file
from backend.ml_pipeline import classifier_instance
from backend.rule_engine import run_statutory_rule_checks, detect_document_domain
from backend.qa_engine import answer_student_question
from backend.pdf_report import generate_pdf_report
from backend.sample_agreements import SAMPLE_AGREEMENTS

client = TestClient(app)

def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["ml_model_ready"] is True

def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    data = response.json()
    assert len(data["samples"]) >= 5

def test_load_sample_internship():
    response = client.get("/api/samples/sample_internship")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "INTERNSHIP"
    assert data["risk_score"] > 60  # Has severe exit bond, non-compete, cert retention
    assert len(data["clauses"]) > 0
    assert len(data["top_issues"]) > 0

def test_load_sample_hostel():
    response = client.get("/api/samples/sample_hostel")
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "HOSTEL_RESIDENCY"
    assert data["risk_score"] > 50

def test_extraction_pdf():
    pdf_path = Path("test_documents/TechNova_Internship_Agreement.pdf")
    if pdf_path.exists():
        extracted = extract_document(pdf_path)
        assert extracted["format"] == "PDF"
        assert len(extracted["text"]) > 200
        assert len(extracted["paragraphs"]) > 5

def test_extraction_docx():
    docx_path = Path("test_documents/TechNova_Internship_Agreement.docx")
    if docx_path.exists():
        extracted = extract_document(docx_path)
        assert extracted["format"] == "DOCX"
        assert len(extracted["text"]) > 200
        assert len(extracted["paragraphs"]) > 5

def test_statutory_rule_engine():
    # Test Section 27 Restraint of Trade
    text = "The Student agrees not to work for any competing business for a period of 24 months globally after departure."
    violations = run_statutory_rule_checks(text, "INTERNSHIP")
    assert any(v["rule_id"] == "RULE_SECTION_27_RESTRAINT_OF_TRADE" for v in violations)

    # Test Certificate retention
    text_cert = "The intern shall deposit their original degree certificates and marks cards as collateral security."
    violations_cert = run_statutory_rule_checks(text_cert, "INTERNSHIP")
    assert any(v["rule_id"] == "RULE_CERTIFICATE_RETENTION_COLLATERAL" for v in violations_cert)

def test_ml_classification():
    assert classifier_instance.is_ready()
    res = classifier_instance.classify_clause("Student agrees to pay liquidated damages of Rs. 1,00,000 for early resignation.")
    assert res["category"] == "FINANCIAL_PENALTY_BOND"
    assert res["ml_risk_level"] == "high"

def test_qa_engine():
    sample = SAMPLE_AGREEMENTS["sample_internship"]
    analysis = analyze_document_text(sample["content"], "Test.txt")
    qa_res = answer_student_question(
        query="Can I resign without paying a fine?",
        analyzed_clauses=analysis["clauses"],
        document_domain="INTERNSHIP",
        overall_fairness=analysis["fairness"]
    )
    assert "penalty" in qa_res["answer"].lower() or "exit" in qa_res["answer"].lower()
    assert len(qa_res["cited_clauses"]) > 0

def test_pdf_report_generation():
    sample = SAMPLE_AGREEMENTS["sample_internship"]
    analysis = analyze_document_text(sample["content"], "Test_Report_Doc.txt")
    pdf_path = generate_pdf_report(analysis, "Test_Unit_Report.pdf")
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 1000

def test_upload_api():
    txt_path = Path("test_documents/GreenValley_Hostel_Residency_Agreement.txt")
    with open(txt_path, "rb") as f:
        response = client.post("/api/upload", files={"file": ("GreenValley_Hostel_Residency_Agreement.txt", f, "text/plain")})
    assert response.status_code == 200
    data = response.json()
    assert data["domain"] == "HOSTEL_RESIDENCY"

if __name__ == "__main__":
    pytest.main(["-v", "test_backend.py"])
