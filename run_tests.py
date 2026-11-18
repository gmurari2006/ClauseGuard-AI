"""
ClauseGuard AI - Fast Direct Verification Script
Directly tests extraction, classification, rules, fairness, QA, and PDF generation.
"""
from pathlib import Path
from backend.extractor import extract_document
from backend.analyzer import analyze_document_text, analyze_file
from backend.ml_pipeline import classifier_instance
from backend.rule_engine import run_statutory_rule_checks, detect_document_domain, calculate_fairness_indicator, compute_overall_risk_score
from backend.qa_engine import answer_student_question
from backend.pdf_report import generate_pdf_report
from backend.sample_agreements import SAMPLE_AGREEMENTS

def run_all_checks():
    print("--- 1. Testing ML Pipeline ---")
    assert classifier_instance.is_ready(), "ML Model not ready"
    metrics = classifier_instance.get_metrics()
    print(f"ML Model: {metrics.get('model_type')}")
    print(f"Holdout Test Accuracy: {metrics.get('test_accuracy') * 100:.1f}%")
    print(f"Weighted F1: {metrics.get('weighted_avg_f1'):.3f}")

    sample_clause = "The Student agrees to serve the Company for a minimum lock-in period of 24 months. In the event of early departure, the Student shall pay liquidated damages of Rs. 1,50,000."
    pred = classifier_instance.classify_clause(sample_clause)
    print(f"Prediction for penalty clause: {pred['category']} (Confidence: {pred['confidence'] * 100:.1f}%, Risk: {pred['ml_risk_level']})")
    assert pred['category'] == "FINANCIAL_PENALTY_BOND"
    assert pred['ml_risk_level'] == "high"

    print("\n--- 2. Testing Statutory Rule Engine ---")
    non_compete = "The Student agrees not to work for any competing business in the same industry for a period of 24 months globally after departure."
    violations = run_statutory_rule_checks(non_compete, "INTERNSHIP")
    print(f"Violations detected on non-compete: {[v['rule_name'] for v in violations]}")
    assert any("RULE_SECTION_27_RESTRAINT_OF_TRADE" == v["rule_id"] for v in violations)

    cert_retention = "The intern shall deposit their original degree certificates, marks cards, and passport with the Company HR as collateral security."
    violations_cert = run_statutory_rule_checks(cert_retention, "INTERNSHIP")
    print(f"Violations on certificate retention: {[v['rule_name'] for v in violations_cert]}")
    assert any("RULE_CERTIFICATE_RETENTION_COLLATERAL" == v["rule_id"] for v in violations_cert)

    print("\n--- 3. Testing Document Extraction ---")
    for ext in ["txt", "docx", "pdf"]:
        doc_file = Path(f"test_documents/TechNova_Internship_Agreement.{ext}")
        assert doc_file.exists(), f"Missing {doc_file}"
        res = extract_document(doc_file)
        print(f"Extracted {ext.upper()}: {res['pages']} pages/sections, {len(res['text'])} chars, {len(res['paragraphs'])} paragraphs")
        assert len(res["text"]) > 200
        assert len(res["paragraphs"]) >= 8

    print("\n--- 4. Testing End-to-End Document Analysis ---")
    for sample_key, sample_data in SAMPLE_AGREEMENTS.items():
        analysis = analyze_document_text(sample_data["content"], sample_data["filename"])
        print(f"[{sample_key}] Domain: {analysis['domain']} | Score: {analysis['risk_score']}/100 ({analysis['risk_category']}) | Student Rights: {analysis['fairness']['student_favor_pct']}% | Top Issues: {len(analysis['top_issues'])}")
        assert analysis["risk_score"] > 0
        assert len(analysis["clauses"]) > 0

    print("\n--- 5. Testing Student Q&A Engine ---")
    intern_analysis = analyze_document_text(SAMPLE_AGREEMENTS["sample_internship"]["content"], "TechNova.txt")
    qa_res = answer_student_question(
        query="Can they legally stop me from working for a competitor or another startup?",
        analyzed_clauses=intern_analysis["clauses"],
        document_domain="INTERNSHIP",
        overall_fairness=intern_analysis["fairness"]
    )
    print(f"Q: Can they legally stop me from working for a competitor?")
    print(f"A: {qa_res['answer'][:180]}...")
    print(f"Cited clauses: {len(qa_res['cited_clauses'])}")
    assert len(qa_res["cited_clauses"]) > 0

    print("\n--- 6. Testing PDF Report Generation ---")
    pdf_out = generate_pdf_report(intern_analysis, "TechNova_Audit_Report.pdf")
    print(f"Generated PDF Report: {pdf_out} (Size: {pdf_out.stat().st_size} bytes)")
    assert pdf_out.exists()
    assert pdf_out.stat().st_size > 5000

    print("\n==========================================")
    print("ALL CORE MODULES VERIFIED AND PASSING 100%!")
    print("==========================================")

if __name__ == "__main__":
    run_all_checks()
