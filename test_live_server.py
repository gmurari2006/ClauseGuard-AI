"""
ClauseGuard AI - End-to-End Live Integration Verification
Tests all endpoints against the active daemon server on http://127.0.0.1:8000
"""
import requests
import json
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def run_live_tests():
    print(f"Connecting to live server at {BASE_URL}...")
    
    # 1. Health check
    r = requests.get(f"{BASE_URL}/api/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health = r.json()
    print("1. Health Check: OK", health)
    assert health["status"] == "healthy"
    assert health["ml_model_ready"] is True

    # 2. Sample contracts endpoint
    r = requests.get(f"{BASE_URL}/api/samples")
    assert r.status_code == 200
    samples = r.json()["samples"]
    print(f"2. Samples API: OK ({len(samples)} student contracts loaded)")
    assert len(samples) == 5

    # 3. Test PDF Upload & Extraction
    pdf_path = Path("test_documents/TechNova_Internship_Agreement.pdf")
    with open(pdf_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/upload", files={"file": (pdf_path.name, f, "application/pdf")})
    assert r.status_code == 200, f"PDF upload failed: {r.text}"
    pdf_data = r.json()
    print(f"3. PDF Upload & Analysis: OK (Domain: {pdf_data['domain']}, Risk Score: {pdf_data['risk_score']}/100, Clauses: {len(pdf_data['clauses'])})")
    assert pdf_data["document_info"]["format"] == "PDF"
    assert pdf_data["risk_score"] > 70
    assert len(pdf_data["clauses"]) > 5

    # 4. Test DOCX Upload & Extraction
    docx_path = Path("test_documents/Apex_Global_Placement_Bond.docx")
    with open(docx_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/upload", files={"file": (docx_path.name, f, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")})
    assert r.status_code == 200, f"DOCX upload failed: {r.text}"
    docx_data = r.json()
    print(f"4. DOCX Upload & Analysis: OK (Domain: {docx_data['domain']}, Risk Score: {docx_data['risk_score']}/100, Clauses: {len(docx_data['clauses'])})")
    assert docx_data["document_info"]["format"] == "DOCX"
    assert docx_data["risk_score"] > 70

    # 5. Test TXT Upload & Extraction
    txt_path = Path("test_documents/GreenValley_Hostel_Residency_Agreement.txt")
    with open(txt_path, "rb") as f:
        r = requests.post(f"{BASE_URL}/api/upload", files={"file": (txt_path.name, f, "text/plain")})
    assert r.status_code == 200, f"TXT upload failed: {r.text}"
    txt_data = r.json()
    print(f"5. TXT Upload & Analysis: OK (Domain: {txt_data['domain']}, Risk Score: {txt_data['risk_score']}/100, Clauses: {len(txt_data['clauses'])})")
    assert txt_data["document_info"]["format"] == "TXT"
    assert txt_data["risk_score"] > 70

    # 6. Test Interactive Q&A Engine
    qa_payload = {
        "query": "Can they hold my original college certificates or marks card as security?",
        "document_id": pdf_data["document_id"]
    }
    r = requests.post(f"{BASE_URL}/api/qa", json=qa_payload)
    assert r.status_code == 200, f"QA failed: {r.text}"
    qa_res = r.json()
    print(f"6. Q&A Engine: OK (Severity: {qa_res['risk_status']}, Citations: {len(qa_res['cited_clauses'])})")
    print(f"   AI Response: {qa_res['answer'][:120]}...")
    assert len(qa_res["cited_clauses"]) > 0
    assert "illegal" in qa_res["answer"].lower() or "violation" in qa_res["answer"].lower()

    # 7. Test PDF Report Generation & Download
    report_payload = {
        "document_id": pdf_data["document_id"],
        "analysis": pdf_data
    }
    r = requests.post(f"{BASE_URL}/api/generate-pdf", json=report_payload)
    assert r.status_code == 200, f"PDF report failed: {r.text}"
    assert r.headers.get("content-type") == "application/pdf"
    pdf_bytes = r.content
    out_pdf = Path("backend/generated_reports/Live_Audit_Report.pdf")
    with open(out_pdf, "wb") as f:
        f.write(pdf_bytes)
    print(f"7. ReportLab PDF Generation: OK ({len(pdf_bytes)} bytes written to {out_pdf})")
    assert len(pdf_bytes) > 5000

    # 8. Test Web Frontend Static Serving
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200
    assert "ClauseGuard AI" in r.text
    assert "Live Insights" in r.text
    print(f"8. Web Frontend UI Serving: OK ({len(r.text)} bytes served at http://127.0.0.1:8000)")

    print("\n========================================================")
    print("ALL 8 END-TO-END SYSTEM CAPABILITIES TESTED & VERIFIED!")
    print("========================================================")

if __name__ == "__main__":
    run_live_tests()
