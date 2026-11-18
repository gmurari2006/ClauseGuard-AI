"""
ClauseGuard AI - Comprehensive FastAPI Integration, API Contract & E2E Compatibility Suite
Verifies all existing and newly upgraded API endpoints, error handling, PDF export headers,
frontend static delivery & honesty checks, cross-module data consistency, and zero ML retraining.
"""
import hashlib
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from main import app, ACTIVE_SESSIONS

client = TestClient(app)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


class TestCoreAndUpgradedAPIContracts:
    def test_health_endpoint(self):
        res = client.get("/api/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "healthy"
        assert data["app"] == "ClauseGuard AI"
        assert data["ml_model_ready"] is True

    def test_samples_list_and_unknown_404(self):
        res = client.get("/api/samples")
        assert res.status_code == 200
        samples = res.json()["samples"]
        assert len(samples) == 5

        missing = client.get("/api/samples/nonexistent_sample_xyz")
        assert missing.status_code == 404

    def test_sample_analysis_includes_cross_clause_summary_and_reconciled_counts(self):
        res = client.get("/api/samples/sample_internship")
        assert res.status_code == 200
        data = res.json()
        assert data["domain"] == "INTERNSHIP"
        assert "cross_clause_summary" in data
        cc = data["cross_clause_summary"]
        assert set(cc.keys()) == {"total_links", "elevated_clauses_count", "edges", "clusters"}
        dist = data["distribution"]
        assert dist["high"] + dist["medium"] + dist["low"] == dist["total"]
        assert dist["total"] == data["document_info"]["clause_count"]

    def test_upload_cloudpulse_pdf_and_extraction_metadata(self):
        pdf_path = Path("test_documents/CloudPulse_Student_Internship_Test_Agreement.pdf")
        assert pdf_path.exists()
        with open(pdf_path, "rb") as f:
            res = client.post(
                "/api/upload",
                files={"file": (pdf_path.name, f, "application/pdf")},
            )
        assert res.status_code == 200
        data = res.json()
        assert data["domain"] == "INTERNSHIP"
        assert data["risk_score"] == 98
        assert data["distribution"]["high"] == 9
        assert data["distribution"]["medium"] == 4
        assert data["distribution"]["low"] == 4
        assert data["distribution"]["total"] == 17
        assert data["document_info"]["structural_count"] == 10
        assert "extraction_metadata" in data["document_info"]
        meta = data["document_info"]["extraction_metadata"]
        assert meta["pages"] >= 1
        assert meta["word_count"] > 200
        assert data["cross_clause_summary"]["total_links"] >= 4
        assert len(data["cross_clause_summary"]["clusters"]) >= 1

    def test_upload_invalid_or_empty_file_returns_400(self):
        res_empty = client.post(
            "/api/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        assert res_empty.status_code == 400
        assert "empty" in res_empty.json()["detail"].lower()

        res_bad_ext = client.post(
            "/api/upload",
            files={"file": ("malware.exe", b"MZ...", "application/octet-stream")},
        )
        assert res_bad_ext.status_code == 400

    def test_qa_presets_default_and_domain_specific(self):
        res_default = client.get("/api/qa/presets")
        assert res_default.status_code == 200
        assert len(res_default.json()["presets"]) >= 5

        res_hostel = client.get("/api/qa/presets?domain=HOSTEL_RESIDENCY")
        assert res_hostel.status_code == 200
        hostel_presets = res_hostel.json()["presets"]
        assert len(hostel_presets) >= 5
        joined = " ".join(p["question"].lower() for p in hostel_presets)
        assert "curfew" in joined or "deposit" in joined or "hostel" in joined

    def test_qa_endpoint_grounded_and_insufficient_evidence(self):
        pdf_path = Path("test_documents/CloudPulse_Student_Internship_Test_Agreement.pdf")
        with open(pdf_path, "rb") as f:
            up = client.post("/api/upload", files={"file": (pdf_path.name, f, "application/pdf")})
        doc_id = up.json()["document_id"]

        res_term = client.post(
            "/api/qa",
            json={
                "query": "Can the company terminate me immediately while requiring me to give 60 days notice?",
                "document_id": doc_id,
            },
        )
        assert res_term.status_code == 200
        body = res_term.json()
        assert body["grounding_status"] == "grounded"
        assert body["confidence"] in ("high", "medium")
        cited_refs = {c["clause_ref"] for c in body["cited_clauses"]}
        assert cited_refs == {"Clause 5.1", "Clause 5.2"}

        res_oob = client.post(
            "/api/qa",
            json={
                "query": "Does this agreement offer dental insurance or 401k matching?",
                "document_id": doc_id,
            },
        )
        assert res_oob.status_code == 200
        oob = res_oob.json()
        assert oob["grounding_status"] == "insufficient_evidence"
        assert oob["confidence"] == "insufficient_evidence"
        assert oob["cited_clauses"] == []

    def test_rules_catalog_endpoint(self):
        res = client.get("/api/rules")
        assert res.status_code == 200
        data = res.json()
        assert data["count"] == len(data["rules"])
        assert data["count"] >= 13
        assert "not constitute formal legal advice" in data["disclaimer"].lower()
        rule_ids = {r["id"] for r in data["rules"]}
        assert "RULE_SECTION_27_RESTRAINT_OF_TRADE" in rule_ids
        assert "RULE_EXCESSIVE_TRAINING_BOND" in rule_ids
        assert "RULE_UNILATERAL_AMENDMENT" in rule_ids
        assert "RULE_FOREIGN_COSTLY_ARBITRATION" in rule_ids

    def test_dataset_status_and_ml_eval_endpoints(self):
        res_ds = client.get("/api/dataset/status")
        assert res_ds.status_code == 200
        ds = res_ds.json()
        assert ds["dataset_ready"] is True
        assert ds["prepared_corpus_size"] >= 300
        assert ds["class_count"] == 10
        assert ds["zero_leakage"] is True
        assert ds["retraining_status"] == "PENDING_AUTHORIZATION"
        assert ds["active_model_retrained"] is False
        assert sum(ds["splits"].values()) == ds["prepared_corpus_size"]

        res_ml = client.get("/api/ml-eval")
        assert res_ml.status_code == 200
        ml = res_ml.json()
        assert ml["dataset_size"] == 125  # Active baseline model unchanged
        assert "weighted_avg_f1" in ml
        assert ml["dataset_readiness"]["prepared_corpus_size"] == ds["prepared_corpus_size"]

    def test_compare_endpoint_and_validation(self):
        r1 = client.get("/api/samples/sample_internship").json()
        r2 = client.get("/api/samples/sample_hostel").json()

        bad = client.post("/api/compare", json={"document_ids": [r1["document_id"]]})
        assert bad.status_code == 400

        res = client.post(
            "/api/compare",
            json={"document_ids": [r1["document_id"], r2["document_id"]]},
        )
        assert res.status_code == 200
        cmp_data = res.json()
        assert cmp_data["compared_count"] == 2
        assert len(cmp_data["agreements"]) == 2
        assert "score_spread" in cmp_data["summary"]

    def test_generate_pdf_clean_filename_and_valid_bytes(self):
        pdf_path = Path("test_documents/CloudPulse_Student_Internship_Test_Agreement.pdf")
        with open(pdf_path, "rb") as f:
            up = client.post("/api/upload", files={"file": (pdf_path.name, f, "application/pdf")})
        doc_id = up.json()["document_id"]

        res = client.post("/api/generate-pdf", json={"document_id": doc_id})
        assert res.status_code == 200
        assert res.headers["content-type"] == "application/pdf"
        cd = res.headers.get("content-disposition", "")
        assert ".pdf.pdf" not in cd.lower()
        assert "ClauseGuard_Report_CloudPulse_Student_Internship_Test_Agreement.pdf" in cd
        assert res.content.startswith(b"%PDF-")
        assert len(res.content) > 5000

    def test_clear_sessions_endpoint(self):
        client.get("/api/samples/sample_hostel")
        assert len(ACTIVE_SESSIONS) >= 1
        res = client.post("/api/sessions/clear")
        assert res.status_code == 200
        assert res.json()["active_sessions"] == 0
        assert len(ACTIVE_SESSIONS) == 0


class TestFrontendDeliveryAndModelGovernance:
    def test_index_html_no_store_and_honest_copy(self):
        res = client.get("/")
        assert res.status_code == 200
        assert "no-store" in res.headers.get("cache-control", "").lower()
        html = res.text
        assert "ML Pipeline: 80% F1" not in html
        assert "AICTE/UGC" not in html
        assert "mlPipelineBtnLabel" in html
        assert "btnCompareDocs" in html
        assert "compareModal" in html
        assert "Upload an agreement to begin analysis." in html
        assert "Insights will appear after document analysis." in html

    def test_baseline_ml_models_not_retrained(self):
        """Guarantees baseline ML artifacts were NOT altered during this upgrade phase."""
        expected = {
            "clause_classifier.joblib": "940C74A1241D22D3C64391406F25FCDA82B4ADD475AD6A2EF63AC334D00AD4B0",
            "tfidf_vectorizer.joblib": "1E81C8D23266020EA7765637DF04A002017666BC7F7B1EF835FC17DC5D470AE0",
            "metrics.json": "2237D823C025AEA5DC0CCC9A7C521496244779B72A777DBAD6F59E48B70D5B26",
        }
        for filename, expected_hash in expected.items():
            path = Path("backend/models") / filename
            assert path.exists(), f"Missing baseline model artifact: {path}"
            assert _sha256(path) == expected_hash, f"Baseline model artifact was modified: {filename}"
