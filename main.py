"""
ClauseGuard AI - FastAPI Web Application
Integrates document extraction, NLP ML classification, rule-based statutory analysis,
interactive contract viewer, Q&A reasoning, and PDF audit report generation.
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import BASE_DIR, FRONTEND_DIR, UPLOADS_DIR, REPORTS_DIR
from backend.analyzer import analyze_document_text, analyze_file
from backend.sample_agreements import SAMPLE_AGREEMENTS, get_all_samples_metadata, get_sample_by_id
from backend.qa_engine import (
    answer_student_question,
    PRESET_STUDENT_QUESTIONS,
    get_preset_questions_for_domain,
)
from backend.rule_engine import get_rule_catalog
from backend.dataset_pipeline import get_dataset_status
from backend.pdf_report import generate_pdf_report
from backend.ml_pipeline import classifier_instance

app = FastAPI(
    title="ClauseGuard AI",
    description="Smart Agreement Analysis & Risk Detection for Students and Educational Institutions",
    version="1.0.0"
)

# Enable CORS for development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def disable_static_cache(request: Request, call_next):
    """Prevent stale browser caching of HTML/JS/CSS frontend assets."""
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".html", ".js", ".css")):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# In-memory session store for current active analyses (keyed by document id/filename)
ACTIVE_SESSIONS: Dict[str, Dict[str, Any]] = {}


class QARequest(BaseModel):
    query: str
    document_id: Optional[str] = None
    clauses: Optional[list] = None
    domain: Optional[str] = "INTERNSHIP"


class TextAnalyzeRequest(BaseModel):
    text: str
    filename: Optional[str] = "Pasted_Agreement.txt"


class PDFReportRequest(BaseModel):
    document_id: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None


class CompareRequest(BaseModel):
    document_ids: Optional[List[str]] = None
    analyses: Optional[List[Dict[str, Any]]] = None


def _build_dataset_status_response() -> Dict[str, Any]:
    """Return canonical dataset readiness metadata compatible with both API clients and UI."""
    raw = get_dataset_status()
    splits = raw.get("split_counts") or {}
    leakage = raw.get("leakage_checks") or {}
    zero_group = leakage.get("zero_group_leakage")
    zero_text = leakage.get("zero_text_overlap")
    zero_leakage = bool(zero_group is True and zero_text is True) if leakage else None

    if zero_leakage is True:
        leakage_status = "Passed (0 Duplicates / 0 Leakage)"
    elif zero_leakage is False:
        leakage_status = "Attention Needed"
    else:
        leakage_status = "Not Built"

    badge = (
        "Model Retraining Pending Authorization"
        if raw.get("retraining_status") == "PENDING_AUTHORIZATION"
        else str(raw.get("retraining_status") or "Model Retraining Pending Authorization")
    )

    return {
        **raw,
        "prepared_corpus_size": raw.get("total_clauses", 0),
        "class_count": raw.get("num_categories", 0),
        "splits": splits,
        "zero_leakage": zero_leakage,
        "leakage_status": leakage_status,
        "pipeline_status_badge": badge,
        "governance_note": raw.get("note", ""),
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "app": "ClauseGuard AI",
        "ml_model_ready": classifier_instance.is_ready()
    }


@app.get("/api/samples")
def list_samples():
    """List available realistic student contracts."""
    return {
        "samples": get_all_samples_metadata()
    }


@app.get("/api/samples/{sample_id}")
def load_sample(sample_id: str):
    """Load and analyze a specific sample agreement explicitly selected by the user."""
    if sample_id not in SAMPLE_AGREEMENTS:
        raise HTTPException(status_code=404, detail=f"Sample agreement '{sample_id}' not found.")
    sample = get_sample_by_id(sample_id)
    analysis = analyze_document_text(
        text=sample["content"],
        filename=sample["filename"],
        format_type="TXT"
    )
    doc_id = f"sample_{sample_id}"
    ACTIVE_SESSIONS[doc_id] = analysis
    analysis["document_id"] = doc_id
    analysis["sample_title"] = sample["title"]
    return analysis


@app.post("/api/sessions/clear")
def clear_active_sessions():
    """Clear backend in-memory active analysis sessions."""
    ACTIVE_SESSIONS.clear()
    return {"status": "cleared", "active_sessions": 0}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF, DOCX, or TXT document and run full analysis pipeline."""
    filename = file.filename or "uploaded_agreement.txt"
    ext = Path(filename).suffix.lower()
    
    if ext not in [".pdf", ".docx", ".doc", ".txt", ".md", ".rtf"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Please upload PDF, DOCX, or TXT."
        )
        
    save_path = UPLOADS_DIR / filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        analysis = analyze_file(save_path)
        doc_id = f"doc_{int(os.path.getmtime(save_path))}_{filename}"
        analysis["document_id"] = doc_id
        ACTIVE_SESSIONS[doc_id] = analysis
        return analysis
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


@app.post("/api/analyze-text")
def analyze_raw_text(req: TextAnalyzeRequest):
    """Analyze manually entered or pasted agreement text."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty.")
        
    analysis = analyze_document_text(
        text=req.text,
        filename=req.filename or "Contract.txt",
        format_type="TXT"
    )
    doc_id = f"text_{hash(req.text[:50])}"
    analysis["document_id"] = doc_id
    ACTIVE_SESSIONS[doc_id] = analysis
    return analysis


@app.get("/api/qa/presets")
def get_qa_presets(domain: Optional[str] = Query(default=None)):
    """Retrieve student contract Q&A presets (optionally tailored to the active document domain)."""
    presets = get_preset_questions_for_domain(domain) if domain else PRESET_STUDENT_QUESTIONS
    return {"presets": presets, "domain": domain}


@app.post("/api/qa")
def ask_question(req: QARequest):
    """Perform local, API-free Q&A over the agreement with grounded citations and plain English explanation."""
    analysis = None
    if req.document_id and req.document_id in ACTIVE_SESSIONS:
        analysis = ACTIVE_SESSIONS[req.document_id]
        
    clauses = req.clauses or (analysis.get("clauses") if analysis else [])
    domain = req.domain or (analysis.get("domain") if analysis else "INTERNSHIP")
    fairness = analysis.get("fairness") if analysis else {}
    
    if not clauses:
        raise HTTPException(status_code=400, detail="No active document found to query. Please analyze a document first.")
        
    result = answer_student_question(
        query=req.query,
        analyzed_clauses=clauses,
        document_domain=domain,
        overall_fairness=fairness
    )
    return result


@app.get("/api/rules")
def list_protection_rules():
    """Return the transparent deterministic student protection and statutory context rule catalog."""
    rules = get_rule_catalog()
    return {
        "rules": rules,
        "count": len(rules),
        "disclaimer": (
            "ClauseGuard AI applies deterministic student-protection heuristics and Indian statutory "
            "context as an educational decision aid. Output does not constitute formal legal advice."
        )
    }


@app.get("/api/dataset/status")
def dataset_preparation_status():
    """Return real, audited status of the prepared clause dataset (model retraining pending authorization)."""
    return _build_dataset_status_response()


@app.get("/api/ml-eval")
def get_ml_evaluation():
    """Retrieve real ML pipeline metrics, classification report, confusion matrix, and dataset readiness."""
    metrics = dict(classifier_instance.get_metrics() or {})
    metrics["dataset_readiness"] = _build_dataset_status_response()
    return metrics


@app.post("/api/compare")
def compare_agreements(req: CompareRequest):
    """Compare two or more analyzed agreements side by side using real metrics."""
    selected: List[Dict[str, Any]] = []
    if req.document_ids:
        for doc_id in req.document_ids:
            if doc_id in ACTIVE_SESSIONS:
                selected.append(ACTIVE_SESSIONS[doc_id])
    if req.analyses:
        selected.extend(req.analyses)

    if len(selected) < 2:
        raise HTTPException(
            status_code=400,
            detail="Please provide at least two analyzed agreements to compare."
        )

    items = []
    for a in selected:
        doc_info = a.get("document_info", {})
        dist = a.get("distribution", {})
        fairness = a.get("fairness", {})
        cc_summary = a.get("cross_clause_summary", {})
        items.append({
            "document_id": a.get("document_id"),
            "filename": doc_info.get("filename", "Agreement"),
            "domain": a.get("domain", "GENERAL_STUDENT_AGREEMENT"),
            "risk_score": a.get("risk_score", 0),
            "risk_category": a.get("risk_category", "Evaluated"),
            "student_favor_pct": fairness.get("student_favor_pct", 50),
            "counterparty_favor_pct": fairness.get("counterparty_favor_pct", 50),
            "fairness_label": fairness.get("fairness_label", ""),
            "substantive_clauses": dist.get("total", doc_info.get("clause_count", 0)),
            "high_count": dist.get("high", 0),
            "medium_count": dist.get("medium", 0),
            "low_count": dist.get("low", 0),
            "structural_count": doc_info.get("structural_count", 0),
            "top_issues_count": len(a.get("top_issues", [])),
            "missing_terms_count": len(a.get("missing_information", [])),
            "cross_clause_links": cc_summary.get("total_links", doc_info.get("cross_clause_links", 0)),
            "cross_clause_clusters": len(cc_summary.get("clusters", [])),
        })

    safest = min(items, key=lambda x: x["risk_score"])
    riskiest = max(items, key=lambda x: x["risk_score"])
    return {
        "compared_count": len(items),
        "agreements": items,
        "summary": {
            "lowest_risk_filename": safest["filename"],
            "lowest_risk_score": safest["risk_score"],
            "highest_risk_filename": riskiest["filename"],
            "highest_risk_score": riskiest["risk_score"],
            "score_spread": riskiest["risk_score"] - safest["risk_score"],
        }
    }


@app.post("/api/generate-pdf")
def create_pdf_report(req: PDFReportRequest):
    """Generate and return a downloadable PDF audit report."""
    analysis = None
    if req.document_id and req.document_id in ACTIVE_SESSIONS:
        analysis = ACTIVE_SESSIONS[req.document_id]
    elif req.analysis:
        raw_text = req.analysis.get("raw_text")
        if raw_text:
            # Re-analyze with the latest backend rules/cross-clause engine if session predates server restart
            doc_info = req.analysis.get("document_info", {})
            analysis = analyze_document_text(
                text=raw_text,
                filename=doc_info.get("filename", "Agreement.txt"),
                format_type=doc_info.get("format", "TXT")
            )
            if req.document_id:
                analysis["document_id"] = req.document_id
                ACTIVE_SESSIONS[req.document_id] = analysis
        else:
            analysis = req.analysis
        
    if not analysis:
        raise HTTPException(status_code=400, detail="No document analysis data provided.")

    raw_doc_name = analysis.get("document_info", {}).get("filename", "Agreement")
    clean_stem = Path(raw_doc_name).stem.replace(" ", "_") or "Agreement"
    filename = f"ClauseGuard_Report_{clean_stem}.pdf"
    pdf_path = generate_pdf_report(analysis, filename)
    
    return FileResponse(
        path=str(pdf_path),
        filename=filename,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"
        }
    )


# Mount static frontend directory
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
