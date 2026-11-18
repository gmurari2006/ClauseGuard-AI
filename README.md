# ClauseGuard AI
**Smart Agreement Analysis and Risk Detection for Students and Educational Institutions**

ClauseGuard AI is a 100% local, API-free contract analysis and student risk-detection platform built for students, campus placement cells, and university legal-aid reviewers. It combines document extraction, clause segmentation, local machine learning classification, deterministic student-protection rules, cross-clause risk modeling, grounded local Q&A, and PDF audit report generation.

> **Educational Disclaimer:** ClauseGuard AI is an automated academic decision-support tool and does not constitute formal legal advice. Legal enforceability depends on governing law, jurisdiction, and specific facts.

---

## 1. Workspace Architecture & Primary Layout

Maintains a three-panel legal-tech workspace:

- **Left Panel (Document Management):**
  - Neutral empty startup state (no preloaded sample contract).
  - Drag-and-drop upload supporting **PDF, DOCX, and TXT**, plus **Paste Agreement Text**.
  - Multi-document list with file size, detected domain, and risk score.
  - **Compare Agreements** modal for side-by-side comparison of 2+ loaded agreements using real analyzed metrics.
  - Explicit **Sample Contracts** dropdown with 5 realistic student agreements and **Clear All** full-workspace reset.

- **Centre Panel (Original Agreement Viewer & Multi-Tab Audit):**
  - `Document`: Interactive contract reader with preserved headings, `STRUCTURAL / INFORMATIONAL` exclusion badges (`INFO`), color-coded substantive clauses (**High**, **Medium**, **Low**), clickable cross-clause links, text search, and risk filter chips.
  - `Clause Inspector`: Slide-out inspection drawer showing category, severity, attribution source, original quote, cross-clause references, student explanation, and copyable counter-clause.
  - `Summary`: Executive brief, **Transparent Risk Score Calculation** breakdown, **Cross-Clause Risk Clusters** with jump-to-clause chips, and educational disclaimer.
  - `Contract Review`: Structured clause-by-clause audit table with inline severity/structural filter chips and one-click **Copy Negotiation Checklist**.
  - `Q&A`: 100% local, API-free student Q&A with domain-specific presets, direct clause lookup (`"Explain Clause 4.2"`), grounding/confidence pills, cross-clause context notes, clickable clause citations, and honest `insufficient_evidence` refusal when an agreement lacks information.

- **Right Panel (Live Risk Insights & Analytics):**
  - **Overall Risk Score (10–98 bounded scale)** circular gauge and category label.
  - **Student vs Counterparty Balance Meter** with mutually exclusive clause counts (`unilateral + moderate_caution + bilateral == substantive clauses`).
  - **Clause Distribution** bar, category tags, and active cross-clause links indicator.
  - **Top Contract Vulnerabilities**, **Missing or Unclear Terms** (with severity badges & recommendations), and **Student Negotiation Playbook**.
  - **Download PDF Report**: Multi-page ReportLab audit report reconciled with all live UI metrics.

---

## 2. Design System Tokens

- **Deep Navy:** `#102B35`
- **Forest Teal:** `#245B52`
- **Sage:** `#D7E6DF`
- **Muted Gold:** `#E8C78B`
- **Off-White / Warm Background:** `#F3F5F2`
- **Risk Accents:** High `#DC2626` / `#FEE2E2` · Medium `#D97706` / `#FEF3C7` · Low `#16A34A` / `#DCFCE7`

---

## 3. Hybrid Analysis, Dataset Pipeline & Legal Context

### A. Active Baseline ML Classifier (`backend/ml_pipeline.py`)
- **Architecture:** Sublinear TF-IDF (1–2 N-grams, 2,500 features) + Calibrated Multiclass Logistic Regression.
- **Active Baseline Corpus:** 125 annotated clauses across 10 `CLAUSE_CATEGORIES` (Holdout Accuracy: `80.0%`, 5-Fold CV Mean: `67.2%`, Weighted F1: `0.798` — see [evaluation_report.md](backend/models/evaluation_report.md) and [metrics.json](backend/models/metrics.json)).
- **Retraining Governance:** Active baseline artifacts in `backend/models/` have **not** been retrained yet (`retraining_status: "PENDING_AUTHORIZATION"`).

### B. Auditable Dataset Preparation Pipeline (`backend/dataset_pipeline.py` & `datasets/`)
- Prepared **360 validated, deduplicated substantive clauses** (exactly **36 per class** across all 10 `CLAUSE_CATEGORIES` and all 6 `AGREEMENT_DOMAINS`) plus **40 structural/informational examples** (`datasets/structural_examples.jsonl`).
- Enforces automated quality gates: schema validation, length bounds (35–1,500 chars), exact hash deduplication, near-duplicate Jaccard filtering (`>= 0.85`), and stratified group-aware `train / val / test` splits (`245 / 56 / 59`) verified for **zero duplicate overlap** and **zero source-group leakage**.
- Full documentation in [DATASET_CARD.md](datasets/DATASET_CARD.md) and machine-readable metrics in [dataset_report.json](datasets/dataset_report.json).

### C. Deterministic Rules (`backend/rule_engine.py`) & Cross-Clause Engine (`backend/cross_clause.py`)
- **Weighted Multi-Signal Domain Detection** across 6 student agreement domains.
- **13 Deterministic Student Protection & Statutory Context Rules** (inspectable via `GET /api/rules`), citing only verified Indian statutory provisions where applicable (Indian Contract Act, 1872 — Sec. 27, 74, 124; Copyright Act, 1957 — Sec. 17 & 57; Consumer Protection Act, 2019; Arbitration and Conciliation Act, 1996) alongside clearly labeled student-protection heuristics.
- **Cross-Clause Analysis (`CC-1`–`CC-7`)**: Condition-aware escalation (e.g., mandatory lock-in linked to a severe financial exit penalty elevates to HIGH, whereas a standalone no-penalty lock-in stays MEDIUM), career/portfolio NDA+IP lock-out linking, stipend-to-withholding cross-references, and connected compound risk clusters (`cross_clause_summary`).

---

## 4. API Endpoints

- `GET /api/health` — Application & ML model readiness check
- `GET /api/samples` & `GET /api/samples/{sample_id}` — Sample contract catalog and analysis
- `POST /api/upload` — Upload and analyze PDF, DOCX, or TXT agreements (returns `400` on empty/corrupted/encrypted files)
- `POST /api/analyze-text` — Analyze pasted agreement text
- `POST /api/sessions/clear` — Clear in-memory server sessions
- `GET /api/qa/presets?domain=...` — Default or domain-tailored student Q&A presets
- `POST /api/qa` — Local API-free Q&A with grounding status, confidence, citations, and cross-clause notes
- `GET /api/rules` — Deterministic protection rule catalog & disclaimer
- `GET /api/dataset/status` — Audited dataset preparation status & split metrics
- `GET /api/ml-eval` — Baseline ML evaluation metrics + dataset readiness
- `POST /api/compare` — Side-by-side comparison across 2+ analyzed agreements
- `POST /api/generate-pdf` — Generate and download reconciled PDF audit report

---

## 5. Running the Application & Tests

```bash
pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### Verification Suites
```bash
# Full automated pytest suite (extraction, dataset hygiene, rules, cross-clause, Q&A reliability, API & E2E)
pytest -v test_backend.py tests/

# Legacy direct module smoke check
python run_tests.py

# Regenerate dataset artifacts (does NOT retrain ML model)
python -m backend.dataset_pipeline
```

---

## 6. Project Structure

```text
ClauseGuard-AI/
├── backend/
│   ├── analyzer.py           # Unified analysis orchestrator
│   ├── config.py             # Paths, palette tokens, domains & categories
│   ├── cross_clause.py       # Cross-clause linking (CC-1..CC-7) & risk cluster builder
│   ├── dataset_pipeline.py   # Auditable dataset preparation & leakage-free splitting
│   ├── extractor.py          # PDF, DOCX, TXT extraction, artifact stripping & segmentation
│   ├── ml_pipeline.py        # Baseline TF-IDF + Logistic Regression inference wrapper
│   ├── pdf_report.py         # Multi-page ReportLab PDF audit report generator
│   ├── qa_engine.py          # Local API-free Q&A retrieval, grounding & synthesis
│   ├── rule_engine.py        # Weighted domain detector, 13 rules, fairness & risk math
│   ├── sample_agreements.py  # 5 realistic student agreements
│   ├── train_eval_model.py   # Baseline training script (untouched pending authorization)
│   └── models/               # Saved baseline model artifacts & evaluation report
├── datasets/                 # Prepared 360-clause master corpus, splits & DATASET_CARD.md
├── frontend/                 # Three-panel legal-tech web workspace (HTML/CSS/JS)
├── test_documents/           # Sample PDF, DOCX, TXT test agreements
├── tests/                    # Comprehensive unit, reliability, and API integration suites
├── main.py                   # FastAPI application & API endpoints
├── run_tests.py              # Direct core module check script
├── test_backend.py           # Core backend regression tests
└── requirements.txt
```
