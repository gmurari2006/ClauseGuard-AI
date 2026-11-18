# ClauseGuard AI — Student Agreement Clause Dataset Card

**Dataset Version:** `2.0.0-prepared`  
**Generated At (UTC):** `2026-11-18T14:11:00Z`  
**Total Validated Substantive Clauses:** `360` (expanded from `125` baseline clauses)  
**Separate Structural/Informational Examples:** `40`  
**Distinct Document/Template Source Groups:** `41`  
**Active ML Model Retraining Status:** `PENDING_AUTHORIZATION`  

> **Honest Status Notice:** This dataset has been prepared, validated, and split into leakage-free
> `train` / `val` / `test` partitions for future supervised training and evaluation. The currently
> active baseline classifier in `backend/models/` was trained on the original 125-clause corpus in
> `backend/train_eval_model.py` and has **NOT** been retrained yet (`PENDING_AUTHORIZATION`).

---

## 1. Purpose & Scope

This dataset supports multi-class legal clause categorization and student risk analysis across
six student-facing contract domains: `INTERNSHIP`, `PLACEMENT_EMPLOYMENT`, `HOSTEL_RESIDENCY`,
`EVENT_SPONSORSHIP`, `STUDENT_SERVICE`, and `GENERAL_STUDENT_AGREEMENT`.

A separate file (`datasets/structural_examples.jsonl`) stores non-substantive document blocks
(headers, recitals, metadata references, section titles, signature blocks, and disclaimers) to
benchmark structural filtering without contaminating substantive clause classification.

---

## 2. Schema (`clauses_master.jsonl` & `clauses_master.csv`)

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Deterministic clause identifier (`CG-0001` …) |
| `text` | `string` | Clean contractual clause text (35–1500 chars) |
| `category` | `string` | One of the 10 `CLAUSE_CATEGORIES` from `backend/config.py` |
| `risk_level` | `string` | Per-clause student risk grade (`high`, `medium`, `low`) |
| `domain` | `string` | One of the 6 `AGREEMENT_DOMAINS` from `backend/config.py` |
| `source_type` | `string` | `curated_synthetic` or `sample_agreement_extracted` |
| `source_group` | `string` | Document/template group ID enforced atomically across splits |
| `clause_ref` | `string` | Clause reference label (e.g., `Clause 3.2`) |
| `rationale` | `string` | Auditable annotation justification for category and risk level |
| `split` | `string` | Partition assignment (`train`, `val`, or `test`) |

---

## 3. Category Distribution (All 10 Classes)

| Category | Total | Train | Val | Test |
| :--- | :---: | :---: | :---: | :---: |
| `FINANCIAL_PENALTY_BOND` | 36 | 24 | 7 | 5 |
| `STIPEND_PAYMENT_TERMS` | 36 | 26 | 5 | 5 |
| `DEPOSIT_REFUND` | 36 | 25 | 5 | 6 |
| `TERMINATION_NOTICE` | 36 | 23 | 7 | 6 |
| `CONFIDENTIALITY_NDA` | 36 | 24 | 5 | 7 |
| `INTELLECTUAL_PROPERTY` | 36 | 24 | 6 | 6 |
| `RESTRICTIVE_NON_COMPETE` | 36 | 25 | 5 | 6 |
| `LIABILITY_INDEMNITY` | 36 | 25 | 5 | 6 |
| `DISPUTE_JURISDICTION` | 36 | 26 | 5 | 5 |
| `STANDARD_FAIR` | 36 | 23 | 6 | 7 |

---

## 4. Domain, Risk Level & Provenance Breakdown

### Domains
| Domain | Count |
| :--- | :---: |
| `INTERNSHIP` | 106 |
| `PLACEMENT_EMPLOYMENT` | 72 |
| `HOSTEL_RESIDENCY` | 56 |
| `EVENT_SPONSORSHIP` | 38 |
| `STUDENT_SERVICE` | 51 |
| `GENERAL_STUDENT_AGREEMENT` | 37 |

### Risk Levels
| Risk Level | Count |
| :--- | :---: |
| `high` | 173 |
| `medium` | 41 |
| `low` | 146 |

### Source Provenance
| Source Type | Count |
| :--- | :---: |
| `curated_synthetic` | 292 |
| `sample_agreement_extracted` | 68 |

---

## 5. Automated Quality Gates & Leakage Verification

- **Raw Input Records:** `360`
- **Validated Accepted Records:** `360`
- **Schema Rejected:** `0`
- **Length Rejected (outside 35–1500 chars):** `0`
- **Exact Duplicates Removed:** `0`
- **Near-Duplicates Removed (Token Jaccard >= 0.85):** `0`
- **Split Counts:** `train=245` (68.1%), `val=56` (15.6%), `test=59` (16.4%)
- **Zero Source-Group Leakage Across Splits:** `True`
- **Zero Text Overlap Across Splits:** `True`
- **All 10 Classes Present in Train/Val/Test:** `True`

---

## 6. Ethical & Legal Limitations

- Clauses are synthetic or extracted from fictional sample agreements; no real student PII is included.
- Annotations reflect student-protection educational framing and are not formal legal advice.
- Any future model retraining using these splits must preserve `source_group` boundaries and report held-out `test.jsonl` metrics transparently.
