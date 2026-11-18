"""
ClauseGuard AI - Q&A Retrieval, Citation Precision, Grounding & Reliability Test Suite
Verifies 100% local, API-free Q&A accuracy across:
  - All 4 CloudPulse Student Internship benchmark questions (exact clause precision)
  - Direct clause lookup (single clause, linked cross-clauses, and missing clause refusal)
  - Overall risk summary grounded strictly in document risk counts & fairness metrics
  - Academic exam leave & Hostel rules/curfew intents
  - Multi-domain sample agreements (Hostel, Placement Bond, Internship, Sponsorship, Service)
  - Out-of-scope / unanswerable question rejection (honest insufficient_evidence)
  - Domain-tailored preset questions & backward-compatible response schema
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import analyze_file, analyze_document_text
from backend.sample_agreements import SAMPLE_AGREEMENTS
from backend.qa_engine import (
    PRESET_STUDENT_QUESTIONS,
    answer_student_question,
    get_preset_questions_for_domain,
)


REQUIRED_TOP_LEVEL_KEYS = {
    "query",
    "intent",
    "answer",
    "risk_status",
    "cited_clauses",
    "grounding_status",
    "confidence",
    "retrieval_stats",
    "cross_clause_notes",
    "structured_answer",
}

REQUIRED_CITED_CLAUSE_KEYS = {
    "index",
    "clause_ref",
    "category",
    "risk_level",
    "text",
    "relevance_score",
    "explanation",
    "action",
    "related_clauses",
}

REQUIRED_STRUCTURED_ANSWER_KEYS = {
    "direct_answer",
    "relevant_clauses",
    "supporting_quotes",
    "explanation",
    "recommendation",
}


@pytest.fixture(scope="module")
def cloudpulse_analysis():
    pdf_path = Path("test_documents/CloudPulse_Student_Internship_Test_Agreement.pdf")
    assert pdf_path.exists(), f"Missing benchmark test file: {pdf_path}"
    return analyze_file(pdf_path)


@pytest.fixture(scope="module")
def sample_analyses():
    out = {}
    for sample_id, sample in SAMPLE_AGREEMENTS.items():
        out[sample_id] = analyze_document_text(sample["content"], sample["filename"])
    return out


def _assert_response_schema(res: dict):
    assert REQUIRED_TOP_LEVEL_KEYS.issubset(res.keys())
    assert res["grounding_status"] in ("grounded", "insufficient_evidence")
    assert res["confidence"] in ("high", "medium", "low", "insufficient_evidence")

    stats = res["retrieval_stats"]
    assert set(stats.keys()) == {"candidate_count", "cited_count", "top_score", "secondary_intent"}
    assert isinstance(stats["candidate_count"], int)
    assert stats["cited_count"] == len(res["cited_clauses"])
    assert isinstance(stats["top_score"], float)

    assert isinstance(res["cross_clause_notes"], list)

    sa = res["structured_answer"]
    assert REQUIRED_STRUCTURED_ANSWER_KEYS.issubset(sa.keys())
    assert isinstance(sa["relevant_clauses"], list)
    assert isinstance(sa["supporting_quotes"], list)

    for c in res["cited_clauses"]:
        assert REQUIRED_CITED_CLAUSE_KEYS.issubset(c.keys())
        assert isinstance(c["related_clauses"], list)


# ---------------------------------------------------------------------------
# 1. CloudPulse 4 Benchmark Questions (Exact Clause Precision)
# ---------------------------------------------------------------------------

def test_cloudpulse_benchmark_q1_exit_penalty(cloudpulse_analysis):
    res = answer_student_question(
        query="Can the company charge me ₹75,000 if I leave early?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert refs == ["Clause 4.1", "Clause 4.2", "Clause 4.3"]
    assert res["intent"] == "EXIT_PENALTY"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    assert "75,000" in res["answer"]
    assert "withhold" in res["answer"].lower()
    assert any("Clause 4.1" in n and "Clause 4.2" in n for n in res["cross_clause_notes"])


def test_cloudpulse_benchmark_q2_security_deposit(cloudpulse_analysis):
    res = answer_student_question(
        query="Can the company keep my ₹15,000 security deposit?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert refs == ["Clause 3.1", "Clause 3.2", "Clause 3.3"]
    assert res["intent"] == "SECURITY_DEPOSIT"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    ans_lower = res["answer"].lower()
    assert "45 business days" in ans_lower
    assert "15,000" in ans_lower
    assert "forfeit" in ans_lower or "non-refundable" in ans_lower
    assert any("Clause 3.3" in n for n in res["cross_clause_notes"])


def test_cloudpulse_benchmark_q3_personal_laptop_ip(cloudpulse_analysis):
    res = answer_student_question(
        query="Who owns the code I develop on my personal laptop?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert refs == ["Clause 7.1", "Clause 7.2"]
    assert res["intent"] == "INTELLECTUAL_PROPERTY"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    ans_lower = res["answer"].lower()
    assert "personal devices" in ans_lower
    assert "capstone" in ans_lower or "moral rights" in ans_lower


def test_cloudpulse_benchmark_q4_termination_notice_only(cloudpulse_analysis):
    res = answer_student_question(
        query="Can the company terminate me immediately while requiring me to give 60 days notice?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    # Must be strictly Clauses 5.1 and 5.2 ONLY (no 4.x or 6.x pollution!)
    assert refs == ["Clause 5.1", "Clause 5.2"]
    assert all(not r.startswith(("Clause 4.", "Clause 6.")) for r in refs)
    assert res["intent"] == "TERMINATION_NOTICE"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    ans_lower = res["answer"].lower()
    assert "immediately" in ans_lower
    assert "60 days" in ans_lower
    # Cross-clause notes should surface linked risks without polluting cited_clauses
    assert any("Clause 5.2" in n and "Clause 4.2" in n for n in res["cross_clause_notes"])


# ---------------------------------------------------------------------------
# 2. Direct Clause Lookup
# ---------------------------------------------------------------------------

def test_direct_clause_lookup_4_2(cloudpulse_analysis):
    res = answer_student_question(
        query="Explain Clause 4.2",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "DIRECT_CLAUSE_LOOKUP"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert "Clause 4.2" in refs
    assert "75,000" in res["answer"]
    assert res["structured_answer"]["relevant_clauses"] == refs


def test_direct_clause_lookup_3_3(cloudpulse_analysis):
    res = answer_student_question(
        query="What does Clause 3.3 say?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "DIRECT_CLAUSE_LOOKUP"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert refs == ["Clause 3.3"]
    assert "15,000" in res["answer"]
    assert "80%" in res["answer"]


def test_direct_clause_lookup_includes_linked_related_clauses(cloudpulse_analysis):
    res = answer_student_question(
        query="Tell me about Section 4.1",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "DIRECT_CLAUSE_LOOKUP"
    assert res["grounding_status"] == "grounded"
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert refs[0] == "Clause 4.1"
    # Clause 4.1 explicitly links to Clause 4.2 and Clause 4.3
    assert "Clause 4.2" in refs
    assert "Clause 4.3" in refs
    assert len(res["cross_clause_notes"]) >= 2


def test_direct_clause_lookup_nonexistent_returns_insufficient_evidence(cloudpulse_analysis):
    res = answer_student_question(
        query="Explain Clause 99.9",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["grounding_status"] == "insufficient_evidence"
    assert res["confidence"] == "insufficient_evidence"
    assert res["cited_clauses"] == []
    assert "99.9" in res["answer"]


# ---------------------------------------------------------------------------
# 3. Overall Risk Summary & Academic Exam Leave
# ---------------------------------------------------------------------------

def test_overall_risk_summary_cloudpulse(cloudpulse_analysis):
    res = answer_student_question(
        query="What are the biggest red flags in this agreement?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "OVERALL_RISK_SUMMARY"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    assert len(res["cited_clauses"]) >= 3
    assert all(c["risk_level"] == "high" for c in res["cited_clauses"])
    # Grounded in actual fairness numbers from analysis
    assert f"{cloudpulse_analysis['fairness']['student_favor_pct']}%" in res["answer"]
    assert f"{cloudpulse_analysis['fairness']['counterparty_favor_pct']}%" in res["answer"]


def test_academic_exam_leave_intent_cloudpulse(cloudpulse_analysis):
    res = answer_student_question(
        query="Can I take leave for my semester exams?",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "ACADEMIC_EXAM_LEAVE"
    assert res["grounding_status"] == "grounded"
    refs = [c["clause_ref"] for c in res["cited_clauses"]]
    assert "Clause 5.2" in refs
    assert "exam leave" in res["answer"].lower()


# ---------------------------------------------------------------------------
# 4. Multi-Domain Q&A: Hostel, Placement Bond, Internship, Sponsorship
# ---------------------------------------------------------------------------

def test_hostel_curfew_and_eviction_qa(sample_analyses):
    hostel = sample_analyses["sample_hostel"]
    res = answer_student_question(
        query="What are the curfew hours and can the warden evict me without appeal?",
        analyzed_clauses=hostel["clauses"],
        document_domain=hostel["domain"],
        overall_fairness=hostel["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "HOSTEL_RULES_CURFEW"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] in ("high", "medium")
    assert len(res["cited_clauses"]) >= 1
    ans_lower = res["answer"].lower()
    assert "9:00 pm" in ans_lower
    assert "eviction" in ans_lower or "appeal" in ans_lower


def test_hostel_deposit_qa(sample_analyses):
    hostel = sample_analyses["sample_hostel"]
    res = answer_student_question(
        query="Can the hostel keep my Rs. 35,000 security deposit or deduct arbitrary fees?",
        analyzed_clauses=hostel["clauses"],
        document_domain=hostel["domain"],
        overall_fairness=hostel["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "SECURITY_DEPOSIT"
    assert res["grounding_status"] == "grounded"
    assert res["risk_status"] == "high"
    assert len(res["cited_clauses"]) >= 1
    assert "35,000" in res["answer"]
    assert "non-refundable" in res["answer"].lower() or "forfeit" in res["answer"].lower()


def test_placement_bond_3_year_penalty_qa(sample_analyses):
    placement = sample_analyses["sample_placement_bond"]
    res = answer_student_question(
        query="Do I have to serve a 3-year bond or pay Rs. 2,50,000 if I leave early?",
        analyzed_clauses=placement["clauses"],
        document_domain="PLACEMENT_EMPLOYMENT",
        overall_fairness=placement["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "EXIT_PENALTY"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    assert len(res["cited_clauses"]) >= 1
    assert "2,50,000" in res["answer"]
    assert "36" in res["answer"] or "three years" in res["answer"].lower()


def test_placement_bond_vs_internship_original_degree_certificates_qa(sample_analyses):
    # sample_placement_bond does NOT mention original degree certificates (only a promissory note)
    placement = sample_analyses["sample_placement_bond"]
    res_place = answer_student_question(
        query="Are they allowed to keep my original degree certificates?",
        analyzed_clauses=placement["clauses"],
        document_domain="PLACEMENT_EMPLOYMENT",
        overall_fairness=placement["fairness"],
    )
    _assert_response_schema(res_place)
    assert res_place["grounding_status"] == "insufficient_evidence"
    assert res_place["cited_clauses"] == []

    # sample_internship Clause 3.3 explicitly demands original degree certificates, marks cards, passport
    intern = sample_analyses["sample_internship"]
    res_intern = answer_student_question(
        query="Are they allowed to keep my original degree certificates?",
        analyzed_clauses=intern["clauses"],
        document_domain="INTERNSHIP",
        overall_fairness=intern["fairness"],
    )
    _assert_response_schema(res_intern)
    assert res_intern["intent"] == "CERTIFICATE_RETENTION"
    assert res_intern["grounding_status"] == "grounded"
    assert res_intern["risk_status"] == "high"
    assert len(res_intern["cited_clauses"]) >= 1
    assert "original degree certificates" in res_intern["answer"].lower()


def test_sponsorship_organizer_personal_guarantee_qa(sample_analyses):
    sponsorship = sample_analyses["sample_sponsorship"]
    res = answer_student_question(
        query="Do student club organizers personally guarantee cancellation losses or low attendee turnout?",
        analyzed_clauses=sponsorship["clauses"],
        document_domain=sponsorship["domain"],
        overall_fairness=sponsorship["fairness"],
    )
    _assert_response_schema(res)
    assert res["intent"] == "LIABILITY_INDEMNITY"
    assert res["grounding_status"] == "grounded"
    assert res["confidence"] == "high"
    assert res["risk_status"] == "high"
    assert len(res["cited_clauses"]) >= 1
    ans_lower = res["answer"].lower()
    assert "personally guarantee" in ans_lower
    assert "footfall" in ans_lower or "turnout" in ans_lower


# ---------------------------------------------------------------------------
# 5. Out-of-Scope / Unanswerable Questions (Strict Non-Hallucination)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "query",
    [
        "Does this offer dental insurance?",
        "Does this agreement provide health insurance or dental coverage?",
        "What is the 401k matching rate?",
        "What is the stock option vesting schedule?",
        "Does the company reimburse flight tickets or relocation expenses?",
        "Can I bring my pet dog to the office?",
    ],
)
def test_unanswerable_questions_return_insufficient_evidence(cloudpulse_analysis, query):
    res = answer_student_question(
        query=query,
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain=cloudpulse_analysis["domain"],
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res)
    assert res["grounding_status"] == "insufficient_evidence"
    assert res["confidence"] == "insufficient_evidence"
    assert res["cited_clauses"] == []
    assert res["structured_answer"]["relevant_clauses"] == []
    assert res["structured_answer"]["supporting_quotes"] == []
    assert "does not contain enough information" in res["answer"].lower()


def test_empty_query_and_no_document_edge_cases(cloudpulse_analysis):
    res_empty = answer_student_question(
        query="   ",
        analyzed_clauses=cloudpulse_analysis["clauses"],
        document_domain="INTERNSHIP",
        overall_fairness=cloudpulse_analysis["fairness"],
    )
    _assert_response_schema(res_empty)
    assert res_empty["intent"] == "EMPTY"
    assert res_empty["grounding_status"] == "insufficient_evidence"
    assert res_empty["cited_clauses"] == []

    res_nodoc = answer_student_question(
        query="Can they charge an exit penalty?",
        analyzed_clauses=[],
        document_domain="INTERNSHIP",
        overall_fairness={},
    )
    _assert_response_schema(res_nodoc)
    assert res_nodoc["intent"] == "NO_DOCUMENT"
    assert res_nodoc["grounding_status"] == "insufficient_evidence"
    assert res_nodoc["cited_clauses"] == []


# ---------------------------------------------------------------------------
# 6. Domain-Tailored Preset Questions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "domain",
    [
        "INTERNSHIP",
        "PLACEMENT_EMPLOYMENT",
        "HOSTEL_RESIDENCY",
        "EVENT_SPONSORSHIP",
        "STUDENT_SERVICE",
    ],
)
def test_get_preset_questions_for_every_supported_domain(domain):
    presets = get_preset_questions_for_domain(domain)
    assert isinstance(presets, list)
    assert len(presets) >= 5
    for item in presets:
        assert {"id", "question", "topic"}.issubset(item.keys())
        assert item["question"].strip()
        assert item["topic"].strip()

    # Case-insensitive lookup works identically
    assert get_preset_questions_for_domain(domain.lower()) == presets


def test_get_preset_questions_fallback_for_none_and_unknown():
    assert get_preset_questions_for_domain(None) == PRESET_STUDENT_QUESTIONS
    assert get_preset_questions_for_domain("") == PRESET_STUDENT_QUESTIONS
    assert get_preset_questions_for_domain("UNKNOWN_DOMAIN") == PRESET_STUDENT_QUESTIONS

    # Ensure returned list is a fresh copy so mutation does not corrupt PRESET_STUDENT_QUESTIONS
    copy_list = get_preset_questions_for_domain(None)
    copy_list[0]["topic"] = "MUTATED"
    assert PRESET_STUDENT_QUESTIONS[0]["topic"] != "MUTATED"
