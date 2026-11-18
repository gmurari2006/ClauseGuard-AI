"""
Comprehensive unit and integration tests for Workstream 2:
- Weighted multi-signal domain detection (backend/rule_engine.py)
- All 13 deterministic statutory & student-protection rules + JSON rule catalog
- Enhanced missing-information checks (INTERNSHIP IP carve-out, STUDENT_SERVICE, GENERAL)
- Exact fairness indicator math & risk score breakdown arithmetic identity
- Cross-clause relationship engine (CC-1 through CC-7, CloudPulse regression,
  condition-aware standalone lock-in, and build_cross_clause_summary clusters/edges)
"""
import json
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.analyzer import analyze_document_text, analyze_file
from backend.cross_clause import apply_cross_clause_analysis, build_cross_clause_summary
from backend.rule_engine import (
    STATUTORY_RULES,
    calculate_fairness_indicator,
    check_missing_information,
    compute_overall_risk_score,
    compute_risk_score_breakdown,
    detect_document_domain,
    get_rule_catalog,
    run_statutory_rule_checks,
    score_document_domains,
)
from backend.sample_agreements import SAMPLE_AGREEMENTS


# ==============================================================================
# 1. Weighted Multi-Signal Domain Detection Tests
# ==============================================================================

class TestWeightedDomainDetection:
    @pytest.mark.parametrize(
        "sample_key,expected_domain",
        [
            ("sample_internship", "INTERNSHIP"),
            ("sample_placement_bond", "PLACEMENT_EMPLOYMENT"),
            ("sample_hostel", "HOSTEL_RESIDENCY"),
            ("sample_sponsorship", "EVENT_SPONSORSHIP"),
            ("sample_service", "STUDENT_SERVICE"),
        ],
    )
    def test_all_five_samples_return_expected_domain(self, sample_key, expected_domain):
        sample = SAMPLE_AGREEMENTS[sample_key]
        detected = detect_document_domain(sample["content"])
        scores = score_document_domains(sample["content"])
        assert detected == expected_domain, f"Expected {expected_domain}, got {detected} (scores={scores})"
        assert scores[expected_domain] == max(scores.values())

    def test_cloudpulse_domain_is_internship(self):
        cloudpulse_path = Path("test_documents/CloudPulse_Student_Internship_Test_Agreement.docx")
        assert cloudpulse_path.exists(), "CloudPulse test document missing"
        analysis = analyze_file(cloudpulse_path)
        assert analysis["domain"] == "INTERNSHIP"

    def test_mixed_keyword_placement_bond_mentioning_internship_experience(self):
        text = """
        APEX ENTERPRISES LIMITED
        CAMPUS PLACEMENT OFFER OF APPOINTMENT AND EMPLOYMENT BOND

        Dear Candidate,
        We offer you full-time employment as a Graduate Engineer Trainee with an Annual CTC of INR 7,50,000
        and a 6-month probationary period. Candidates with prior internship experience may receive faster onboarding.
        A 2-year service bond applies to this employment agreement.
        """
        assert detect_document_domain(text) == "PLACEMENT_EMPLOYMENT"

    def test_mixed_keyword_student_service_mentioning_student_and_stipend(self):
        text = """
        EDUTUTOR LABS
        INDEPENDENT STUDENT SERVICE AGREEMENT AND FREELANCE CONTENT CREATOR CONTRACT

        This Independent Contractor Agreement is entered into with the Student Contractor for online tutoring
        services and video tutorials. Compensation is paid per completed lesson against milestone deliverables.
        Unlike a traditional internship stipend, payments are invoiced per deliverable under this service agreement.
        """
        assert detect_document_domain(text) == "STUDENT_SERVICE"

    def test_mixed_keyword_hostel_mentioning_internship_semester(self):
        text = """
        CAMPUS RESIDENCY & HOSTEL ROOM ALLOTMENT AGREEMENT
        Hostel Management allots Room 202 to the Resident Student for the academic term including any summer
        internship semester. Caution deposit, mess fee, curfew hours, and warden rules apply strictly.
        """
        assert detect_document_domain(text) == "HOSTEL_RESIDENCY"

    def test_mixed_keyword_sponsorship_mentioning_student_interns(self):
        text = """
        TECHFEST 2026 CORPORATE SPONSORSHIP AGREEMENT
        Between BrandCorp (Sponsor) and the Student Branch Organizing Committee for the annual technical symposium.
        Sponsor receives an exhibition stall, main stage banner, and sponsor logo visibility. Student interns
        volunteering at the fest desk will assist with registration.
        """
        assert detect_document_domain(text) == "EVENT_SPONSORSHIP"

    def test_generic_text_falls_back_to_general_student_agreement(self):
        generic = (
            "This Mutual Understanding Document records general collaboration guidelines and "
            "communication expectations between Party A and Party B for calendar year 2026."
        )
        assert detect_document_domain(generic) == "GENERAL_STUDENT_AGREEMENT"
        assert detect_document_domain("") == "GENERAL_STUDENT_AGREEMENT"


# ==============================================================================
# 2. All 13 Deterministic Rules & Rule Catalog Tests
# ==============================================================================

class TestDeterministicRulesAndCatalog:
    EXPECTED_RULE_IDS = [
        "RULE_SECTION_27_RESTRAINT_OF_TRADE",
        "RULE_CERTIFICATE_RETENTION_COLLATERAL",
        "RULE_EXCESSIVE_TRAINING_BOND",
        "RULE_UNILATERAL_IMMEDIATE_TERMINATION",
        "RULE_TOTAL_IP_SURRENDER_PREEXISTING",
        "RULE_HOSTEL_NON_REFUNDABLE_DEPOSIT",
        "RULE_UNLIMITED_STUDENT_INDEMNITY",
        "RULE_GAG_DISPARAGEMENT_CLAUSE",
        "RULE_CONTINGENT_STIPEND_PAYOUT",
        "RULE_UNILATERAL_AMENDMENT",
        "RULE_FOREIGN_COSTLY_ARBITRATION",
        "RULE_NO_ACADEMIC_LEAVE",
        "RULE_DEVICE_PRIVACY_OVERREACH",
    ]

    def test_rule_catalog_completeness_and_json_serializability(self):
        catalog = get_rule_catalog()
        assert len(catalog) == 13
        assert len(STATUTORY_RULES) == 13

        catalog_ids = [item["id"] for item in catalog]
        for expected_id in self.EXPECTED_RULE_IDS:
            assert expected_id in catalog_ids

        # Must serialize cleanly to JSON without compiled regexes
        serialized = json.dumps(catalog)
        deserialized = json.loads(serialized)
        assert len(deserialized) == 13

        required_keys = {
            "id", "name", "severity", "applies_to",
            "statute", "explanation", "action", "rule_type"
        }
        for entry in deserialized:
            assert required_keys.issubset(entry.keys())
            assert entry["severity"] in {"low", "medium", "high"}
            assert entry["rule_type"] in {"STATUTORY_CONTEXT", "STUDENT_PROTECTION_HEURISTIC"}
            assert isinstance(entry["applies_to"], list) and len(entry["applies_to"]) > 0
            # Legal accuracy guardrails: never invent AICTE/UGC sections or absolute illegality
            combined_text = f"{entry['statute']} {entry['explanation']}".lower()
            assert "aicte section" not in combined_text
            assert "ugc section" not in combined_text
            assert "automatically illegal" not in combined_text

    @pytest.mark.parametrize(
        "rule_id,domain,sample_clause",
        [
            (
                "RULE_SECTION_27_RESTRAINT_OF_TRADE",
                "INTERNSHIP",
                "The student agrees not to work for any competing business for 24 months after exit.",
            ),
            (
                "RULE_CERTIFICATE_RETENTION_COLLATERAL",
                "INTERNSHIP",
                "The Company shall withhold the internship completion certificate and academic evaluation until exit penalties are paid.",
            ),
            (
                "RULE_EXCESSIVE_TRAINING_BOND",
                "PLACEMENT_EMPLOYMENT",
                "Early departure triggers a training bond and liquidated damages of Rs. 1,50,000.",
            ),
            (
                "RULE_UNILATERAL_IMMEDIATE_TERMINATION",
                "INTERNSHIP",
                "The Company may terminate this Agreement at any time, without cause, immediately upon notice.",
            ),
            (
                "RULE_TOTAL_IP_SURRENDER_PREEXISTING",
                "INTERNSHIP",
                "The intern assigns worldwide ownership and waives all moral rights over deliverables.",
            ),
            (
                "RULE_HOSTEL_NON_REFUNDABLE_DEPOSIT",
                "HOSTEL_RESIDENCY",
                "The security deposit of Rs. 35,000 is strictly non-refundable under any circumstances.",
            ),
            (
                "RULE_UNLIMITED_STUDENT_INDEMNITY",
                "EVENT_SPONSORSHIP",
                "Student organizers personally guarantee payment and indemnify the sponsor for all losses.",
            ),
            (
                "RULE_GAG_DISPARAGEMENT_CLAUSE",
                "INTERNSHIP",
                "Intern is prohibited from mentioning the company name on their CV, LinkedIn, or posting Glassdoor reviews.",
            ),
            (
                "RULE_CONTINGENT_STIPEND_PAYOUT",
                "INTERNSHIP",
                "Stipend disbursement is contingent entirely upon client milestone sign-off and receipt of funds.",
            ),
            (
                "RULE_UNILATERAL_AMENDMENT",
                "INTERNSHIP",
                "The Company reserves the right to unilaterally modify the monthly stipend, fees, or material terms of this agreement without written student consent.",
            ),
            (
                "RULE_FOREIGN_COSTLY_ARBITRATION",
                "PLACEMENT_EMPLOYMENT",
                "All disputes shall be resolved via binding arbitration in Singapore under SIAC rules, with arbitration costs borne solely by the student.",
            ),
            (
                "RULE_NO_ACADEMIC_LEAVE",
                "INTERNSHIP",
                "No academic exam leave may be availed during the notice period, and absence for mandatory university examinations shall incur a penalty.",
            ),
            (
                "RULE_DEVICE_PRIVACY_OVERREACH",
                "INTERNSHIP",
                "The student must install continuous monitoring software and grant remote wipe access over their personal laptop and mobile phone.",
            ),
        ],
    )
    def test_each_of_13_rules_triggers_accurately(self, rule_id, domain, sample_clause):
        violations = run_statutory_rule_checks(sample_clause, domain)
        matched_ids = [v["rule_id"] for v in violations]
        assert rule_id in matched_ids, f"Expected {rule_id} to trigger on: {sample_clause}"

    def test_new_rules_do_not_false_positive_on_fair_clauses(self):
        # Fair mutual amendment
        fair_amend = "Any amendment or modification to this Agreement shall be valid only if made in writing and signed by both parties."
        assert not any(v["rule_id"] == "RULE_UNILATERAL_AMENDMENT" for v in run_statutory_rule_checks(fair_amend, "INTERNSHIP"))

        # Fair local Indian jurisdiction
        fair_jurisdiction = "This Agreement is governed by the laws of India, and courts in Bengaluru, Karnataka shall have exclusive jurisdiction."
        assert not any(v["rule_id"] == "RULE_FOREIGN_COSTLY_ARBITRATION" for v in run_statutory_rule_checks(fair_jurisdiction, "INTERNSHIP"))

        # Fair academic leave accommodation
        fair_exam_leave = "The Intern shall be granted unpaid academic leave for scheduled university examinations upon 7 days prior written notice."
        assert not any(v["rule_id"] == "RULE_NO_ACADEMIC_LEAVE" for v in run_statutory_rule_checks(fair_exam_leave, "INTERNSHIP"))

        # Fair company-issued hardware security
        fair_device = "The Company will issue a corporate laptop pre-configured with standard endpoint security for project work."
        assert not any(v["rule_id"] == "RULE_DEVICE_PRIVACY_OVERREACH" for v in run_statutory_rule_checks(fair_device, "INTERNSHIP"))


# ==============================================================================
# 3. Enhanced Missing Information Detection Tests
# ==============================================================================

class TestMissingInformationChecks:
    def test_student_service_missing_payment_timeline_and_acceptance_criteria(self):
        text = "Student Contractor agrees to build 10 tutorials. Disputes go to local courts."
        missing = check_missing_information(text, "STUDENT_SERVICE")
        topics = {m["topic"] for m in missing}
        assert "Payment Timeline" in topics
        assert "Deliverable Acceptance Criteria" in topics

        # When both are present, neither should be flagged
        complete_text = (
            "Student Contractor agrees to build 10 tutorials. Objective acceptance criteria apply with a "
            "5-day review period after which deliverables are deemed accepted. Payment shall be disbursed "
            "within 15 days of invoice submission. Disputes go to local jurisdiction."
        )
        missing_complete = check_missing_information(complete_text, "STUDENT_SERVICE")
        topics_complete = {m["topic"] for m in missing_complete}
        assert "Payment Timeline" not in topics_complete
        assert "Deliverable Acceptance Criteria" not in topics_complete

    def test_general_student_agreement_missing_termination_notice(self):
        text = "This general agreement governs campus lab collaboration. Local jurisdiction applies."
        missing = check_missing_information(text, "GENERAL_STUDENT_AGREEMENT")
        topics = {m["topic"] for m in missing}
        assert "Termination Notice Terms" in topics

        with_notice = text + " Either party may terminate this agreement with 14 days written notice."
        missing2 = check_missing_information(with_notice, "GENERAL_STUDENT_AGREEMENT")
        assert "Termination Notice Terms" not in {m["topic"] for m in missing2}

    def test_internship_ip_carveout_missing_and_present(self):
        no_carveout = (
            "Working hours are 40 hours per week with a mentor. Stipend paid on the 5th of each month. "
            "Exam leave is provided. Local jurisdiction applies. All intellectual property and inventions "
            "created during the internship belong exclusively to the company."
        )
        missing = check_missing_information(no_carveout, "INTERNSHIP")
        assert any("Carve-Out" in m["topic"] for m in missing)

        with_carveout = (
            no_carveout + " Explicit carve-out: pre-existing code, personal projects, and open-source contributions "
            "shall remain the property of the intern."
        )
        missing_with = check_missing_information(with_carveout, "INTERNSHIP")
        assert not any("Carve-Out" in m["topic"] for m in missing_with)


# ==============================================================================
# 4. Fairness Math & Risk Score Breakdown Arithmetic Identity Tests
# ==============================================================================

class TestFairnessAndRiskScoreMath:
    def test_fairness_indicator_mutually_exclusive_exact_counts(self):
        clauses = [
            {"final_risk_level": "high", "rule_violations": [{"rule_id": "R1"}]},
            {"final_risk_level": "high", "rule_violations": []},
            {"final_risk_level": "medium", "rule_violations": [{"rule_id": "R2"}]},  # rule violation -> unilateral
            {"final_risk_level": "medium", "rule_violations": []},
            {"final_risk_level": "medium", "rule_violations": []},
            {"final_risk_level": "low", "rule_violations": []},
            {"final_risk_level": "low", "rule_violations": []},
            {"final_risk_level": "low", "rule_violations": []},
        ]
        fairness = calculate_fairness_indicator(clauses, "INTERNSHIP")
        assert (
            fairness["unilateral_clause_count"]
            + fairness["moderate_caution_clause_count"]
            + fairness["bilateral_clause_count"]
            == fairness["total_classified_clauses"]
            == len(clauses)
        )
        assert fairness["unilateral_clause_count"] == 3
        assert fairness["moderate_caution_clause_count"] == 2
        assert fairness["bilateral_clause_count"] == 3
        assert fairness["student_favor_pct"] + fairness["counterparty_favor_pct"] == 100
        assert round(fairness["student_points"] + fairness["counterparty_points"], 1) == fairness["total_points"]

    def test_fairness_indicator_empty_clauses(self):
        fairness = calculate_fairness_indicator([], "INTERNSHIP")
        assert fairness["student_favor_pct"] == 50
        assert fairness["counterparty_favor_pct"] == 50
        assert fairness["total_classified_clauses"] == 0

    @pytest.mark.parametrize(
        "high_n,med_n,low_n,rule_n,missing_n",
        [
            (0, 0, 10, 0, 0),   # Floor cap at 10
            (1, 2, 3, 1, 2),    # Uncapped middle range
            (6, 3, 2, 5, 3),    # Ceiling cap at 98
        ],
    )
    def test_risk_score_breakdown_arithmetic_identity(self, high_n, med_n, low_n, rule_n, missing_n):
        clauses = (
            [{"final_risk_level": "high", "rule_violations": [{"id": "x"}] if i < rule_n else []} for i in range(high_n)]
            + [{"final_risk_level": "medium", "rule_violations": []} for _ in range(med_n)]
            + [{"final_risk_level": "low", "rule_violations": []} for _ in range(low_n)]
        )
        missing_items = [{"topic": f"Missing {i}"} for i in range(missing_n)]

        bd = compute_risk_score_breakdown(clauses, missing_items)
        expected_raw = (
            bd["baseline"]
            + bd["high_points"]
            + bd["rule_flag_points"]
            + bd["medium_points"]
            + bd["missing_points"]
            + bd["low_offset"]
        )
        assert bd["raw_score"] == expected_raw
        assert bd["final_score"] == max(10, min(98, expected_raw))
        assert bd["was_capped"] == (bd["raw_score"] != bd["final_score"])

        score, category = compute_overall_risk_score(clauses, missing_items)
        assert score == bd["final_score"]
        assert category == bd["category"]


# ==============================================================================
# 5. Cross-Clause Relationships (CC-1 to CC-7), CloudPulse & Summary Tests
# ==============================================================================

def _make_clause(idx, text, risk="low", category="STANDARD_FAIR", rules=None):
    return {
        "index": idx,
        "original_text": text,
        "is_heading": False,
        "is_structural": False,
        "category": category,
        "ml_confidence": 0.8,
        "ml_risk_level": risk,
        "rule_violations": rules or [],
        "final_risk_level": risk,
        "source": "ML_CLASSIFIER",
        "explanation": "Initial clause explanation.",
        "recommended_action": "Initial action.",
        "related_clauses": [],
        "cross_clause_risk_elevated": False,
    }


class TestCrossClauseRelationships:
    def test_cloudpulse_exact_cross_clause_and_distribution_behavior(self):
        cloudpulse_path = Path("test_documents/CloudPulse_Student_Internship_Test_Agreement.docx")
        analysis = analyze_file(cloudpulse_path)

        # Exact 9H / 4M / 4L substantive counts preserved
        dist = analysis["distribution"]
        assert dist["high"] == 9
        assert dist["medium"] == 4
        assert dist["low"] == 4
        assert dist["total"] == 17

        by_num = {
            c.get("clause_number"): c
            for c in analysis["clauses"]
            if not c["is_heading"] and not c["is_structural"]
        }

        # Clause 4.1 -> linked to 4.2 & 4.3, elevated to HIGH
        c41 = by_num["4.1"]
        assert c41["final_risk_level"] == "high"
        assert c41["cross_clause_risk_elevated"] is True
        c41_refs = {r["target_clause_ref"] for r in c41["related_clauses"]}
        assert "Clause 4.2" in c41_refs
        assert "Clause 4.3" in c41_refs

        # Clause 4.2 (₹75,000 exit penalty, HIGH) & Clause 3.3 (₹15,000 forfeiture, HIGH)
        assert by_num["4.2"]["final_risk_level"] == "high"
        assert by_num["3.3"]["final_risk_level"] == "high"

        # Clause 4.3 -> 4.2
        assert "Clause 4.2" in {r["target_clause_ref"] for r in by_num["4.3"]["related_clauses"]}

        # Clause 5.2 -> 4.2 & 5.1
        c52_refs = {r["target_clause_ref"] for r in by_num["5.2"]["related_clauses"]}
        assert "Clause 4.2" in c52_refs
        assert "Clause 5.1" in c52_refs

        # Clause 3.1 & 3.2 -> 3.3
        assert "Clause 3.3" in {r["target_clause_ref"] for r in by_num["3.1"]["related_clauses"]}
        assert "Clause 3.3" in {r["target_clause_ref"] for r in by_num["3.2"]["related_clauses"]}

        # CC-7: Clause 2.1 -> 2.2 without falsely escalating Clause 2.1
        c21 = by_num["2.1"]
        assert c21["final_risk_level"] == "low"
        assert c21["cross_clause_risk_elevated"] is False
        assert any(
            r["display_text"] == "Related to Clause 2.2 – Contingent Pay & Fee Forfeiture"
            for r in c21["related_clauses"]
        )
        assert "Clause 2.2" in c21["explanation"]

        # CC-6: Clause 6.2 <-> Clause 7.1 & 7.2 bidirectional career/portfolio lock-out
        c62 = by_num["6.2"]
        c71 = by_num["7.1"]
        c72 = by_num["7.2"]
        assert any(
            r["target_clause_ref"] == "Clause 7.2" and r["relationship"] == "Overbroad IP Assignment"
            for r in c62["related_clauses"]
        )
        assert any(
            r["target_clause_ref"] == "Clause 6.2" and r["relationship"] == "Portfolio & Confidentiality Restriction"
            for r in c71["related_clauses"]
        )
        assert any(
            r["target_clause_ref"] == "Clause 6.2" and r["relationship"] == "Portfolio & Confidentiality Restriction"
            for r in c72["related_clauses"]
        )

        # build_cross_clause_summary on CloudPulse
        summary = build_cross_clause_summary(analysis["clauses"])
        assert summary["total_links"] == len(summary["edges"])
        assert summary["total_links"] >= 10
        assert summary["elevated_clauses_count"] >= 1
        assert len(summary["clusters"]) >= 3
        json.dumps(summary)  # Must be cleanly JSON-serializable

        for cluster in summary["clusters"]:
            for key in ("cluster_id", "title", "severity", "clause_refs", "clause_indices", "summary"):
                assert key in cluster
            assert len(cluster["clause_refs"]) >= 2

    def test_standalone_lock_in_without_penalty_stays_medium_not_high(self):
        original_lock_in = "4.1 The Intern commits to a mandatory minimum lock-in period of 6 months."
        clauses = [
            _make_clause(0, "1.1 The Intern shall receive a monthly stipend of INR 15,000 paid on the 5th of each month.", risk="low"),
            _make_clause(1, original_lock_in, risk="low", category="STANDARD_FAIR"),
            _make_clause(2, "5.1 Either party may terminate this agreement with 15 days written notice with zero exit penalty.", risk="low"),
        ]
        updated = apply_cross_clause_analysis(clauses, "INTERNSHIP")
        lock_in_clause = updated[1]

        assert lock_in_clause["final_risk_level"] == "medium"
        assert lock_in_clause["final_risk_level"] != "high"
        assert lock_in_clause["related_clauses"] == []
        assert lock_in_clause["original_text"] == original_lock_in

    def test_cc6_career_portfolio_lockout_bidirectional_link(self):
        clauses = [
            _make_clause(
                0,
                "6.2 Intern is strictly prohibited from mentioning the company name, project, or skills on their CV, LinkedIn profile, or GitHub repository.",
                risk="high",
                category="CONFIDENTIALITY_NDA",
            ),
            _make_clause(
                1,
                "7.1 All code and designs created on personal devices during weekends and personal time belong exclusively to the Company, and the Intern waives all moral rights.",
                risk="high",
                category="INTELLECTUAL_PROPERTY",
            ),
        ]
        updated = apply_cross_clause_analysis(clauses, "INTERNSHIP")
        gag_c, ip_c = updated[0], updated[1]

        assert any(
            r["target_clause_ref"] == "Clause 7.1" and r["relationship"] == "Overbroad IP Assignment"
            for r in gag_c["related_clauses"]
        )
        assert any(
            r["target_clause_ref"] == "Clause 6.2" and r["relationship"] == "Portfolio & Confidentiality Restriction"
            for r in ip_c["related_clauses"]
        )
        assert "career and academic lock-out" in gag_c["explanation"].lower()
        assert "career and academic lock-out" in ip_c["explanation"].lower()

        summary = build_cross_clause_summary(updated)
        assert summary["total_links"] == 2
        assert len(summary["clusters"]) == 1
        assert summary["clusters"][0]["clause_refs"] == ["Clause 6.2", "Clause 7.1"]

    def test_cc7_stipend_linked_to_contingent_withholding_without_escalation(self):
        clauses = [
            _make_clause(
                0,
                "2.1 The Company shall pay the Intern a monthly stipend of INR 20,000.",
                risk="low",
                category="STANDARD_FAIR",
            ),
            _make_clause(
                1,
                "2.2 Stipend disbursement is contingent entirely upon client milestone sign-off, and the Company may withhold, delay, or reduce the monthly stipend at its sole discretion.",
                risk="high",
                category="STIPEND_PAYMENT_TERMS",
                rules=[{"rule_id": "RULE_CONTINGENT_STIPEND_PAYOUT"}],
            ),
        ]
        updated = apply_cross_clause_analysis(clauses, "INTERNSHIP")
        stipend_c = updated[0]

        assert stipend_c["final_risk_level"] == "low"
        assert stipend_c["cross_clause_risk_elevated"] is False
        assert len(stipend_c["related_clauses"]) == 1
        assert stipend_c["related_clauses"][0]["display_text"] == "Related to Clause 2.2 – Contingent Pay & Fee Forfeiture"
        assert "Clause 2.2" in stipend_c["explanation"]

    def test_cross_clause_summary_empty_when_no_links(self):
        clauses = [
            _make_clause(0, "1.1 Standard role description for 6 months.", risk="low"),
            _make_clause(1, "2.1 Standard monthly stipend paid on the 5th of each month.", risk="low"),
            _make_clause(2, "3.1 Local courts in Pune shall have jurisdiction.", risk="low"),
        ]
        updated = apply_cross_clause_analysis(clauses, "INTERNSHIP")
        summary = build_cross_clause_summary(updated)
        assert summary == {
            "total_links": 0,
            "elevated_clauses_count": 0,
            "edges": [],
            "clusters": [],
        }
