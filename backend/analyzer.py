"""
ClauseGuard AI - Unified Contract Analysis Orchestrator
Coordinates extraction, ML classification, statutory rule compliance checks,
fairness calculation, missing term detection, and executive summaries.
"""
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from backend.extractor import extract_document, segment_into_paragraphs
from backend.ml_pipeline import (
    classifier_instance,
    CATEGORY_STUDENT_EXPLANATIONS,
    CATEGORY_RECOMMENDED_ACTIONS
)
from backend.rule_engine import (
    detect_document_domain,
    run_statutory_rule_checks,
    check_missing_information,
    calculate_fairness_indicator,
    compute_overall_risk_score,
    compute_risk_score_breakdown,
)
from backend.cross_clause import apply_cross_clause_analysis, build_cross_clause_summary

# Structural / informational patterns (NOT substantive contractual obligations)
RE_DOC_REFERENCE = re.compile(
    r'^(?:document\s+reference\s*:|ref\s*:|reference\s*:|date\s*:|to\s*:|from\s*:|subject\s*:|dear\s+[a-z]+)',
    re.IGNORECASE
)
RE_INTRO_RECITAL = re.compile(
    r'^(?:this\s+(?:[a-z\s]{0,40})?agreement\s+(?:\("agreement"\)\s*)?(?:is\s+entered|is\s+made)|whereas\b|terms\s+and\s+conditions\s+of\b)',
    re.IGNORECASE
)
RE_SIGNATURE_OR_PARTY = re.compile(
    r'(?:in\s+witness\s+whereof|sincerely\b|for\s+the\s+(?:intern|company|student|sponsor|resident|student\s+intern)|^for\s+[A-Z][\w\s\.\-&]{2,50}(?:ltd|inc|llp|corp|committee|solutions|\(fictional\))|authori[sz]ed\s+signatory|hostel\s+warden|accepted\s+(?:and\s+agreed|by\s*:)|_{5,}|operations\s+manager\s+.*?(?:contractor|student)|marketing\s+lead\s+.*?convener)',
    re.IGNORECASE | re.DOTALL
)
RE_DISCLAIMER_NOTICE = re.compile(
    r'(?:^disclaimer\s*:|fictional\s+test\s+data|simulated\s+and\s+have\s+zero\s+legal\s+effect|for\s+evaluation\s+and\s+testing\s+purposes\s+only)',
    re.IGNORECASE
)
RE_PREAMBLE_TITLES = re.compile(
    r'^(?:[A-Z0-9\s,\.\-&()]{4,}(?:LIMITED|INC\.|LLP|SOLUTIONS|CORP|SYMPOSIUM|AGREEMENT|BOND|OFFER|DATA\))\s*)+$',
    re.IGNORECASE
)
RE_NUMBERED_CLAUSE_START = re.compile(r'^(?:\d+(?:\.\d+)*[\.\)]?\s+|(?:section|article|clause)\s+\d+)', re.IGNORECASE)


def _classify_structural_entry(clause_text: str):
    """
    Detect non-substantive document references, introductory recitals, party/signature blocks,
    and disclaimers. Never classifies actual numbered contractual clauses as structural.
    Returns (explanation, recommended_action) or None if substantive.
    """
    txt = clause_text.strip()
    if RE_NUMBERED_CLAUSE_START.match(txt):
        return None

    if RE_DISCLAIMER_NOTICE.search(txt):
        return (
            "Informational disclaimer or fictional test-data notice; not a binding contractual obligation.",
            "Informational notice only — excluded from contract risk scoring."
        )
    if RE_DOC_REFERENCE.search(txt):
        return (
            "Document reference, date, or addressee metadata header.",
            "Verify that the reference number, effective date, and addressee details match your official offer."
        )
    if RE_INTRO_RECITAL.search(txt):
        return (
            "Introductory party identification or background recital framing the agreement.",
            "Confirm the exact legal name of the organization, your full name, and the effective date."
        )
    if RE_SIGNATURE_OR_PARTY.search(txt):
        return (
            "Execution, witness, or party signature block.",
            "Ensure authorized representatives of both parties sign and date the final agreed copy."
        )
    if len(txt.splitlines()) <= 3 and len(txt) <= 140 and RE_PREAMBLE_TITLES.match(txt):
        return (
            "Document title and organization header block.",
            "Structural document header — excluded from contract risk scoring."
        )
    return None


def _specific_low_risk_advice(clause_text: str, category: str):
    """
    Generate clause-specific explanations and recommendations for low-risk contractual clauses
    instead of repeating generic boilerplate across unrelated clauses.
    """
    tl = clause_text.lower()

    if category == "DISPUTE_JURISDICTION" or any(k in tl for k in ["governing law", "jurisdiction", "conciliation", "arbitration"]):
        if "conciliation" in tl or "good-faith" in tl or "mutual consultation" in tl:
            return (
                "Specifies governing law and good-faith conciliation/consultation prior to court proceedings.",
                "Verify that the named court location is reasonably accessible for you if a formal dispute ever arises."
            )
        if any(k in tl for k in ["new york", "singapore", "delaware", "siac"]):
            return (
                "Sets a distant foreign governing law or arbitration venue, which can make enforcing student rights expensive.",
                "Ask to replace the distant foreign venue with local courts in your city or university jurisdiction."
            )
        return (
            "Defines the governing law and court jurisdiction for resolving disputes under this agreement.",
            "Confirm the named jurisdiction is local and practical for you, and consider adding amicable conciliation first."
        )

    if any(k in tl for k in ["hours per week", "working hours", "monday to friday", "hybrid", "mentor", "reporting"]):
        return (
            "Defines weekly working hours, work arrangement (hybrid/in-office), and mentorship reporting structure.",
            "Confirm weekly hours will not conflict with university classes or exam schedules, and note your assigned mentor's name."
        )

    if any(k in tl for k in ["appointed as", "fixed duration", "commencing on", "concluding on", "initial term", "probationary period", "shall join"]):
        return (
            "Establishes the internship/job role title, duration, and start/end dates or probation period.",
            "Confirm the exact start date, end date, role title, and alignment with your university internship credit calendar."
        )

    if any(k in tl for k in ["monthly stipend", "pay the intern", "disburse a monthly stipend", "annual ctc"]):
        return (
            "States the base monthly stipend or compensation figure before any separate withholding or milestone conditions.",
            "Verify the net take-home stipend after statutory deductions, confirm payment mode, and cross-check any linked payout conditions."
        )

    if any(k in tl for k in ["allots room", "room occupancy", "hostel management allots"]):
        return (
            "Specifies the assigned hostel room number, occupancy period, and personal occupancy condition.",
            "Verify the room number, occupancy dates, and included utilities against a signed move-in inventory checklist."
        )

    if any(k in tl for k in ["sponsorship", "exhibition stall", "deliverables", "record 40 video", "tutoring", "tutorials"]):
        return (
            "Defines the core deliverables, branding commitments, or work scope agreed between the parties.",
            "Confirm exact deliverable counts, quality acceptance criteria, and timelines in writing to prevent scope disputes."
        )

    if any(k in tl for k in ["media rights", "photographs", "recordings", "presentations"]):
        return (
            "Allocates commercial media and recording rights for event photography or student presentations.",
            "Confirm that participating students retain attribution credit and permission to showcase their own project presentations."
        )

    return (
        "Standard contractual provision defining general operational terms without elevated risk indicators.",
        "Verify that all dates, figures, and obligations in this clause match your agreed offer details."
    )


def analyze_document_text(
    text: str,
    filename: str = "document.txt",
    format_type: str = "TXT",
    extraction_metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Main pipeline entry point: Analyzes raw or extracted agreement text.
    Provides unified single-source-of-truth risk distribution, clause classifications,
    statutory rule detections, cross-clause clusters, student fairness metrics,
    and top vulnerability highlights.
    """
    # 1. Detect Domain
    domain = detect_document_domain(text)
    
    # 2. Segment into paragraphs / clauses
    paragraphs = segment_into_paragraphs(text)
    
    # 3. Process each clause through ML model and Statutory Rule Engine
    analyzed_clauses = []
    
    for p in paragraphs:
        clause_text = p["text"].strip()
        if len(clause_text) < 15 and p["is_heading"]:
            continue  # Short title lines don't need heavy risk grading

        # Pure standalone section headings are structural dividers, not risk-graded clauses
        if p["is_heading"]:
            analyzed_clauses.append({
                "index": p["index"],
                "original_text": clause_text,
                "char_start": p["char_start"],
                "char_end": p["char_end"],
                "is_heading": True,
                "is_structural": True,
                "category": "SECTION_HEADING",
                "ml_confidence": 1.0,
                "ml_risk_level": "informational",
                "rule_violations": [],
                "final_risk_level": "informational",
                "source": "DOCUMENT_STRUCTURE",
                "explanation": "Section heading.",
                "recommended_action": "",
                "related_clauses": [],
                "cross_clause_risk_elevated": False
            })
            continue
            
        # Check non-substantive document references, recitals, party/signature blocks, disclaimers
        structural_info = _classify_structural_entry(clause_text)

        if structural_info is not None:
            exp_txt, act_txt = structural_info
            analyzed_clauses.append({
                "index": p["index"],
                "original_text": clause_text,
                "char_start": p["char_start"],
                "char_end": p["char_end"],
                "is_heading": False,
                "is_structural": True,
                "category": "STRUCTURAL_INFORMATIONAL",
                "ml_confidence": 1.0,
                "ml_risk_level": "informational",
                "rule_violations": [],
                "final_risk_level": "informational",
                "source": "DOCUMENT_STRUCTURE",
                "explanation": exp_txt,
                "recommended_action": act_txt,
                "related_clauses": [],
                "cross_clause_risk_elevated": False
            })
            continue

        # Actual contractual clause: run ML Inference + Statutory Rule Engine
        ml_res = classifier_instance.classify_clause(clause_text)
        rule_violations = run_statutory_rule_checks(clause_text, domain)
        
        category = ml_res["category"]
        ml_conf = ml_res["confidence"]
        ml_risk = ml_res["ml_risk_level"]
        explanation = ml_res["explanation"]
        recommended_action = ml_res["recommended_action"]
        
        # Synthesize final risk and attribution
        if rule_violations:
            final_risk = "high"
            source = "STATUTORY_RULE" if ml_conf < 0.4 else "COMBINED_EVALUATION"
            explanation = rule_violations[0]["explanation"]
            recommended_action = rule_violations[0]["action"]
            # Align category with specific statutory rule if ML missed it
            r_id = rule_violations[0]["rule_id"]
            if "NON_COMPETE" in r_id or "RESTRAINT" in r_id:
                category = "RESTRICTIVE_NON_COMPETE"
            elif "BOND" in r_id or "CERTIFICATE" in r_id:
                category = "FINANCIAL_PENALTY_BOND"
            elif "INDEMNITY" in r_id:
                category = "LIABILITY_INDEMNITY"
            elif "DEPOSIT" in r_id:
                category = "DEPOSIT_REFUND"
            elif "GAG" in r_id or "PRIVACY" in r_id:
                category = "CONFIDENTIALITY_NDA"
            elif "TERMINATION" in r_id or "ACADEMIC_LEAVE" in r_id:
                category = "TERMINATION_NOTICE"
            elif "STIPEND" in r_id:
                category = "STIPEND_PAYMENT_TERMS"
            elif "ARBITRATION" in r_id or "JURISDICTION" in r_id:
                category = "DISPUTE_JURISDICTION"
        elif ml_conf < 0.28:
            # Low confidence fallback: treat as low-risk contractual clause with clause-specific advice
            category = "STANDARD_FAIR"
            final_risk = "low"
            source = "ML_CLASSIFIER"
            explanation, recommended_action = _specific_low_risk_advice(clause_text, category)
        elif category == "RESTRICTIVE_NON_COMPETE":
            # Confident non-compete is inherently high risk for students
            final_risk = "high"
            source = "ML_CLASSIFIER"
        elif category == "FINANCIAL_PENALTY_BOND":
            # Exit bond / penalty is inherently high risk for students
            final_risk = "high"
            source = "ML_CLASSIFIER"
        elif ml_risk == "high" and ml_conf >= 0.35:
            final_risk = "high"
            source = "ML_CLASSIFIER"
        elif ml_risk == "medium":
            final_risk = "medium"
            source = "ML_CLASSIFIER"
        else:
            final_risk = "low"
            source = "ML_CLASSIFIER"
            explanation, recommended_action = _specific_low_risk_advice(clause_text, category)
            
        analyzed_clauses.append({
            "index": p["index"],
            "original_text": clause_text,
            "char_start": p["char_start"],
            "char_end": p["char_end"],
            "is_heading": False,
            "is_structural": False,
            "category": category,
            "ml_confidence": ml_conf,
            "ml_risk_level": ml_risk,
            "rule_violations": rule_violations,
            "final_risk_level": final_risk,
            "source": source,
            "explanation": explanation,
            "recommended_action": recommended_action,
            "related_clauses": [],
            "cross_clause_risk_elevated": False
        })

    # 3.5 Cross-clause relationship detection, contextual risk escalation & cluster summary
    analyzed_clauses = apply_cross_clause_analysis(analyzed_clauses, domain)
    cross_clause_summary = build_cross_clause_summary(analyzed_clauses)

    # Compute ALL risk metrics strictly over substantive contractual clauses
    # (excluding both pure section headings and structural/informational blocks)
    substantive_clauses = [c for c in analyzed_clauses if not c["is_heading"] and not c["is_structural"]]
    structural_entries = [c for c in analyzed_clauses if c["is_structural"] and not c["is_heading"]]
    heading_clauses = [c for c in analyzed_clauses if c["is_heading"]]
        
    # 4. Check missing information
    missing_items = check_missing_information(text, domain)
    
    # 5. Compute fairness indicator
    fairness = calculate_fairness_indicator(substantive_clauses, domain)
    
    # 6. Overall risk score & transparent breakdown
    risk_breakdown = compute_risk_score_breakdown(substantive_clauses, missing_items)
    risk_score = risk_breakdown["final_score"]
    risk_category = risk_breakdown["category"]
    
    # 7. Summary statistics (Guaranteed unified source of truth)
    high_count = sum(1 for c in substantive_clauses if c["final_risk_level"] == "high")
    med_count = sum(1 for c in substantive_clauses if c["final_risk_level"] == "medium")
    low_count = sum(1 for c in substantive_clauses if c["final_risk_level"] == "low")
    cross_ref_count = sum(1 for c in substantive_clauses if c.get("related_clauses"))
    
    # Category distribution breakdown (substantive clauses only)
    category_counts = {}
    for c in substantive_clauses:
        cat = c["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
        
    # 8. Extract top issues (severe red flags directly derived from high-risk substantive clauses)
    top_issues = []
    for c in substantive_clauses:
        if c["final_risk_level"] == "high":
            clause_ref = f"Clause {c['clause_number']}" if c.get("clause_number") else f"Clause #{c['index']}"
            top_issues.append({
                "clause_index": c["index"],
                "clause_ref": clause_ref,
                "category": c["category"],
                "title": f"{clause_ref}: {c['category'].replace('_', ' ').title()}",
                "snippet": c["original_text"][:140] + ("..." if len(c["original_text"]) > 140 else ""),
                "explanation": c["explanation"],
                "action": c["recommended_action"],
                "source": c["source"],
                "related_clauses": c.get("related_clauses", [])
            })
            
    word_count = len(text.split())
    
    doc_info: Dict[str, Any] = {
        "filename": filename,
        "format": format_type,
        "word_count": word_count,
        "clause_count": len(substantive_clauses),
        "classified_clause_count": len(substantive_clauses),
        "structural_count": len(structural_entries),
        "heading_count": len(heading_clauses),
        "extracted_blocks_count": len(analyzed_clauses),
        "paragraphs_count": len(paragraphs),
        "cross_clause_links": cross_ref_count
    }
    if extraction_metadata is not None:
        doc_info["extraction_metadata"] = extraction_metadata

    return {
        "document_info": doc_info,
        "domain": domain,
        "risk_score": risk_score,
        "risk_category": risk_category,
        "risk_score_breakdown": risk_breakdown,
        "fairness": fairness,
        "distribution": {
            "high": high_count,
            "medium": med_count,
            "low": low_count,
            "total": len(substantive_clauses),
            "categories": category_counts
        },
        "cross_clause_summary": cross_clause_summary,
        "top_issues": top_issues[:8],
        "missing_information": missing_items,
        "clauses": analyzed_clauses,
        "raw_text": text,
        "ml_metadata": classifier_instance.get_metrics()
    }


def analyze_file(file_path: Path) -> Dict[str, Any]:
    """Extract and analyze a file from disk."""
    extracted = extract_document(file_path)
    return analyze_document_text(
        text=extracted["text"],
        filename=file_path.name,
        format_type=extracted["format"],
        extraction_metadata=extracted.get("extraction_metadata")
    )
