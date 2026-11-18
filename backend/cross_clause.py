"""
ClauseGuard AI - Cross-Clause Relationship & Contextual Risk Engine
Detects semantic, structural, and explicit cross-clause references across an agreement
(e.g., mandatory lock-in periods linked to financial exit penalties, termination notice
terms linked to service bonds, deposit terms linked to forfeiture clauses).

Principle: Do NOT blindly escalate every lock-in or duration clause to high risk.
Evaluate actual contractual conditions and linked financial/coercive enforcement clauses.
Preserves original_text untouched.
"""
import re
from typing import Dict, List, Any, Optional, Tuple


# Clause & section numbering patterns
RE_SUBCLAUSE_NUM = re.compile(r'^(?:clause\s+|section\s+)?(\d+\.\d+(?:\.\d+)?)\b', re.IGNORECASE)
RE_SECTION_NUM = re.compile(r'^(?:(?:section|article|clause)\s+)?(\d+)[\.\)]\s+', re.IGNORECASE)
RE_EXPLICIT_REF = re.compile(
    r'\b(?:clause|section|article|paragraph|para\.?)\s+(\d+(?:\.\d+)?)\b',
    re.IGNORECASE
)

# Currency / monetary amount extractor (supports Indian 1,50,000 and standard 75,000 formats)
RE_MONEY = re.compile(
    r'(?:Rs\.?|INR|₹|USD|\$|EUR|€)\s*([0-9][0-9,]*(?:\.\d{1,2})?)',
    re.IGNORECASE
)

# Functional legal role patterns
RE_LOCK_IN = re.compile(
    r'\b(?:mandatory\s+(?:minimum\s+)?lock-?in|minimum\s+lock-?in|lock-?in\s+period|mandatory\s+(?:service\s+)?tenure|minimum\s+service\s+period|commit(?:s|ted)?\s+to\s+(?:a\s+)?mandatory|compulsory\s+(?:service|tenure|lock-?in))\b',
    re.IGNORECASE
)

RE_FINANCIAL_PENALTY = re.compile(
    r'\b(?:exit\s+penalty|early\s+exit\s+penalty|liquidated\s+damages|financial\s+penalty|penalty\s+fee|recruitment\s+penalty|service\s+bond|training\s+bond|promissory\s+note|training\s+fee\s+recovery)\b',
    re.IGNORECASE
)

RE_DEPOSIT_FORFEITURE = re.compile(
    r'\b(?:strictly\s+non-refundable|forfeited\s+to\s+the\s+(?:company|hostel)|forfeiture\s+condition|entire\s+security\s+deposit.*?forfeited|deposits?\s+will\s+not\s+be\s+refunded|booking\s+cancellation\s+penalty)\b',
    re.IGNORECASE | re.DOTALL
)

RE_DEPOSIT_REQUIREMENT = re.compile(
    r'\b(?:security\s+deposit|caution\s+deposit|refund\s+conditions?|deposit\s+shall\s+be\s+refunded)\b',
    re.IGNORECASE
)

RE_CERT_WITHHOLDING = re.compile(
    r'\b(?:withhold\b.*?\b(?:completion\s+certificate|academic\s+evaluation|experience\s+letter|relieving\s+letter|certificates?)|deposit\s+(?:their\s+)?original\s+(?:degree|certificates?))\b',
    re.IGNORECASE | re.DOTALL
)

RE_TERMINATION_CLAUSE = re.compile(
    r'\b(?:termination|terminate|resign|resignation|early\s+departure|notice\s+period|prior\s+written\s+notice|summary\s+dismissal|eviction|expelled)\b',
    re.IGNORECASE
)

RE_UNILATERAL_FIRING = re.compile(
    r'\b(?:without\s+cause,?\s*immediately|terminate.*?at\s+any\s+time.*?without\s+(?:cause|notice|compensation)|dismiss.*?instantly)\b',
    re.IGNORECASE | re.DOTALL
)

RE_FEE_FORFEITURE = re.compile(
    r'\b(?:accrued\s+fees\s+shall\s+be\s+forfeited|contingent\s+entirely\s+upon\s+client)\b',
    re.IGNORECASE
)

RE_FORCE_MAJEURE_REFUND = re.compile(
    r'\b(?:force\s+majeure|cancellation)\b.*?\b(?:refund\s+all\s+expended|reimburse|marketing\s+costs)\b',
    re.IGNORECASE | re.DOTALL
)

RE_ORGANIZER_INDEMNITY = re.compile(
    r'\b(?:personally\s+guarantee|organizer\s+indemnity|liquidated\s+damages.*?footfall)\b',
    re.IGNORECASE | re.DOTALL
)

RE_OVERBROAD_IP = re.compile(
    r'\b(?:on\s+personal\s+devices|personal\s+time\s+during\s+weekends|weekends\s+and\s+personal\s+time|waives?\s+all\s+moral\s+rights|academic\s+(?:capstone|thesis|theses)|university\s+capstone\s+projects?)\b',
    re.IGNORECASE
)

RE_PORTFOLIO_GAG = re.compile(
    r'(?:\b(?:prohibited|barred|restricted|shall\s+not|may\s+not)\b[^.;]{0,120}\b(?:mentioning|disclosing|listing|showcasing|publishing)\b[^.;]{0,120}\b(?:cv|resume|linkedin|github|portfolio)\b|\b(?:portfolio\s+and\s+gag\s+restriction|prohibited\s+from\s+mentioning\s+the\s+company\s+name)\b|\b(?:never\s+to\s+post\s+negative\s+comments|glassdoor)\b[^.;]{0,120}\b(?:cv|resume|linkedin|github|portfolio)\b)',
    re.IGNORECASE
)

RE_BASE_COMPENSATION = re.compile(
    r'\b(?:monthly\s+stipend|pay\s+the\s+intern|disburse\s+a\s+monthly\s+stipend|payment\s+of\s+(?:Rs\.?|INR|₹|\$)|annual\s+ctc|monthly\s+(?:salary|remuneration|compensation|fee))\b',
    re.IGNORECASE
)

RE_CONTINGENT_WITHHOLDING = re.compile(
    r'\b(?:contingent\s+entirely\s+upon\s+client|withhold,\s*delay,\s*or\s*reduce\s+(?:the\s+)?(?:monthly\s+)?(?:stipend|compensation|payment|fees?)|accrued\s+fees\s+shall\s+be\s+forfeited|disbursed\s+90\s+days\s+in\s+arrears|sole\s+discretion\s+(?:to\s+)?(?:withhold|delay|reduce))\b',
    re.IGNORECASE
)


RISK_ORDER = {"low": 1, "medium": 2, "high": 3}


def _extract_max_amount(text: str) -> float:
    """Extract the largest numeric monetary figure referenced in a clause."""
    max_val = 0.0
    for m in RE_MONEY.finditer(text):
        raw = m.group(1).replace(",", "")
        try:
            val = float(raw)
            if val > max_val:
                max_val = val
        except ValueError:
            continue
    return max_val


def _parse_clause_numbers(clauses: List[Dict[str, Any]]) -> None:
    """Annotate each clause with clause_number, section_number, and explicit_refs."""
    current_section: Optional[str] = None

    for c in clauses:
        text = c["original_text"].strip()
        clause_num: Optional[str] = None
        sec_num: Optional[str] = None

        m_sub = RE_SUBCLAUSE_NUM.match(text)
        m_sec = RE_SECTION_NUM.match(text)

        if m_sub:
            clause_num = m_sub.group(1)
            sec_num = clause_num.split(".")[0]
            current_section = sec_num
        elif m_sec:
            clause_num = m_sec.group(1)
            sec_num = clause_num
            current_section = sec_num
        else:
            sec_num = current_section

        refs = []
        for rm in RE_EXPLICIT_REF.finditer(text):
            ref_num = rm.group(1)
            if ref_num != clause_num and ref_num not in refs:
                refs.append(ref_num)

        c["clause_number"] = clause_num
        c["section_number"] = sec_num
        c["explicit_refs"] = refs
        c["related_clauses"] = []
        c["cross_clause_risk_elevated"] = False


def _format_clause_ref(clause: Dict[str, Any]) -> str:
    """Human-readable clause reference like 'Clause 4.2' or 'Clause 3'."""
    num = clause.get("clause_number")
    if num:
        return f"Clause {num}"
    return f"Clause #{clause['index']}"


def _infer_anchor_label(clause: Dict[str, Any], role: str) -> str:
    """Generate a concise descriptive title for a linked anchor clause."""
    text_lower = clause["original_text"].lower()
    if role == "PENALTY_ANCHOR":
        if "early exit penalty" in text_lower or "exit penalty" in text_lower:
            return "Early Exit Penalty"
        if "service bond" in text_lower or "mandatory tenure" in text_lower:
            return "Service Bond Penalty"
        if "liquidated damages" in text_lower:
            return "Liquidated Damages Bond"
        return "Financial Exit Penalty"
    if role == "FORFEITURE_ANCHOR":
        return "Deposit Forfeiture"
    if role == "WITHHOLDING_ANCHOR":
        return "Certificate & Evaluation Withholding"
    if role == "UNILATERAL_TERMINATION_ANCHOR":
        return "Unilateral Immediate Termination"
    if role == "FEE_FORFEITURE_ANCHOR":
        return "Contingent Pay & Fee Forfeiture"
    if role == "ORGANIZER_INDEMNITY_ANCHOR":
        return "Organizer Indemnity & Penalty"
    if role == "PORTFOLIO_GAG_ANCHOR":
        return "Portfolio & Confidentiality Restriction"
    if role == "OVERBROAD_IP_ANCHOR":
        return "Overbroad IP Assignment"
    return clause.get("category", "Related Provision").replace("_", " ").title()


def _add_cross_reference(clause: Dict[str, Any], target: Dict[str, Any], role: str) -> None:
    """Attach a deduplicated cross-reference link to a clause."""
    if clause["index"] == target["index"]:
        return
    target_ref = _format_clause_ref(target)
    rel_label = _infer_anchor_label(target, role)
    display_text = f"Related to {target_ref} – {rel_label}"

    for existing in clause["related_clauses"]:
        if existing["target_index"] == target["index"]:
            return

    clause["related_clauses"].append({
        "target_index": target["index"],
        "target_clause_ref": target_ref,
        "relationship": rel_label,
        "display_text": display_text
    })


def _elevate_risk(clause: Dict[str, Any], target_level: str, new_category: Optional[str] = None) -> bool:
    """Elevate clause risk level monotonically (never lowers existing higher risk)."""
    curr = clause.get("final_risk_level", "low")
    elevated = False
    if RISK_ORDER.get(target_level, 1) > RISK_ORDER.get(curr, 1):
        clause["final_risk_level"] = target_level
        clause["cross_clause_risk_elevated"] = True
        elevated = True
        if clause.get("source") in ("ML_CLASSIFIER", "DOCUMENT_STRUCTURE"):
            clause["source"] = "CROSS_CLAUSE_ANALYSIS"
    if new_category and clause.get("category") in ("STANDARD_FAIR", None):
        clause["category"] = new_category
    return elevated


def _find_matching_anchors(
    clause: Dict[str, Any],
    anchors: List[Dict[str, Any]],
    allow_doc_wide: bool = True
) -> List[Dict[str, Any]]:
    """
    Find linked anchors prioritized by:
    1. Explicit clause/section citation (e.g. 'Clause 4' matches 4, 4.1, 4.2)
    2. Same section number (sibling sub-clauses in Section 4)
    3. Document-wide anchors (when allow_doc_wide=True)
    """
    c_idx = clause["index"]
    c_sec = clause.get("section_number")
    c_refs = clause.get("explicit_refs", [])

    explicit_matches = []
    same_section_matches = []
    doc_wide_matches = []

    for a in anchors:
        if a["index"] == c_idx:
            continue
        a_num = a.get("clause_number") or ""
        a_sec = a.get("section_number") or ""

        # Explicit citation match (exact clause number or section number)
        if any(ref == a_num or ref == a_sec for ref in c_refs):
            explicit_matches.append(a)
        elif c_sec and a_sec and c_sec == a_sec:
            same_section_matches.append(a)
        elif allow_doc_wide:
            doc_wide_matches.append(a)

    if explicit_matches:
        return explicit_matches
    if same_section_matches:
        return same_section_matches
    return doc_wide_matches


def apply_cross_clause_analysis(analyzed_clauses: List[Dict[str, Any]], domain: str) -> List[Dict[str, Any]]:
    """
    Second-pass cross-clause relationship detection and contextual risk refinement.
    Operates in-place over analyzed_clauses and returns the updated list.
    Preserves original_text on every clause.
    """
    _parse_clause_numbers(analyzed_clauses)

    substantive = [c for c in analyzed_clauses if not c.get("is_heading") and not c.get("is_structural")]

    # 1. Identify risk anchor clauses across the document
    penalty_anchors: List[Dict[str, Any]] = []
    forfeiture_anchors: List[Dict[str, Any]] = []
    withholding_anchors: List[Dict[str, Any]] = []
    unilateral_term_anchors: List[Dict[str, Any]] = []
    fee_forfeiture_anchors: List[Dict[str, Any]] = []
    organizer_indemnity_anchors: List[Dict[str, Any]] = []
    ip_overreach_anchors: List[Dict[str, Any]] = []
    portfolio_gag_anchors: List[Dict[str, Any]] = []
    contingent_pay_anchors: List[Dict[str, Any]] = []

    for c in substantive:
        txt = c["original_text"]
        amt = _extract_max_amount(txt)
        c["_max_amount"] = amt
        rule_ids = {rv.get("rule_id") for rv in c.get("rule_violations", [])}

        if RE_FINANCIAL_PENALTY.search(txt) and (amt > 0 or c["final_risk_level"] in ("high", "medium")):
            # Determine severity of the linked penalty anchor
            c["_penalty_severity"] = (
                "severe"
                if (c["final_risk_level"] == "high" or amt >= 25000 or bool(c.get("rule_violations")))
                else "moderate"
            )
            penalty_anchors.append(c)

        if RE_DEPOSIT_FORFEITURE.search(txt) and c["final_risk_level"] in ("high", "medium"):
            forfeiture_anchors.append(c)

        if RE_CERT_WITHHOLDING.search(txt):
            withholding_anchors.append(c)

        if RE_UNILATERAL_FIRING.search(txt):
            unilateral_term_anchors.append(c)

        if RE_FEE_FORFEITURE.search(txt) and c["final_risk_level"] == "high":
            fee_forfeiture_anchors.append(c)

        if RE_ORGANIZER_INDEMNITY.search(txt) and c["final_risk_level"] == "high":
            organizer_indemnity_anchors.append(c)

        if RE_OVERBROAD_IP.search(txt) or "RULE_TOTAL_IP_SURRENDER_PREEXISTING" in rule_ids:
            ip_overreach_anchors.append(c)

        if RE_PORTFOLIO_GAG.search(txt):
            portfolio_gag_anchors.append(c)

        if (
            RE_CONTINGENT_WITHHOLDING.search(txt)
            or "RULE_CONTINGENT_STIPEND_PAYOUT" in rule_ids
            or c in fee_forfeiture_anchors
        ):
            contingent_pay_anchors.append(c)

    # 2. Evaluate cross-clause relationships for each substantive clause
    for c in substantive:
        txt = c["original_text"]
        txt_lower = txt.lower()

        # ------------------------------------------------------------------
        # RULE CC-1: Mandatory Lock-In / Minimum Tenure linked to Penalty/Bond
        # Condition-aware: checks actual linked penalties, not blind escalation.
        # ------------------------------------------------------------------
        if RE_LOCK_IN.search(txt) and c not in penalty_anchors:
            linked_penalties = _find_matching_anchors(c, penalty_anchors, allow_doc_wide=True)
            linked_withholdings = _find_matching_anchors(c, withholding_anchors, allow_doc_wide=False)
            linked_forfeitures = _find_matching_anchors(c, forfeiture_anchors, allow_doc_wide=True)

            if linked_penalties:
                primary = linked_penalties[0]
                _add_cross_reference(c, primary, "PENALTY_ANCHOR")
                for w in linked_withholdings[:1]:
                    _add_cross_reference(c, w, "WITHHOLDING_ANCHOR")

                # Severe financial exit penalty or coercive withholding -> HIGH RISK
                # Moderate/small financial penalty -> MEDIUM RISK
                is_severe = (
                    primary.get("_penalty_severity") == "severe"
                    or bool(linked_withholdings)
                )
                target_level = "high" if is_severe else "medium"
                _elevate_risk(c, target_level, new_category="FINANCIAL_PENALTY_BOND")

                ref_name = _format_clause_ref(primary)
                amt_note = f" (₹{int(primary['_max_amount']):,})" if primary.get("_max_amount", 0) > 0 else ""
                c["explanation"] = (
                    f"This mandatory lock-in period is directly enforced by {ref_name}{amt_note}, "
                    f"turning a fixed internship/employment tenure into a financially coercive restriction "
                    f"that penalizes early exit even for academic conflicts or better career opportunities."
                )
                c["recommended_action"] = (
                    f"Negotiate removal of the mandatory lock-in or decouple it from {ref_name}'s financial exit penalty. "
                    f"Replace with a mutual 14–30 day notice period without exit fines."
                )
            elif linked_forfeitures:
                primary = linked_forfeitures[0]
                _add_cross_reference(c, primary, "FORFEITURE_ANCHOR")
                _elevate_risk(c, "medium", new_category="FINANCIAL_PENALTY_BOND")
                ref_name = _format_clause_ref(primary)
                c["explanation"] = (
                    f"This mandatory lock-in period is linked to {ref_name} (deposit forfeiture on early departure), "
                    f"creating financial loss if you need to exit before completion."
                )
                c["recommended_action"] = (
                    f"Clarify that early exit with reasonable notice will not trigger forfeiture under {ref_name}."
                )
            else:
                # Standalone lock-in with NO linked financial penalty or forfeiture:
                # Do NOT classify as high risk — flag as moderate advisory caution only.
                _elevate_risk(c, "medium", new_category="TERMINATION_NOTICE")
                c["explanation"] = (
                    "This clause states a mandatory lock-in period. While no direct financial exit penalty "
                    "was found linked to it, binding lock-in wording can still restrict academic flexibility."
                )
                c["recommended_action"] = (
                    "Confirm in writing that there is no financial penalty for early withdrawal due to academic or personal emergencies."
                )

        # ------------------------------------------------------------------
        # RULE CC-2: Coercive Certificate / Evaluation Withholding linked to Exit Penalty
        # ------------------------------------------------------------------
        if c in withholding_anchors:
            linked_penalties = _find_matching_anchors(c, penalty_anchors, allow_doc_wide=True)
            if linked_penalties:
                primary = linked_penalties[0]
                _add_cross_reference(c, primary, "PENALTY_ANCHOR")
                _elevate_risk(c, "high", new_category="FINANCIAL_PENALTY_BOND")
                ref_name = _format_clause_ref(primary)
                if "cross-clause" not in c["explanation"].lower() and ref_name not in c["explanation"]:
                    c["explanation"] = (
                        f"{c['explanation']} This withholding mechanism is tied to {ref_name} to coerce "
                        f"payment of exit penalties."
                    )

        # ------------------------------------------------------------------
        # RULE CC-3: Termination / Notice / Eviction linked to Penalties, Bonds,
        # Deposit Forfeiture, Asymmetric Firing, or Fee Forfeiture
        # ------------------------------------------------------------------
        if RE_TERMINATION_CLAUSE.search(txt):
            # 3a. Explicit or document-wide link to financial exit penalty / service bond
            linked_penalties = _find_matching_anchors(c, penalty_anchors, allow_doc_wide=True)
            explicit_penalty_link = any(
                any(ref == (p.get("clause_number") or "") or ref == (p.get("section_number") or "") for ref in c["explicit_refs"])
                for p in linked_penalties
            )
            # Only link termination clause to a document-wide penalty if the termination clause
            # governs student resignation/exit/notice (not purely company-side firing alone)
            is_student_exit_or_mutual = any(
                k in txt_lower for k in [
                    "intern must provide", "either party may terminate", "resign", "early departure",
                    "termination by intern", "termination covenants", "90 days written notice", "60 (sixty) days"
                ]
            )
            if linked_penalties and (explicit_penalty_link or is_student_exit_or_mutual):
                primary = linked_penalties[0]
                _add_cross_reference(c, primary, "PENALTY_ANCHOR")
                target_level = "high" if primary.get("_penalty_severity") == "severe" else "medium"
                elevated = _elevate_risk(c, target_level, new_category="TERMINATION_NOTICE")
                ref_name = _format_clause_ref(primary)
                if elevated or explicit_penalty_link:
                    c["explanation"] = (
                        f"This termination/notice clause interacts directly with {ref_name} "
                        f"({_infer_anchor_label(primary, 'PENALTY_ANCHOR')}). Even if you serve notice, "
                        f"leaving before the bond/lock-in period ends can still trigger heavy financial exit penalties."
                    )
                    c["recommended_action"] = (
                        f"Add explicit language confirming that providing mutual written notice (e.g., 14–30 days) "
                        f"terminates the agreement cleanly without triggering {ref_name}."
                    )

            # 3b. Asymmetric termination pairing (student long notice vs sibling company instant dismissal)
            if c not in unilateral_term_anchors and any(k in txt_lower for k in ["days prior written notice", "days written notice", "no academic exam leave", "no leave"]):
                sibling_firing = _find_matching_anchors(c, unilateral_term_anchors, allow_doc_wide=False)
                if sibling_firing:
                    _add_cross_reference(c, sibling_firing[0], "UNILATERAL_TERMINATION_ANCHOR")
                    _elevate_risk(c, "high", new_category="TERMINATION_NOTICE")

            # 3c. Hostel / Residency eviction or cancellation linked to Deposit Forfeiture
            if forfeiture_anchors and any(k in txt_lower for k in ["deposit", "eviction", "expelled", "cancellation", "vacates"]):
                linked_forf = _find_matching_anchors(c, forfeiture_anchors, allow_doc_wide=True)
                if linked_forf:
                    primary = linked_forf[0]
                    _add_cross_reference(c, primary, "FORFEITURE_ANCHOR")
                    if "will not be refunded" in txt_lower or "expelled" in txt_lower:
                        _elevate_risk(c, "high", new_category="DEPOSIT_REFUND")
                    else:
                        _elevate_risk(c, "medium", new_category="TERMINATION_NOTICE")

            # 3d. At-will termination linked to Contingent Pay & Fee Forfeiture
            if fee_forfeiture_anchors and c in unilateral_term_anchors:
                primary = fee_forfeiture_anchors[0]
                if primary["index"] != c["index"]:
                    _add_cross_reference(c, primary, "FEE_FORFEITURE_ANCHOR")
                    ref_name = _format_clause_ref(primary)
                    if ref_name not in c["explanation"]:
                        c["explanation"] = (
                            f"{c['explanation']} Combined with {ref_name}, immediate termination without cause "
                            f"risks forfeiture of work already completed."
                        )

        # ------------------------------------------------------------------
        # RULE CC-4: Security Deposit / Conditional Refund linked to Deposit Forfeiture
        # ------------------------------------------------------------------
        if RE_DEPOSIT_REQUIREMENT.search(txt) and c not in forfeiture_anchors:
            sibling_forf = _find_matching_anchors(c, forfeiture_anchors, allow_doc_wide=False)
            if sibling_forf:
                primary = sibling_forf[0]
                _add_cross_reference(c, primary, "FORFEITURE_ANCHOR")
                _elevate_risk(c, "medium", new_category="DEPOSIT_REFUND")
                ref_name = _format_clause_ref(primary)
                c["explanation"] = (
                    f"Although framed as a refundable deposit/refund process, this clause is governed by "
                    f"{ref_name}, which allows full forfeiture of the deposit upon early withdrawal or subjective review."
                )
                c["recommended_action"] = (
                    f"Limit any deposit deductions strictly to proven physical asset damage and remove the "
                    f"blanket forfeiture trigger in {ref_name}."
                )

        # ------------------------------------------------------------------
        # RULE CC-5: Force Majeure / Cancellation One-Sided Refund linked to Organizer Indemnity
        # ------------------------------------------------------------------
        if RE_FORCE_MAJEURE_REFUND.search(txt) and organizer_indemnity_anchors:
            primary = organizer_indemnity_anchors[0]
            if primary["index"] != c["index"]:
                _add_cross_reference(c, primary, "ORGANIZER_INDEMNITY_ANCHOR")
                _elevate_risk(c, "medium", new_category="LIABILITY_INDEMNITY")
                ref_name = _format_clause_ref(primary)
                c["explanation"] = (
                    f"This cancellation/force majeure clause still obligates student organizers to refund "
                    f"all expended marketing costs within 14 days, compounding the personal liability risk in {ref_name}."
                )
                c["recommended_action"] = (
                    f"Make force majeure mutually excusable with zero refund penalty for unpreventable events, "
                    f"and remove personal student guarantees in {ref_name}."
                )

        # ------------------------------------------------------------------
        # RULE CC-7: Stipend / Compensation Linked to Contingent Withholding
        # Informational cross-reference ONLY — does NOT falsely escalate
        # standard base-stipend clauses so risk counts remain clean and honest.
        # ------------------------------------------------------------------
        if RE_BASE_COMPENSATION.search(txt) and c not in contingent_pay_anchors:
            sibling_contingent = _find_matching_anchors(c, contingent_pay_anchors, allow_doc_wide=False)
            if sibling_contingent:
                primary = sibling_contingent[0]
                _add_cross_reference(c, primary, "FEE_FORFEITURE_ANCHOR")
                ref_name = _format_clause_ref(primary)
                if ref_name not in c.get("explanation", ""):
                    base_exp = c.get("explanation", "").strip()
                    note = (
                        f"Note: Base stipend/compensation certainty directly depends on {ref_name}, "
                        f"which permits contingent or discretionary withholding/delay of payouts."
                    )
                    c["explanation"] = f"{base_exp} {note}".strip()

    # ------------------------------------------------------------------
    # RULE CC-6: Career / Portfolio Lock-Out Compound Risk
    # Bidirectional linkage when overbroad personal-device / academic IP
    # assignment coexists with broad confidentiality / portfolio gag terms.
    # ------------------------------------------------------------------
    if ip_overreach_anchors and portfolio_gag_anchors:
        for ip_clause in ip_overreach_anchors:
            linked_gags = [g for g in portfolio_gag_anchors if g["index"] != ip_clause["index"]]
            if not linked_gags:
                continue
            for gag_clause in linked_gags:
                _add_cross_reference(ip_clause, gag_clause, "PORTFOLIO_GAG_ANCHOR")
            if ip_clause.get("final_risk_level") == "low":
                _elevate_risk(ip_clause, "medium", new_category="INTELLECTUAL_PROPERTY")
            gag_refs = ", ".join(_format_clause_ref(g) for g in linked_gags)
            if gag_refs not in ip_clause.get("explanation", ""):
                base_exp = ip_clause.get("explanation", "").strip()
                ip_clause["explanation"] = (
                    f"{base_exp} Combined with {gag_refs} (Portfolio & Confidentiality Restriction), "
                    f"this creates a career and academic lock-out: the organization claims ownership over "
                    f"personal or academic work while simultaneously barring you from showcasing the "
                    f"experience, role, or skills on your CV, LinkedIn, or GitHub."
                ).strip()

        for gag_clause in portfolio_gag_anchors:
            linked_ips = [ip for ip in ip_overreach_anchors if ip["index"] != gag_clause["index"]]
            if not linked_ips:
                continue
            for ip_clause in linked_ips:
                _add_cross_reference(gag_clause, ip_clause, "OVERBROAD_IP_ANCHOR")
            if gag_clause.get("final_risk_level") == "low":
                _elevate_risk(gag_clause, "medium", new_category="CONFIDENTIALITY_NDA")
            ip_refs = ", ".join(_format_clause_ref(ip) for ip in linked_ips)
            if ip_refs not in gag_clause.get("explanation", ""):
                base_exp = gag_clause.get("explanation", "").strip()
                gag_clause["explanation"] = (
                    f"{base_exp} Combined with {ip_refs} (Overbroad IP Assignment), this creates a "
                    f"career and academic lock-out: you cannot reference your work or skills on your "
                    f"resume, LinkedIn, or GitHub while losing ownership and attribution rights over "
                    f"related personal or academic projects."
                ).strip()

    # Clean up internal temporary keys
    for c in analyzed_clauses:
        c.pop("_max_amount", None)
        c.pop("_penalty_severity", None)

    return analyzed_clauses


def _classify_cluster_theme(
    comp_edges: List[Dict[str, Any]],
    members: List[Dict[str, Any]]
) -> str:
    """Derive a concise, grounded human-readable title for a cross-clause cluster."""
    rel_text = " ".join(e["relationship"].lower() for e in comp_edges)
    has_penalty = any(k in rel_text for k in ["penalty", "service bond", "liquidated damages", "withholding"])
    has_termination = "termination" in rel_text
    has_stipend = "contingent pay" in rel_text or "fee forfeiture" in rel_text
    has_deposit = "deposit forfeiture" in rel_text
    has_career_ip = "portfolio" in rel_text or "overbroad ip" in rel_text
    has_indemnity = "organizer indemnity" in rel_text

    if has_penalty and has_termination and has_stipend:
        return "Exit Penalty, Asymmetric Termination & Stipend Withholding Trap"
    if has_penalty and has_termination:
        return "Exit Penalty, Lock-In & Termination Enforcement Trap"
    if has_penalty:
        return "Financial Exit Penalty & Coercive Lock-In Cluster"
    if has_deposit:
        return "Security Deposit & Blanket Forfeiture Risk"
    if has_career_ip:
        return "Career, Portfolio & IP Lock-Out Compound Risk"
    if has_stipend:
        return "Stipend Certainty & Contingent Withholding Risk"
    if has_indemnity:
        return "Event Cancellation & Personal Organizer Liability"
    return "Compound Cross-Clause Risk Cluster"


def build_cross_clause_summary(analyzed_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a structured, JSON-serializable summary of all detected cross-clause
    relationships and grouped compound risk clusters derived strictly from real
    detected `related_clauses` in the analyzed document.
    """
    substantive = [
        c for c in analyzed_clauses
        if not c.get("is_heading") and not c.get("is_structural")
    ]
    clause_by_index: Dict[int, Dict[str, Any]] = {c["index"]: c for c in substantive}

    edges: List[Dict[str, Any]] = []
    adjacency: Dict[int, set] = {}

    for c in substantive:
        src_idx = c["index"]
        src_ref = _format_clause_ref(c)
        for rel in c.get("related_clauses", []):
            tgt_idx = rel["target_index"]
            edge = {
                "source_index": src_idx,
                "source_ref": src_ref,
                "target_index": tgt_idx,
                "target_ref": rel["target_clause_ref"],
                "relationship": rel["relationship"],
                "display_text": rel["display_text"],
            }
            edges.append(edge)
            adjacency.setdefault(src_idx, set()).add(tgt_idx)
            adjacency.setdefault(tgt_idx, set()).add(src_idx)

    elevated_clauses_count = sum(
        1 for c in substantive if c.get("cross_clause_risk_elevated")
    )

    # Group connected components into compound risk clusters (strictly from real edges)
    clusters: List[Dict[str, Any]] = []
    if edges:
        visited = set()
        components: List[List[int]] = []

        for node in sorted(adjacency.keys()):
            if node in visited:
                continue
            comp = []
            queue = [node]
            visited.add(node)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nxt in sorted(adjacency.get(curr, set())):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            comp.sort()
            components.append(comp)

        components.sort(key=lambda comp: comp[0])

        for idx, comp_indices in enumerate(components, start=1):
            comp_set = set(comp_indices)
            members = [clause_by_index[i] for i in comp_indices if i in clause_by_index]
            comp_edges = [
                e for e in edges
                if e["source_index"] in comp_set and e["target_index"] in comp_set
            ]

            clause_refs = [
                _format_clause_ref(clause_by_index[i]) if i in clause_by_index else f"Clause #{i}"
                for i in comp_indices
            ]

            # Severity is the maximum risk level present among clauses in this cluster
            max_risk_rank = 1
            for m in members:
                rank = RISK_ORDER.get(m.get("final_risk_level", "low"), 1)
                if rank > max_risk_rank:
                    max_risk_rank = rank
            severity = {1: "low", 2: "medium", 3: "high"}[max_risk_rank]

            title = _classify_cluster_theme(comp_edges, members)
            rel_labels = []
            for e in comp_edges:
                if e["relationship"] not in rel_labels:
                    rel_labels.append(e["relationship"])

            high_in_cluster = sum(1 for m in members if m.get("final_risk_level") == "high")
            elevated_in_cluster = sum(1 for m in members if m.get("cross_clause_risk_elevated"))

            refs_str = ", ".join(clause_refs)
            rels_str = "; ".join(rel_labels)
            summary = (
                f"Links {refs_str} across {len(comp_edges)} cross-clause relationship(s) "
                f"({rels_str}). Contains {high_in_cluster} high-risk clause(s)"
                + (
                    f" and {elevated_in_cluster} contextually risk-elevated clause(s)."
                    if elevated_in_cluster > 0
                    else "."
                )
            )

            clusters.append({
                "cluster_id": f"CLUSTER_{idx}",
                "title": title,
                "severity": severity,
                "clause_refs": clause_refs,
                "clause_indices": comp_indices,
                "summary": summary,
            })

    return {
        "total_links": len(edges),
        "elevated_clauses_count": elevated_clauses_count,
        "edges": edges,
        "clusters": clusters,
    }
