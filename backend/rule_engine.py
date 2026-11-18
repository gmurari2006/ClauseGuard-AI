"""
ClauseGuard AI - Rule-Based Student Protection & Statutory Compliance Engine
Separated explicitly from the ML classifier.
Applies deterministic legal rules, statutory safeguards, fairness calculations,
and missing-clause detection specifically for student agreements.
"""
import re
from typing import Dict, List, Any, Tuple


# Automated Student Protection & Legal Context Rules
STATUTORY_RULES = [
    {
        "id": "RULE_SECTION_27_RESTRAINT_OF_TRADE",
        "name": "Post-Engagement Non-Compete / Restraint of Trade Flag",
        "severity": "high",
        "rule_type": "STATUTORY_CONTEXT",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:not\s+compete|non-compete|competing\s+business|restraint\s+of\s+trade|barred\s+from\s+joining|shall\s+not\s+work\s+for\s+any\s+competitor)\b',
            re.IGNORECASE
        ),
        "explanation": "Restricts where a student can work or study after the engagement ends. Where Indian law applies, Section 27 of the Indian Contract Act, 1872 generally treats agreements in restraint of a lawful profession, trade, or business as void to that extent (subject to narrow statutory exceptions and specific governing-law facts).",
        "statute": "Indian Contract Act, 1872 — Section 27 (Restraint of Trade; subject to governing law & facts)",
        "action": "Request removal of post-engagement non-compete restrictions and limit obligations strictly to protecting genuine confidential trade secrets."
    },
    {
        "id": "RULE_CERTIFICATE_RETENTION_COLLATERAL",
        "name": "Original Certificate / Completion Credential Withholding Flag",
        "severity": "high",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'(?:\b(?:deposit|surrender|submit|retain|retention\s+of|hold)\b.*?\b(?:degrees?|certificates?|marks\s*cards?|passports?|marksheets?)\b.*?\b(?:collateral|security|with\s+(?:the\s+)?company|until|hr)\b|\bwithhold\b.*?\b(?:completion\s+certificate|academic\s+evaluation|experience\s+letter|relieving\s+letter|certificates?)\b.*?\b(?:until|penalt|damages|paid)\b)',
            re.IGNORECASE | re.DOTALL
        ),
        "explanation": "Requiring surrender of original academic documents or withholding internship completion certificates and academic evaluation signatures as leverage for exit penalties poses severe academic and career risk to students. (Regulatory advisories against retaining original certificates primarily govern educational institutions; enforceability in private contracts depends on applicable law and facts.)",
        "statute": "Automated Student Protection Rule (Credential & Document Withholding Flag — not a legal determination)",
        "action": "Refuse to surrender original academic certificates or passports (offer attested copies only), and request removal of any term conditioning academic evaluation signatures or completion certificates on exit penalty payments."
    },
    {
        "id": "RULE_EXCESSIVE_TRAINING_BOND",
        "name": "Financial Lock-in / Liquidated Damages Exit Penalty Flag",
        "severity": "high",
        "rule_type": "STATUTORY_CONTEXT",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:liquidated\s+damages|exit\s+penalty|training\s+bond|minimum\s+lock-in|service\s+bond|promissory\s+note|indemnity\s+bond|reimburse\s+all\s+training\s+costs?|recruitment\s+penalty\s+fee)\b.*?(?:Rs\.?|INR|₹|\$|EUR|\d{4,})',
            re.IGNORECASE | re.DOTALL
        ),
        "explanation": "Imposes a fixed financial penalty or liquidated-damages bond for early exit. Where Indian law applies, Section 74 of the Indian Contract Act, 1872 provides for reasonable compensation for actual loss (not exceeding the stated sum) rather than automatically rendering every bond or penalty enforceable or automatically unenforceable; enforceability depends on actual facts and demonstrable training costs.",
        "statute": "Indian Contract Act, 1872 — Section 74 (Reasonable Compensation; subject to applicable legal facts)",
        "action": "Request removal of arbitrary exit penalties or negotiate capping any recovery strictly to actual, documented external training expenses with monthly pro-rata reduction."
    },
    {
        "id": "RULE_UNILATERAL_IMMEDIATE_TERMINATION",
        "name": "Asymmetric Termination Rights Flag",
        "severity": "medium",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:without\s+cause,?\s*immediately|terminate.*?at\s+any\s+time.*?without\s+notice|dismiss.*?instantly|(?:terminate|dismiss|cancel).*?sole\s+and\s+unappealable\s+discretion)\b',
            re.IGNORECASE
        ),
        "explanation": "Allows the counterparty to terminate immediately without cause or notice pay while imposing longer notice obligations or exit restrictions on the student.",
        "statute": "Automated Student Protection Rule (Termination Symmetry Flag)",
        "action": "Negotiate mutual notice parity (for example, 14–30 days' written notice for both parties, or stipend in lieu of notice)."
    },
    {
        "id": "RULE_TOTAL_IP_SURRENDER_PREEXISTING",
        "name": "Overbroad Intellectual Property Assignment Flag",
        "severity": "high",
        "rule_type": "STATUTORY_CONTEXT",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:whether\s+created\s+during\s+office\s+hours\s+or\s+on\s+personal\s+devices|personal\s+time\s+during\s+weekends|academic\s+thesis.*?sole\s+proprietary|waives\s+all\s+moral\s+rights)\b',
            re.IGNORECASE
        ),
        "explanation": "Extends company IP ownership beyond assigned workplace deliverables to personal devices, off-hours work, moral rights, or university capstone/thesis projects. Where Indian law applies, Sections 17 and 57 of the Copyright Act, 1957 address first ownership of work created under a contract of service and authors' special (moral) rights, subject to contract terms and facts.",
        "statute": "Copyright Act, 1957 (India) — Sections 17 & 57 (Ownership & Author's Special Rights; subject to facts)",
        "action": "Insert an explicit written carve-out protecting pre-existing student work, off-hours personal projects, open-source contributions, and academic coursework/theses."
    },
    {
        "id": "RULE_HOSTEL_NON_REFUNDABLE_DEPOSIT",
        "name": "Broad Security Deposit Forfeiture Flag",
        "severity": "high",
        "rule_type": "STATUTORY_CONTEXT",
        "applies_to": ["HOSTEL_RESIDENCY", "INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:deposit\s+(?:of\s+.*?)?is\s+strictly\s+non-refundable|deposits?\s+will\s+not\s+be\s+refunded|arbitrary\s+amounts|deduct\s+arbitrary|forfeited\s+to\s+the\s+(?:hostel|company)|entire\s+security\s+deposit.*?forfeited)\b',
            re.IGNORECASE | re.DOTALL
        ),
        "explanation": "Allows blanket forfeiture of the security deposit on early withdrawal, subjective internal scoring, or unitemized deductions. While not every forfeiture term is automatically invalid, disputes in India over forfeiture of sums named in a contract are typically examined under Section 74 of the Indian Contract Act, 1872 (reasonable compensation for actual loss) and, where a consumer relationship exists, unfair contract terms under the Consumer Protection Act, 2019 — depending on the facts.",
        "statute": "Indian Contract Act, 1872 — Section 74 / Consumer Protection Act, 2019 (where applicable; subject to facts)",
        "action": "Negotiate a definite 15–30 day refund window after handover and limit deductions strictly to itemized, documented physical damage verified jointly."
    },
    {
        "id": "RULE_UNLIMITED_STUDENT_INDEMNITY",
        "name": "Disproportionate Student Indemnification Flag",
        "severity": "high",
        "rule_type": "STATUTORY_CONTEXT",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "EVENT_SPONSORSHIP", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:student\s+shall\s+fully\s+indemnify|personally\s+guarantee\s+payment\s+and\s+indemnify|assumes\s+full\s+personal\s+financial\s+liability|treble\s+damages)\b',
            re.IGNORECASE
        ),
        "explanation": "Shifts broad commercial, third-party, or operational outage liability onto an individual student or student committee. Under Section 124 of the Indian Contract Act, 1872, an indemnity obligates the indemnifier to save the promisee from covered loss, making uncapped personal indemnities financially hazardous for students.",
        "statute": "Indian Contract Act, 1872 — Section 124 (Contract of Indemnity)",
        "action": "Limit student liability strictly to willful misconduct or gross negligence and cap aggregate monetary exposure at the total compensation received."
    },
    {
        "id": "RULE_GAG_DISPARAGEMENT_CLAUSE",
        "name": "Broad Non-Disparagement / Portfolio Restriction Flag",
        "severity": "medium",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:never\s+to\s+post\s+negative\s+comments|glassdoor|prohibited\s+from\s+mentioning\s+the\s+company\s+name.*?on\s+their\s+cv|resume|linkedin)\b',
            re.IGNORECASE
        ),
        "explanation": "Restricts students from listing legitimate internship experience, role title, or non-confidential skills on resumes/LinkedIn and imposes broad anti-review restrictions.",
        "statute": "Automated Student Protection Rule (Career & Portfolio Restriction Flag)",
        "action": "Clarify in writing that the student may list the company name, internship title, dates, and high-level non-confidential skills on resumes and LinkedIn."
    },
    {
        "id": "RULE_CONTINGENT_STIPEND_PAYOUT",
        "name": "Contingent or Discretionary Stipend Flag",
        "severity": "medium",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'\b(?:contingent\s+entirely\s+upon\s+client|unpaid\s+voluntary\s+basis|disbursed\s+90\s+days\s+in\s+arrears|withhold,\s*delay,\s*or\s*reduce\s+the\s+monthly\s+stipend)\b',
            re.IGNORECASE
        ),
        "explanation": "Conditions student stipend payouts on external client sign-off, subjective evaluation, or 90-day arrears, creating payment uncertainty for student living costs.",
        "statute": "Automated Student Protection Rule (Compensation Certainty Flag)",
        "action": "Request a fixed monthly disbursement date (e.g., 5th of each month) independent of external client invoices or subjective discretionary reductions."
    },
    {
        "id": "RULE_UNILATERAL_AMENDMENT",
        "name": "Unilateral Modification of Agreement Terms Flag",
        "severity": "medium",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "HOSTEL_RESIDENCY", "EVENT_SPONSORSHIP", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'(?:\bunilaterally\s+(?:amend|modify|alter|revise|change|update)\b|\breserves?\s+the\s+(?:unilateral\s+|sole\s+)?right\s+to\s+(?:amend|modify|alter|revise|change|update)[^.;]{0,120}\b(?:terms|stipend|fees?|rent|compensation|agreement|conditions|schedule|policy|policies)\b|\b(?:may|shall\s+have\s+the\s+right\s+to)\s+(?:amend|modify|alter|revise|change)[^.;]{0,120}\b(?:terms|stipend|fees?|rent|compensation|agreement)\b[^.;]{0,120}\b(?:without\s+(?:prior\s+)?(?:written\s+)?(?:consent|approval|notice)|at\s+its\s+sole\s+discretion)\b)',
            re.IGNORECASE
        ),
        "explanation": "Allows the counterparty to unilaterally alter stipend, fees, rent, or material contractual obligations without the student's written consent, undermining certainty of the agreed deal.",
        "statute": "Automated Student Protection Rule (Unilateral Contract Modification Flag — not a legal determination)",
        "action": "Request a mutual amendment clause requiring any material change to stipend, fees, tenure, or core obligations to be agreed in writing and signed by both parties."
    },
    {
        "id": "RULE_FOREIGN_COSTLY_ARBITRATION",
        "name": "Foreign Jurisdiction / Costly Student-Borne Arbitration Flag",
        "severity": "high",
        "rule_type": "STATUTORY_CONTEXT",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "HOSTEL_RESIDENCY", "EVENT_SPONSORSHIP", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'(?:\b(?:exclusive\s+jurisdiction|courts?\s+(?:located\s+)?in|laws?\s+of|binding\s+arbitration\s+in|arbitration\s+(?:seated|held|conducted)\s+in|seated\s+in)[^.;]{0,120}\b(?:new\s+york|delaware|singapore|siac|london|lcia|california|england\s+and\s+wales|united\s+states|usa|uk)\b|\b(?:siac|lcia|icc\s+arbitration)\b|\barbitration\s+costs?\s+(?:shall\s+be\s+)?borne\s+(?:solely|exclusively|entirely)\s+by\s+the\s+(?:student|intern|employee|resident|contractor)\b)',
            re.IGNORECASE
        ),
        "explanation": "Subjects disputes to distant foreign courts or international arbitration seats (such as New York, Delaware, Singapore/SIAC, or London) or shifts all arbitration expenses solely onto the student. Where Indian parties contract domestically, practical access to remedies and cost allocation under the Arbitration and Conciliation Act, 1996 depend on the chosen seat, governing law, and contract terms; foreign venues can make pursuing legitimate student claims financially prohibitive.",
        "statute": "Arbitration and Conciliation Act, 1996 (India — Domestic Forum & Cost Allocation Considerations; subject to governing law & facts)",
        "action": "Negotiate Indian governing law, local courts or domestic arbitration in the student's or university's city, and equal or statutory sharing of arbitration costs."
    },
    {
        "id": "RULE_NO_ACADEMIC_LEAVE",
        "name": "Prohibition or Penalization of Academic / Exam Leave Flag",
        "severity": "medium",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'(?:\bno\s+(?:academic|exam(?:ination)?|semester|university|study)\s+(?:exam\s+)?leave\b|\b(?:prohibit(?:ed|s)?|bar(?:red|s)?|not\s+permit(?:ted)?|shall\s+not\s+(?:avail|take|be\s+granted)|denied)[^.;]{0,100}\b(?:academic\s+leave|exam(?:ination)?\s+leave|study\s+leave|university\s+exam(?:ination)?s?|mandatory\s+exam(?:ination)?s?)\b|\b(?:penal(?:ty|iz|is)|deduct(?:ion|ed)?|forfeit(?:ure|ed)?|terminat(?:ed|ion)|disciplinary\s+action)[^.;]{0,100}\b(?:university\s+exam(?:ination)?s?|mandatory\s+exam(?:ination)?s?|academic\s+exam(?:ination)?s?|semester\s+exam(?:ination)?s?)\b|\b(?:academic|exam(?:ination)?|study)\s+leave\b[^.;]{0,80}\b(?:strictly\s+prohibited|will\s+not\s+be\s+granted|shall\s+not\s+be\s+permitted|not\s+allowed)\b)',
            re.IGNORECASE
        ),
        "explanation": "Explicitly prohibits academic or examination leave, or penalizes a student for attending mandatory university examinations, creating a direct conflict with degree progression.",
        "statute": "Automated Student Protection Rule (Academic Accommodation & Exam Leave Flag)",
        "action": "Request a written academic accommodation clause permitting unpaid or adjusted leave for scheduled university examinations, lab vivas, and mandatory academic evaluations with reasonable prior notice."
    },
    {
        "id": "RULE_DEVICE_PRIVACY_OVERREACH",
        "name": "Personal Device Spyware / Remote-Wipe Overreach Flag",
        "severity": "high",
        "rule_type": "STUDENT_PROTECTION_HEURISTIC",
        "applies_to": ["INTERNSHIP", "PLACEMENT_EMPLOYMENT", "HOSTEL_RESIDENCY", "STUDENT_SERVICE"],
        "pattern": re.compile(
            r'(?:\b(?:spyware|keylogger|screen\s+capture|webcam\s+monitoring|continuous\s+monitoring|unrestricted\s+(?:monitoring|access|surveillance)|remote\s+wipe|factory\s+reset|mdm\s+software|monitoring\s+(?:software|agent|tool))[^.;]{0,120}\b(?:personal\s+(?:laptop|device|computer|phone|mobile|smartphone|hardware)|byod|own\s+device)\b|\b(?:personal\s+(?:laptop|device|computer|phone|mobile|smartphone|hardware)|byod|own\s+device)[^.;]{0,120}\b(?:spyware|keylogger|screen\s+capture|webcam\s+monitoring|unrestricted\s+(?:monitoring|access|surveillance)|remote\s+wipe|factory\s+reset|install\s+(?:tracking|monitoring|surveillance)\s+(?:software|agent|tool)|full\s+administrative\s+access)\b)',
            re.IGNORECASE
        ),
        "explanation": "Demands invasive surveillance, monitoring/spyware installation, or remote-wipe authority over the student's personal laptop or mobile phone, risking unauthorized access to personal files and irreversible loss of academic or private data.",
        "statute": "Automated Student Protection Rule (Personal Device Privacy & Data Autonomy Flag — not a legal determination)",
        "action": "Request company-issued hardware or an isolated virtual machine/workspace for organizational tasks, and refuse remote-wipe or unrestricted monitoring software on personal devices."
    }
]


# Weighted multi-signal patterns for transparent document domain classification
DOMAIN_SIGNALS: Dict[str, List[Tuple[re.Pattern, float]]] = {
    "HOSTEL_RESIDENCY": [
        (re.compile(r'\bhostel\s+(?:room|residency|occupancy|management|allotment|agreement|warden|premises)\b', re.IGNORECASE), 5.0),
        (re.compile(r'\b(?:room\s+allotment|hostel\s+warden|mess\s+fee|curfew\s+hours|dormitory|student\s+residences?)\b', re.IGNORECASE), 4.5),
        (re.compile(r'\b(?:campus\s+residency|room\s+occupancy|room\s+rent|hostel\s+resident|resident\s+student)\b', re.IGNORECASE), 3.5),
        (re.compile(r'\b(?:hostel|warden|move-in\s+date|caution\s+deposit)\b', re.IGNORECASE), 2.5),
        (re.compile(r'\bresidency\b', re.IGNORECASE), 1.5),
    ],
    "EVENT_SPONSORSHIP": [
        (re.compile(r'\b(?:corporate\s+|title\s+|event\s+|club\s+)?sponsorship\s+(?:agreement|contract|deliverables|tranches?)\b', re.IGNORECASE), 5.0),
        (re.compile(r'\b(?:student\s+(?:branch\s+)?organizing\s+committee|student\s+club\s+organizers?|exhibition\s+stall|sponsor\s+logo|target\s+footfall|technical\s+symposium)\b', re.IGNORECASE), 4.5),
        (re.compile(r'\b(?:sponsorship|campusfest|symposium|student\s+organizers?|student\s+club)\b', re.IGNORECASE), 3.0),
        (re.compile(r'\b(?:sponsor|main\s+stage|event\s+photographs)\b', re.IGNORECASE), 2.0),
        (re.compile(r'\b(?:fest|stall|banner)\b', re.IGNORECASE), 1.5),
    ],
    "INTERNSHIP": [
        (re.compile(r'\b(?:student\s+)?internship\s+(?:and\s+training\s+)?agreement\b', re.IGNORECASE), 5.0),
        (re.compile(r'\b(?:software\s+engineering\s+intern|data\s+engineering\s+intern|internship\s+completion\s+certificate|academic\s+internship|summer\s+intern(?:ship)?|summer\s+associate)\b', re.IGNORECASE), 4.5),
        (re.compile(r'\b(?:student\s+intern|internship\s+(?:duration|term|scope)|engineering\s+mentor|monthly\s+stipend)\b', re.IGNORECASE), 3.0),
        (re.compile(r'\binternship\b', re.IGNORECASE), 2.5),
        (re.compile(r'\b(?:intern|stipend)\b', re.IGNORECASE), 1.0),
        (re.compile(r'\btrainee\b', re.IGNORECASE), 0.75),
    ],
    "PLACEMENT_EMPLOYMENT": [
        (re.compile(r'\b(?:campus\s+placement|placement\s+(?:appointment|offer|bond)|graduate\s+engineer\s+trainee|offer\s+of\s+appointment|employee\s+service\s+agreement)\b', re.IGNORECASE), 5.0),
        (re.compile(r'\b(?:employment\s+(?:agreement|bond|contract|offer)|full-time\s+(?:placement\s+)?offer|annual\s+ctc|recruitment\s+penalty)\b', re.IGNORECASE), 4.5),
        (re.compile(r'\b(?:probationary\s+period|probation|service\s+bond|hr\s+director|remuneration)\b', re.IGNORECASE), 3.0),
        (re.compile(r'\b(?:placement|employment)\b', re.IGNORECASE), 2.5),
        (re.compile(r'\b(?:employee|salary|full-time|annual\s+package)\b', re.IGNORECASE), 1.5),
    ],
    "STUDENT_SERVICE": [
        (re.compile(r'\b(?:independent\s+(?:student\s+)?(?:tutor|contractor)(?:\s+and\s+content\s+creator)?\s+agreement|student\s+service\s+agreement|freelance\s+(?:service\s+)?agreement|content\s+creator\s+agreement)\b', re.IGNORECASE), 5.5),
        (re.compile(r'\b(?:student\s+contractor|independent\s+contractor|tutoring\s+(?:agreement|services?|industry)|per\s+completed\s+lesson|video\s+tutorials?)\b', re.IGNORECASE), 4.5),
        (re.compile(r'\b(?:service\s+agreement|freelance|content\s+developer|statement\s+of\s+work|consulting\s+services?)\b', re.IGNORECASE), 3.0),
        (re.compile(r'\b(?:contractor|tutoring|tutor|service\s+provider)\b', re.IGNORECASE), 2.0),
        (re.compile(r'\b(?:deliverables|milestone\s+billing)\b', re.IGNORECASE), 1.25),
    ],
}

DOMAIN_HEADER_BOOST = 2.5
DOMAIN_MIN_SCORE_THRESHOLD = 2.0


def score_document_domains(text: str) -> Dict[str, float]:
    """
    Compute transparent weighted multi-signal domain scores across all supported
    student agreement domains, weighting document titles/headers more heavily than
    incidental body mentions and capping per-pattern repetition.
    """
    header_zone = text[:450]
    scores: Dict[str, float] = {}

    for domain, patterns in DOMAIN_SIGNALS.items():
        score = 0.0
        for pattern, weight in patterns:
            matches = pattern.findall(text)
            if matches:
                # Cap repeated mentions per pattern to avoid noise overwhelming title/structure signals
                capped_count = min(len(matches), 3)
                score += weight * capped_count
                if pattern.search(header_zone):
                    score += weight * DOMAIN_HEADER_BOOST
        scores[domain] = round(score, 2)

    return scores


def detect_document_domain(text: str) -> str:
    """Classify the overall domain of the student agreement using weighted multi-signal scoring."""
    if not text or not text.strip():
        return "GENERAL_STUDENT_AGREEMENT"

    scores = score_document_domains(text)
    best_domain = max(scores, key=lambda d: scores[d])
    if scores[best_domain] < DOMAIN_MIN_SCORE_THRESHOLD:
        return "GENERAL_STUDENT_AGREEMENT"
    return best_domain


def get_rule_catalog() -> List[Dict[str, Any]]:
    """
    Return clean JSON-serializable metadata for every deterministic rule in the engine,
    explicitly distinguishing statutory context rules from student protection heuristics.
    """
    catalog: List[Dict[str, Any]] = []
    for rule in STATUTORY_RULES:
        catalog.append({
            "id": rule["id"],
            "name": rule["name"],
            "severity": rule["severity"],
            "applies_to": list(rule["applies_to"]),
            "statute": rule["statute"],
            "explanation": rule["explanation"],
            "action": rule["action"],
            "rule_type": rule.get("rule_type", "STUDENT_PROTECTION_HEURISTIC"),
        })
    return catalog


def run_statutory_rule_checks(paragraph_text: str, domain: str) -> List[Dict[str, Any]]:
    """Check a paragraph against statutory student protection rules."""
    violations = []
    
    for rule in STATUTORY_RULES:
        # Check domain applicability
        if domain != "GENERAL_STUDENT_AGREEMENT" and rule["applies_to"] and domain not in rule["applies_to"]:
            continue
            
        if rule["pattern"].search(paragraph_text):
            violations.append({
                "rule_id": rule["id"],
                "rule_name": rule["name"],
                "severity": rule["severity"],
                "explanation": rule["explanation"],
                "statute": rule["statute"],
                "action": rule["action"],
                "rule_type": rule.get("rule_type", "STUDENT_PROTECTION_HEURISTIC"),
            })
            
    return violations


def check_missing_information(full_text: str, domain: str) -> List[Dict[str, Any]]:
    """
    Check whether crucial student protections are absent from the agreement.
    Missing clauses can be just as dangerous as unfair ones.
    """
    text_lower = full_text.lower()
    missing_items = []
    
    # Universal check: Dispute resolution
    if not any(k in text_lower for k in ["dispute", "arbitration", "jurisdiction", "governing law", "conciliation"]):
        missing_items.append({
            "topic": "Dispute Resolution & Jurisdiction",
            "severity": "medium",
            "description": "No governing law or dispute resolution clause is defined, leaving legal venue ambiguous.",
            "recommendation": "Specify governing law and local courts in the student's or university's jurisdiction."
        })
        
    if domain in ["INTERNSHIP", "PLACEMENT_EMPLOYMENT"]:
        # Check working hours
        if not any(k in text_lower for k in ["hours per week", "working hours", "9:00", "schedule", "work hours"]):
            missing_items.append({
                "topic": "Working Hours & Schedule Definition",
                "severity": "medium",
                "description": "The contract does not specify maximum weekly working hours or daily schedules.",
                "recommendation": "Request a clause limiting commitment to 20 hrs/week during semesters or 40 hrs/week during vacations."
            })
            
        # Check stipend payment date
        if not any(k in text_lower for k in ["5th", "1st", "end of each month", "disbursed on", "calendar month", "bi-weekly"]):
            missing_items.append({
                "topic": "Stipend Disbursement Date",
                "severity": "medium",
                "description": "No explicit calendar date is set for monthly stipend disbursement.",
                "recommendation": "Insert: 'Stipend shall be disbursed on or before the 5th day of every calendar month.'"
            })
            
        # Check mentor / supervisor designation
        if not any(k in text_lower for k in ["mentor", "supervisor", "reporting manager", "reporting to"]):
            missing_items.append({
                "topic": "Reporting Manager / Mentorship Support",
                "severity": "low",
                "description": "No reporting manager or educational mentor is specified for the student trainee.",
                "recommendation": "Ensure a designated technical mentor is assigned to sign internship evaluation rubrics."
            })
            
        # Check academic emergency leave
        if not any(k in text_lower for k in ["exam", "examination", "academic leave", "semester", "study leave"]):
            missing_items.append({
                "topic": "Exam Leave & Academic Accommodation",
                "severity": "high",
                "description": "No provision allowing time off for university examinations or academic evaluations.",
                "recommendation": "Include an academic accommodation clause allowing study leave during university exam windows."
            })

        # Check IP personal-project / academic carve-out for internships with IP assignment
        if domain == "INTERNSHIP":
            has_ip_assignment = any(k in text_lower for k in [
                "intellectual property", "inventions", "copyright",
                "work made for hire", "assigns worldwide ownership",
                "belong exclusively to the company"
            ])
            has_personal_carveout = any(k in text_lower for k in [
                "carve-out", "carve out", "pre-existing", "preexisting",
                "prior inventions", "retain ownership", "excluded from assignment",
                "exclude personal projects", "except for personal",
                "shall remain the property of the intern",
                "shall remain the property of the student",
                "open-source contributions", "open source contributions"
            ])
            if has_ip_assignment and not has_personal_carveout:
                missing_items.append({
                    "topic": "IP Personal-Project & Academic Carve-Out",
                    "severity": "high",
                    "description": "The agreement includes intellectual property assignment language without an explicit carve-out protecting pre-existing student work, personal off-hours projects, or university coursework/theses.",
                    "recommendation": "Add a written carve-out stating that pre-existing code, off-hours personal projects, open-source contributions, and academic coursework/theses remain the student's sole property."
                })

    elif domain == "HOSTEL_RESIDENCY":
        # Check deposit return deadline
        if not any(k in text_lower for k in ["within 15 days", "within 30 days", "refunded within", "return of deposit"]):
            missing_items.append({
                "topic": "Security Deposit Refund Timeline",
                "severity": "high",
                "description": "No definite timeline is specified for returning your caution deposit after move-out.",
                "recommendation": "Require the hostel to refund the deposit within 15 business days of room handover."
            })
            
        # Check room inspection protocol
        if not any(k in text_lower for k in ["inspection", "handover", "inventory", "checklist"]):
            missing_items.append({
                "topic": "Joint Move-in/Move-out Inspection Checklist",
                "severity": "medium",
                "description": "No requirement for a signed joint inventory checklist before damage deductions.",
                "recommendation": "Demand that any damage deduction requires a signed move-in condition report and photos."
            })

    elif domain == "EVENT_SPONSORSHIP":
        if not any(k in text_lower for k in ["advance", "initial payment", "50%", "tranche"]):
            missing_items.append({
                "topic": "Sponsorship Advance Payment Schedule",
                "severity": "high",
                "description": "No advance payment schedule is stated, risking delayed payment after event expenses are incurred.",
                "recommendation": "Request at least 50% advance disbursement prior to the event date."
            })

    elif domain == "STUDENT_SERVICE":
        # Check payment timeline
        if not any(k in text_lower for k in [
            "within 7 days", "within 14 days", "within 15 days", "within 30 days",
            "net 15", "net 30", "days of invoice", "days of acceptance",
            "payment timeline", "disbursed within", "5th of", "1st of"
        ]):
            missing_items.append({
                "topic": "Payment Timeline",
                "severity": "high",
                "description": "No definite payment timeline (e.g., within 15–30 days of invoice or lesson delivery) is specified for completed student services.",
                "recommendation": "Insert a fixed payout window (for example, payment within 15 days of invoice submission) independent of downstream client collections."
            })

        # Check deliverable acceptance criteria
        if not any(k in text_lower for k in [
            "acceptance criteria", "review period", "deemed accepted",
            "approval within", "written feedback within", "revision limit",
            "number of revisions", "acceptance window"
        ]):
            missing_items.append({
                "topic": "Deliverable Acceptance Criteria",
                "severity": "medium",
                "description": "No objective acceptance criteria, review turnaround window, or cap on revision rounds is defined for deliverables.",
                "recommendation": "Define objective acceptance standards, a 5–7 business day review window after which work is deemed accepted, and a maximum of two revision rounds."
            })

    elif domain == "GENERAL_STUDENT_AGREEMENT":
        # Check termination notice terms
        if not any(k in text_lower for k in [
            "termination", "notice period", "written notice",
            "days notice", "days' notice", "terminate this agreement"
        ]):
            missing_items.append({
                "topic": "Termination Notice Terms",
                "severity": "medium",
                "description": "No clear termination clause or mutual notice period is defined, leaving exit rights and wind-down obligations ambiguous.",
                "recommendation": "Add a mutual termination clause allowing either party to exit cleanly with 14–30 days' prior written notice."
            })
            
    return missing_items


def calculate_fairness_indicator(
    clauses_analysis: List[Dict[str, Any]],
    domain: str
) -> Dict[str, Any]:
    """
    Computes a Student vs Counterparty fairness indicator based strictly
    on the classified substantive clauses, keeping all counts mutually exclusive
    so unilateral + moderate_caution + bilateral == total classified clauses.
    """
    student_favorable_points = 0.0
    counterparty_favorable_points = 0.0
    unilateral_clauses = 0
    moderate_caution_clauses = 0
    bilateral_clauses = 0

    for item in clauses_analysis:
        risk = item.get("final_risk_level", "low")
        rule_violation = bool(item.get("rule_violations"))

        if risk == "high" or rule_violation:
            counterparty_favorable_points += 3.0
            unilateral_clauses += 1
        elif risk == "medium":
            counterparty_favorable_points += 1.5
            moderate_caution_clauses += 1
        else:
            student_favorable_points += 1.5
            bilateral_clauses += 1

    total_classified = unilateral_clauses + moderate_caution_clauses + bilateral_clauses
    total_points = student_favorable_points + counterparty_favorable_points
    if total_points == 0:
        student_pct = 50
        counterparty_pct = 50
    else:
        raw_counterparty_pct = round((counterparty_favorable_points / total_points) * 100)
        counterparty_pct = min(92, max(8, int(raw_counterparty_pct)))
        student_pct = 100 - counterparty_pct

    if counterparty_pct >= 75:
        fairness_label = "Highly Asymmetric (Heavily Favors Counterparty)"
        summary_verdict = "The agreement imposes heavy unilateral obligations, exit penalties, or indemnities onto the student with minimal reciprocal protections."
    elif counterparty_pct >= 60:
        fairness_label = "Moderately Unbalanced"
        summary_verdict = "The agreement leans in favor of the counterparty with several restrictive or ambiguous covenants requiring student negotiation."
    else:
        fairness_label = "Fair & Balanced"
        summary_verdict = "The agreement exhibits reasonable reciprocity, standard notice terms, and acceptable student safeguards."

    return {
        "student_favor_pct": student_pct,
        "counterparty_favor_pct": counterparty_pct,
        "fairness_label": fairness_label,
        "summary_verdict": summary_verdict,
        "unilateral_clause_count": unilateral_clauses,
        "moderate_caution_clause_count": moderate_caution_clauses,
        "bilateral_clause_count": bilateral_clauses,
        "total_classified_clauses": total_classified,
        "student_points": round(student_favorable_points, 1),
        "counterparty_points": round(counterparty_favorable_points, 1),
        "total_points": round(total_points, 1),
        "weights": {
            "high_unilateral": 3.0,
            "medium_caution": 1.5,
            "low_standard": 1.5
        }
    }


def compute_risk_score_breakdown(
    clauses_analysis: List[Dict[str, Any]],
    missing_items: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes the overall contract risk score (bounded 10–98) and returns
    the exact itemized breakdown of every term in the actual calculation.
    """
    high_count = sum(1 for c in clauses_analysis if c.get("final_risk_level") == "high")
    med_count = sum(1 for c in clauses_analysis if c.get("final_risk_level") == "medium")
    low_count = sum(1 for c in clauses_analysis if c.get("final_risk_level") == "low")
    rule_flag_count = sum(len(c.get("rule_violations", [])) for c in clauses_analysis)
    missing_count = len(missing_items)

    baseline = 15
    high_points = high_count * 18
    rule_flag_points = rule_flag_count * 8
    medium_points = med_count * 6
    missing_points = missing_count * 4
    low_offset = -(low_count * 2)

    raw_score = baseline + high_points + rule_flag_points + medium_points + missing_points + low_offset
    final_score = max(10, min(98, raw_score))

    if final_score >= 75:
        category = "Critical Risk (Severe Red Flags)"
    elif final_score >= 55:
        category = "Elevated Risk"
    elif final_score >= 35:
        category = "Moderate Caution"
    else:
        category = "Low Risk / Standard"

    return {
        "baseline": baseline,
        "high_count": high_count,
        "high_weight": 18,
        "high_points": high_points,
        "rule_flag_count": rule_flag_count,
        "rule_flag_weight": 8,
        "rule_flag_points": rule_flag_points,
        "medium_count": med_count,
        "medium_weight": 6,
        "medium_points": medium_points,
        "missing_count": missing_count,
        "missing_weight": 4,
        "missing_points": missing_points,
        "low_count": low_count,
        "low_weight": -2,
        "low_offset": low_offset,
        "raw_score": raw_score,
        "final_score": final_score,
        "was_capped": raw_score != final_score,
        "bounds": [10, 98],
        "category": category
    }


def compute_overall_risk_score(
    clauses_analysis: List[Dict[str, Any]],
    missing_items: List[Dict[str, Any]]
) -> Tuple[int, str]:
    """
    Computes overall contract risk score (0-100) and risk category.
    Higher score = Higher danger for the student.
    """
    breakdown = compute_risk_score_breakdown(clauses_analysis, missing_items)
    return breakdown["final_score"], breakdown["category"]
