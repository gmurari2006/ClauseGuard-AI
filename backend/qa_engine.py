"""
ClauseGuard AI - Grounded Agreement Q&A Engine
100% local, API-free retrieval and grounded synthesis engine.
Identifies student question intent (or direct clause lookup / overall risk summary),
retrieves the most relevant substantive clauses using hybrid TF-IDF semantic similarity
+ intent/category + lexical/numeric grounding, verifies evidence sufficiency, and
synthesizes accurate, document-grounded answers with:
  1. Direct answer
  2. Relevant clause numbers
  3. Exact supporting contract quotations
  4. Simple student-friendly explanation (with cross-clause context)
  5. Practical recommendation

Explicitly returns `insufficient_evidence` when the uploaded agreement does not contain
enough information — never hallucinates or fabricates legal facts.
"""
import re
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PRESET_STUDENT_QUESTIONS = [
    {
        "id": "q_termination",
        "question": "Can the company terminate me immediately while requiring me to give 60 days notice?",
        "topic": "Termination & Notice Period"
    },
    {
        "id": "q_penalty",
        "question": "What happens if I leave early, and do I have to pay the exit penalty?",
        "topic": "Exit Penalties & Lock-In"
    },
    {
        "id": "q_deposit",
        "question": "What are the security deposit refund and forfeiture conditions?",
        "topic": "Deposit Refund & Forfeiture"
    },
    {
        "id": "q_ip",
        "question": "Who owns the code, personal projects, or academic capstone I build during this period?",
        "topic": "Intellectual Property Ownership"
    },
    {
        "id": "q_noncompete",
        "question": "Can they restrict me from working for a competing company after I leave?",
        "topic": "Non-Compete Restrictions"
    },
    {
        "id": "q_stipend",
        "question": "When and how will my stipend or compensation be paid?",
        "topic": "Stipend & Payment Schedule"
    }
]


DOMAIN_PRESET_QUESTIONS: Dict[str, List[Dict[str, str]]] = {
    "INTERNSHIP": [
        {
            "id": "intern_termination",
            "question": "Can the company terminate me immediately while requiring me to give 60 days notice?",
            "topic": "Termination & Notice Period"
        },
        {
            "id": "intern_penalty",
            "question": "What happens if I leave early, and do I have to pay an exit penalty?",
            "topic": "Exit Penalties & Lock-In"
        },
        {
            "id": "intern_deposit",
            "question": "What are the security deposit refund and forfeiture conditions?",
            "topic": "Deposit Refund & Forfeiture"
        },
        {
            "id": "intern_ip",
            "question": "Who owns the code I develop on my personal laptop or for my university capstone?",
            "topic": "Personal Laptop & Capstone IP"
        },
        {
            "id": "intern_exams",
            "question": "Can I take leave for my university semester exams during the internship?",
            "topic": "Academic Exam Leave"
        },
        {
            "id": "intern_redflags",
            "question": "What are the biggest red flags in this agreement?",
            "topic": "Overall Risk Summary"
        }
    ],
    "PLACEMENT_EMPLOYMENT": [
        {
            "id": "place_bond",
            "question": "Do I have to serve a mandatory service bond or pay liquidated damages if I resign early?",
            "topic": "Service Bond & Lock-In"
        },
        {
            "id": "place_certs",
            "question": "Can the employer hold my original degree certificates or promissory note as collateral?",
            "topic": "Certificates & Collateral"
        },
        {
            "id": "place_noncompete",
            "question": "Can they bar me from joining a competing company or graduate studies after I leave?",
            "topic": "Non-Compete Restrictions"
        },
        {
            "id": "place_termination",
            "question": "What are the notice period and immediate dismissal rules?",
            "topic": "Termination & Notice"
        },
        {
            "id": "place_ip",
            "question": "Who owns intellectual property I create on personal time or weekends?",
            "topic": "Personal Time IP Ownership"
        },
        {
            "id": "place_redflags",
            "question": "What are the biggest red flags I should negotiate before signing?",
            "topic": "Overall Risk Summary"
        }
    ],
    "HOSTEL_RESIDENCY": [
        {
            "id": "hostel_curfew",
            "question": "What are the curfew rules and can the warden evict me without appeal?",
            "topic": "Curfew & Disciplinary Eviction"
        },
        {
            "id": "hostel_deposit",
            "question": "Can the hostel forfeit my security deposit or make arbitrary deductions?",
            "topic": "Security Deposit & Deductions"
        },
        {
            "id": "hostel_damage",
            "question": "What is my personal financial liability for room or furniture damage?",
            "topic": "Room Damage & Liability"
        },
        {
            "id": "hostel_exit",
            "question": "How much notice do I need to vacate the hostel and is there a cancellation penalty?",
            "topic": "Early Exit & Cancellation"
        },
        {
            "id": "hostel_redflags",
            "question": "What are the biggest red flags in this hostel agreement?",
            "topic": "Overall Risk Summary"
        }
    ],
    "EVENT_SPONSORSHIP": [
        {
            "id": "spon_guarantee",
            "question": "Do student organizers personally guarantee losses or low attendee turnout?",
            "topic": "Personal Guarantee & Indemnity"
        },
        {
            "id": "spon_payment",
            "question": "When will the sponsorship funds be paid and is there any advance?",
            "topic": "Payment Delay & Milestones"
        },
        {
            "id": "spon_cancel",
            "question": "What happens if the event is cancelled due to weather or force majeure?",
            "topic": "Cancellation & Force Majeure"
        },
        {
            "id": "spon_media",
            "question": "Who owns the commercial media rights to event recordings and student presentations?",
            "topic": "Media & Presentation Rights"
        },
        {
            "id": "spon_redflags",
            "question": "What are the biggest red flags for student club organizers?",
            "topic": "Overall Risk Summary"
        }
    ],
    "STUDENT_SERVICE": [
        {
            "id": "svc_payment",
            "question": "When will I get paid and can my earned fees be withheld or forfeited?",
            "topic": "Fee Payment & Forfeiture"
        },
        {
            "id": "svc_ip",
            "question": "Who owns code or educational modules I create on my personal laptop?",
            "topic": "IP & Moral Rights Waiver"
        },
        {
            "id": "svc_noncompete",
            "question": "Can they restrict me from tutoring or working in edtech after this contract?",
            "topic": "Non-Compete Restrictions"
        },
        {
            "id": "svc_gag",
            "question": "Am I barred from posting reviews or discussing my work online?",
            "topic": "Gag & Non-Disparagement"
        },
        {
            "id": "svc_termination",
            "question": "Can the company terminate my service contract immediately without compensation?",
            "topic": "Termination at Will"
        },
        {
            "id": "svc_redflags",
            "question": "What are the biggest red flags in this service agreement?",
            "topic": "Overall Risk Summary"
        }
    ]
}


def get_preset_questions_for_domain(domain: Optional[str] = None) -> List[Dict[str, str]]:
    """
    Return tailored student preset questions for a specific contract domain.
    Supports INTERNSHIP, PLACEMENT_EMPLOYMENT, HOSTEL_RESIDENCY, EVENT_SPONSORSHIP,
    and STUDENT_SERVICE; falls back to PRESET_STUDENT_QUESTIONS for unknown/None domains.
    """
    if not domain or not isinstance(domain, str):
        return [dict(q) for q in PRESET_STUDENT_QUESTIONS]
    key = domain.strip().upper()
    presets = DOMAIN_PRESET_QUESTIONS.get(key)
    if not presets:
        return [dict(q) for q in PRESET_STUDENT_QUESTIONS]
    return [dict(q) for q in presets]


# Intent configuration: lexical signals, target clause categories, and regex anchors
INTENT_CONFIG: Dict[str, Dict[str, Any]] = {
    "OVERALL_RISK_SUMMARY": {
        "categories": [],
        "query_terms": [
            ("red flags", 5.0), ("red flag", 4.5), ("biggest risks", 5.0),
            ("major risks", 4.5), ("main risks", 4.5), ("key risks", 4.5),
            ("should i sign", 5.0), ("safe to sign", 5.0), ("worth signing", 4.5),
            ("summarize the risks", 5.0), ("summarize the major risks", 5.0),
            ("risk summary", 5.0), ("overall risk", 5.0), ("overall fairness", 4.5),
            ("is this fair", 4.5), ("is this contract fair", 5.0),
            ("is this agreement fair", 5.0), ("biggest issues", 4.5),
            ("worst clauses", 4.5), ("most dangerous", 4.5),
            ("what should i negotiate", 4.5), ("top issues", 4.5),
            ("major concerns", 4.0), ("summarize this agreement", 4.0)
        ],
        "clause_anchor": re.compile(r'.+', re.IGNORECASE)
    },
    "TERMINATION_NOTICE": {
        "categories": ["TERMINATION_NOTICE"],
        "query_terms": [
            ("terminate", 3.0), ("termination", 3.0), ("notice period", 3.5),
            ("days notice", 3.5), ("prior written notice", 3.5), ("notice", 2.5),
            ("immediately", 2.0), ("without cause", 3.0), ("without notice", 3.0),
            ("fire", 2.5), ("dismiss", 2.5), ("dismissal", 2.5),
            ("severance", 2.0), ("vacate", 2.5),
            ("60 days", 2.5), ("90 days", 2.5), ("30 days", 2.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:terminate\s+this|termination\s+by|termination\s+covenants?|termination\s+at\s+will|prior\s+written\s+notice|advance\s+written\s+notice|notice\s+period|without\s+cause|summary\s+dismissal|eviction|severance)\b',
            re.IGNORECASE
        )
    },
    "EXIT_PENALTY": {
        "categories": ["FINANCIAL_PENALTY_BOND"],
        "query_terms": [
            ("exit penalty", 4.0), ("early exit", 3.5), ("liquidated damages", 4.0),
            ("penalty", 3.0), ("penalties", 3.0), ("fine", 2.5), ("bond", 3.5),
            ("service bond", 4.0), ("employment bond", 4.0), ("3-year bond", 4.5),
            ("3 year bond", 4.5), ("three years", 2.5), ("36 months", 3.0),
            ("lock-in", 3.5), ("lock in", 3.5), ("mandatory tenure", 3.5),
            ("leave early", 2.5), ("quit", 2.0), ("resign", 2.0),
            ("training fee", 3.0), ("promissory note", 3.5), ("withhold", 2.0),
            ("75,000", 3.5), ("75000", 3.5), ("1,50,000", 3.5), ("2,50,000", 3.5),
            ("250000", 3.5), ("1,20,000", 3.5), ("2,00,000", 3.5)
        ],
        "clause_anchor": re.compile(
            r'\b(?:exit\s+penalty|liquidated\s+damages|lock-?in|service\s+bond|employment\s+bond|training\s+bond|promissory\s+note|recruitment\s+penalty|mandatory\s+financial\s+penalty|withhold\b.*?certificate)\b',
            re.IGNORECASE | re.DOTALL
        )
    },
    "SECURITY_DEPOSIT": {
        "categories": ["DEPOSIT_REFUND"],
        "query_terms": [
            ("security deposit", 4.5), ("caution deposit", 4.5), ("deposit", 3.5),
            ("refund", 3.5), ("refunded", 3.5), ("non-refundable", 4.0),
            ("forfeit", 3.5), ("forfeited", 3.5), ("forfeiture", 3.5),
            ("deduct", 2.5), ("deductions", 2.5), ("45 business days", 3.0),
            ("renovation fee", 3.5), ("15,000", 3.0), ("15000", 3.0), ("35,000", 3.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:security\s+deposit|caution\s+deposit|deposit|refund\s+conditions?|refunded|forfeiture|forfeited|non-refundable)\b',
            re.IGNORECASE
        )
    },
    "INTELLECTUAL_PROPERTY": {
        "categories": ["INTELLECTUAL_PROPERTY"],
        "query_terms": [
            ("intellectual property", 4.5), ("ip", 3.0), ("who owns", 3.5),
            ("code ownership", 4.5), ("ownership", 2.5), ("owns the code", 4.5),
            ("code", 2.5), ("copyright", 3.5), ("patent", 3.5),
            ("moral rights", 4.0), ("thesis", 3.5), ("capstone", 3.5),
            ("side project", 3.5), ("side-project", 3.5), ("personal project", 3.5),
            ("personal device", 3.5), ("personal laptop", 4.0), ("my laptop", 4.0),
            ("own laptop", 4.0), ("own device", 4.0), ("personal computer", 4.0),
            ("weekend", 2.5), ("inventions", 3.0), ("media rights", 4.0),
            ("recordings", 3.0), ("presentations", 2.5)
        ],
        "clause_anchor": re.compile(
            r'\b(?:intellectual\s+property|inventions?|software\s+code|copyright|patent|moral\s+rights|authorship|capstone|thesis|theses|work\s+made\s+for\s+hire|media\s+rights|personal\s+devices?)\b',
            re.IGNORECASE
        )
    },
    "ACADEMIC_EXAM_LEAVE": {
        "categories": ["TERMINATION_NOTICE", "STANDARD_FAIR"],
        "anchor_required": True,
        "query_terms": [
            ("semester exam", 5.0), ("semester exams", 5.0), ("exam leave", 5.0),
            ("academic exam", 5.0), ("university exam", 5.0), ("college exam", 4.5),
            ("study leave", 4.5), ("academic leave", 4.5), ("academic conflict", 4.5),
            ("university schedule", 4.5), ("class schedule", 4.0), ("exams", 3.5),
            ("university classes", 4.0), ("exam schedule", 4.5)
        ],
        "clause_anchor": re.compile(
            r'\b(?:academic\s+exam\s+leave|exam\s+leave|semester\s+exams?|university\s+(?:exams?|schedule|classes)|study\s+leave|academic\s+leave|no\s+leave\s+may\s+be\s+availed|personal\s+leave)\b',
            re.IGNORECASE
        )
    },
    "HOSTEL_RULES_CURFEW": {
        "categories": ["DEPOSIT_REFUND", "TERMINATION_NOTICE", "STANDARD_FAIR", "LIABILITY_INDEMNITY"],
        "anchor_required": True,
        "query_terms": [
            ("curfew", 5.0), ("curfew hours", 5.0), ("warden", 4.5),
            ("visitors", 4.0), ("visitor", 3.5), ("mess fee", 4.5),
            ("mess charges", 4.5), ("hostel rules", 4.5), ("code of conduct", 4.0),
            ("eviction", 4.0), ("evict", 4.0), ("evicted", 4.0),
            ("expelled", 3.5), ("expulsion", 3.5), ("noise complaints", 4.0),
            ("room rules", 4.0), ("9:00 pm", 3.5), ("hostel", 2.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:curfew|warden|visitors?|mess\s+(?:fee|charges?)|hostel\s+rules?|code\s+of\s+conduct|disciplinary\s+(?:eviction|proceedings)|eviction|expelled|noise\s+complaints?|insubordination\s+to\s+the\s+warden)\b',
            re.IGNORECASE
        )
    },
    "CONFIDENTIALITY_NDA": {
        "categories": ["CONFIDENTIALITY_NDA"],
        "query_terms": [
            ("confidential", 3.5), ("confidentiality", 4.0), ("nda", 4.0),
            ("non-disclosure", 4.0), ("trade secret", 3.5), ("resume", 3.0),
            ("cv", 3.0), ("linkedin", 3.5), ("github", 3.5), ("portfolio", 3.0),
            ("glassdoor", 3.5), ("review", 2.5), ("reviews", 2.5),
            ("negative comments", 3.5), ("gag", 3.5), ("disparagement", 3.5),
            ("mention the company", 3.5)
        ],
        "clause_anchor": re.compile(
            r'\b(?:confidential|non-disclosure|trade\s+secrets?|resume|cv|linkedin|github|glassdoor|gag\s+clause|negative\s+comments|disparagement)\b',
            re.IGNORECASE
        )
    },
    "STIPEND_PAYMENT": {
        "categories": ["STIPEND_PAYMENT_TERMS"],
        "query_terms": [
            ("stipend", 4.5), ("salary", 3.5), ("ctc", 3.5), ("compensation", 3.5),
            ("payment", 3.0), ("get paid", 3.5), ("be paid", 3.0),
            ("disbursed", 3.5), ("disbursement", 3.5), ("withhold stipend", 4.0),
            ("delayed", 2.5), ("milestone", 2.5), ("bonus", 2.5),
            ("sponsorship funds", 3.5), ("advance payment", 3.5),
            ("earned fees", 3.5), ("accrued fees", 3.5), ("18,000", 3.0), ("20,000", 3.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:stipend|remuneration|annual\s+ctc|monthly\s+stipend|disburs(?:e|ed|ement)|payment\s+of\s+rs|compensation|milestone\s+sign-off|bonus|90\s+days\s+in\s+arrears|accrued\s+fees)\b',
            re.IGNORECASE
        )
    },
    "CERTIFICATE_RETENTION": {
        "categories": ["FINANCIAL_PENALTY_BOND"],
        "anchor_required": True,
        "query_terms": [
            ("certificate", 3.5), ("certificates", 3.5), ("marksheet", 4.0),
            ("marksheets", 4.0), ("marks card", 4.0), ("marks cards", 4.0),
            ("original degree", 4.5), ("degree", 3.0), ("passport", 4.0),
            ("completion certificate", 4.5), ("experience letter", 4.0),
            ("relieving letter", 4.0), ("academic evaluation", 4.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:certificate|certificates|marksheet|marksheets|marks\s*cards?|degree|passport|experience\s+letter|relieving\s+letter|academic\s+evaluation)\b',
            re.IGNORECASE
        )
    },
    "NON_COMPETE": {
        "categories": ["RESTRICTIVE_NON_COMPETE"],
        "anchor_required": True,
        "query_terms": [
            ("non-compete", 4.5), ("non compete", 4.5), ("compete", 3.5),
            ("competitor", 4.0), ("competing", 4.0), ("rival", 3.5),
            ("restraint of trade", 4.5), ("section 27", 4.0), ("solicit", 3.5),
            ("barred from joining", 4.0), ("graduate studies", 3.5)
        ],
        "clause_anchor": re.compile(
            r'\b(?:non-?compete|competing\s+business|competitor|restraint\s+of\s+trade|solicit|barred\s+from\s+joining|section\s+27)\b',
            re.IGNORECASE
        )
    },
    "DURATION_SCHEDULE": {
        "categories": ["STANDARD_FAIR"],
        "anchor_required": True,
        "query_terms": [
            ("duration", 3.5), ("how long", 3.0), ("working hours", 4.0),
            ("hours per week", 4.0), ("work schedule", 4.0), ("working schedule", 4.0),
            ("office timings", 3.5), ("hybrid", 3.5), ("remote", 3.0),
            ("start date", 3.0), ("end date", 3.0), ("mentor", 3.0), ("probation", 3.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:fixed\s+duration|hours\s+per\s+week|hybrid|commencing|concluding|probation|mentor|initial\s+term)\b',
            re.IGNORECASE
        )
    },
    "GOVERNING_LAW": {
        "categories": ["DISPUTE_JURISDICTION"],
        "anchor_required": True,
        "query_terms": [
            ("governing law", 4.5), ("jurisdiction", 4.5), ("court", 3.5),
            ("courts", 3.5), ("arbitration", 4.5), ("dispute", 3.5),
            ("conciliation", 4.0), ("legal venue", 4.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:governing\s+law|jurisdiction|arbitration|conciliation|courts?\s+located|disputes?)\b',
            re.IGNORECASE
        )
    },
    "LIABILITY_INDEMNITY": {
        "categories": ["LIABILITY_INDEMNITY"],
        "query_terms": [
            ("indemnify", 4.5), ("indemnity", 4.5), ("liability", 4.0),
            ("liable", 3.5), ("lawsuit", 3.5), ("bugs", 3.0), ("outage", 3.0),
            ("personal guarantee", 4.5), ("personally guarantee", 4.5),
            ("personally liable", 4.5), ("treble damages", 4.5),
            ("attorney fees", 3.5), ("footfall", 4.0), ("turnout", 4.0),
            ("low attendee turnout", 4.5), ("cancellation losses", 4.5),
            ("force majeure", 4.0), ("room damage", 4.0), ("furniture damage", 4.0),
            ("organizer", 3.0), ("organizers", 3.0)
        ],
        "clause_anchor": re.compile(
            r'\b(?:indemnif|liability|liable|lawsuits?|attorney\s+fees|treble\s+damages|personally\s+guarantee|personal\s+guarantee|force\s+majeure|target\s+footfall|cancellation\s+losses|low\s+attendee\s+turnout)\b',
            re.IGNORECASE
        )
    }
}


RE_NUM_TOKENS = re.compile(r'\b\d[\d,]*\b')
RE_MONEY_EXACT = re.compile(r'(?:Rs\.?|INR|₹|\$|EUR)\s*[0-9][0-9,]*(?:\.\d{1,2})?', re.IGNORECASE)
RE_DURATION_DAYS = re.compile(r'\b(\d+)\s*(?:\([a-z\s-]+\)\s*)?(business\s+days?|calendar\s+days?|days?|months?|years?)\b', re.IGNORECASE)
RE_DIRECT_CLAUSE_LOOKUP = re.compile(
    r'\b(?:clause|section|article|para(?:graph)?\.?|provision)\s+(\d+(?:\.\d+)*)\b',
    re.IGNORECASE
)
RE_STATUTE_CONTEXT = re.compile(
    r'\b(?:indian\s+contract\s+act|contract\s+act|companies\s+act|labour\s+act|labor\s+act|constitution|statute|ipc|act\s+of\s+\d{4})\b',
    re.IGNORECASE
)
RE_WORD_TOKENS = re.compile(r'[a-z0-9]{2,}')

# Specific benefit/HR/equity/perk concepts that MUST actually appear in the agreement
# if explicitly asked about; prevents generic terms ("pay", "schedule", "provide") from
# falsely matching unrelated clauses.
SPECIFIC_CONCEPT_ANCHORS: List[Tuple[str, re.Pattern]] = [
    ("health insurance", re.compile(r'\b(?:health\s+insurance|medical\s+insurance|mediclaim)\b', re.IGNORECASE)),
    ("dental insurance", re.compile(r'\b(?:dental\s+(?:insurance|coverage|plan|benefits?)|dental)\b', re.IGNORECASE)),
    ("vision insurance", re.compile(r'\b(?:vision\s+(?:insurance|coverage|plan|benefits?))\b', re.IGNORECASE)),
    ("life insurance", re.compile(r'\b(?:life\s+insurance)\b', re.IGNORECASE)),
    ("401k", re.compile(r'\b(?:401\s*\(?k\)?|401k|retirement\s+matching)\b', re.IGNORECASE)),
    ("stock options / vesting", re.compile(r'\b(?:stock\s+options?|esops?|espp|vesting|rsus?|equity\s+grants?)\b', re.IGNORECASE)),
    ("maternity / paternity leave", re.compile(r'\b(?:maternity|paternity|parental\s+leave)\b', re.IGNORECASE)),
    ("pension / provident fund / gratuity", re.compile(r'\b(?:pension|provident\s+fund|epf|gratuity)\b', re.IGNORECASE)),
    ("relocation / travel allowance", re.compile(r'\b(?:relocation|flight\s+tickets?|airfare|travel\s+allowance)\b', re.IGNORECASE)),
    ("paid vacation / PTO / sick leave", re.compile(r'\b(?:vacation\s+days?|paid\s+time\s+off|\bpto\b|sick\s+leave|casual\s+leave)\b', re.IGNORECASE)),
    ("dress code", re.compile(r'\b(?:dress\s+code|uniform)\b', re.IGNORECASE)),
    ("cafeteria / gym / childcare / parking", re.compile(r'\b(?:cafeteria|gym\s+membership|childcare|parking\s+permit)\b', re.IGNORECASE)),
]

# English + generic contract query stopwords used for lexical overlap verification
STOPWORDS_AND_GENERIC: Set[str] = {
    "the", "and", "for", "with", "from", "into", "onto", "over", "under", "about",
    "between", "during", "before", "after", "while", "where", "when", "what", "which",
    "who", "whom", "whose", "why", "how", "whether", "this", "that", "these", "those",
    "there", "their", "they", "them", "then", "than", "your", "yours", "ours", "mine",
    "have", "has", "had", "does", "did", "done", "doing", "will", "would", "could",
    "should", "shall", "must", "may", "might", "can", "cannot", "are", "was", "were",
    "been", "being", "not", "nor", "but", "yet", "any", "all", "some", "such", "only",
    "also", "just", "very", "much", "more", "most", "other", "each", "both", "few",
    "agreement", "contract", "clause", "clauses", "section", "document", "policy",
    "company", "intern", "student", "employee", "employer", "resident", "sponsor",
    "party", "parties", "term", "terms", "provide", "provides", "provided", "providing",
    "offer", "offers", "offered", "include", "includes", "included", "including",
    "mention", "mentions", "mentioned", "say", "says", "state", "states", "stated",
    "explain", "tell", "mean", "means", "give", "get", "make", "take", "keep",
    "charge", "allow", "allowed", "schedule", "rate", "plan", "rules", "conditions",
    "provisions", "obligations", "rights", "details", "information", "regarding",
    "pay", "paid", "paying"
}


def _clause_label(clause: Dict[str, Any]) -> str:
    num = clause.get("clause_number")
    if num:
        return f"Clause {num}"
    m = re.match(r'^(\d+(?:\.\d+)*)\b', clause.get("original_text", "").strip())
    if m:
        return f"Clause {m.group(1)}"
    return f"Clause #{clause.get('index', '?')}"


def _is_boilerplate_or_heading(clause: Dict[str, Any]) -> bool:
    if clause.get("is_heading") or clause.get("is_structural"):
        return True
    txt = clause.get("original_text", "").strip()
    if len(txt) < 20:
        return True
    lower = txt.lower()
    if lower.startswith(("disclaimer:", "in witness whereof", "____", "document reference:", "fictional test data")):
        return True
    if "authorised signatory" in lower or "authorized signatory" in lower:
        return True
    return False


def _format_cited_clause(clause: Dict[str, Any], sim: float, score: float) -> Dict[str, Any]:
    """Consistent cited-clause payload preserving all legacy keys plus related_clauses."""
    return {
        "index": clause["index"],
        "clause_number": clause.get("clause_number"),
        "clause_ref": _clause_label(clause),
        "category": clause.get("category", "STANDARD_FAIR"),
        "risk_level": clause.get("final_risk_level", "low"),
        "text": " ".join(clause.get("original_text", "").split()),
        "explanation": clause.get("explanation", ""),
        "action": clause.get("recommended_action", ""),
        "related_clauses": clause.get("related_clauses", []),
        "similarity": round(float(sim), 3),
        "relevance_score": round(float(score), 3)
    }


def _distinctive_tokens(text: str) -> Set[str]:
    """Extract distinctive topical tokens (normalized singular/base form) excluding generic words."""
    tokens: Set[str] = set()
    for tok in RE_WORD_TOKENS.findall(text.lower()):
        if len(tok) < 3 or tok in STOPWORDS_AND_GENERIC:
            continue
        if tok.isdigit():
            continue
        tokens.add(tok)
    return tokens


def _token_matches_text(tok: str, text_tokens: Set[str], text_lower: str) -> bool:
    if tok in text_tokens:
        return True
    if tok.endswith("s") and len(tok) > 4 and tok[:-1] in text_tokens:
        return True
    if f"{tok}s" in text_tokens:
        return True
    if tok.endswith("ed") and len(tok) > 4 and tok[:-2] in text_tokens:
        return True
    if tok.endswith("ing") and len(tok) > 5 and tok[:-3] in text_tokens:
        return True
    if len(tok) >= 5 and tok in text_lower:
        return True
    return False


def _check_missing_specific_concepts(query: str, candidates: List[Dict[str, Any]]) -> List[str]:
    """
    Detect if the query explicitly asks about specific HR/benefit/equity concepts
    (e.g., dental insurance, 401k, stock option vesting) that do NOT appear anywhere
    in the agreement's substantive clauses.
    """
    queried_concepts: List[Tuple[str, re.Pattern]] = []
    for label, pattern in SPECIFIC_CONCEPT_ANCHORS:
        if pattern.search(query):
            queried_concepts.append((label, pattern))

    if not queried_concepts:
        return []

    corpus = " ".join(c.get("original_text", "") for c in candidates)
    missing = [label for label, pat in queried_concepts if not pat.search(corpus)]
    # Only flag missing if ALL queried specific concepts are absent from the agreement
    if len(missing) == len(queried_concepts):
        return missing
    return []


def _extract_direct_clause_targets(query: str) -> List[str]:
    """
    Detect explicit clause/section number lookups such as:
    'Explain Clause 4.2', 'What does Clause 5.1 mean?', 'Tell me about Section 3.3'.
    Ignores statutory act citations like 'Section 27 of the Indian Contract Act'.
    """
    if RE_STATUTE_CONTEXT.search(query):
        return []
    targets: List[str] = []
    for m in RE_DIRECT_CLAUSE_LOOKUP.finditer(query):
        num = m.group(1)
        # Skip known Indian Contract Act sections when explicitly paired with restraint/non-compete
        if num == "27" and any(w in query.lower() for w in ["non-compete", "non compete", "restraint of trade", "competitor"]):
            continue
        if num == "74" and any(w in query.lower() for w in ["liquidated damages", "enforceable", "unenforceable"]):
            continue
        if num not in targets:
            targets.append(num)
    return targets


def _lookup_direct_clauses(
    target_numbers: List[str],
    analyzed_clauses: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str], bool]:
    """
    Directly resolve requested clause numbers by `clause_number` (or embedded subclause / section),
    plus any clauses explicitly linked in their `related_clauses`.
    Returns: (selected_cited_clauses, matched_requested_refs, exact_match_found)
    """
    candidates = [c for c in analyzed_clauses if not _is_boilerplate_or_heading(c)]
    by_index = {c["index"]: c for c in candidates}

    requested_clauses: List[Dict[str, Any]] = []
    seen_indices: Set[int] = set()
    exact_match_found = False

    for target in target_numbers:
        found_for_target: List[Dict[str, Any]] = []
        # 1. Exact clause_number match
        for c in candidates:
            if (c.get("clause_number") or "").strip() == target:
                found_for_target.append(c)
                exact_match_found = True

        # 2. Subclause embedded inside a grouped section paragraph (e.g. "2.2" inside Clause 2)
        if not found_for_target:
            sub_re = re.compile(rf'(?:^|\s){re.escape(target)}\s+')
            for c in candidates:
                if sub_re.search(c.get("original_text", "")):
                    found_for_target.append(c)
                    exact_match_found = True

        # 3. Top-level section match (e.g. "Explain Section 4" -> 4.1, 4.2, 4.3)
        if not found_for_target and "." not in target:
            for c in candidates:
                if (c.get("section_number") or "").strip() == target:
                    found_for_target.append(c)

        for c in found_for_target:
            if c["index"] not in seen_indices:
                seen_indices.add(c["index"])
                requested_clauses.append(c)

    if not requested_clauses:
        return [], [], False

    requested_clauses.sort(key=lambda c: c["index"])
    matched_requested_refs = [_clause_label(c) for c in requested_clauses]

    selected: List[Dict[str, Any]] = [
        _format_cited_clause(c, sim=1.0, score=1.5 if exact_match_found else 1.2)
        for c in requested_clauses
    ]

    # Attach explicitly linked clauses from `related_clauses` (without duplicating)
    linked_clauses: List[Dict[str, Any]] = []
    for c in requested_clauses:
        for rel in c.get("related_clauses", []):
            t_idx = rel.get("target_index")
            if t_idx in by_index and t_idx not in seen_indices:
                seen_indices.add(t_idx)
                linked_clauses.append(by_index[t_idx])

    linked_clauses.sort(key=lambda c: c["index"])
    for lc in linked_clauses[:3]:
        selected.append(_format_cited_clause(lc, sim=0.65, score=0.95))

    return selected, matched_requested_refs, exact_match_found


def detect_question_intent(query: str) -> Tuple[str, Optional[str], Dict[str, float]]:
    """
    Identify primary (and optional secondary) question intent before clause retrieval.
    Uses weighted phrase matching with disambiguation rules.
    """
    q_lower = query.lower()
    scores: Dict[str, float] = {}

    for intent, cfg in INTENT_CONFIG.items():
        score = 0.0
        for term, weight in cfg["query_terms"]:
            if len(term) <= 3 and term.isalpha():
                if re.search(rf'\b{re.escape(term)}\b', q_lower):
                    score += weight
            else:
                if term in q_lower:
                    score += weight
        if score > 0:
            scores[intent] = score

    if not scores:
        return "GENERAL", None, scores

    # Disambiguation 1: immediate termination & notice period without explicit penalty/bond words
    if "TERMINATION_NOTICE" in scores and "EXIT_PENALTY" in scores:
        has_explicit_penalty_word = any(
            w in q_lower for w in [
                "penalty", "penalties", "fine", "bond", "liquidated",
                "lock-in", "lock in", "75,000", "75000", "2,50,000", "pay to leave"
            ]
        )
        if not has_explicit_penalty_word:
            scores["TERMINATION_NOTICE"] += 3.0

    # Disambiguation 2: explicit security deposit / refund question boosts SECURITY_DEPOSIT
    if "SECURITY_DEPOSIT" in scores and "deposit" in q_lower:
        scores["SECURITY_DEPOSIT"] += 2.0

    # Disambiguation 3: curfew/warden/eviction in hostel context boosts HOSTEL_RULES_CURFEW
    if "HOSTEL_RULES_CURFEW" in scores and any(w in q_lower for w in ["curfew", "warden", "mess fee", "hostel rules", "evict", "eviction"]):
        scores["HOSTEL_RULES_CURFEW"] += 2.0

    # Disambiguation 4: explicit academic/semester exam leave boosts ACADEMIC_EXAM_LEAVE
    if "ACADEMIC_EXAM_LEAVE" in scores and any(w in q_lower for w in ["exam", "semester", "study leave", "academic"]):
        scores["ACADEMIC_EXAM_LEAVE"] += 2.5

    # Disambiguation 5: explicit overall summary / red flags / should I sign
    if "OVERALL_RISK_SUMMARY" in scores and any(
        w in q_lower for w in ["red flag", "should i sign", "safe to sign", "overall risk", "biggest risk", "major risk", "is this contract fair", "is this agreement fair"]
    ):
        scores["OVERALL_RISK_SUMMARY"] += 3.0

    # Disambiguation 6: personal laptop / code ownership boosts INTELLECTUAL_PROPERTY
    if "INTELLECTUAL_PROPERTY" in scores and any(
        w in q_lower for w in ["personal laptop", "my laptop", "own laptop", "own device", "personal computer", "owns the code", "who owns"]
    ):
        scores["INTELLECTUAL_PROPERTY"] += 2.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    primary = ranked[0][0]
    secondary = (
        ranked[1][0]
        if len(ranked) > 1 and ranked[1][1] >= 3.0 and ranked[1][1] >= ranked[0][1] * 0.65
        else None
    )
    return primary, secondary, scores


def _retrieve_overall_risk_clauses(analyzed_clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Select the document's top risk-bearing substantive clauses for OVERALL_RISK_SUMMARY,
    prioritizing statutory rule violations, cross-clause links, and category diversity.
    """
    candidates = [c for c in analyzed_clauses if not _is_boilerplate_or_heading(c)]
    if not candidates:
        return []

    risk_rank = {"high": 3, "medium": 2, "low": 1}
    cat_priority = {
        "FINANCIAL_PENALTY_BOND": 6,
        "DEPOSIT_REFUND": 5,
        "TERMINATION_NOTICE": 5,
        "LIABILITY_INDEMNITY": 5,
        "RESTRICTIVE_NON_COMPETE": 5,
        "INTELLECTUAL_PROPERTY": 4,
        "STIPEND_PAYMENT_TERMS": 4,
        "CONFIDENTIALITY_NDA": 3,
        "DISPUTE_JURISDICTION": 2,
        "STANDARD_FAIR": 1
    }

    def sort_key(c: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
        r_score = risk_rank.get(c.get("final_risk_level", "low"), 1)
        rule_hits = len(c.get("rule_violations", []))
        rel_hits = len(c.get("related_clauses", []))
        cat_score = cat_priority.get(c.get("category", "STANDARD_FAIR"), 1)
        return (r_score, 1 if rule_hits > 0 else 0, cat_score, rel_hits, -c.get("index", 0))

    high_clauses = [c for c in candidates if c.get("final_risk_level") == "high"]
    pool = high_clauses if high_clauses else [c for c in candidates if c.get("final_risk_level") == "medium"]
    if not pool:
        pool = candidates

    ranked = sorted(pool, key=sort_key, reverse=True)

    # Prioritize distinct risk categories first so the summary spans all major red flags
    selected_raw: List[Dict[str, Any]] = []
    seen_cats: Set[str] = set()
    for c in ranked:
        cat = c.get("category", "STANDARD_FAIR")
        if cat not in seen_cats:
            selected_raw.append(c)
            seen_cats.add(cat)
        if len(selected_raw) >= 5:
            break

    # Fill up to 5 with remaining highest-priority risk clauses if fewer than 5 categories
    if len(selected_raw) < 5:
        selected_indices = {c["index"] for c in selected_raw}
        for c in ranked:
            if c["index"] not in selected_indices:
                selected_raw.append(c)
                selected_indices.add(c["index"])
            if len(selected_raw) >= 5:
                break

    selected_raw.sort(key=lambda c: c["index"])
    return [
        _format_cited_clause(
            c,
            sim=0.85 if c.get("final_risk_level") == "high" else 0.60,
            score=1.35 if c.get("final_risk_level") == "high" else 0.90
        )
        for c in selected_raw
    ]


def retrieve_relevant_clauses(
    query: str,
    analyzed_clauses: List[Dict[str, Any]],
    primary_intent: str,
    secondary_intent: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval combining:
      1. TF-IDF cosine similarity (1-2 ngrams)
      2. Primary/secondary intent category alignment
      3. Intent lexical anchor regex match
      4. Exact numeric/figure overlap (e.g. 60, 45, 15,000, 75,000)
      5. Suppression of unrelated categories and ungrounded generic matches
    """
    if primary_intent == "OVERALL_RISK_SUMMARY":
        return _retrieve_overall_risk_clauses(analyzed_clauses)

    candidates = [c for c in analyzed_clauses if not _is_boilerplate_or_heading(c)]
    if not candidates:
        return []

    # Reject queries explicitly asking about HR/benefit/equity concepts absent from the document
    if _check_missing_specific_concepts(query, candidates):
        return []

    texts = [c["original_text"] for c in candidates]
    sims = np.zeros(len(candidates))
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
        mat = vec.fit_transform(texts + [query])
        sims = cosine_similarity(mat[-1], mat[:-1])[0]
    except Exception:
        pass

    q_nums = {n.replace(",", "") for n in RE_NUM_TOKENS.findall(query)}
    q_distinctive = _distinctive_tokens(query)
    primary_cfg = INTENT_CONFIG.get(primary_intent)
    secondary_cfg = INTENT_CONFIG.get(secondary_intent) if secondary_intent else None
    anchor_required = bool(primary_cfg and primary_cfg.get("anchor_required", False))

    scored: List[Tuple[float, float, Dict[str, Any], int]] = []
    for idx, c in enumerate(candidates):
        sim = float(sims[idx])
        txt = c["original_text"]
        txt_lower = txt.lower()
        cat = c.get("category", "")
        c_nums = {n.replace(",", "") for n in RE_NUM_TOKENS.findall(txt)}
        c_tokens = set(RE_WORD_TOKENS.findall(txt_lower))
        distinctive_hits = sum(
            1 for tok in q_distinctive if _token_matches_text(tok, c_tokens, txt_lower)
        )

        score = sim * 1.4

        if primary_cfg:
            in_primary_cat = cat in primary_cfg["categories"]
            matches_primary_anchor = bool(primary_cfg["clause_anchor"].search(txt))

            if in_primary_cat:
                score += 0.55
            if matches_primary_anchor:
                score += 0.45

            # Anchor-strict intents MUST explicitly match their lexical anchor
            if anchor_required and not matches_primary_anchor:
                score -= 0.85

            # EXIT_PENALTY retrieves sibling lock-in, penalty, & certificate withholding in same section
            if primary_intent == "EXIT_PENALTY":
                if cat == "FINANCIAL_PENALTY_BOND":
                    score += 0.25

            # SECURITY_DEPOSIT retrieves all deposit/refund/forfeiture clauses (e.g. 3.1, 3.2, 3.3)
            if primary_intent == "SECURITY_DEPOSIT" and cat == "DEPOSIT_REFUND":
                score += 0.25

            # TERMINATION_NOTICE prioritizes actual termination/notice clauses (5.1, 5.2),
            # NOT exit penalty clauses (4.1, 4.2, 4.3) unless EXIT_PENALTY was also asked about
            if primary_intent == "TERMINATION_NOTICE" and secondary_intent != "EXIT_PENALTY":
                if cat == "FINANCIAL_PENALTY_BOND" and not in_primary_cat:
                    score -= 0.65

            # Suppress completely unrelated clauses when primary intent is clear
            if not in_primary_cat and not matches_primary_anchor:
                if secondary_cfg and (cat in secondary_cfg["categories"] or secondary_cfg["clause_anchor"].search(txt)):
                    score += 0.20
                else:
                    score -= 0.30
        else:
            # GENERAL query: rely strictly on semantic similarity + distinctive topical grounding
            score = sim

        # Exact number overlap bonus (e.g. "60", "75000", "15000", "45")
        if q_nums and (q_nums & c_nums):
            score += 0.25

        scored.append((score, sim, c, distinctive_hits))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)

    # Relevance gating + relative cutoff so incidental word mentions never sneak in
    min_threshold = 0.45 if primary_cfg else 0.14
    top_score = scored[0][0] if scored else 0.0
    effective_cutoff = max(min_threshold, top_score * 0.55) if primary_cfg else min_threshold

    selected: List[Dict[str, Any]] = []
    for score, sim, c, distinctive_hits in scored:
        if score < effective_cutoff:
            continue
        if primary_cfg:
            in_cat = c.get("category") in primary_cfg["categories"]
            in_sec_cat = bool(secondary_cfg and c.get("category") in secondary_cfg["categories"])
            anchor_hit = bool(primary_cfg["clause_anchor"].search(c["original_text"]))
            sec_anchor_hit = bool(secondary_cfg and secondary_cfg["clause_anchor"].search(c["original_text"]))
            if anchor_required:
                if not (anchor_hit or (secondary_cfg and (in_sec_cat or sec_anchor_hit))):
                    continue
            elif not (in_cat or in_sec_cat or anchor_hit):
                continue
        else:
            # Hardened GENERAL grounding: require real topical lexical overlap, not just boilerplate
            if q_distinctive:
                if distinctive_hits == 0:
                    continue
                if len(q_distinctive) >= 2 and distinctive_hits < 2 and sim < 0.24:
                    continue
            elif sim < 0.20:
                continue

        selected.append(_format_cited_clause(c, sim=sim, score=score))
        if len(selected) >= 3:
            break

    # Sort selected clauses in document order for clean reading (e.g., 5.1 then 5.2, or 3.1, 3.2, 3.3)
    selected.sort(key=lambda item: item["index"])
    return selected


def _join_clause_refs(clauses: List[Dict[str, Any]]) -> str:
    refs = [c["clause_ref"] for c in clauses]
    if not refs:
        return "None found"
    if len(refs) == 1:
        return refs[0]
    if len(refs) == 2:
        return f"{refs[0]} and {refs[1]}"
    return ", ".join(refs[:-1]) + f", and {refs[-1]}"


def _short_quote(text: str, max_len: int = 380) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[:max_len].rstrip() + "..."


def _extract_money_mentions(clauses: List[Dict[str, Any]]) -> List[str]:
    found = []
    for c in clauses:
        for m in RE_MONEY_EXACT.findall(c["text"]):
            val = m.strip()
            if val not in found:
                found.append(val)
    return found


def _extract_durations(text: str) -> List[str]:
    return [f"{m.group(1)} {m.group(2)}" for m in RE_DURATION_DAYS.finditer(text)]


def _build_cross_clause_notes(clauses: List[Dict[str, Any]]) -> List[str]:
    """
    Build concise human-readable cross-clause notes from retrieved clauses' `related_clauses`
    without polluting `cited_clauses` with unrelated clauses.
    """
    notes: List[str] = []
    seen: Set[Tuple[str, str]] = set()
    for c in clauses:
        src_ref = c["clause_ref"]
        for rel in c.get("related_clauses", []):
            target_ref = rel.get("target_clause_ref")
            relationship = rel.get("relationship")
            if target_ref and relationship:
                key = (src_ref, target_ref)
                if key not in seen:
                    seen.add(key)
                    notes.append(f"{src_ref} is cross-referenced with {target_ref} – {relationship}")
            elif rel.get("display_text"):
                key = (src_ref, rel["display_text"])
                if key not in seen:
                    seen.add(key)
                    notes.append(f"{src_ref}: {rel['display_text']}")
    return notes


def _append_cross_clause_context(explanation: str, cross_clause_notes: List[str]) -> str:
    """Concisely mention linked cross-clause risks in the explanation when not already stated."""
    if not cross_clause_notes:
        return explanation
    exp_lower = explanation.lower()
    unmentioned: List[str] = []
    for note in cross_clause_notes:
        # Check if the target clause reference is already explicitly discussed in explanation
        m = re.search(r'with\s+(Clause\s+\S+)', note, re.IGNORECASE)
        target_token = m.group(1).lower() if m else ""
        if target_token and target_token in exp_lower:
            continue
        unmentioned.append(note)
    if not unmentioned:
        return explanation
    snippet = "; ".join(unmentioned[:2])
    sep = " " if explanation and not explanation.endswith(" ") else ""
    return f"{explanation}{sep}Cross-clause link: {snippet}."


def _compute_confidence_and_grounding(
    query: str,
    primary_intent: str,
    secondary_intent: Optional[str],
    matched_clauses: List[Dict[str, Any]],
    direct_exact_hit: bool = False
) -> Tuple[str, str]:
    """
    Transparently compute (`grounding_status`, `confidence`) from:
      - top retrieval score
      - intent/category/anchor alignment
      - direct lexical & numeric overlap between query and retrieved clauses
    Never hardcoded; returns ("insufficient_evidence", "insufficient_evidence") when ungrounded.
    """
    if not matched_clauses:
        return "insufficient_evidence", "insufficient_evidence"

    top_score = max(float(c.get("relevance_score", 0.0)) for c in matched_clauses)

    if primary_intent == "DIRECT_CLAUSE_LOOKUP":
        if direct_exact_hit:
            return "grounded", "high"
        return "grounded", "medium"

    if primary_intent == "OVERALL_RISK_SUMMARY":
        if len(matched_clauses) >= 2 and top_score >= 0.85:
            return "grounded", "high"
        return "grounded", "medium"

    # Compute lexical & numeric overlap factor
    combined_lower = " ".join(c["text"] for c in matched_clauses).lower()
    combined_tokens = set(RE_WORD_TOKENS.findall(combined_lower))
    q_distinctive = _distinctive_tokens(query)
    lex_ratio = (
        sum(1 for tok in q_distinctive if _token_matches_text(tok, combined_tokens, combined_lower)) / len(q_distinctive)
        if q_distinctive else 0.0
    )

    q_nums = {n.replace(",", "") for n in RE_NUM_TOKENS.findall(query)}
    c_nums = {n.replace(",", "") for n in RE_NUM_TOKENS.findall(combined_lower)}
    num_bonus = 0.20 if (q_nums and (q_nums & c_nums)) else 0.0
    overlap_factor = min(1.0, lex_ratio + num_bonus)

    primary_cfg = INTENT_CONFIG.get(primary_intent)
    secondary_cfg = INTENT_CONFIG.get(secondary_intent) if secondary_intent else None

    if primary_cfg:
        best_align = 0.0
        for c in matched_clauses:
            in_cat = c["category"] in primary_cfg["categories"]
            anchor_hit = bool(primary_cfg["clause_anchor"].search(c["text"]))
            if in_cat and anchor_hit:
                best_align = max(best_align, 1.0)
            elif in_cat or anchor_hit:
                best_align = max(best_align, 0.72)
            elif secondary_cfg and (c["category"] in secondary_cfg["categories"] or secondary_cfg["clause_anchor"].search(c["text"])):
                best_align = max(best_align, 0.45)

        score_norm = min(top_score / 1.35, 1.0)
        composite = 0.45 * score_norm + 0.35 * best_align + 0.20 * overlap_factor
    else:
        score_norm = min(top_score / 0.50, 1.0)
        composite = 0.55 * score_norm + 0.45 * overlap_factor

    if composite >= 0.72:
        conf = "high"
    elif composite >= 0.45:
        conf = "medium"
    else:
        conf = "low"

    return "grounded", conf


def _build_structured_answer(
    direct_answer: str,
    clauses: List[Dict[str, Any]],
    explanation: str,
    recommendation: str
) -> Tuple[str, Dict[str, Any]]:
    refs_list = [c["clause_ref"] for c in clauses]
    refs_str = _join_clause_refs(clauses)
    quotes_items = [
        {"clause_ref": c["clause_ref"], "quote": _short_quote(c["text"])}
        for c in clauses
    ]
    quotes_lines = "\n".join(
        f'- **{q["clause_ref"]}:** "{q["quote"]}"'
        for q in quotes_items
    )
    markdown = (
        f"**Direct Answer:** {direct_answer}\n\n"
        f"**Relevant Clauses:** {refs_str}\n\n"
        f"**Supporting Contract Quotations:**\n{quotes_lines}\n\n"
        f"**Simple Explanation:** {explanation}\n\n"
        f"**Practical Recommendation:** {recommendation}"
    )
    structured = {
        "direct_answer": direct_answer,
        "relevant_clauses": refs_list,
        "supporting_quotes": quotes_items,
        "explanation": explanation,
        "recommendation": recommendation
    }
    return markdown, structured


def _synthesize_termination_answer(query: str, clauses: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    combined = " ".join(c["text"] for c in clauses)
    lower = combined.lower()

    company_immediate = any(
        k in lower for k in [
            "without cause, immediately", "at any time, with or without cause",
            "immediately upon written", "dismiss the intern instantly", "without notice or severance"
        ]
    )
    notice_matches = re.findall(r'(\d+)\s*(?:\([a-z\s-]+\)\s*)?days?\s*(?:prior\s+)?written\s+notice', lower)
    student_notice = f"{notice_matches[0]} days" if notice_matches else None
    no_leave = "no academic exam leave" in lower or "no leave may be availed" in lower
    linked_penalty = any(
        "penalty" in c["text"].lower() or "penalties" in c["text"].lower()
        or any("Penalty" in r.get("display_text", "") or "Bond" in r.get("display_text", "") for r in c.get("related_clauses", []))
        for c in clauses
    )

    if company_immediate and student_notice:
        direct = (
            f"Yes. As written, the agreement permits the company to terminate you immediately without cause "
            f"or severance, while requiring you to give at least {student_notice}' prior written notice to exit."
        )
    elif company_immediate:
        direct = "Yes. The agreement allows the company to terminate the contract immediately without cause or compensation."
    elif student_notice:
        direct = f"The agreement requires {student_notice}' prior written notice to terminate."
    else:
        direct = f"Termination conditions are governed by {_join_clause_refs(clauses)} in this agreement."

    exp_bits = []
    for c in clauses:
        cl = c["text"].lower()
        if "without cause" in cl and "immediately" in cl:
            exp_bits.append(f"{c['clause_ref']} lets the company end the agreement immediately with or without cause and with zero notice pay or severance.")
        elif "notice" in cl:
            extra = []
            if student_notice:
                extra.append(f"requires you to serve {student_notice} prior written notice")
            if no_leave:
                extra.append("prohibits taking academic exam leave or personal leave during the notice period")
            if "clause 4" in cl or "exit penalties" in cl:
                extra.append("states that early departure remains subject to the Clause 4 exit penalties")
            if extra:
                exp_bits.append(f"{c['clause_ref']} " + ", ".join(extra) + ".")
            else:
                exp_bits.append(f"{c['clause_ref']} sets out the notice and exit requirements.")
        elif "summary dismissal" in cl or "eviction" in cl:
            exp_bits.append(f"{c['clause_ref']} permits immediate summary dismissal/eviction without appeal.")

    if not exp_bits:
        exp_bits.append("These clauses define unequal exit obligations between you and the counterparty.")
    if linked_penalty and not any("clause 4" in b.lower() for b in exp_bits):
        exp_bits.append("Resigning before the agreed term may also trigger linked service bond or exit penalty clauses.")

    explanation = " ".join(exp_bits)
    recommendation = (
        "Request mutual notice parity (for example, 14–30 days' written notice for both parties), "
        "ensure academic exam leave is allowed during notice, and ask for written confirmation that "
        "resigning with notice will not trigger early exit penalties."
    )
    risk = "high" if any(c["risk_level"] == "high" for c in clauses) else "medium"
    return direct, explanation, recommendation, risk


def _synthesize_deposit_answer(query: str, clauses: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    combined = " ".join(c["text"] for c in clauses)
    lower = combined.lower()
    amounts = _extract_money_mentions(clauses)
    amt_str = amounts[0] if amounts else "the required security deposit"

    refund_time_match = re.search(r'refunded\s+within\s+(\d+\s+(?:business\s+|working\s+|calendar\s+)?days?)', lower)
    refund_timeline = refund_time_match.group(1) if refund_time_match else None

    score_match = re.search(r'exit\s+score\s+of\s+(\d+%)', lower)
    exit_score = score_match.group(1) if score_match else None

    has_forfeiture = any(k in lower for k in ["forfeited", "strictly non-refundable", "forfeiture", "will not be refunded"])
    has_arbitrary = "arbitrary amounts" in lower or "without itemized proof" in lower
    has_renovation = "renovation fee" in lower

    if has_forfeiture and refund_timeline:
        direct = (
            f"You must pay {amt_str}, which is only refunded within {refund_timeline} after completing the full term "
            f"and passing exit checks — and the full {amt_str} is strictly non-refundable and forfeited if you leave early"
            + (f", are dismissed, or score below {exit_score} on an internal exit review." if exit_score else " or are dismissed.")
        )
    elif has_forfeiture:
        extra_str = " and permits arbitrary deductions without itemized proof" if has_arbitrary else ""
        direct = (
            f"Yes, as written, the agreement makes {amt_str} strictly non-refundable / subject to forfeiture "
            f"upon early departure or disciplinary dismissal{extra_str}."
        )
    elif refund_timeline:
        direct = f"The deposit ({amt_str}) is scheduled to be refunded within {refund_timeline} after completion and clearance."
    else:
        direct = f"Security deposit and refund obligations are governed by {_join_clause_refs(clauses)}."

    exp_bits = []
    for c in clauses:
        cl = c["text"].lower()
        c_amts = _extract_money_mentions([c])
        c_amt = c_amts[0] if c_amts else amt_str
        if "prior to onboarding" in cl or ("pay a" in cl and "deposit" in cl and "forfeit" not in cl and "non-refundable" not in cl):
            exp_bits.append(f"{c['clause_ref']} requires paying {c_amt} upfront as a security deposit.")
        elif "refunded within" in cl or "refund conditions" in cl:
            conds = ["full completion of the internship duration"]
            if "no-dues" in cl:
                conds.append("clearance of all no-dues checklists")
            if "undamaged" in cl:
                conds.append("return of hardware in undamaged working order")
            tl = refund_timeline or "the stated post-exit period"
            exp_bits.append(
                f"{c['clause_ref']} delays the refund until {tl} after the final exit audit, conditioned on "
                + ", ".join(conds) + "."
            )
        elif "forfeit" in cl or "non-refundable" in cl or "will not be refunded" in cl:
            triggers = []
            if "withdraws before" in cl or "leaves" in cl or "before the end" in cl:
                triggers.append("withdrawing before completing the full term")
            if "disciplinary" in cl or "expelled" in cl:
                triggers.append("disciplinary dismissal or expulsion")
            if exit_score and exit_score in cl:
                triggers.append(f"failing to achieve an internal exit score of {exit_score}")
            trig_str = ", ".join(triggers) if triggers else "early exit or management discretion"
            extra_notes = []
            if "arbitrary amounts" in cl or "without itemized proof" in cl:
                extra_notes.append("allows arbitrary deductions without itemized proof")
            if "renovation fee" in cl:
                extra_notes.append("retains a mandatory non-refundable facility renovation fee upon check-out")
            extra_suffix = (" It also " + " and ".join(extra_notes) + ".") if extra_notes else ""
            exp_bits.append(
                f"{c['clause_ref']} states that the {c_amt} deposit is strictly non-refundable and forfeited "
                f"upon {trig_str}.{extra_suffix}"
            )

    explanation = " ".join(exp_bits) if exp_bits else "The deposit terms place disproportionate financial risk on the student."
    recommendation = (
        "Ask to remove any upfront deposit or negotiate written amendments: "
        "(1) shorten the refund timeline to 15–30 calendar days after clearance/asset return, "
        "(2) remove blanket forfeiture tied to early withdrawal, expulsion without appeal, or subjective exit scores, and "
        "(3) limit deductions strictly to itemized, documented physical damage verified via a joint handover checklist."
    )
    risk = "high" if any(c["risk_level"] == "high" for c in clauses) else "medium"
    return direct, explanation, recommendation, risk


def _synthesize_exit_penalty_answer(query: str, clauses: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    combined = " ".join(c["text"] for c in clauses)
    lower = combined.lower()
    amounts = _extract_money_mentions(clauses)
    amt_str = amounts[0] if amounts else "a financial exit penalty"

    has_cert_withhold = any(
        k in lower for k in ["withhold", "completion certificate", "academic evaluation", "experience letter", "original degree"]
    )
    has_promissory = "promissory note" in lower

    direct = f"Yes, as written, leaving before the end of the committed term triggers an exit penalty / service bond liability of {amt_str}"
    if has_cert_withhold:
        direct += " and allows the company to withhold your completion certificate, academic evaluation signature, and experience letter until paid in full."
    elif has_promissory:
        direct += " alongside a mandatory promissory note deposit."
    else:
        direct += "."

    exp_bits = []
    for c in clauses:
        cl = c["text"].lower()
        c_amts = _extract_money_mentions([c])
        c_amt = c_amts[0] if c_amts else amt_str
        if any(k in cl for k in ["withhold", "completion certificate", "academic evaluation", "original degree", "experience letter", "relieving letter"]):
            exp_bits.append(
                f"{c['clause_ref']} states that the company will withhold your completion certificate, academic evaluation signature, or original documents until all exit penalties and liquidated damages are paid in full — creating direct risk to university credit requirements."
            )
        elif "lock-in" in cl and "rs." not in cl and "inr" not in cl and "₹" not in cl:
            exp_bits.append(
                f"{c['clause_ref']} binds you to a mandatory lock-in period covering the full tenure, which operates together with the linked financial exit penalty."
            )
        elif any(k in cl for k in ["exit penalty", "liquidated damages", "service bond", "penalty fee", "promissory note", "mandatory financial penalty"]):
            extra_amts = f" (referencing {', '.join(c_amts)})" if len(c_amts) > 1 else ""
            exp_bits.append(
                f"{c['clause_ref']} demands {c_amt}{extra_amts} in exit penalty / liquidated damages if you resign or discontinue early."
            )

    exp_bits.append(
        "Under Section 74 of the Indian Contract Act (where Indian law applies), courts assess reasonable compensation "
        "for actual loss rather than automatically making every contractual penalty enforceable or automatically unenforceable. "
        "Even where a large fixed penalty may be contested, signing it exposes you to dispute risk and withheld academic credentials."
    )

    explanation = " ".join(exp_bits)
    recommendation = (
        f"Before signing, request removal of the mandatory lock-in, the {amt_str} exit penalty, and any clause allowing "
        f"withholding of completion certificates, promissory notes, or academic evaluation signatures. If the company provides genuine paid "
        f"external training, propose capping recovery strictly to actual third-party receipts on a pro-rata basis."
    )
    return direct, explanation, recommendation, "high"


def _synthesize_ip_answer(query: str, clauses: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    combined = " ".join(c["text"] for c in clauses)
    lower = combined.lower()
    personal_devices = "personal devices" in lower or "weekends" in lower or "personal time" in lower
    academic_claim = "capstone" in lower or "thesis" in lower or "theses" in lower or "academic" in lower
    moral_waiver = "moral rights" in lower
    media_rights = "media rights" in lower or "participant recordings" in lower

    if personal_devices or academic_claim:
        direct = (
            "As drafted, the company claims exclusive ownership of all code, algorithms, and designs you create "
            "during the term — including work created on personal devices during weekends/personal time"
            + (" and even academic capstone projects or theses incorporating internship concepts." if academic_claim else ".")
        )
    elif media_rights:
        direct = "As drafted, the sponsor retains exclusive commercial media rights to all event photographs, recordings, and student project pitch presentations."
    else:
        direct = "The agreement assigns work product created during your engagement to the company."

    exp_bits = []
    for c in clauses:
        cl = c["text"].lower()
        if "personal devices" in cl or "personal time" in cl or "weekends" in cl:
            exp_bits.append(
                f"{c['clause_ref']} extends company ownership beyond office hours to personal devices and weekend work."
            )
        elif "work made for hire" in cl:
            exp_bits.append(f"{c['clause_ref']} assigns company-directed work made on company equipment to the employer.")
        elif "media rights" in cl:
            exp_bits.append(f"{c['clause_ref']} grants the sponsor exclusive commercial media rights over event recordings and student project presentations.")
        if "moral rights" in cl or "capstone" in cl or "thesis" in cl or "theses" in cl:
            exp_bits.append(
                f"{c['clause_ref']} assigns worldwide rights and waives moral rights/authorship attribution"
                + (", explicitly reaching university capstone projects or academic theses." if ("capstone" in cl or "thesis" in cl or "theses" in cl) else ".")
            )

    if not exp_bits:
        exp_bits.append("Review the scope of IP ownership to verify whether personal and university work is protected.")

    explanation = " ".join(exp_bits)
    recommendation = (
        "Request a written carve-out stating that the counterparty only owns deliverables created using company resources "
        "within assigned scope, and that you retain full ownership of pre-existing projects, personal off-hours "
        "code on your own laptop, open-source work, and university capstone/thesis submissions."
    )
    risk = "high" if (personal_devices or academic_claim or moral_waiver) else "medium"
    return direct, explanation, recommendation, risk


def _synthesize_academic_exam_answer(query: str, clauses: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    combined = " ".join(c["text"] for c in clauses).lower()
    refs = _join_clause_refs(clauses)
    blocks_exam_leave = "no academic exam leave" in combined or "no leave may be availed" in combined

    if blocks_exam_leave:
        direct = (
            f"No — {refs} explicitly prohibits taking academic exam leave or personal leave during the notice period, "
            f"and the agreement contains no protective guarantee for university semester exam leave."
        )
        risk = "high"
    else:
        direct = f"Academic leave or schedule conditions referenced in this agreement appear in {refs}."
        risk = "medium"

    exp_bits = []
    for c in clauses:
        cl = c["text"].lower()
        if "no academic exam leave" in cl:
            exp_bits.append(
                f"{c['clause_ref']} bars taking academic exam leave or personal leave during the required notice period, creating a direct conflict with university examination calendars."
            )
        elif "no leave may be availed" in cl:
            exp_bits.append(
                f"{c['clause_ref']} forbids taking any leave during the notice period, which can conflict with semester exams."
            )
        elif c.get("explanation"):
            exp_bits.append(f"{c['clause_ref']}: {c['explanation']}")

    explanation = " ".join(exp_bits) if exp_bits else f"See {refs} for the stated leave restrictions."
    recommendation = (
        "Request a written academic accommodation clause guaranteeing unpaid or flexible leave for scheduled university "
        "semester exams, viva/project evaluations, and mandatory academic submissions without penalty or notice extension."
    )
    return direct, explanation, recommendation, risk


def _synthesize_hostel_rules_answer(query: str, clauses: List[Dict[str, Any]]) -> Tuple[str, str, str, str]:
    combined = " ".join(c["text"] for c in clauses)
    lower = combined.lower()
    refs = _join_clause_refs(clauses)

    curfew_match = re.search(r'curfew\s+(?:hours\s+of\s+)?(\d{1,2}:\d{2}\s*(?:am|pm))', lower, re.IGNORECASE)
    curfew_time = curfew_match.group(1).upper() if curfew_match else None
    has_no_appeal_eviction = "without right of appeal" in lower or "summary dismissal and eviction" in lower
    has_expulsion_forfeiture = "will not be refunded if the resident is expelled" in lower

    parts = []
    if curfew_time:
        parts.append(f"enforces a strict {curfew_time} daily curfew")
    if has_no_appeal_eviction:
        parts.append("allows immediate summary eviction without right of appeal for alleged misconduct, noise complaints, or warden disputes")
    if has_expulsion_forfeiture:
        parts.append("forfeits the entire security deposit upon disciplinary expulsion regardless of appeal status")

    if parts:
        direct = f"Under {refs}, the hostel agreement " + ", ".join(parts) + "."
    else:
        direct = f"Hostel occupancy and disciplinary rules are governed by {refs}."

    exp_bits = []
    for c in clauses:
        cl = c["text"].lower()
        if "curfew" in cl or "eviction" in cl or "warden" in cl:
            exp_bits.append(
                f"{c['clause_ref']} combines strict disciplinary rules with unilateral warden eviction powers"
                + (" and automatic deposit forfeiture without due process." if "will not be refunded" in cl else ".")
            )
        elif c.get("explanation"):
            exp_bits.append(f"{c['clause_ref']}: {c['explanation']}")

    explanation = " ".join(exp_bits) if exp_bits else f"See {refs} for hostel conduct and eviction provisions."
    recommendation = (
        "Ask for written due-process safeguards before eviction (written warning, opportunity to be heard, and a minimum "
        "7–14 day vacate period except for serious safety violations), and decouple security deposit refunds from disputed disciplinary proceedings."
    )
    risk = "high" if any(c["risk_level"] == "high" for c in clauses) else "medium"
    return direct, explanation, recommendation, risk


def _synthesize_overall_risk_summary(
    clauses: List[Dict[str, Any]],
    analyzed_clauses: List[Dict[str, Any]],
    document_domain: str,
    overall_fairness: Dict[str, Any]
) -> Tuple[str, str, str, str]:
    substantive = [c for c in analyzed_clauses if not _is_boilerplate_or_heading(c)]
    high_cnt = sum(1 for c in substantive if c.get("final_risk_level") == "high")
    med_cnt = sum(1 for c in substantive if c.get("final_risk_level") == "medium")
    low_cnt = sum(1 for c in substantive if c.get("final_risk_level") == "low")
    total_cnt = len(substantive)

    fairness = overall_fairness or {}
    student_pct = fairness.get("student_favor_pct")
    counterparty_pct = fairness.get("counterparty_favor_pct")
    fairness_label = fairness.get("fairness_label")

    fairness_clause = ""
    if student_pct is not None and counterparty_pct is not None:
        fairness_clause = f" Overall fairness is rated {student_pct}% student-favor vs. {counterparty_pct}% counterparty-favor"
        if fairness_label:
            fairness_clause += f" ({fairness_label})"
        fairness_clause += "."

    if high_cnt >= 3:
        direct = (
            f"Caution: Do not sign as drafted without written amendments. This {document_domain.replace('_', ' ').lower()} agreement "
            f"contains {high_cnt} high-risk, {med_cnt} medium-risk, and {low_cnt} low-risk substantive clauses (out of {total_cnt})."
            f"{fairness_clause}"
        )
        risk = "high"
    elif high_cnt >= 1:
        direct = (
            f"This agreement has {high_cnt} high-risk and {med_cnt} medium-risk clauses out of {total_cnt} substantive clauses "
            f"that warrant negotiation before signing.{fairness_clause}"
        )
        risk = "high"
    elif med_cnt >= 1:
        direct = (
            f"No high-risk red flags were detected, though {med_cnt} medium-risk clauses (out of {total_cnt}) merit review before signing."
            f"{fairness_clause}"
        )
        risk = "medium"
    else:
        direct = (
            f"All {total_cnt} substantive clauses were assessed as low-risk standard provisions with no severe red flags detected."
            f"{fairness_clause}"
        )
        risk = "low"

    exp_bits = []
    for c in clauses:
        cat_label = c["category"].replace("_", " ").title()
        summary_line = c.get("explanation") or f"{cat_label} ({c['risk_level'].upper()} risk)."
        exp_bits.append(f"{c['clause_ref']} ({cat_label} – {c['risk_level'].upper()} risk): {summary_line}")

    explanation = " ".join(exp_bits) if exp_bits else "Review all cited terms carefully before executing."

    actions = []
    for c in clauses[:3]:
        act = (c.get("action") or "").strip()
        if act and act not in actions:
            actions.append(f"{c['clause_ref']}: {act}")
    recommendation = (
        "Prioritize negotiating these top items in writing before signing: " + " ".join(actions)
        if actions else "Verify all key dates, financial terms, and exit conditions in writing before signing."
    )
    return direct, explanation, recommendation, risk


def _synthesize_direct_clause_answer(
    requested_refs: List[str],
    clauses: List[Dict[str, Any]]
) -> Tuple[str, str, str, str]:
    primary_clauses = [c for c in clauses if c["clause_ref"] in requested_refs] or clauses[:1]
    linked_clauses = [c for c in clauses if c["clause_ref"] not in requested_refs]

    primary_desc = []
    for c in primary_clauses:
        cat_readable = c["category"].replace("_", " ").title()
        primary_desc.append(f"{c['clause_ref']} ({cat_readable}, {c['risk_level'].upper()} risk)")

    first = primary_clauses[0]
    first_exp = (first.get("explanation") or "").strip()
    if len(primary_clauses) == 1:
        direct = f"{primary_desc[0]}: {first_exp}" if first_exp else f"{primary_desc[0]} governs the obligations quoted below."
    else:
        direct = f"The requested provisions ({', '.join(primary_desc)}) set out the obligations quoted below."

    if linked_clauses:
        linked_refs = _join_clause_refs(linked_clauses)
        direct += f" It is directly linked to {linked_refs}."

    exp_bits = []
    for c in primary_clauses:
        if c.get("explanation"):
            exp_bits.append(f"{c['clause_ref']}: {c['explanation']}")
        else:
            exp_bits.append(f"{c['clause_ref']} falls under {c['category'].replace('_', ' ').title()} ({c['risk_level'].upper()} risk).")
    for lc in linked_clauses:
        if lc.get("explanation"):
            exp_bits.append(f"Linked {lc['clause_ref']} ({lc['risk_level'].upper()} risk): {lc['explanation']}")

    explanation = " ".join(exp_bits)
    rec_bits = [c["action"] for c in primary_clauses if c.get("action")]
    recommendation = " ".join(dict.fromkeys(rec_bits)) or "Review the exact clause language and request written clarification or amendment before signing."
    risk = "high" if any(c["risk_level"] == "high" for c in clauses) else (
        "medium" if any(c["risk_level"] == "medium" for c in clauses) else "low"
    )
    return direct, explanation, recommendation, risk


def _synthesize_general_intent_answer(
    intent: str,
    query: str,
    clauses: List[Dict[str, Any]]
) -> Tuple[str, str, str, str]:
    """Document-grounded synthesis for confidentiality, stipend, non-compete, certificates, liability, duration, jurisdiction, etc."""
    combined = " ".join(c["text"] for c in clauses)
    lower = combined.lower()
    amounts = _extract_money_mentions(clauses)
    refs = _join_clause_refs(clauses)
    max_risk = "high" if any(c["risk_level"] == "high" for c in clauses) else (
        "medium" if any(c["risk_level"] == "medium" for c in clauses) else "low"
    )

    if intent == "STIPEND_PAYMENT":
        amt = amounts[0] if amounts else "the stated compensation"
        contingent = any(k in lower for k in ["contingent", "withhold, delay, or reduce", "90 days in arrears", "forfeited", "no advance payment"])
        direct = (
            f"The agreement specifies {amt}, but makes payment contingent on milestone sign-off, delayed in arrears, or subject to unilateral withholding/forfeiture."
            if contingent else f"Compensation is stated as {amt} under {refs}."
        )
        explanation = " ".join(dict.fromkeys(c["explanation"] for c in clauses if c.get("explanation"))) or f"See {refs} for exact payout conditions."
        recommendation = "Request a guaranteed fixed payment schedule independent of external client sign-offs, no forfeiture of earned fees, and partial upfront disbursement where applicable."
    elif intent == "CONFIDENTIALITY_NDA":
        has_gag = any(k in lower for k in ["glassdoor", "negative comments", "resume", "linkedin", "github", "gag clause"])
        direct = (
            f"Yes, {refs} imposes strict confidentiality and also restricts mentioning the company/tech stack on resumes, LinkedIn, or GitHub and bars posting negative reviews."
            if has_gag else f"Confidentiality obligations are defined in {refs}."
        )
        explanation = " ".join(dict.fromkeys(c["explanation"] for c in clauses if c.get("explanation")))
        recommendation = "Keep legitimate trade-secret confidentiality, but request written permission to list your role title, company name, and non-confidential skills on your CV/LinkedIn and remove broad gag language."
    elif intent == "NON_COMPETE":
        direct = f"Yes, {refs} contains post-departure non-compete / restrictive covenant language."
        explanation = (
            f"{refs} restricts working with competing businesses after departure. Where Indian law applies, "
            f"Section 27 of the Indian Contract Act heavily restricts post-employment restraints of trade; "
            f"however, signing overly broad restrictions can still create legal friction."
        )
        recommendation = "Request deletion of post-internship/post-employment non-compete clauses and limit obligations to confidentiality of actual trade secrets."
    elif intent == "CERTIFICATE_RETENTION":
        direct = f"Yes, {refs} allows withholding or retaining certificates, marksheets, or academic evaluation documents."
        explanation = " ".join(dict.fromkeys(c["explanation"] for c in clauses if c.get("explanation")))
        recommendation = "Refuse surrendering original certificates and ask to remove any condition tying academic evaluation signatures or completion certificates to financial penalties."
    elif intent == "LIABILITY_INDEMNITY":
        if "personally guarantee" in lower or "organizer indemnity" in lower:
            direct = (
                f"Yes, {refs} makes student organizers personally guarantee payment and indemnify the sponsor "
                f"against cancellation losses, weather delays, or low attendee turnout"
                + (" (plus liquidated damages if target footfall is missed)." if "footfall" in lower else ".")
            )
        elif "treble damages" in lower:
            direct = f"Yes, {refs} imposes personal liability for treble (3x) damages to hostel property and disclaims management responsibility for on-campus accidents."
        elif "attorney fees" in lower:
            direct = f"Yes, {refs} shifts the counterparty's attorney fees and litigation costs onto the student while capping company liability."
        else:
            direct = f"Yes, {refs} places broad personal indemnity or financial liability obligations on the student."
        explanation = " ".join(dict.fromkeys(c["explanation"] for c in clauses if c.get("explanation"))) or (
            f"{refs} transfers disproportionate financial and legal liability onto the student."
        )
        recommendation = (
            "Remove personal student guarantees, treble damages, and unlimited indemnity clauses. "
            "Limit any liability strictly to documented direct losses caused by willful misconduct, capped at the contract fee or club budget."
        )
    else:
        top = clauses[0]
        direct = f"Based on the uploaded agreement, this is addressed in {refs}."
        explanation = " ".join(dict.fromkeys(c["explanation"] for c in clauses if c.get("explanation"))) or (
            f"These provisions fall under {top['category'].replace('_', ' ').title()} ({top['risk_level'].upper()} risk)."
        )
        recommendation = top.get("action") or "Review the cited clause wording carefully and request written clarification for any ambiguous terms."

    return direct, explanation, recommendation, max_risk


def _build_insufficient_evidence_response(
    query_clean: str,
    intent: str,
    candidate_count: int,
    secondary_intent: Optional[str],
    missing_concepts: Optional[List[str]] = None,
    missing_clause_refs: Optional[List[str]] = None
) -> Dict[str, Any]:
    if missing_clause_refs:
        refs_txt = ", ".join(f"Clause {n}" for n in missing_clause_refs)
        direct = f"This agreement does not contain {refs_txt}, so there is not enough information in the uploaded document to explain it."
        explanation = (
            f"No substantive provision numbered {refs_txt} was found in the uploaded contract. "
            f"Because ClauseGuard AI answers strictly from the uploaded agreement, it will not guess or invent missing clauses."
        )
    elif missing_concepts:
        topic_txt = ", ".join(missing_concepts)
        direct = f"This agreement does not contain enough information about {topic_txt} to answer your question."
        explanation = (
            f"No clause in the uploaded contract mentions {topic_txt}. "
            f"Because ClauseGuard AI only answers from the uploaded contract text, it will not assume or fabricate benefits or terms that are absent."
        )
    else:
        intent_readable = intent.replace("_", " ").lower() if intent not in ("GENERAL", "DIRECT_CLAUSE_LOOKUP") else "that topic"
        direct = f"This agreement does not contain enough information about {intent_readable} to answer your question directly."
        explanation = (
            f"No clause in the current agreement explicitly addresses \"{query_clean}\". "
            f"Because ClauseGuard AI only answers from the uploaded contract text, it will not guess or invent missing terms."
        )

    recommendation = "Ask the company or institution to add explicit written terms covering this point before you sign."
    answer = (
        f"**Direct Answer:** {direct}\n\n"
        f"**Relevant Clauses:** None found in the uploaded document.\n\n"
        f"**Simple Explanation:** {explanation}\n\n"
        f"**Practical Recommendation:** {recommendation}"
    )
    return {
        "query": query_clean,
        "intent": intent,
        "answer": answer,
        "risk_status": "low",
        "cited_clauses": [],
        "grounding_status": "insufficient_evidence",
        "confidence": "insufficient_evidence",
        "retrieval_stats": {
            "candidate_count": candidate_count,
            "cited_count": 0,
            "top_score": 0.0,
            "secondary_intent": secondary_intent
        },
        "cross_clause_notes": [],
        "structured_answer": {
            "direct_answer": direct,
            "relevant_clauses": [],
            "supporting_quotes": [],
            "explanation": explanation,
            "recommendation": recommendation
        }
    }


def answer_student_question(
    query: str,
    analyzed_clauses: List[Dict[str, Any]],
    document_domain: str,
    overall_fairness: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main Q&A entry point:
      1. Support direct clause number lookups ('Explain Clause 4.2') and overall risk summaries.
      2. Identify question intent before retrieval.
      3. Retrieve only relevant clauses via hybrid semantic + intent/category + numeric matching.
      4. Verify grounding & confidence transparently and extract cross-clause notes.
      5. Explicitly return `insufficient_evidence` when the agreement lacks information.
    """
    query_clean = (query or "").strip()
    if not query_clean:
        msg = "Please enter a question about your uploaded agreement."
        return {
            "query": query,
            "intent": "EMPTY",
            "answer": msg,
            "risk_status": "low",
            "cited_clauses": [],
            "grounding_status": "insufficient_evidence",
            "confidence": "insufficient_evidence",
            "retrieval_stats": {
                "candidate_count": 0,
                "cited_count": 0,
                "top_score": 0.0,
                "secondary_intent": None
            },
            "cross_clause_notes": [],
            "structured_answer": {
                "direct_answer": msg,
                "relevant_clauses": [],
                "supporting_quotes": [],
                "explanation": "No question text was submitted.",
                "recommendation": "Enter a question about a specific clause, risk, or obligation in your agreement."
            }
        }

    if not analyzed_clauses:
        msg = "No agreement is currently loaded. Please upload a PDF, DOCX, or TXT contract or select a sample agreement first."
        return {
            "query": query_clean,
            "intent": "NO_DOCUMENT",
            "answer": msg,
            "risk_status": "neutral",
            "cited_clauses": [],
            "grounding_status": "insufficient_evidence",
            "confidence": "insufficient_evidence",
            "retrieval_stats": {
                "candidate_count": 0,
                "cited_count": 0,
                "top_score": 0.0,
                "secondary_intent": None
            },
            "cross_clause_notes": [],
            "structured_answer": {
                "direct_answer": msg,
                "relevant_clauses": [],
                "supporting_quotes": [],
                "explanation": "No contract clauses are available to search.",
                "recommendation": "Upload a contract or load a sample agreement to enable grounded Q&A."
            }
        }

    candidates = [c for c in analyzed_clauses if not _is_boilerplate_or_heading(c)]
    candidate_count = len(candidates)

    # 1. Check Direct Clause Number Lookup (e.g. "Explain Clause 4.2", "What does Clause 3.3 say?")
    direct_targets = _extract_direct_clause_targets(query_clean)
    if direct_targets:
        matched_clauses, requested_refs, exact_hit = _lookup_direct_clauses(direct_targets, analyzed_clauses)
        if not matched_clauses:
            return _build_insufficient_evidence_response(
                query_clean=query_clean,
                intent="DIRECT_CLAUSE_LOOKUP",
                candidate_count=candidate_count,
                secondary_intent=None,
                missing_clause_refs=direct_targets
            )
        cross_clause_notes = _build_cross_clause_notes(matched_clauses)
        direct, explanation, recommendation, risk_status = _synthesize_direct_clause_answer(requested_refs, matched_clauses)
        explanation = _append_cross_clause_context(explanation, cross_clause_notes)
        full_answer, structured_answer = _build_structured_answer(direct, matched_clauses, explanation, recommendation)
        grounding_status, confidence = _compute_confidence_and_grounding(
            query=query_clean,
            primary_intent="DIRECT_CLAUSE_LOOKUP",
            secondary_intent=None,
            matched_clauses=matched_clauses,
            direct_exact_hit=exact_hit
        )
        top_score = round(max(float(c["relevance_score"]) for c in matched_clauses), 3)
        return {
            "query": query_clean,
            "intent": "DIRECT_CLAUSE_LOOKUP",
            "answer": full_answer,
            "risk_status": risk_status,
            "cited_clauses": matched_clauses,
            "grounding_status": grounding_status,
            "confidence": confidence,
            "retrieval_stats": {
                "candidate_count": candidate_count,
                "cited_count": len(matched_clauses),
                "top_score": top_score,
                "secondary_intent": None
            },
            "cross_clause_notes": cross_clause_notes,
            "structured_answer": structured_answer
        }

    # 2. Check explicit missing specific concepts (e.g. dental insurance, 401k, stock option vesting)
    missing_concepts = _check_missing_specific_concepts(query_clean, candidates)
    primary_intent, secondary_intent, _ = detect_question_intent(query_clean)

    if missing_concepts and primary_intent != "OVERALL_RISK_SUMMARY":
        return _build_insufficient_evidence_response(
            query_clean=query_clean,
            intent=primary_intent,
            candidate_count=candidate_count,
            secondary_intent=secondary_intent,
            missing_concepts=missing_concepts
        )

    # 3. Hybrid clause retrieval
    matched_clauses = retrieve_relevant_clauses(
        query=query_clean,
        analyzed_clauses=analyzed_clauses,
        primary_intent=primary_intent,
        secondary_intent=secondary_intent
    )

    if not matched_clauses:
        return _build_insufficient_evidence_response(
            query_clean=query_clean,
            intent=primary_intent,
            candidate_count=candidate_count,
            secondary_intent=secondary_intent
        )

    # 4. Cross-clause context extraction (without polluting cited_clauses)
    cross_clause_notes = _build_cross_clause_notes(matched_clauses)

    # 5. Route to intent-specific grounded synthesizer
    if primary_intent == "OVERALL_RISK_SUMMARY":
        direct, explanation, recommendation, risk_status = _synthesize_overall_risk_summary(
            clauses=matched_clauses,
            analyzed_clauses=analyzed_clauses,
            document_domain=document_domain or "AGREEMENT",
            overall_fairness=overall_fairness or {}
        )
    elif primary_intent == "TERMINATION_NOTICE":
        direct, explanation, recommendation, risk_status = _synthesize_termination_answer(query_clean, matched_clauses)
    elif primary_intent == "SECURITY_DEPOSIT":
        direct, explanation, recommendation, risk_status = _synthesize_deposit_answer(query_clean, matched_clauses)
    elif primary_intent == "EXIT_PENALTY":
        direct, explanation, recommendation, risk_status = _synthesize_exit_penalty_answer(query_clean, matched_clauses)
    elif primary_intent == "INTELLECTUAL_PROPERTY":
        direct, explanation, recommendation, risk_status = _synthesize_ip_answer(query_clean, matched_clauses)
    elif primary_intent == "ACADEMIC_EXAM_LEAVE":
        direct, explanation, recommendation, risk_status = _synthesize_academic_exam_answer(query_clean, matched_clauses)
    elif primary_intent == "HOSTEL_RULES_CURFEW":
        direct, explanation, recommendation, risk_status = _synthesize_hostel_rules_answer(query_clean, matched_clauses)
    else:
        direct, explanation, recommendation, risk_status = _synthesize_general_intent_answer(primary_intent, query_clean, matched_clauses)

    explanation = _append_cross_clause_context(explanation, cross_clause_notes)
    full_answer, structured_answer = _build_structured_answer(direct, matched_clauses, explanation, recommendation)

    # 6. Transparent confidence & grounding verification
    grounding_status, confidence = _compute_confidence_and_grounding(
        query=query_clean,
        primary_intent=primary_intent,
        secondary_intent=secondary_intent,
        matched_clauses=matched_clauses
    )
    top_score = round(max(float(c["relevance_score"]) for c in matched_clauses), 3)

    return {
        "query": query_clean,
        "intent": primary_intent,
        "answer": full_answer,
        "risk_status": risk_status,
        "cited_clauses": matched_clauses,
        "grounding_status": grounding_status,
        "confidence": confidence,
        "retrieval_stats": {
            "candidate_count": candidate_count,
            "cited_count": len(matched_clauses),
            "top_score": top_score,
            "secondary_intent": secondary_intent
        },
        "cross_clause_notes": cross_clause_notes,
        "structured_answer": structured_answer
    }
