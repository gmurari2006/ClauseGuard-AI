"""
ClauseGuard AI - Auditable Student Agreement Dataset Preparation Pipeline

Constructs a balanced, deduplicated, leakage-free dataset of student contract
clauses across all 10 CLAUSE_CATEGORIES and 6 AGREEMENT_DOMAINS, plus a separate
set of non-substantive structural/informational blocks for structural-filter
evaluation.

IMPORTANT:
- This module prepares and validates dataset artifacts ONLY.
- It does NOT retrain the active baseline ML classifier or modify backend/models/*.
"""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from backend.config import AGREEMENT_DOMAINS, BASE_DIR, CLAUSE_CATEGORIES

DEFAULT_DATASETS_DIR = BASE_DIR / "datasets"
VALID_RISK_LEVELS = {"high", "medium", "low"}
VALID_SOURCE_TYPES = {"curated_synthetic", "sample_agreement_extracted"}
MIN_CLAUSE_CHARS = 35
MAX_CLAUSE_CHARS = 1500
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85
BASELINE_MODEL_DATASET_SIZE = 125
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

REQUIRED_CLAUSE_FIELDS = (
    "id",
    "text",
    "category",
    "risk_level",
    "domain",
    "source_type",
    "source_group",
    "clause_ref",
    "rationale",
    "split",
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_text_for_hash(text: str) -> str:
    """Case- and whitespace-normalized text canonical form for exact deduplication."""
    collapsed = re.sub(r"\s+", " ", text.strip().lower())
    return collapsed


def compute_exact_hash(text: str) -> str:
    """Deterministic SHA-256 hash over normalized clause text."""
    return hashlib.sha256(normalize_text_for_hash(text).encode("utf-8")).hexdigest()


def tokenize_for_jaccard(text: str) -> Set[str]:
    """Extract lowercase alphanumeric token set for near-duplicate Jaccard checks."""
    return set(_TOKEN_RE.findall(text.lower()))


def compute_token_jaccard(tokens_a: Set[str], tokens_b: Set[str]) -> float:
    """Compute Jaccard similarity |A ∩ B| / |A ∪ B| between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    if not tokens_a or not tokens_b:
        return 0.0
    union_len = len(tokens_a | tokens_b)
    return len(tokens_a & tokens_b) / union_len if union_len else 0.0


# ---------------------------------------------------------------------------
# Curated & Sample-Extracted Clause Seed Corpus
# Tuple format:
#   (text, category, risk_level, domain, source_type, source_group, clause_ref, rationale)
# ---------------------------------------------------------------------------
RAW_CLAUSE_SEED: List[Tuple[str, str, str, str, str, str, str, str]] = [
    # =========================================================================
    # 1. FINANCIAL_PENALTY_BOND (36 clauses)
    # =========================================================================
    (
        "3.2 In the event of breach or early departure prior to completion of the 24-month term, the Student shall pay liquidated damages of Rs. 1,50,000 as training fee recovery and operational disruption compensation.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 3.2",
        "Imposes a Rs. 1,50,000 liquidated-damages exit bond on an intern for departing early.",
    ),
    (
        "3.3 The intern shall deposit their original degree certificates, marks cards, and passport with the Company HR as collateral security until completion of the 18-month service bond.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 3.3",
        "Coercively retains original academic credentials and passport as bond collateral.",
    ),
    (
        "2.1 Employee service agreement: A mandatory financial penalty of Rs. 2,50,000 shall be levied if the trainee departs before completing 36 continuous months of employment.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 2.1",
        "Enforces a three-year service bond backed by a Rs. 2,50,000 financial exit penalty.",
    ),
    (
        "2.2 The employee executes an employment bond committing to serve for three years, with a promissory note of Rs. 1,20,000 deposited with the company HR department upon joining.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 2.2",
        "Requires a graduate trainee to sign a Rs. 1,20,000 promissory note to enforce a multi-year bond.",
    ),
    (
        "2.3 Should the candidate fail to accept the full-time placement offer following graduation, a recruitment penalty fee of Rs. 2,00,000 shall become immediately due and payable.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 2.3",
        "Penalizes a graduating student with Rs. 2,00,000 for declining a campus placement offer.",
    ),
    (
        "4.2 Early Exit Penalty: If the Intern resigns, discontinues, or abandons the internship prior to 30th June 2027 for any reason whatsoever, the Intern shall pay liquidated damages of INR 75,000 within 10 days.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 4.2",
        "Assesses INR 75,000 liquidated damages for any early internship withdrawal regardless of cause.",
    ),
    (
        "4.3 The Company shall withhold the Internship Completion Certificate, academic evaluation signature, and experience letter until all early exit penalties and alleged damages are settled in full.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 4.3",
        "Uses academic credit signatures and completion certificates as leverage to collect exit penalties.",
    ),
    (
        "3.2 Liquidated damages shall be assessed against the student for any failure to achieve the target footfall of 3,000 registered students, deductible directly from future club sponsorship tranches.",
        "FINANCIAL_PENALTY_BOND", "high", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 3.2",
        "Levies liquidated damages on student organizers when campus event turnout falls short of sponsor targets.",
    ),
    (
        "Should the summer analyst discontinue the rotational program before completing twelve calendar months, a fixed onboarding recovery fee of INR 95,000 becomes payable immediately.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 4.1",
        "Creates an unitemized onboarding recovery penalty tied to early internship exit.",
    ),
    (
        "Upon signing this offer, the campus recruit must furnish an undated security cheque of Rs. 1,80,000 drawn in favor of the employer to guarantee adherence to the 24-month lock-in.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 5.2",
        "Collects an undated security cheque as coercive financial collateral for a service lock-in.",
    ),
    (
        "The graduate trainee's parents or legal guardian must co-sign a personal surety bond of Rs. 3,00,000 indemnifying the employer against premature resignation during the first two years.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_core_02", "Clause 3.1",
        "Binds the student's family as financial guarantors for a Rs. 3,00,000 employment bond.",
    ),
    (
        "Any intern withdrawing to pursue postgraduate admission or competitive examinations before the project handover date shall reimburse USD 3,000 in flat mentorship and software costs.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 6.1",
        "Imposes a flat USD 3,000 penalty even when a student leaves for academic progression.",
    ),
    (
        "Leaving the residential hall mid-semester triggers an automatic contract breakage fine equal to four months of hostel fees in addition to forfeiting advance mess charges.",
        "FINANCIAL_PENALTY_BOND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 5.3",
        "Charges a punitive four-month breakage fine when a student exits a hostel mid-term.",
    ),
    (
        "If the student freelance tutor fails to deliver all thirty recorded lectures by the semester deadline, a flat default penalty of Rs. 45,000 will be billed to the student.",
        "FINANCIAL_PENALTY_BOND", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 4.4",
        "Assesses a steep flat monetary penalty for incomplete freelance educational deliverables.",
    ),
    (
        "The student research fellow must repay 150% of all laboratory bench fees and stipend amounts previously received if the fellowship is terminated before the 18-month grant cycle concludes.",
        "FINANCIAL_PENALTY_BOND", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 7.2",
        "Demands punitive 150% clawback of past fellowship stipends upon early withdrawal.",
    ),
    (
        "A mandatory administrative exit charge of Rs. 60,000 will be recovered from the intern's bank account via standing instruction if notice is served during a client sprint.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 5.1",
        "Authorizes automatic bank recovery of an arbitrary Rs. 60,000 exit charge.",
    ),
    (
        "Failure to convert from the six-month winter internship into the subsequent two-year full-time analyst track obligates the student to pay Rs. 1,10,000 in recruitment damages.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_consulting_03", "Clause 2.4",
        "Forces internship-to-full-time conversion through a Rs. 1,10,000 penalty.",
    ),
    (
        "Original Class XII marksheets and university migration certificates shall remain locked in the institution's accounts vault until the student clears every hostel and lab bond.",
        "FINANCIAL_PENALTY_BOND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_campus_02", "Clause 4.2",
        "Withholds original academic transcripts as collateral for hostel financial obligations.",
    ),
    (
        "If the student organizing society postpones the cultural festival by more than seven days, the committee shall pay liquidated damages equal to twice the committed branding fee.",
        "FINANCIAL_PENALTY_BOND", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 6.2",
        "Imposes double-fee liquidated damages on a student club for event scheduling delays.",
    ),
    (
        "The apprentice shall pay Rs. 85,000 as liquidated compensation for cloud certification vouchers if employment ends within twenty-four months, without pro-rata reduction.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 4.3",
        "Recovers fixed certification bond fees with zero pro-rata depreciation over two years.",
    ),
    (
        "Withdrawal from the paid university boot camp after week two incurs a non-negotiable cancellation penalty of Rs. 40,000 irrespective of documented medical hospitalization.",
        "FINANCIAL_PENALTY_BOND", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 3.3",
        "Enforces a rigid financial cancellation penalty even in documented medical emergencies.",
    ),
    (
        "Student campus ambassadors who miss two monthly promotional quotas must pay a brand-reputation liquidated fee of Rs. 25,000 deductible from pending commissions.",
        "FINANCIAL_PENALTY_BOND", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 5.1",
        "Fines student brand ambassadors Rs. 25,000 for missing marketing quotas.",
    ),
    (
        "In the event the trainee resigns during the 90-day classroom induction, all relocation allowances plus a flat Rs. 70,000 training recovery surcharge must be wired within five business days.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_banking_05", "Clause 6.1",
        "Adds an unitemized Rs. 70,000 surcharge on top of relocation reimbursement for early exit.",
    ),
    (
        "A minimum lock-in obligation of fifteen months binds the design intern; any premature separation triggers forfeiture of accrued stipends and a Rs. 55,000 breach charge.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 3.2",
        "Combines unpaid stipend forfeiture with a Rs. 55,000 breach penalty for a 15-month lock-in.",
    ),
    (
        "Should the student hostel occupant vacate during the summer vacation window without prior warden clearance, a liquidated vacancy penalty of Rs. 30,000 shall apply.",
        "FINANCIAL_PENALTY_BOND", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 7.1",
        "Imposes a Rs. 30,000 vacancy fine on student residents leaving during semester breaks.",
    ),
    (
        "Where the company funds an external accredited certification costing over Rs. 20,000, the trainee agrees to reimburse the documented invoice value on a monthly pro-rata declining scale over 12 months.",
        "FINANCIAL_PENALTY_BOND", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 4.5",
        "Structured pro-rata reimbursement of documented external training costs; moderate financial lock-in.",
    ),
    (
        "If the student contractor abandons an active website milestone without seven days' handover notice, a fixed transition penalty of Rs. 15,000 may be offset against unpaid invoices.",
        "FINANCIAL_PENALTY_BOND", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 6.2",
        "Offsets a Rs. 15,000 transition penalty for abrupt freelance handover failure.",
    ),
    (
        "The robotics club shall forfeit its Rs. 50,000 hardware grant and pay a 20% administrative recovery charge if the prototype demonstration is absent from the sponsor expo.",
        "FINANCIAL_PENALTY_BOND", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 5.4",
        "Attaches grant clawback plus a 20% administrative surcharge to student hackathon deliverables.",
    ),
    (
        "Relocation reimbursement of up to Rs. 25,000 paid at joining shall be recovered proportionally if the graduate engineer voluntarily resigns within six months of onboarding.",
        "FINANCIAL_PENALTY_BOND", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 3.4",
        "Proportionate short-term relocation clawback; moderate financial exit obligation.",
    ),
    (
        "The engineering intern must execute an indemnity bond of EUR 2,000 guaranteed by a local bank before receiving access to the company's staging repositories.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 4.2",
        "Demands a EUR 2,000 bank-guaranteed bond merely to onboard onto software repositories.",
    ),
    (
        "Failure to attend mandatory weekend hackathons organized by the employer will trigger a Rs. 10,000 contractual breach fine per missed event deducted from the intern's stipend.",
        "FINANCIAL_PENALTY_BOND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 5.3",
        "Levies punitive Rs. 10,000 fines against an intern for missing weekend events.",
    ),
    (
        "Students withdrawing from the international exchange consortium after visa issuance must pay a flat withdrawal penalty of USD 1,800 beyond non-refundable airline tickets.",
        "FINANCIAL_PENALTY_BOND", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 8.1",
        "Imposes a steep USD 1,800 flat withdrawal fine above actual travel expenses.",
    ),
    (
        "A service continuity bond of Rs. 1,40,000 applies to every management trainee; early separation for any reason renders the entire sum recoverable as an undisputed debt.",
        "FINANCIAL_PENALTY_BOND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fmcg_08", "Clause 2.2",
        "Frames a Rs. 1,40,000 service bond as an automatic undisputed debt on early departure.",
    ),
    (
        "The student content creator agrees to a liquidated damage fee of Rs. 35,000 if published course notes are removed from the platform prior to the 12-month hosting commitment.",
        "FINANCIAL_PENALTY_BOND", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 7.1",
        "Penalizes student creators Rs. 35,000 for withdrawing study material before one year.",
    ),
    (
        "Early surrender of the hostel meal plan without dean approval incurs a flat contract cancellation fee of Rs. 18,000 regardless of remaining semester weeks.",
        "FINANCIAL_PENALTY_BOND", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_mess_04", "Clause 6.3",
        "Applies a flat Rs. 18,000 meal-plan cancellation charge with no pro-rata adjustment.",
    ),
    (
        "If the student hackathon host fails to display the gold sponsor's banner on all livestream overlays, a flat contractual penalty of Rs. 60,000 becomes payable within fourteen days.",
        "FINANCIAL_PENALTY_BOND", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 4.2",
        "Assesses disproportionate Rs. 60,000 liquidated damages for minor branding omissions.",
    ),

    # =========================================================================
    # 2. STIPEND_PAYMENT_TERMS (36 clauses)
    # =========================================================================
    (
        "2.2 Stipend payments are contingent entirely upon client milestone sign-off and receipt of client funds by the company, with no guaranteed monthly schedule. The Company reserves the unilateral right to withhold, delay, or reduce the monthly stipend at its sole discretion based on subjective evaluation of work output.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 2.2",
        "Conditions intern stipend on third-party client funds and allows arbitrary discretionary withholding.",
    ),
    (
        "2.2 Stipend payments are scheduled for the 10th business day of the following month; however, stipend disbursement is contingent upon client project acceptance and subjective manager satisfaction with weekly velocity.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 2.2",
        "Makes monthly stipend payout uncertain by tying it to client acceptance and subjective velocity scores.",
    ),
    (
        "2. Payment shall be disbursed 90 days in arrears following final approval of all project deliverables and event conclusion by the sponsor marketing lead. No advance payment shall be released prior to the event.",
        "STIPEND_PAYMENT_TERMS", "high", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 2",
        "Delays sponsorship payout 90 days post-event with zero advance funding for student organizers.",
    ),
    (
        "2.1 Payment of Rs. 600 per completed lesson shall be made upon technical acceptance.",
        "STIPEND_PAYMENT_TERMS", "low", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service", "Clause 2.1",
        "Defines clear per-lesson compensation rate for a student content developer.",
    ),
    (
        "3. Annual CTC is INR 6,50,000 per annum. Trainee remuneration is subject to statutory tax deduction at source (TDS). Performance bonus and incentives advertised during recruitment are non-binding and payable strictly at the sole and unappealable discretion of management.",
        "STIPEND_PAYMENT_TERMS", "medium", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 3",
        "States annual CTC and TDS while making advertised recruitment bonuses non-binding and discretionary.",
    ),
    (
        "2.1 The Company shall pay the Intern a monthly stipend of INR 18,000 (Rupees Eighteen Thousand only), subject to applicable statutory deductions, payable via NEFT to the Intern's bank account.",
        "STIPEND_PAYMENT_TERMS", "low", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 2.1",
        "Establishes base monthly stipend amount and electronic transfer payment method.",
    ),
    (
        "The research internship is completely unpaid; the student shall receive no stipend, housing allowance, commuter pass, or meal subsidy during the entire twenty-week placement.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 2.1",
        "Unpaid extended internship offering zero compensation or basic expense support.",
    ),
    (
        "Monthly stipend of Rs. 28,000 shall be credited directly to the intern's bank account on or before the 5th calendar day of each succeeding month without deductions for public holidays.",
        "STIPEND_PAYMENT_TERMS", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 2.1",
        "Clear, fixed monthly stipend schedule with predictable disbursement date.",
    ),
    (
        "Any unauthorized absence or sick leave exceeding one day in a calendar month triggers a punitive deduction equal to 250% of the intern's daily stipend rate.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 2.3",
        "Applies a disproportionate 250% wage deduction penalty for student sick leave.",
    ),
    (
        "The initial sixty days of training shall remain unpaid, and stipend accrual begins only after the intern clears an internal panel review conducted at management's convenience.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 2.2",
        "Withholds compensation for two full months pending an unscheduled internal review.",
    ),
    (
        "Company shall reimburse pre-approved intercity travel, hackathon registration fees, and lodging expenses within 14 calendar days of receiving digital receipts.",
        "STIPEND_PAYMENT_TERMS", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 2.4",
        "Fair, time-bound reimbursement commitment for approved student expenses.",
    ),
    (
        "The part-time student developer shall earn USD 22 per logged hour up to 20 hours weekly during academic semesters, settled bi-weekly via wire transfer.",
        "STIPEND_PAYMENT_TERMS", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 2.1",
        "Transparent hourly compensation aligned with part-time academic hour limits.",
    ),
    (
        "Fifty percent of the monthly trainee salary shall be retained in a deferred retention pool and released only upon completing eighteen continuous months of service.",
        "STIPEND_PAYMENT_TERMS", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 2.3",
        "Withholds half of earned monthly wages as a disguised retention lock-in.",
    ),
    (
        "Hours worked beyond 45 hours per week, including weekend release deployments, shall neither attract overtime pay nor qualify for compensatory time off.",
        "STIPEND_PAYMENT_TERMS", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 3.2",
        "Mandates uncompensated overtime and weekend work without compensatory leave.",
    ),
    (
        "Hostel rent of Rs. 9,500 per month is payable quarterly in advance by the 7th of the first month; payments delayed beyond ten days attract a compounding late fee of Rs. 500 per day.",
        "STIPEND_PAYMENT_TERMS", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 2.1",
        "Imposes steep daily late payment charges of Rs. 500/day on quarterly hostel rent.",
    ),
    (
        "Sponsor shall release 50% of the INR 2,00,000 title sponsorship upon signing this MOU and the remaining 50% within 10 working days after submission of the post-event photo report.",
        "STIPEND_PAYMENT_TERMS", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 2.1",
        "Balanced two-tranche sponsorship disbursement providing upfront working capital to students.",
    ),
    (
        "All freelance invoices submitted by the student designer are payable on net-120 day terms and may be reduced unilaterally if end-client engagement metrics fall below target.",
        "STIPEND_PAYMENT_TERMS", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 3.1",
        "Combines excessive 120-day payment delay with unilateral reductions tied to client metrics.",
    ),
    (
        "The teaching assistant shall receive a consolidated monthly fellowship of Rs. 16,000 credited on the last working day of each month alongside statutory TDS statements.",
        "STIPEND_PAYMENT_TERMS", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 2.1",
        "Predictable monthly academic fellowship schedule with transparent tax documentation.",
    ),
    (
        "Management retains absolute authority to convert the paid internship into an unpaid credit-only engagement at any point during the quarter if startup funding rounds are delayed.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 2.4",
        "Allows retroactive or unilateral elimination of stipend based on company fundraising.",
    ),
    (
        "Commission payouts to student campus ambassadors shall be disbursed only once accumulated verified referrals exceed Rs. 15,000; balances below that threshold expire each semester.",
        "STIPEND_PAYMENT_TERMS", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 2.2",
        "High payout threshold paired with semester-end forfeiture of earned student commissions.",
    ),
    (
        "Graduate engineers shall receive base salary via direct deposit on the final banking day of every month, plus an itemized payslip detailing provident fund and tax deductions.",
        "STIPEND_PAYMENT_TERMS", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 2.1",
        "Standard, transparent monthly salary disbursement with itemized statutory deductions.",
    ),
    (
        "Electricity and air-conditioning charges in the student residence shall be billed monthly at Rs. 18 per kWh—significantly above local utility tariffs—with no sub-meter inspection right.",
        "STIPEND_PAYMENT_TERMS", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 2.4",
        "Inflated private utility billing with no verification access for student residents.",
    ),
    (
        "The EdTech platform shall settle verified student tutor earnings twice monthly on the 1st and 16th via UPI or bank transfer with zero minimum withdrawal threshold.",
        "STIPEND_PAYMENT_TERMS", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 2.1",
        "Student-friendly bi-monthly payout schedule without minimum balance traps.",
    ),
    (
        "Variable joining bonus of Rs. 1,00,000 shall be paid only after completing twelve months of satisfactory employment and is subject to full recovery if the employee exits within twenty-four months.",
        "STIPEND_PAYMENT_TERMS", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_consulting_03", "Clause 3.1",
        "Delayed bonus disbursement coupled with a two-year clawback condition.",
    ),
    (
        "Sponsor contributions shall be delivered exclusively as redeemable merchandise vouchers rather than cash unless student registrations exceed 5,000 unique verified accounts.",
        "STIPEND_PAYMENT_TERMS", "medium", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 2.2",
        "Replaces cash sponsorship with restrictive vouchers unless aggressive registration hurdles are met.",
    ),
    (
        "Student research stipend payments pause automatically during university examination breaks exceeding five days, even when remote literature review continues.",
        "STIPEND_PAYMENT_TERMS", "medium", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 3.1",
        "Suspends research stipend during exam weeks despite ongoing project work.",
    ),
    (
        "If an invoice submitted by the student videographer contains any formatting discrepancy, the 60-day payment clock resets completely from the date of resubmission.",
        "STIPEND_PAYMENT_TERMS", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 2.3",
        "Uses technical invoice resets to delay student contractor payments indefinitely.",
    ),
    (
        "The university hostel reserves the right to revise monthly mess and room charges upward by up to 25% mid-semester without prior written consultation with residents.",
        "STIPEND_PAYMENT_TERMS", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_mess_04", "Clause 2.2",
        "Permits unilateral mid-semester rent and mess fee hikes up to 25%.",
    ),
    (
        "Trainees relocating from outside the state shall receive a one-time settling-in allowance of Rs. 15,000 disbursed alongside their first month's payroll.",
        "STIPEND_PAYMENT_TERMS", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 2.2",
        "Clear relocation allowance benefit with explicit payout timing.",
    ),
    (
        "Payment for student hackathon prize pools shall be issued within 30 business days of winner verification via direct bank transfer after collecting PAN and bank details.",
        "STIPEND_PAYMENT_TERMS", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 2.3",
        "Reasonable, clearly documented prize disbursement schedule.",
    ),
    (
        "The bootcamp installment plan requires automated NACH debit authorization; failed debits incur Rs. 1,500 bounce charges plus 24% annualized interest from the due date.",
        "STIPEND_PAYMENT_TERMS", "medium", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 2.2",
        "Attaches high bounce penalties and 24% interest to student tuition installments.",
    ),
    (
        "Intern compensation consists exclusively of unvested startup equity options with a four-year cliff and zero cash stipend during the six-month summer engagement.",
        "STIPEND_PAYMENT_TERMS", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 2.1",
        "Substitutes illiquid unvested equity with a four-year cliff for cash intern compensation.",
    ),
    (
        "Monthly stipend shall be prorated strictly according to actual calendar days served in the first and final months, paid via bank transfer within seven working days of handover.",
        "STIPEND_PAYMENT_TERMS", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 2.2",
        "Fair, transparent proration and prompt final settlement rule.",
    ),
    (
        "Hostel tuition-linked fee rebates shall be credited directly to the resident's ledger within 15 days of submitting valid merit scholarship confirmation.",
        "STIPEND_PAYMENT_TERMS", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 2.3",
        "Predictable scholarship fee adjustment process for hostel students.",
    ),
    (
        "Disputed timesheet entries submitted by the student contractor will freeze payment of the entire monthly invoice—not merely the contested hours—until resolved.",
        "STIPEND_PAYMENT_TERMS", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 2.4",
        "Allows withholding of undisputed student earnings whenever a small subset of hours is questioned.",
    ),
    (
        "The employer may deduct the full retail replacement price of any damaged office peripheral directly from the trainee's monthly payroll without depreciation or written consent.",
        "STIPEND_PAYMENT_TERMS", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_core_02", "Clause 2.5",
        "Permits unilateral full-retail payroll deductions without depreciation or consent.",
    ),

    # =========================================================================
    # 3. DEPOSIT_REFUND (36 clauses)
    # =========================================================================
    (
        "2.1 The Resident shall pay a security deposit of Rs. 35,000 upon hostel admission. The deposit is strictly non-refundable under any circumstances whatsoever.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 2.1",
        "Declares a Rs. 35,000 hostel security deposit strictly non-refundable under all circumstances.",
    ),
    (
        "2.2 If the student leaves the hostel residency before the end of the academic year, the entire security deposit and remaining annual room rent shall be forfeited.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 2.2",
        "Forfeits both the full security deposit and all remaining annual rent upon early departure.",
    ),
    (
        "2.3 The hostel management reserves the right to deduct arbitrary amounts from the caution deposit for general campus maintenance without itemized proof.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 2.3",
        "Authorizes arbitrary, unitemized deductions from the student's caution deposit.",
    ),
    (
        "2.4 A mandatory non-refundable facility renovation fee of Rs. 10,000 will be retained from the security deposit upon student check-out.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 2.4",
        "Automatically retains Rs. 10,000 for general renovation regardless of actual room condition.",
    ),
    (
        "3.1 Prior to onboarding, the Intern shall pay a training and equipment security deposit of INR 15,000 (Rupees Fifteen Thousand only) via bank transfer.",
        "DEPOSIT_REFUND", "medium", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 3.1",
        "Requires an upfront INR 15,000 security deposit from a student intern before starting work.",
    ),
    (
        "3.2 Refund Conditions: Subject to full completion of the 6-month internship duration, clearance of all no-dues checklists, and satisfactory evaluation, the deposit shall be refunded within 90 days of exit.",
        "DEPOSIT_REFUND", "medium", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 3.2",
        "Delays deposit refund up to 90 days and conditions return on subjective performance evaluation.",
    ),
    (
        "3.3 Forfeiture Condition: In the event the Intern withdraws before completing the 6-month term, is dismissed for disciplinary reasons, or fails to meet internal performance metrics, the entire security deposit shall be forfeited to the Company.",
        "DEPOSIT_REFUND", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 3.3",
        "Allows total deposit forfeiture for subjective performance ratings or early withdrawal.",
    ),
    (
        "5.2 In the event of cancellation prior to move-in date, 50% of the deposit shall be retained as a booking cancellation penalty.",
        "DEPOSIT_REFUND", "medium", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 5.2",
        "Retains half of the hostel deposit if a student cancels prior to moving in.",
    ),
    (
        "The caution money of Rs. 20,000 shall be refunded in full to the resident's bank account within 15 business days of key handover following a signed joint room inspection.",
        "DEPOSIT_REFUND", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 3.1",
        "Balanced 15-day deposit refund term anchored to a joint written inspection.",
    ),
    (
        "Deductions from the student's hostel deposit may only cover documented physical damage beyond ordinary wear and tear, evidenced by timestamped move-in and move-out photographs.",
        "DEPOSIT_REFUND", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 3.2",
        "Protects students against arbitrary deductions by requiring photographic proof and excluding fair wear.",
    ),
    (
        "A flat repainting, mattress sanitization, and deep-cleaning charge of Rs. 8,500 shall be automatically deducted from every resident's security deposit regardless of occupancy length.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 3.2",
        "Imposes automatic flat deductions for normal turnover maintenance from the deposit.",
    ),
    (
        "Refund of the university hostel caution deposit requires physical clearance signatures from nine campus offices within 14 days of final exams, failing which the deposit lapses permanently.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_campus_02", "Clause 3.4",
        "Creates burdensome multi-office clearance hurdles with permanent forfeiture if missed.",
    ),
    (
        "Before receiving a company laptop, the student trainee must remit a Rs. 25,000 hardware caution deposit refundable only 120 days after returning the device in factory-sealed condition.",
        "DEPOSIT_REFUND", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 3.1",
        "Requires an upfront Rs. 25,000 laptop deposit with a 120-day delay and impossible factory-sealed standard.",
    ),
    (
        "If the student resident fails to claim the refundable room deposit within 45 days of semester end, the management may absorb the full balance into its general welfare fund.",
        "DEPOSIT_REFUND", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 3.3",
        "Sets an unfairly short 45-day forfeiture window for unclaimed student deposits.",
    ),
    (
        "The student club must lodge a Rs. 40,000 auditorium damage deposit with the venue operator; refunds will be processed within 21 working days minus actual verified repair invoices.",
        "DEPOSIT_REFUND", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 3.1",
        "Reasonable venue deposit with clear 21-day refund timeline and invoice-backed deductions.",
    ),
    (
        "Any breach of hostel curfew or visitor guidelines automatically empowers the warden to forfeit 100% of the resident's Rs. 30,000 security deposit without a disciplinary hearing.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_campus_02", "Clause 3.1",
        "Weaponizes full deposit forfeiture for minor hostel rules without due process.",
    ),
    (
        "Students enrolling in the coding fellowship must pay a Rs. 12,000 seat-blocking deposit that is strictly non-refundable even if the program curriculum or batch dates are altered.",
        "DEPOSIT_REFUND", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 4.1",
        "Retains seat deposit even when the provider unilaterally changes batch dates or curriculum.",
    ),
    (
        "Lost RFID access cards or room keys incur an automatic Rs. 6,000 replacement deduction from the security deposit, irrespective of actual locksmith replacement cost.",
        "DEPOSIT_REFUND", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 3.5",
        "Charges a disproportionate Rs. 6,000 key replacement fee far above actual replacement cost.",
    ),
    (
        "Security deposit refunds for off-campus co-living rooms shall be credited via NEFT within 10 working days of move-out, accompanied by an itemized settlement ledger.",
        "DEPOSIT_REFUND", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 3.1",
        "Student-protective 10-day electronic deposit refund with itemized statement.",
    ),
    (
        "The placement training agency shall retain 100% of the Rs. 20,000 refundable placement guarantee deposit if the student misses a single mock interview or declines any listed role.",
        "DEPOSIT_REFUND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_agency_09", "Clause 3.2",
        "Uses strict technicalities to forfeit a student's supposedly refundable placement deposit.",
    ),
    (
        "Laboratory glassware and sensor kit caution money of Rs. 5,000 will be returned upon semester completion, subject solely to verified breakage priced at institutional procurement cost.",
        "DEPOSIT_REFUND", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 4.2",
        "Fair academic lab deposit policy limited to verified procurement-cost breakage.",
    ),
    (
        "If the student cancels hostel admission at least 15 days prior to check-in due to university counseling reallocation, 95% of the security deposit and advance rent shall be refunded within 10 days.",
        "DEPOSIT_REFUND", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 3.4",
        "Student-friendly cancellation refund rule accommodating university admission reallocations.",
    ),
    (
        "Hostel management may hold the resident's security deposit indefinitely without interest until a replacement tenant signs a twelve-month lease for the vacated bed.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 3.5",
        "Indefinitely blocks deposit refund by shifting vacancy risk onto the departing student.",
    ),
    (
        "An equipment usage deposit of Rs. 10,000 paid by the student videographer shall be refunded within 7 business days after returning cameras and microphones in working order.",
        "DEPOSIT_REFUND", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 3.2",
        "Clear, short 7-day refund window for returned student contractor gear.",
    ),
    (
        "The exhibition stall booking deposit of Rs. 25,000 paid by the student entrepreneurship cell is non-refundable even if the venue operator cancels the expo for commercial reasons.",
        "DEPOSIT_REFUND", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 3.3",
        "Unfairly withholds student deposit even when cancellation is caused by the counterparty.",
    ),
    (
        "No security deposit or caution fee shall be collected from the intern as a condition of receiving company software credentials or onboarding equipment.",
        "DEPOSIT_REFUND", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 3.1",
        "Explicitly protects interns from upfront security deposit demands.",
    ),
    (
        "Refund of the co-living security deposit is delayed until 60 days post-vacation and subject to a mandatory 15% administrative processing deduction.",
        "DEPOSIT_REFUND", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_mess_04", "Clause 3.2",
        "Combines a 60-day refund delay with an unjustified 15% processing deduction.",
    ),
    (
        "The library and maker-space caution deposit of Rs. 3,000 shall be refunded automatically to the graduating student's registered UPI account upon clearing overdue book records.",
        "DEPOSIT_REFUND", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 3.3",
        "Transparent automated refund policy for campus maker-space caution deposits.",
    ),
    (
        "In case a group of four students shares a hostel suite, management may deduct collective corridor maintenance charges from all residents' deposits without establishing individual responsibility.",
        "DEPOSIT_REFUND", "medium", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_mess_04", "Clause 3.4",
        "Applies collective unattributed deposit deductions across all student roommates.",
    ),
    (
        "The overseas study agency's USD 1,000 housing reservation deposit becomes non-refundable 48 hours after payment, even if the student's study visa is refused by the embassy.",
        "DEPOSIT_REFUND", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 3.2",
        "Forfeits USD 1,000 student housing deposit despite embassy visa refusal outside student control.",
    ),
    (
        "Security deposit refunds shall bear interest at 6% per annum if the hostel operator delays repayment beyond thirty calendar days after completed handover.",
        "DEPOSIT_REFUND", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 3.5",
        "Incentivizes timely deposit return by awarding students interest on late refunds.",
    ),
    (
        "The tutoring portal retains 20% of the student tutor's first five payouts as a rolling quality deposit, forfeited automatically if average student ratings dip below 4.6 stars.",
        "DEPOSIT_REFUND", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 3.4",
        "Seizes a rolling earnings deposit based on volatile subjective star ratings.",
    ),
    (
        "A refundable uniform and safety-gear deposit of Rs. 4,000 collected from the industrial trainee shall be returned on the final day of plant training upon gear return.",
        "DEPOSIT_REFUND", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 3.3",
        "Prompt same-day refund of a modest industrial safety-gear deposit.",
    ),
    (
        "If the student tenant transfers to another hostel block mid-term, the existing security deposit is forfeited rather than transferred to the newly allotted room.",
        "DEPOSIT_REFUND", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 3.6",
        "Unfairly forfeits deposit during an internal room transfer within the same hostel.",
    ),
    (
        "Sponsorship registration deposits collected from participating college teams shall be refunded 100% within 7 days if the organizer cancels or reschedules the competition.",
        "DEPOSIT_REFUND", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 3.2",
        "Clear full-refund guarantee when competition organizers cancel or reschedule.",
    ),
    (
        "The company requires a Rs. 15,000 background-verification security deposit from selected campus hires, refundable only after completion of the 12-month probation window.",
        "DEPOSIT_REFUND", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_agency_09", "Clause 3.4",
        "Shifts routine employer verification costs onto campus hires as a one-year locked deposit.",
    ),

    # =========================================================================
    # 4. TERMINATION_NOTICE (36 clauses)
    # =========================================================================
    (
        "7.1 The Company may terminate this internship agreement at any time, with or without cause, immediately upon written or verbal notice, without any compensation.",
        "TERMINATION_NOTICE", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 7.1",
        "Grants employer unilateral immediate termination without cause, written notice, or compensation.",
    ),
    (
        "7.2 The intern must provide 90 days written notice to resign, during which no leave may be availed, whereas the employer may dismiss the intern instantly without notice or severance pay.",
        "TERMINATION_NOTICE", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 7.2",
        "Imposes severely asymmetric 90-day intern notice with leave ban vs instant employer firing.",
    ),
    (
        "5.1 Termination by Company: The Company may terminate this Agreement at any time, with or without cause, immediately upon written notice and without severance or notice pay.",
        "TERMINATION_NOTICE", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 5.1",
        "Allows immediate without-cause termination by the company with zero notice pay.",
    ),
    (
        "5.2 Termination by Intern: The Intern must provide at least 60 (sixty) days prior written notice to terminate this Agreement. No academic exam leave may be taken during the notice period.",
        "TERMINATION_NOTICE", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 5.2",
        "Requires 60 days' intern notice and bans university exam leave during notice.",
    ),
    (
        "6.1 Either party may terminate this agreement by providing thirty (30) days prior written notice, or payment of stipend in lieu thereof.",
        "TERMINATION_NOTICE", "low", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 6.1",
        "Balanced mutual 30-day written notice clause with buyout option.",
    ),
    (
        "6.2 Immediate summary dismissal shall apply without right of appeal in cases of alleged misconduct, breach of confidentiality, or insubordination.",
        "TERMINATION_NOTICE", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 6.2",
        "Permits summary dismissal upon mere allegation with no hearing or appeal right.",
    ),
    (
        "3.2 Immediate summary dismissal and eviction shall apply without right of appeal in cases of alleged misconduct, noise complaints, or insubordination to the warden.",
        "TERMINATION_NOTICE", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 3.2",
        "Enables immediate hostel eviction over minor noise complaints without appeal.",
    ),
    (
        "5.1 The resident may terminate hostel occupancy at the end of the semester by submitting a one-month advance written notice to the warden.",
        "TERMINATION_NOTICE", "low", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 5.1",
        "Standard semester-end hostel notice procedure for students.",
    ),
    (
        "Either party may end this internship for convenience by delivering fourteen (14) calendar days' written email notice, with full pro-rata stipend paid through the final working day.",
        "TERMINATION_NOTICE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 6.1",
        "Symmetric 14-day notice ensuring earned stipend is settled through the last day.",
    ),
    (
        "In the event of documented medical emergencies or mandatory university examination conflicts, the student may exit the engagement upon 7 calendar days' written notice without penalty.",
        "TERMINATION_NOTICE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 7.1",
        "Student-protective academic and medical emergency exit right.",
    ),
    (
        "Absence from the office for two consecutive business days without prior HR portal approval constitutes automatic job abandonment, forfeiting all accrued unpaid stipend.",
        "TERMINATION_NOTICE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 7.2",
        "Treats brief absence as automatic abandonment with forfeiture of already-earned wages.",
    ),
    (
        "The employer may revoke the campus placement offer at any time prior to the joining date without notice or compensation, even after the student has withdrawn from campus placement drives.",
        "TERMINATION_NOTICE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 6.1",
        "Leaves graduating students stranded by allowing uncompensated pre-joining offer revocation.",
    ),
    (
        "During the six-month probationary term, either party may terminate employment with fifteen (15) days' written notice, accompanied by written performance feedback if initiated by the employer.",
        "TERMINATION_NOTICE", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 6.1",
        "Fair mutual probationary notice paired with transparent performance feedback.",
    ),
    (
        "The graduate trainee is bound by a 90-day mandatory notice period with no option to pay salary in lieu of notice, and management may extend the notice period unilaterally by another 30 days.",
        "TERMINATION_NOTICE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_consulting_03", "Clause 6.2",
        "Traps trainees in up to 120 days of non-buyable notice at management's sole discretion.",
    ),
    (
        "Hostel management must issue at least 30 days' written eviction notice specifying concrete grounds and provide a 7-day opportunity to cure non-severe violations before terminating residency.",
        "TERMINATION_NOTICE", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 5.1",
        "Ensures due process, written notice, and cure window before any hostel eviction.",
    ),
    (
        "If the warden cancels room allotment on 24 hours' verbal notice for any perceived breach of house decorum, the resident must vacate immediately without refund of prepaid rent.",
        "TERMINATION_NOTICE", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 5.1",
        "Allows 24-hour verbal eviction and forfeiture of prepaid hostel rent.",
    ),
    (
        "The corporate sponsor may cancel its funding commitment up to 48 hours before the symposium for any shifting marketing priority without reimbursing expenses already incurred by the student club.",
        "TERMINATION_NOTICE", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 5.1",
        "Permits last-minute sponsor pullout 48 hours pre-event while leaving students with sunk costs.",
    ),
    (
        "Either the sponsor or the student organizing committee may terminate this sponsorship MOU with 30 days' prior written notice, provided documented non-cancellable vendor costs are settled fairly.",
        "TERMINATION_NOTICE", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 6.1",
        "Balanced cancellation clause protecting student organizers against sunk vendor expenses.",
    ),
    (
        "The platform may deactivate the student tutor's account immediately without explanation or appeal while freezing pending payout balances for 180 days.",
        "TERMINATION_NOTICE", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 6.1",
        "Unilateral account termination combined with a 180-day freeze on earned tutor fees.",
    ),
    (
        "Either party may end this freelance student engagement upon 10 business days' written notice, and all completed milestones delivered prior to termination shall be paid in full.",
        "TERMINATION_NOTICE", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 6.1",
        "Mutual 10-day termination notice guaranteeing full payment for completed work.",
    ),
    (
        "This agreement may be dissolved at any time by mutual written agreement signed electronically by both the student fellow and the faculty supervisor.",
        "TERMINATION_NOTICE", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 6.1",
        "Standard consensual termination clause preserving flexibility for both parties.",
    ),
    (
        "Failure to pass any unannounced technical audit during probation allows instant discharge without notice pay, relieving letter, or experience certificate.",
        "TERMINATION_NOTICE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 6.3",
        "Denies relieving documentation and notice pay upon subjective unannounced testing.",
    ),
    (
        "The student resident may terminate the co-living contract mid-year by giving 30 days' written notice via the resident portal without forfeiting the security deposit.",
        "TERMINATION_NOTICE", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 5.2",
        "Fair mid-year exit option with reasonable 30-day notice and deposit protection.",
    ),
    (
        "Should the startup undergo acquisition or restructuring, the intern's agreement terminates automatically on zero notice while all post-termination restrictions remain fully binding.",
        "TERMINATION_NOTICE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 6.2",
        "Asymmetric zero-notice termination on restructuring while keeping restrictive covenants alive.",
    ),
    (
        "Notice of termination by the student contractor is ineffective unless submitted on physical stamped paper sent by registered post to the company's overseas headquarters.",
        "TERMINATION_NOTICE", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 6.4",
        "Creates artificial procedural obstacles to impede valid student resignation notice.",
    ),
    (
        "The university exchange coordinator may withdraw program nomination upon 14 days' written notice if the student fails to maintain minimum academic standing, subject to a formal appeal panel.",
        "TERMINATION_NOTICE", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 6.1",
        "Clear academic-standing exit rule backed by written notice and formal appeal.",
    ),
    (
        "Employer reserves the sole right to place the trainee on unpaid indefinite furlough without terminating the contract, barring the trainee from accepting alternative employment.",
        "TERMINATION_NOTICE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_banking_05", "Clause 6.4",
        "Locks trainee into unpaid indefinite furlough without pay or freedom to leave.",
    ),
    (
        "If the company fails to disburse monthly stipend within 15 days of the scheduled pay date, the intern may terminate immediately for good cause without serving notice or incurring penalty.",
        "TERMINATION_NOTICE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 6.3",
        "Protects students by allowing immediate penalty-free exit when stipend is unpaid.",
    ),
    (
        "Cancellation of the coding bootcamp by the student within the first seven days entitles the student to a 100% tuition refund processed within ten working days.",
        "TERMINATION_NOTICE", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 5.1",
        "Fair seven-day cooling-off cancellation policy with full tuition refund.",
    ),
    (
        "The student ambassador must provide 45 days' advance written notice to step down, whereas the brand may revoke ambassador status immediately via text message.",
        "TERMINATION_NOTICE", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 6.1",
        "Asymmetric 45-day student notice compared to instant company cancellation via text.",
    ),
    (
        "Upon expiration of the fixed six-month term, this internship concludes automatically unless extended in writing by mutual consent signed at least 14 days prior to the end date.",
        "TERMINATION_NOTICE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 6.1",
        "Clean automatic expiry rule preventing unwanted auto-renewal lock-ins.",
    ),
    (
        "This hostel residency agreement automatically renews for a full twelve-month term unless the student submits a notarized opt-out letter ninety days before semester completion.",
        "TERMINATION_NOTICE", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 5.1",
        "Predatory auto-renewal trap requiring notarized notice three months ahead.",
    ),
    (
        "Termination for material breach requires written specification of the breach and a 15-day cure period before either the club or sponsor may cancel the event partnership.",
        "TERMINATION_NOTICE", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 5.1",
        "Balanced breach-and-cure termination procedure for student event contracts.",
    ),
    (
        "Company may extend the trainee's probation period repeatedly at its sole discretion, during which instant dismissal without severance remains continuously applicable.",
        "TERMINATION_NOTICE", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fmcg_08", "Clause 6.2",
        "Indefinite probation extensions prolonging asymmetric at-will dismissal exposure.",
    ),
    (
        "If the student's university withdraws internship credit authorization, the student may conclude the placement on 10 days' written notice with a certificate reflecting actual weeks completed.",
        "TERMINATION_NOTICE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 6.2",
        "Aligns internship exit fairly with university academic credit requirements.",
    ),
    (
        "Discontinuation of laboratory access by the faculty supervisor requires written safety or compliance findings and seven days' notice to archive student research data.",
        "TERMINATION_NOTICE", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 5.2",
        "Reasonable lab access termination clause preserving student data backup rights.",
    ),

    # =========================================================================
    # 5. CONFIDENTIALITY_NDA (36 clauses)
    # =========================================================================
    (
        "6.1 The Intern agrees to keep secret and confidential all technical data, client lists, algorithms, source code, and commercial secrets indefinitely.",
        "CONFIDENTIALITY_NDA", "medium", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 6.1",
        "Imposes a perpetual, time-unlimited NDA across broad categories of information.",
    ),
    (
        "6.2 Intern is strictly prohibited from mentioning the company name, project title, or technologies used on their CV, LinkedIn profile, or GitHub repository without prior written CEO permission.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 6.2",
        "Blocks students from listing legitimate internship experience and tech stack on resumes.",
    ),
    (
        "6.3 Gag clause: The intern agrees never to post negative comments, reviews, or Glassdoor ratings regarding the company or its executives.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 6.3",
        "Broad non-disparagement gag prohibiting truthful workplace reviews.",
    ),
    (
        "6.2 Portfolio and Gag Restriction: The Intern is strictly prohibited from mentioning the Company name, project architecture, or technologies used on their resume, CV, LinkedIn profile, or academic viva presentation, and agrees never to post negative comments or Glassdoor reviews about the Company.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 6.2",
        "Prevents academic viva presentation, resume listing, and public commentary.",
    ),
    (
        "5. Gag clause: The intern agrees never to post negative comments, reviews, or Glassdoor ratings regarding the company or its executives. The student shall not discuss compensation or dispute terms online.",
        "CONFIDENTIALITY_NDA", "high", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service", "Clause 5",
        "Combines a permanent non-disparagement gag with pay and dispute secrecy.",
    ),
    (
        "6.1 The Intern agrees to hold in strict confidence all proprietary source code, data pipelines, client lists, pricing models, and business strategies indefinitely, even after termination of this Agreement.",
        "CONFIDENTIALITY_NDA", "medium", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 6.1",
        "Perpetual confidentiality obligation with no expiration window or public-domain carve-out.",
    ),
    (
        "Confidentiality obligations apply solely to non-public documents explicitly marked 'Confidential' or reasonably understood as trade secrets, for a period of two (2) years following completion.",
        "CONFIDENTIALITY_NDA", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 5.1",
        "Well-scoped, time-bounded two-year NDA protecting genuine trade secrets.",
    ),
    (
        "Confidential Information expressly excludes information already in the public domain, independently developed by the student without company data, or required to be disclosed by law.",
        "CONFIDENTIALITY_NDA", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 5.2",
        "Includes essential standard NDA exclusions protecting students from overbroad claims.",
    ),
    (
        "The student may accurately list the Company name, role title, internship dates, and high-level non-proprietary technologies used on resumes, LinkedIn profiles, and university placement portfolios.",
        "CONFIDENTIALITY_NDA", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 5.2",
        "Explicitly safeguards student career portfolio and resume disclosure rights.",
    ),
    (
        "The trainee shall not disclose salary figures, bond terms, working hours, or workplace grievances to university placement officers, peers, or regulatory authorities.",
        "CONFIDENTIALITY_NDA", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 7.1",
        "Silences students from discussing pay or reporting unfair contract terms to campus placement cells.",
    ),
    (
        "The student researcher shall not submit any academic thesis, conference paper, or open preprint derived from internship work without obtaining irrevocable written clearance from the Chief Legal Officer.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 5.3",
        "Gives corporate counsel open-ended veto power over a student's university thesis defense.",
    ),
    (
        "Company shall review the student's draft university internship report within 10 business days solely to redact proprietary customer identifiers and shall not unreasonably withhold academic submission approval.",
        "CONFIDENTIALITY_NDA", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 5.2",
        "Balances corporate secrecy with timely approval for university academic grading.",
    ),
    (
        "Upon concluding the engagement, the intern must return all issued laptops, access badges, and hardware tokens, and permanently delete local clones of private repositories.",
        "CONFIDENTIALITY_NDA", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 5.1",
        "Standard, reasonable post-exit return of company devices and proprietary repositories.",
    ),
    (
        "Any unauthorized screenshot, email forwarding, or cloud backup of internal customer datasets constitutes a material confidentiality breach entitling the company to immediate injunctive relief.",
        "CONFIDENTIALITY_NDA", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_banking_05", "Clause 5.1",
        "Protects sensitive customer datasets while introducing injunctive relief exposure.",
    ),
    (
        "Residents are strictly forbidden from recording audio or photographs of hostel dining conditions, maintenance defects, or warden interactions and posting them on social media or student forums.",
        "CONFIDENTIALITY_NDA", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 6.1",
        "Imposes an anti-transparency gag preventing hostel residents from documenting living conditions.",
    ),
    (
        "Hostel management shall treat student identity records, emergency parent contacts, and medical declarations as strictly confidential and share them only with authorized campus authorities.",
        "CONFIDENTIALITY_NDA", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 6.1",
        "Protects student residents' personal and medical privacy data.",
    ),
    (
        "The student organizing committee agrees not to disclose the sponsor's custom pricing tier or internal marketing campaign roadmap to rival sponsors for twelve months following the festival.",
        "CONFIDENTIALITY_NDA", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 4.1",
        "Narrow, time-limited commercial confidentiality appropriate for event sponsorship.",
    ),
    (
        "Student club members must execute perpetual individual non-disclosure and non-disparagement deeds prohibiting any public critique of the sponsor's products for life.",
        "CONFIDENTIALITY_NDA", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 4.2",
        "Demands lifetime personal gag agreements from volunteer student club organizers.",
    ),
    (
        "The student contractor shall keep confidential all unpublished exam question banks and proprietary curriculum scripts provided by the EdTech client for three years.",
        "CONFIDENTIALITY_NDA", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 5.1",
        "Reasonable protection of proprietary educational question banks for a defined period.",
    ),
    (
        "The freelance student developer is barred for five years from showing redacted UI wireframes or describing general software problem-solving methods used on the project during job interviews.",
        "CONFIDENTIALITY_NDA", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 5.2",
        "Overbroad five-year secrecy ban crippling a student's ability to interview for jobs.",
    ),
    (
        "Research assistants handling human-subject survey datasets must anonymize all personal identifiers and store files exclusively on encrypted university servers per ethics board protocols.",
        "CONFIDENTIALITY_NDA", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 5.1",
        "Standard academic data privacy and IRB confidentiality compliance.",
    ),
    (
        "A breach of any confidentiality or non-disparagement provision triggers automatic liquidated damages of Rs. 5,00,000 without the employer needing to prove actual commercial loss.",
        "CONFIDENTIALITY_NDA", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_consulting_03", "Clause 5.3",
        "Attaches massive Rs. 5,00,000 automatic damages to broad NDA and gag terms.",
    ),
    (
        "Both parties agree to maintain mutual confidentiality regarding non-public business plans, API credentials, and personal financial details exchanged during the engagement for two years.",
        "CONFIDENTIALITY_NDA", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 5.1",
        "Reciprocal, balanced two-year confidentiality clause protecting both sides.",
    ),
    (
        "The student bootcamp participant may not share, resell, or publicly upload proprietary recorded workshop videos or paid assignment solutions to public GitHub repositories.",
        "CONFIDENTIALITY_NDA", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 6.1",
        "Reasonable protection of paid course recordings and proprietary assessments.",
    ),
    (
        "Interns must install continuous keystroke-logging and screen-capture surveillance software on their personal laptops to monitor compliance with internal data secrecy rules.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 6.2",
        "Invasive personal-device surveillance justified under confidentiality enforcement.",
    ),
    (
        "Employee agrees that all general industry know-how, software design patterns, and architectural skills learned during employment constitute confidential trade secrets of the employer.",
        "CONFIDENTIALITY_NDA", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 5.2",
        "Improperly classifies general professional skills and know-how as employer trade secrets.",
    ),
    (
        "Nothing in this non-disclosure section restricts the student from reporting unlawful conduct, harassment, or statutory labor violations to competent government authorities.",
        "CONFIDENTIALITY_NDA", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 5.3",
        "Protects whistleblower and statutory reporting rights within an NDA.",
    ),
    (
        "The campus ambassador shall not reveal unreleased product pricing sheets or promotional coupon algorithms shared prior to official launch dates.",
        "CONFIDENTIALITY_NDA", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 5.2",
        "Targeted pre-launch embargo appropriate for campus marketing programs.",
    ),
    (
        "The intern agrees not to discuss any aspect of their daily tasks with faculty advisors, even at a conceptual summary level required for mandatory university internship grading.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 6.1",
        "Conflicts directly with mandatory university faculty supervision and credit evaluation.",
    ),
    (
        "Participants in the sponsored university hackathon retain full freedom to publish open-source code created during the 36-hour event, excluding confidential sponsor API keys.",
        "CONFIDENTIALITY_NDA", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 4.3",
        "Preserves open-source hackathon portfolio rights while safeguarding API secrets.",
    ),
    (
        "Resident student agrees that all correspondence regarding hostel maintenance complaints or deposit disputes is strictly confidential and may never be shared with the student union.",
        "CONFIDENTIALITY_NDA", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 6.2",
        "Blocks student residents from seeking support from campus student unions.",
    ),
    (
        "Confidentiality obligations survive termination for three (3) years with respect to proprietary algorithms and client pricing, while trade secrets remain protected as long as legally non-public.",
        "CONFIDENTIALITY_NDA", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 5.1",
        "Standard commercial NDA duration distinguishing general confidential data from trade secrets.",
    ),
    (
        "The student agrees not to upload confidential laboratory schematics or unpublished clinical datasets to third-party generative AI prompts or external personal cloud drives.",
        "CONFIDENTIALITY_NDA", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 6.1",
        "Sensible modern data-hygiene rule safeguarding unpublished research data.",
    ),
    (
        "The exchange scholar shall maintain confidentiality over host-institution peer review comments and proprietary grant proposals reviewed during the visiting semester.",
        "CONFIDENTIALITY_NDA", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 5.1",
        "Appropriate academic peer-review confidentiality expectation.",
    ),
    (
        "The student content writer must never publicly claim authorship or disclose collaboration with the platform on personal blogs, even for non-confidential public articles.",
        "CONFIDENTIALITY_NDA", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 5.1",
        "Broad ghostwriting secrecy restricting portfolio attribution for published work.",
    ),
    (
        "Non-disclosure obligations under this Web3 fellowship extend permanently to all tokenomics models, private Discord channel logs, and founder communications without statutory exceptions.",
        "CONFIDENTIALITY_NDA", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 5.1",
        "Perpetual NDA lacking standard public-domain or legal-disclosure carve-outs.",
    ),

    # =========================================================================
    # 6. INTELLECTUAL_PROPERTY (36 clauses)
    # =========================================================================
    (
        "5.1 All inventions, patents, code, copyright, and designs created by the student, whether created during office hours or on personal devices during the internship term, belong exclusively to the Company.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 5.1",
        "Claims blanket ownership of all student creations including personal-device off-hours work.",
    ),
    (
        "5.2 The intern waives all moral rights, authorship attribution, and patent inventorship claims for any software module contributed during the internship, including academic capstone projects utilizing company architecture.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 5.2",
        "Strips moral rights and appropriates university capstone projects.",
    ),
    (
        "5.1 Work created specifically under company direction using company equipment shall be deemed 'work made for hire' belonging to the employer.",
        "INTELLECTUAL_PROPERTY", "low", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 5.1",
        "Narrow, standard work-made-for-hire assignment limited to directed company work.",
    ),
    (
        "5.2 Student agrees that all intellectual property conceived on personal time during weekends while employed shall automatically vest in the company.",
        "INTELLECTUAL_PROPERTY", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 5.2",
        "Captures off-hours weekend inventions unrelated to company duties.",
    ),
    (
        "7.1 All inventions, software code, scripts, machine learning models, documentation, and designs created by the Intern during the internship term—whether created during office hours or on personal devices outside working hours—shall belong exclusively to the Company.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 7.1",
        "Overbroad 24/7 IP assignment reaching personal laptops and non-working hours.",
    ),
    (
        "7.2 The Intern hereby assigns worldwide ownership and waives all moral rights, authorship attribution, and inventorship claims for any work or academic capstone project referencing concepts learned during the internship.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 7.2",
        "Captures university capstone work and waives moral attribution rights.",
    ),
    (
        "4. Sponsor retains exclusive commercial media rights to all event photographs, participant recordings, and student project pitch presentations recorded during the symposium.",
        "INTELLECTUAL_PROPERTY", "medium", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 4",
        "Transfers commercial media rights over student pitch recordings to the corporate sponsor.",
    ),
    (
        "3.2 The intern waives all moral rights, authorship attribution, and patent inventorship claims for any educational module contributed.",
        "INTELLECTUAL_PROPERTY", "high", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service", "Clause 3.2",
        "Forces complete waiver of moral authorship rights over student-authored course content.",
    ),
    (
        "The Student retains exclusive ownership of all pre-existing open-source repositories, independent coursework, university theses, and personal side projects created outside working hours without company resources.",
        "INTELLECTUAL_PROPERTY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 4.1",
        "Gold-standard student IP carve-out protecting pre-existing, academic, and personal projects.",
    ),
    (
        "Intellectual property created solely within the scope of assigned internship deliverables using company hardware and data shall vest in the Company upon full payment of stipend.",
        "INTELLECTUAL_PROPERTY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 4.2",
        "Balanced scope-limited IP assignment conditioned on actual stipend payment.",
    ),
    (
        "Any invention, algorithm, or mobile application conceived by the trainee within twelve months after leaving employment shall be presumed to belong to the Company unless proven otherwise.",
        "INTELLECTUAL_PROPERTY", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 4.1",
        "Trailer IP clause claiming ownership over inventions created a full year after departure.",
    ),
    (
        "The student grants the Company an irrevocable, perpetual, royalty-free worldwide license to commercialize all pre-existing software libraries listed or inadvertently used during the internship.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 4.3",
        "Captures broad commercial license over pre-existing student open-source libraries.",
    ),
    (
        "All patentable discoveries arising from the student's university-funded B.Tech thesis shall become sole corporate property if the intern consulted any company mentor during thesis ideation.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 4.1",
        "Appropriates university-funded thesis IP merely because of informal mentor discussions.",
    ),
    (
        "Pre-existing student tools and academic prototypes disclosed in Schedule B remain the sole intellectual property of the trainee and are excluded from employer assignment.",
        "INTELLECTUAL_PROPERTY", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 4.1",
        "Clear prior-inventions schedule safeguarding student's existing portfolio.",
    ),
    (
        "The Company grants the Intern a non-exclusive license to describe high-level project architecture and personal learnings in their university internship report and oral defense.",
        "INTELLECTUAL_PROPERTY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 4.2",
        "Explicitly supports mandatory university reporting and defense obligations.",
    ),
    (
        "Participating student hackathon teams retain 100% ownership of source code, hardware designs, and startup prototypes built during the event; sponsors receive no equity or IP rights.",
        "INTELLECTUAL_PROPERTY", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 4.2",
        "Protects student innovators' ownership over hackathon builds.",
    ),
    (
        "By submitting a prototype to the sponsored campus challenge, student teams irrevocably assign all worldwide patent, copyright, and commercialization rights to the title sponsor without royalty.",
        "INTELLECTUAL_PROPERTY", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 4.3",
        "Strips student hackathon teams of all IP rights to their own inventions without compensation.",
    ),
    (
        "Upon full payment of milestone invoices, copyright in the bespoke client website deliverables transfers to the client, while the student developer retains reuse rights to generic utility components.",
        "INTELLECTUAL_PROPERTY", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 4.1",
        "Fair freelance IP transfer tied to payment while preserving reusable developer utilities.",
    ),
    (
        "The EdTech platform owns exclusive worldwide copyright in all recorded tutoring sessions, slides, and personal teaching methodologies developed by the student tutor, even before payment clears.",
        "INTELLECTUAL_PROPERTY", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 4.1",
        "Transfers ownership before payment and overreaches into personal teaching methods.",
    ),
    (
        "Jointly authored research papers resulting from the student fellowship shall list the student as co-author according to academic contribution norms, and the university retains publication rights.",
        "INTELLECTUAL_PROPERTY", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 4.1",
        "Respects academic authorship credit and scholarly publication standards.",
    ),
    (
        "The student appoints the Company as irrevocable attorney-in-fact to execute patent assignments and copyright filings in the student's name without notice or additional compensation.",
        "INTELLECTUAL_PROPERTY", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 4.2",
        "Broad power-of-attorney IP execution clause lacking student notification or compensation.",
    ),
    (
        "Ownership of any software script written by a student resident while connected to the hostel Wi-Fi network shall vest jointly with the hostel technology affiliate.",
        "INTELLECTUAL_PROPERTY", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 6.3",
        "Absurdly overreaching IP grab tied merely to using residential Wi-Fi.",
    ),
    (
        "Hostel management makes no claim of ownership over any academic projects, software, or creative works produced by residents on personal devices within hostel premises.",
        "INTELLECTUAL_PROPERTY", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 6.1",
        "Explicitly affirms residents retain full ownership of all personal and academic work.",
    ),
    (
        "All promotional reels, campus photography, and meme templates created by the student ambassador become the brand's sole property in perpetuity, including the right to use the student's likeness in paid ads.",
        "INTELLECTUAL_PROPERTY", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 4.1",
        "Transfers perpetual commercial likeness and content rights without ongoing royalty.",
    ),
    (
        "Where a trainee's work results in a granted company patent, the employer shall acknowledge the trainee as named inventor and provide a standard Rs. 50,000 patent filing honorarium.",
        "INTELLECTUAL_PROPERTY", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 4.2",
        "Recognizes inventorship attribution and provides an inventor honorarium.",
    ),
    (
        "The student intern shall not contribute to any public open-source software repository—even unrelated weekend projects—during the six-month term without prior legal department clearance.",
        "INTELLECTUAL_PROPERTY", "medium", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 4.2",
        "Restricts unrelated weekend open-source participation by students.",
    ),
    (
        "Student bootcamp projects submitted for capstone grading remain the intellectual property of the student, with a limited license for the academy to display project thumbnails with attribution.",
        "INTELLECTUAL_PROPERTY", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 6.2",
        "Preserves student capstone ownership while allowing attributed showcase display.",
    ),
    (
        "Any neural network weights, prompt templates, or synthetic datasets fine-tuned by the intern on personal GPU credits during the engagement automatically assign to the employer.",
        "INTELLECTUAL_PROPERTY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 4.1",
        "Appropriates AI models trained on the student's own personal compute resources.",
    ),
    (
        "The freelance student writer assigns publication copyright for commissioned study guides upon receipt of cleared payment, retaining the right to link published samples in a personal portfolio.",
        "INTELLECTUAL_PROPERTY", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 4.1",
        "Fair payment-conditioned copyright transfer preserving portfolio showcase rights.",
    ),
    (
        "All laboratory notebooks, physical prototypes, and CAD drawings generated under the funded university grant belong to the university, with student inventors sharing royalty proceeds under campus IP policy.",
        "INTELLECTUAL_PROPERTY", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 4.1",
        "Balanced academic grant IP framework including student royalty sharing.",
    ),
    (
        "The event organizer grants the sponsor a non-exclusive license to use approved festival photographs for post-event recap posts for twelve months, crediting the student photography club.",
        "INTELLECTUAL_PROPERTY", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 4.1",
        "Time-bounded, attributed event media license respecting student creators.",
    ),
    (
        "The trainee surrenders all worldwide rights to any domain names, technical blogs, or YouTube tutorials published on topics adjacent to the employer's software stack during employment.",
        "INTELLECTUAL_PROPERTY", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fmcg_08", "Clause 4.3",
        "Overreaches into personal educational blogs and technical channels created outside work.",
    ),
    (
        "Inventions created strictly using employer proprietary source code and during compensated work hours are assigned to the employer; statutory moral rights remain preserved where non-waivable.",
        "INTELLECTUAL_PROPERTY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 4.1",
        "Narrow work-scope IP assignment respecting statutory moral rights.",
    ),
    (
        "If the student contractor incorporates any open-source library without written sign-off, the contractor must assign all copyright in their personal code library as remedial ownership.",
        "INTELLECTUAL_PROPERTY", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 4.3",
        "Punitive IP forfeiture clause capturing unrelated personal code libraries.",
    ),
    (
        "Visiting exchange students retain copyright in personal research dissertations while granting the host university a non-exclusive archival license for library preservation.",
        "INTELLECTUAL_PROPERTY", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 4.2",
        "Standard scholarly dissertation ownership with non-exclusive library archival rights.",
    ),
    (
        "All design concepts, Figma files, and rejected mockups drafted by the intern—even those never adopted by the company—remain locked under exclusive corporate ownership forever.",
        "INTELLECTUAL_PROPERTY", "medium", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 4.4",
        "Prevents design interns from showcasing even rejected or unused exploratory concepts.",
    ),

    # =========================================================================
    # 7. RESTRICTIVE_NON_COMPETE (36 clauses)
    # =========================================================================
    (
        "4.1 The Student agrees not to work for, consult, or be associated with any competing business in the software or financial technology industry for a period of 24 months globally after departure.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 4.1",
        "Imposes a 24-month global post-internship non-compete barring software/fintech careers.",
    ),
    (
        "4.2 Intern shall not solicit any employee, customer, vendor, or partner of the company for a period of 12 months following termination.",
        "RESTRICTIVE_NON_COMPETE", "medium", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 4.2",
        "Post-internship non-solicitation restriction covering employees, clients, and vendors.",
    ),
    (
        "4.1 Restraint of trade: Intern is prohibited from accepting employment with any competitor within the territory for one full year following departure.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 4.1",
        "Restrains a graduating student from working for any competitor for one year.",
    ),
    (
        "4.3 For 18 months post-internship, the student is barred from joining any fintech startup, software company, or entering graduate studies in the same domain.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 4.3",
        "Extreme restraint barring both industry employment and graduate university studies.",
    ),
    (
        "4. The Student agrees not to work for, consult, or be associated with any competing business in the edtech or online tutoring industry for a period of 24 months globally after departure.",
        "RESTRICTIVE_NON_COMPETE", "high", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service", "Clause 4",
        "Global 24-month non-compete prohibiting a student tutor from teaching anywhere else.",
    ),
    (
        "For thirty-six (36) months following completion of the summer internship, the student shall not accept an internship, full-time job, or freelance contract with any enterprise offering cloud analytics.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 6.1",
        "Three-year industry-wide post-internship non-compete blocking student employability.",
    ),
    (
        "The graduate trainee covenants not to join any direct or indirect client of the IT consultancy for two years post-resignation, even via independent campus recruitment.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 4.2",
        "Bars campus hires from joining major client companies for two years post-exit.",
    ),
    (
        "No post-employment or post-internship non-compete restriction shall apply to the student; after departure, the student is completely free to work for any organization or pursue higher education.",
        "RESTRICTIVE_NON_COMPETE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 7.1",
        "Explicitly waives post-internship non-compete restrictions to protect student career mobility.",
    ),
    (
        "During the active internship period only, the student shall not simultaneously perform paid engineering work for a direct business competitor without written disclosure.",
        "RESTRICTIVE_NON_COMPETE", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 6.2",
        "Reasonable conflict-of-interest rule limited strictly to the active engagement period.",
    ),
    (
        "The student agrees not to co-found, advise, or invest in any university incubator startup operating in artificial intelligence or SaaS for 36 months after concluding the internship.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 7.1",
        "Blocks student entrepreneurship and university incubator participation for three years.",
    ),
    (
        "Campus ambassadors are prohibited from representing any other consumer brand, student app, or college festival sponsor across India for eighteen months after the campaign ends.",
        "RESTRICTIVE_NON_COMPETE", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 7.1",
        "Overbroad 18-month exclusivity restraint on part-time student campus promoters.",
    ),
    (
        "The student organizing committee agrees not to invite any competing beverage or telecom brand as a co-sponsor for the same annual festival edition.",
        "RESTRICTIVE_NON_COMPETE", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 5.1",
        "Standard single-event category exclusivity appropriate for a title festival sponsor.",
    ),
    (
        "The university technical society is barred for five consecutive academic years from accepting sponsorships from any technology company deemed competitive by the current sponsor.",
        "RESTRICTIVE_NON_COMPETE", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 5.3",
        "Multi-year five-year exclusivity lock restricting future cohorts of student clubs.",
    ),
    (
        "Upon leaving the EdTech platform, the student tutor shall not conduct private one-on-one tuition, publish free YouTube coding lessons, or teach university workshops for 18 months.",
        "RESTRICTIVE_NON_COMPETE", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 7.1",
        "Restrains a student from free teaching, YouTube education, and campus workshops.",
    ),
    (
        "The trainee shall not recruit, entice, or refer any former colleague or fellow university intern away from the Company for 24 months after separation.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_consulting_03", "Clause 4.4",
        "Broad two-year anti-referral clause restricting normal peer networking among classmates.",
    ),
    (
        "Where permitted by applicable governing law, any post-termination non-compete shall be limited to 3 months within a 15 km radius and enforceable only if the employer pays 100% full salary as garden leave.",
        "RESTRICTIVE_NON_COMPETE", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 6.1",
        "Narrowly tailored garden-leave non-compete with full salary compensation; moderate caution.",
    ),
    (
        "During residency, hostel tenants are prohibited from operating unauthorized commercial subletting or rival student brokerage services from hostel rooms.",
        "RESTRICTIVE_NON_COMPETE", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 7.1",
        "Standard residential non-commercial use rule.",
    ),
    (
        "Former hostel residents agree not to rent accommodation from any competing private PG or student residence within 5 kilometers of the campus for one academic year after vacating.",
        "RESTRICTIVE_NON_COMPETE", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 7.2",
        "Unconscionable housing non-compete restricting where a student may live near campus.",
    ),
    (
        "The freelance student developer shall not bid on Upwork, Fiverr, or university freelance boards in web development for twelve months following completion of this single project.",
        "RESTRICTIVE_NON_COMPETE", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 7.1",
        "Prohibits a student freelancer from working on public freelance marketplaces for a year.",
    ),
    (
        "The student research intern is barred from applying to competing corporate AI labs or joining PhD research groups sponsored by industry rivals for two years.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 7.2",
        "Interferes with doctoral admissions and academic lab placements for two years.",
    ),
    (
        "For six months after contract expiration, the student contractor agrees not to directly solicit the specific named end-client introduced by the agency solely to bypass agency fees.",
        "RESTRICTIVE_NON_COMPETE", "medium", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 7.1",
        "Narrowly scoped 6-month non-circumvention limited to one introduced client.",
    ),
    (
        "Any breach of this global non-compete covenant entitles the employer to an ex-parte injunction plus liquidated damages equal to twenty-four months of gross CTC.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_banking_05", "Clause 4.2",
        "Backs a global restraint of trade with two years' salary in liquidated damages.",
    ),
    (
        "The intern shall not participate in competitive programming contests, external hackathons, or open-source summer programs like GSoC during or for 6 months after the internship.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 7.1",
        "Bans students from educational hackathons and open-source fellowships post-internship.",
    ),
    (
        "Exclusivity during the coding bootcamp prevents the student from enrolling in any competing upskilling platform or attending rival placement workshops for twelve months.",
        "RESTRICTIVE_NON_COMPETE", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 7.1",
        "Restricts a student's educational freedom to take external courses or placement prep.",
    ),
    (
        "The graduate engineer agrees to refrain from accepting employment with any enterprise listed in Annexure C (comprising 150 global software firms) for eighteen months post-departure.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 7.1",
        "Blacklists 150 major technology employers for a fresh graduate for 18 months.",
    ),
    (
        "Under Section 27 of the Indian Contract Act, post-separation restraints of trade are generally void; accordingly, this agreement imposes no post-internship non-compete on the student.",
        "RESTRICTIVE_NON_COMPETE", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 7.1",
        "Acknowledges statutory law and explicitly avoids unlawful post-exit restraints.",
    ),
    (
        "Student note-creators agree not to upload competing syllabus summaries to rival study portals for twenty-four months after uploading a single module.",
        "RESTRICTIVE_NON_COMPETE", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 6.1",
        "Locks student authors out of other study platforms for two full years.",
    ),
    (
        "The Web3 intern covenants not to contribute code to any decentralized protocol, blockchain foundation, or fintech wallet worldwide for 24 months after offboarding.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 7.1",
        "Worldwide two-year industry ban across blockchain and fintech sectors.",
    ),
    (
        "During active full-time employment, the trainee shall disclose any outside consulting engagement exceeding five hours per week to avoid direct client conflicts.",
        "RESTRICTIVE_NON_COMPETE", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 6.2",
        "Reasonable conflict-of-interest disclosure rule limited to active employment.",
    ),
    (
        "The student placement agency prohibits registered candidates from applying directly to campus off-cycle job openings without routing offers through the agency for 12 months.",
        "RESTRICTIVE_NON_COMPETE", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_agency_09", "Clause 4.1",
        "Restricts graduating students from applying independently to job openings.",
    ),
    (
        "Sponsor category exclusivity applies strictly to the 48-hour hackathon venue and digital event page, and shall not restrict the student club's other semester workshops.",
        "RESTRICTIVE_NON_COMPETE", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 6.1",
        "Cleanly bounded 48-hour event exclusivity preserving year-round club autonomy.",
    ),
    (
        "The trainee shall not solicit or service any customer account with which the trainee had direct material contact for a narrow period of six months post-exit.",
        "RESTRICTIVE_NON_COMPETE", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fmcg_08", "Clause 7.1",
        "Focused six-month client non-solicitation restricted to directly handled accounts.",
    ),
    (
        "Students participating in the industry-sponsored capstone lab are barred for 12 months from publishing comparative benchmarks or working with rival hardware vendors.",
        "RESTRICTIVE_NON_COMPETE", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 7.1",
        "Restricts graduating students from joining hardware employers after an academic capstone.",
    ),
    (
        "For fifteen months after completing the UI/UX internship, the student shall not design mobile interfaces for any e-commerce, delivery, or social networking company.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 7.1",
        "Sweeping 15-month restriction barring student designers from major consumer tech sectors.",
    ),
    (
        "Exchange program fellows may not accept internships in the host country for twelve months following return unless sponsored exclusively by the originating placement consortium.",
        "RESTRICTIVE_NON_COMPETE", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 7.2",
        "Restricts international career opportunities for returning exchange students.",
    ),
    (
        "The student agrees not to accept any job offer from an organization employing any former director of the Company for a period of 24 months after the internship ends.",
        "RESTRICTIVE_NON_COMPETE", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 7.3",
        "Vague, sweeping two-year employment ban tied to former executives' affiliations.",
    ),

    # =========================================================================
    # 8. LIABILITY_INDEMNITY (36 clauses)
    # =========================================================================
    (
        "8.1 The Student shall fully indemnify, defend, and hold harmless the Company from any and all damages, claims, third-party lawsuits, legal costs, and lost profits arising from any code errors or bugs.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 8.1",
        "Imposes unlimited personal indemnification on an intern for ordinary software bugs and lost profits.",
    ),
    (
        "8.2 The intern assumes full personal financial liability for any accidental bugs, production outages, or cybersecurity breaches occurring during their project.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 8.2",
        "Makes a student intern personally liable for corporate production outages and cyber incidents.",
    ),
    (
        "7.1 The student agrees to pay all attorney fees and litigation costs incurred by the employer in enforcing any clause of this agreement.",
        "LIABILITY_INDEMNITY", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 7.1",
        "One-sided fee-shifting forcing a student to fund all employer litigation costs.",
    ),
    (
        "7.2 Company's total aggregate liability arising out of this agreement shall be limited to the total fees paid to the student in the preceding one month.",
        "LIABILITY_INDEMNITY", "medium", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 7.2",
        "Asymmetrically caps employer liability at one month's pay while leaving student liability uncapped.",
    ),
    (
        "4.1 Hostel resident shall be personally liable for treble damages for any damage caused to hostel furniture, shared amenities, or electrical installations.",
        "LIABILITY_INDEMNITY", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 4.1",
        "Imposes punitive 3x treble damages on students for hostel furniture or fixture damage.",
    ),
    (
        "4.2 Company disclaims all responsibility for on-campus student accidents, lab injuries, or equipment hazards incurred during off-site assignments or inside hostel premises.",
        "LIABILITY_INDEMNITY", "high", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 4.2",
        "Blanket disclaimer attempting to escape liability for student safety hazards and injuries.",
    ),
    (
        "3.1 The student club organizers personally guarantee payment and indemnify the corporate sponsor against any cancellation losses, weather delays, or low attendee turnout.",
        "LIABILITY_INDEMNITY", "high", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 3.1",
        "Binds individual student club leaders as personal financial guarantors for event turnout and weather.",
    ),
    (
        "Neither party shall be liable to the other for any indirect, incidental, consequential, special, or punitive damages, and the student's aggregate liability is strictly capped at total stipend received.",
        "LIABILITY_INDEMNITY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 8.1",
        "Fair mutual exclusion of consequential damages with student liability capped at stipend.",
    ),
    (
        "The Intern shall bear zero personal financial liability for unintentional coding errors, staging environment downtime, or good-faith mistakes committed under normal supervision.",
        "LIABILITY_INDEMNITY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 8.1",
        "Explicitly shields student interns from operational liability for ordinary mistakes.",
    ),
    (
        "The graduate trainee shall personally reimburse the employer for any SLA penalty charged by an enterprise client due to delayed tickets or regression defects in modules touched by the trainee.",
        "LIABILITY_INDEMNITY", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 8.1",
        "Passes enterprise client SLA penalties onto an individual fresh graduate.",
    ),
    (
        "Each party agrees to indemnify the other solely for direct third-party losses proven by final court judgment to result from its own willful misconduct or gross negligence.",
        "LIABILITY_INDEMNITY", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 8.1",
        "Balanced mutual indemnity limited to proven gross negligence or willful misconduct.",
    ),
    (
        "Student interns traveling for field surveys waive all legal claims against the organization for road accidents, unsafe site conditions, or employer-negligent equipment failures.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 8.2",
        "Forces students to waive injury claims even for employer negligence during field work.",
    ),
    (
        "The hostel operator disclaims all liability for theft, electrical fires, contaminated drinking water, or structural ceiling leaks inside the residential block, regardless of operator negligence.",
        "LIABILITY_INDEMNITY", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 4.1",
        "Attempts to immunize hostel management from basic habitability and safety failures.",
    ),
    (
        "Resident liability for room furnishings is limited strictly to actual documented repair or depreciated replacement cost for damage caused by the resident's negligence.",
        "LIABILITY_INDEMNITY", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 4.1",
        "Fair, depreciated actual-cost standard for residential property damage.",
    ),
    (
        "The college fest convener and treasurer jointly and severally indemnify the sponsor for up to Rs. 10,00,000 against any intellectual property claim arising from third-party student bands or speakers.",
        "LIABILITY_INDEMNITY", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 6.1",
        "Exposes individual student officers to Rs. 10,00,000 personal indemnity for third-party performers.",
    ),
    (
        "Student organizing committee members act strictly in their institutional representative capacity and assume no personal financial liability for university club obligations.",
        "LIABILITY_INDEMNITY", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 7.1",
        "Protects volunteer student club officers from personal financial liability.",
    ),
    (
        "The freelance student developer provides experimental hackathon and beta prototypes on an 'as-is' basis and disclaims implied warranties of merchantability or uninterrupted uptime.",
        "LIABILITY_INDEMNITY", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 8.1",
        "Standard warranty disclaimer appropriate for student prototype development.",
    ),
    (
        "The student tutor shall indemnify the EdTech company against all parent refund demands, exam score disappointments, regulatory fines, and reputation damages without monetary cap.",
        "LIABILITY_INDEMNITY", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 8.1",
        "Uncapped personal indemnity holding a student tutor liable for end-user exam outcomes.",
    ),
    (
        "In the event of any data breach on the startup's cloud infrastructure, the intern agrees to indemnify the company for all GDPR/DPDP statutory penalties and forensic audit fees.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 8.1",
        "Shifts massive corporate regulatory data-breach fines onto a student intern.",
    ),
    (
        "Liability of either party under this student tutoring agreement shall in no event exceed the total fees actually paid to the student contractor over the preceding three months.",
        "LIABILITY_INDEMNITY", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 8.1",
        "Reasonable, symmetric liability cap proportional to contractor earnings.",
    ),
    (
        "The campus ambassador shall hold the brand harmless against any social media copyright claim, regulatory advertising penalty, or consumer complaint arising from brand-supplied promotional scripts.",
        "LIABILITY_INDEMNITY", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 8.1",
        "Forces students to indemnify the brand even for legal issues in the brand's own scripts.",
    ),
    (
        "Company shall maintain workplace accident insurance covering the industrial trainee during all on-site plant rotations and reimburse emergency medical care for work-related injuries.",
        "LIABILITY_INDEMNITY", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 8.1",
        "Student-protective workplace insurance and medical coverage commitment.",
    ),
    (
        "The student assumes sole financial liability for any accidental cloud billing overrun exceeding USD 50 on company AWS sandboxes, regardless of missing budget alerts.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 8.2",
        "Makes an intern personally liable for uncapped corporate cloud billing overruns.",
    ),
    (
        "Hostel residents sharing a floor are jointly and strictly liable for 200% of repair costs for any common-area vandalism, regardless of proof of individual involvement.",
        "LIABILITY_INDEMNITY", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_campus_02", "Clause 6.2",
        "Imposes collective double-cost strict liability without fault determination.",
    ),
    (
        "The university laboratory provides mandatory personal protective equipment and assumes statutory responsibility for maintaining safe fume hoods, electrical grounding, and chemical storage.",
        "LIABILITY_INDEMNITY", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 8.1",
        "Clear institutional safety responsibility protecting lab students.",
    ),
    (
        "Student participants in the overseas study tour waive every legal right to claim damages even for gross negligence or reckless misconduct by contracted transport vendors.",
        "LIABILITY_INDEMNITY", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 8.2",
        "Overbroad liability waiver attempting to excuse gross negligence.",
    ),
    (
        "The trainee agrees to defend and indemnify the employer against any third-party intellectual property claim arising from knowingly submitting plagiarized proprietary third-party code.",
        "LIABILITY_INDEMNITY", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_cloud_04", "Clause 8.2",
        "Targeted indemnity tied to knowing IP infringement; still warrants a monetary liability cap.",
    ),
    (
        "If the student contractor's website deliverable experiences downtime after handover, the student shall reimburse the client's lost e-commerce revenue and advertising spend.",
        "LIABILITY_INDEMNITY", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 8.2",
        "Exposes a student freelancer to uncapped consequential lost-revenue claims.",
    ),
    (
        "The sponsor and student committee each bear responsibility for their own personnel's negligent acts, and neither party guarantees specific commercial sales conversions from festival footfall.",
        "LIABILITY_INDEMNITY", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 7.1",
        "Balanced allocation of event liability avoiding unfair sales guarantees.",
    ),
    (
        "The intern agrees to personally indemnify the startup's founders, investors, and officers against any smart-contract exploit or token price drop linked to code reviewed by the intern.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 8.1",
        "Shifts catastrophic crypto-market and smart-contract financial risk onto an intern.",
    ),
    (
        "The coding academy disclaims all placement promises and limits its maximum liability for billing errors or canceled classes to the unutilized portion of tuition paid by the student.",
        "LIABILITY_INDEMNITY", "medium", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 8.1",
        "Caps provider liability to unused tuition while disclaiming placement representations.",
    ),
    (
        "Hostel management shall maintain functional CCTV, fire extinguishers, and basic structural safety, and remains liable for direct property losses caused by staff negligence.",
        "LIABILITY_INDEMNITY", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 4.2",
        "Holds hostel operator accountable for safety infrastructure and staff negligence.",
    ),
    (
        "The graduate engineer must reimburse the bank for all regulatory investigation costs and external counsel retainers if any internal audit flags documentation authored by the trainee.",
        "LIABILITY_INDEMNITY", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_banking_05", "Clause 8.1",
        "Shifts corporate regulatory audit and external law-firm fees onto a junior trainee.",
    ),
    (
        "The employer shall indemnify and defend the intern against any third-party lawsuit arising from authorized work performed in good faith within the scope of assigned internship duties.",
        "LIABILITY_INDEMNITY", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 8.1",
        "Employer-to-intern defense and indemnity protection for good-faith assigned work.",
    ),
    (
        "The student design intern assumes full liability for all print-run reprinting charges and marketing launch delays if a typographic error passes client review.",
        "LIABILITY_INDEMNITY", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 8.1",
        "Holds an intern personally responsible for commercial print-run costs even after client review.",
    ),
    (
        "The research institution maintains general liability coverage for approved campus experiments, and fellows are not personally liable for accidental instrument wear during supervised operation.",
        "LIABILITY_INDEMNITY", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 8.1",
        "Protects student researchers from personal financial liability during supervised lab work.",
    ),

    # =========================================================================
    # 9. DISPUTE_JURISDICTION (36 clauses)
    # =========================================================================
    (
        "9. Any dispute arising out of this agreement shall be subject to the exclusive jurisdiction of the courts located in New York, USA, irrespective of student's residence, and all arbitration costs shall be borne solely by the student.",
        "DISPUTE_JURISDICTION", "high", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship", "Clause 9",
        "Forces a local student into New York courts and shifts 100% of arbitration fees onto the student.",
    ),
    (
        "8. All disputes shall be settled by binding arbitration in Singapore under SIAC rules, with arbitration costs borne solely by the student.",
        "DISPUTE_JURISDICTION", "high", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement", "Clause 8",
        "Mandates expensive overseas SIAC arbitration in Singapore paid exclusively by the student.",
    ),
    (
        "6. Exclusive jurisdiction for hostel residency disputes shall rest with the local municipal civil court where the hostel is situated.",
        "DISPUTE_JURISDICTION", "low", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 6",
        "Fair local court jurisdiction where the hostel and student are actually located.",
    ),
    (
        "6. Disputes shall be resolved through good-faith mutual consultation, failing which the courts at the student's university campus city shall have jurisdiction.",
        "DISPUTE_JURISDICTION", "low", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 6",
        "Balanced escalation beginning with good-faith consultation and local campus-city courts.",
    ),
    (
        "7. This agreement shall be governed by and construed in accordance with the laws of the State of Delaware without regard to conflict of law principles.",
        "DISPUTE_JURISDICTION", "high", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service", "Clause 7",
        "Applies distant foreign Delaware governing law to a domestic student contractor.",
    ),
    (
        "8.1 This Agreement shall be governed by and construed in accordance with the laws of India. Any disputes arising hereunder shall first be referred to good-faith conciliation, failing which the competent courts at Bengaluru, Karnataka shall have exclusive jurisdiction.",
        "DISPUTE_JURISDICTION", "low", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 8.1",
        "Reasonable Indian governing law, good-faith conciliation, and local Bengaluru jurisdiction.",
    ),
    (
        "Any claim or controversy must be litigated exclusively in London, United Kingdom under English law, and the student waives all objections to inconvenient forum.",
        "DISPUTE_JURISDICTION", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_agency_03", "Clause 9.1",
        "Imposes prohibitive foreign litigation in London with an inconvenient-forum waiver.",
    ),
    (
        "The student irrevocably waives the right to bring any legal claim more than 60 calendar days after the underlying incident occurs, permanently extinguishing statutory limitation periods.",
        "DISPUTE_JURISDICTION", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_it_01", "Clause 9.1",
        "Drastically shortens the statutory claim window to just 60 days.",
    ),
    (
        "Unresolved disputes shall be referred to a sole independent arbitrator appointed jointly by both parties under the Arbitration and Conciliation Act, 1996, with seat in the student's city and fees shared equally.",
        "DISPUTE_JURISDICTION", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 9.1",
        "Fair domestic arbitration with mutual arbitrator selection and equal fee split.",
    ),
    (
        "The Company shall have the sole and unilateral right to appoint the presiding arbitrator from its retained legal panel, and the arbitrator's award shall be final and unappealable.",
        "DISPUTE_JURISDICTION", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_consulting_03", "Clause 9.1",
        "Compromises neutrality through unilateral employer appointment of the sole arbitrator.",
    ),
    (
        "The student waives all rights to participate in any class action, collective grievance, or consolidated proceeding against the education platform or its affiliates.",
        "DISPUTE_JURISDICTION", "high", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 9.1",
        "Strips students of collective/class action remedies.",
    ),
    (
        "Before initiating formal litigation, the resident and hostel warden shall attempt amicable resolution through the university Student Welfare Grievance Committee within 15 days.",
        "DISPUTE_JURISDICTION", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 8.1",
        "Accessible campus grievance resolution step prior to formal legal proceedings.",
    ),
    (
        "All hostel fee and eviction disputes shall be adjudicated exclusively by courts in Mumbai, even though the residential campus and student are situated in Jaipur.",
        "DISPUTE_JURISDICTION", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_private_01", "Clause 8.1",
        "Selects a distant unrelated city for hostel disputes to deter student claims.",
    ),
    (
        "This Agreement is governed by the laws of India, and courts located in Pune (where the internship office and university are situated) shall exercise exclusive jurisdiction.",
        "DISPUTE_JURISDICTION", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 9.1",
        "Local domestic governing law and accessible jurisdiction.",
    ),
    (
        "Disputes arising from this Web3 internship shall be resolved exclusively via private arbitration in the Cayman Islands under rules chosen by the foundation, with proceedings funded upfront by the claimant.",
        "DISPUTE_JURISDICTION", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_web3_07", "Clause 9.1",
        "Offshore Cayman Islands arbitration financially impossible for a student to pursue.",
    ),
    (
        "Either party may elect online video-conferenced mediation through a neutral campus-approved mediator prior to approaching local civil courts in Hyderabad.",
        "DISPUTE_JURISDICTION", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 9.1",
        "Low-cost remote mediation option paired with local jurisdiction.",
    ),
    (
        "The student contractor agrees that all disputes shall be settled exclusively by courts in San Francisco, California, and consents to service of process by email alone.",
        "DISPUTE_JURISDICTION", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03", "Clause 9.1",
        "Foreign California venue inaccessible to an international or domestic student contractor.",
    ),
    (
        "Mandatory pre-litigation mediation shall be conducted in good faith for 21 days; if unresolved, either party may file suit before the competent courts of Chennai.",
        "DISPUTE_JURISDICTION", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 8.1",
        "Balanced mediation-first dispute clause with local city courts.",
    ),
    (
        "Any legal action initiated by the student club against the sponsor must be preceded by depositing a non-refundable Rs. 50,000 arbitration filing bond with the sponsor's law firm.",
        "DISPUTE_JURISDICTION", "high", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_cultural_01", "Clause 7.2",
        "Erects a punitive Rs. 50,000 financial barrier blocking students from pursuing legal claims.",
    ),
    (
        "Governing law shall be the laws of India without conflict-of-law rules; the parties submit to the non-exclusive jurisdiction of courts at the student's place of permanent residence or workplace.",
        "DISPUTE_JURISDICTION", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 9.1",
        "Student-friendly jurisdiction allowing local filing where the student lives or works.",
    ),
    (
        "The employer retains the unilateral option to pursue court injunctions in any jurisdiction worldwide while confining all employee claims strictly to private arbitration in Dubai.",
        "DISPUTE_JURISDICTION", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_banking_05", "Clause 9.1",
        "One-sided forum clause granting employer worldwide court access while forcing trainee to Dubai arbitration.",
    ),
    (
        "Nothing in this dispute clause prevents the student resident from filing a complaint before the local District Consumer Disputes Redressal Commission.",
        "DISPUTE_JURISDICTION", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 8.1",
        "Preserves students' statutory consumer forum remedies.",
    ),
    (
        "Disputes involving academic authorship, fellowship stipends, or lab supervision shall first be reviewed by the University Dean of Research under institutional ombudsman procedures.",
        "DISPUTE_JURISDICTION", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 9.1",
        "Appropriate university ombudsman dispute escalation channel.",
    ),
    (
        "All controversies under this EdTech tutor contract shall be arbitrated in Zurich under Swiss rules, with English translations and arbitrator travel billed exclusively to the losing student.",
        "DISPUTE_JURISDICTION", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 9.1",
        "Prohibitive foreign Swiss arbitration with crushing cost exposure.",
    ),
    (
        "This agreement is governed by Indian law; any unresolved dispute under Rs. 2,00,000 may be settled via fast-track online dispute resolution (ODR) with fees borne by the Company.",
        "DISPUTE_JURISDICTION", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 9.1",
        "Accessible, company-funded fast-track ODR for smaller student claims.",
    ),
    (
        "The student waives any right to appeal an adverse administrative decision of the hostel disciplinary board to any civil court or university appellate tribunal.",
        "DISPUTE_JURISDICTION", "high", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_campus_02", "Clause 7.2",
        "Attempts to block judicial and university appellate review of disciplinary actions.",
    ),
    (
        "Arbitration proceedings shall be conducted in English in Bengaluru by a three-member tribunal, with each party bearing its own legal counsel fees and splitting tribunal costs 50/50.",
        "DISPUTE_JURISDICTION", "medium", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 9.1",
        "Balanced cost split, though a three-member tribunal can be costly for an individual student.",
    ),
    (
        "Disputes regarding event sponsorship deliverables shall be referred to joint faculty-sponsor mediation within 10 days before pursuing remedies in local campus-city courts.",
        "DISPUTE_JURISDICTION", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 8.1",
        "Practical faculty-assisted mediation and local court venue.",
    ),
    (
        "Any claim brought by the intern against the company must be filed within 30 days of termination in courts at Hong Kong SAR, waiving all statutory remedies in the intern's home country.",
        "DISPUTE_JURISDICTION", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_fintech_01", "Clause 9.1",
        "Combines a 30-day claim bar with distant Hong Kong exclusive jurisdiction.",
    ),
    (
        "Governing law for the international academic exchange shall be the laws of the host university's jurisdiction, with mandatory university ombudsman mediation prior to legal filings.",
        "DISPUTE_JURISDICTION", "medium", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 9.1",
        "Expected host-country governing law for overseas exchange; requires awareness of foreign procedures.",
    ),
    (
        "The campus ambassador agrees that courts in the brand's overseas headquarters city shall have sole jurisdiction and that no small-claims tribunal may entertain stipend claims.",
        "DISPUTE_JURISDICTION", "high", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 9.1",
        "Blocks accessible small-claims tribunals in favor of an overseas court.",
    ),
    (
        "Both parties agree to resolve billing or content-acceptance disputes amicably within 14 days by written notice, failing which courts in Kolkata shall have exclusive jurisdiction.",
        "DISPUTE_JURISDICTION", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 9.1",
        "Clear domestic resolution period and local Kolkata court jurisdiction.",
    ),
    (
        "This agreement shall be governed by Indian statutory law, and disputes arising out of laboratory safety or stipend disbursements shall fall under local district courts.",
        "DISPUTE_JURISDICTION", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 9.1",
        "Clear local jurisdiction protecting campus lab students.",
    ),
    (
        "The employer's internal HR grievance panel decision shall constitute final and binding arbitration, barring the trainee from seeking relief before labor commissioners or courts.",
        "DISPUTE_JURISDICTION", "high", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fmcg_08", "Clause 9.1",
        "Replaces independent courts with the employer's own HR department as final judge.",
    ),
    (
        "All contractual controversies shall be governed by the laws of Singapore, with mandatory confidential arbitration administered exclusively in English in Singapore.",
        "DISPUTE_JURISDICTION", "high", "INTERNSHIP", "curated_synthetic", "tpl_intern_startup_06", "Clause 9.1",
        "Foreign Singapore governing law and arbitration venue unsuitable for student interns.",
    ),
    (
        "In the event of any disagreement concerning room handover inspection, either party may request review by the Dean of Student Affairs prior to approaching local municipal courts.",
        "DISPUTE_JURISDICTION", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03", "Clause 8.1",
        "Provides institutional escalation alongside accessible municipal courts.",
    ),

    # =========================================================================
    # 10. STANDARD_FAIR (36 clauses)
    # =========================================================================
    (
        "1.1 The Intern is appointed as a Data Engineering Intern for a fixed duration of 6 (six) months, commencing on 1st January 2027 and concluding on 30th June 2027.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 1.1",
        "Standard role title and fixed 6-month internship start/end dates.",
    ),
    (
        "1.2 The Intern shall commit 40 hours per week (Monday to Friday, 9:30 AM to 6:30 PM IST) in a hybrid work arrangement and report directly to the Lead Data Architect.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship", "Clause 1.2",
        "Clearly specifies weekly working hours, hybrid mode, and reporting mentor.",
    ),
    (
        "1. The Hostel Management allots Room 304, Block B, for the academic term from 1st September 2026 to 30th June 2027. Room occupancy is strictly personal to the Resident.",
        "STANDARD_FAIR", "low", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel", "Clause 1",
        "Standard room allotment identifying room number and academic occupancy period.",
    ),
    (
        "1.1 The Sponsor shall provide title sponsorship of INR 1,50,000 for CampusFest 2026.",
        "STANDARD_FAIR", "low", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 1.1",
        "States agreed title sponsorship financial contribution.",
    ),
    (
        "1.2 Student Organizers shall provide one 10x10 exhibition stall, sponsor logo on all main stage backdrops, and 15-minute key presentation slot.",
        "STANDARD_FAIR", "low", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship", "Clause 1.2",
        "Defines clear, concrete event branding deliverables for the student club.",
    ),
    (
        "1. Student Contractor agrees to record 40 video tutorials on Data Structures & Algorithms and prepare coding exercises as per syllabus guidelines.",
        "STANDARD_FAIR", "low", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service", "Clause 1",
        "Clearly scopes educational content deliverables for the student contractor.",
    ),
    (
        "This Agreement represents the entire understanding between the parties regarding its subject matter and supersedes all prior informal emails or verbal representations.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 10.1",
        "Standard integration / entire-agreement boilerplate.",
    ),
    (
        "If any court determines a provision of this Agreement to be invalid or unenforceable, that clause shall be severed while all remaining balanced provisions stay in full effect.",
        "STANDARD_FAIR", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 10.1",
        "Standard severability clause.",
    ),
    (
        "No modification, waiver, or amendment to this Agreement shall be binding unless set out in writing and signed electronically or physically by both parties.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 10.1",
        "Protects students against unilateral oral changes by requiring mutual written amendments.",
    ),
    (
        "The Company shall assign a dedicated technical mentor who will hold bi-weekly feedback check-ins and complete required university internship evaluation rubrics on time.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 1.3",
        "Ensures structured mentorship and timely university credit sign-off.",
    ),
    (
        "The Intern is entitled to up to five (5) days of unpaid or flexible academic examination leave during scheduled university semester exams upon giving seven days' advance notice.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 1.3",
        "Explicitly accommodates university examination schedules.",
    ),
    (
        "Neither party shall be liable for delay or non-performance caused by force majeure events such as floods, earthquakes, epidemics, or government lockdowns, without financial penalty to either side.",
        "STANDARD_FAIR", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 9.1",
        "Balanced mutual force majeure clause without one-sided refund penalties.",
    ),
    (
        "The Company shall issue an official Internship Completion Certificate and verifiable experience letter within seven (7) business days of completing the internship term.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_saas_02", "Clause 1.4",
        "Guarantees prompt delivery of internship completion credentials.",
    ),
    (
        "The Company shall provide the Intern with a configured workstation, licensed software tools, and secure VPN credentials necessary to perform assigned tasks at zero cost to the Intern.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 1.2",
        "Commits employer to supply required tools and software without student charges.",
    ),
    (
        "Official notices under this Agreement shall be sent in writing via registered email to the designated contact addresses listed in Schedule A, deemed received on the next business day.",
        "STANDARD_FAIR", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 10.1",
        "Clear, modern electronic notice procedure.",
    ),
    (
        "Section headings and clause captions are included solely for organizational convenience and shall not alter the legal interpretation of any contractual obligation.",
        "STANDARD_FAIR", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 10.2",
        "Standard boilerplate headings clause.",
    ),
    (
        "During active university semesters, part-time work commitments shall not exceed 20 hours per week, with flexible scheduling agreed weekly to avoid timetable conflicts.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08", "Clause 1.2",
        "Student-centric part-time weekly hour cap aligned with academic classes.",
    ),
    (
        "The graduate trainee shall accrue 1.5 days of paid earned leave per completed month plus statutory public holidays and 10 days of annual sick leave.",
        "STANDARD_FAIR", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 1.4",
        "Clear, statutory-aligned leave entitlement for new graduates.",
    ),
    (
        "Hostel residency includes high-speed Wi-Fi, potable filtered drinking water, regular housekeeping of common corridors, and 24-hour emergency security assistance.",
        "STANDARD_FAIR", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05", "Clause 1.2",
        "Enumerates core hostel utilities and student safety amenities clearly.",
    ),
    (
        "A signed joint move-in inventory checklist recording furniture, electrical fixtures, and meter readings shall be completed on arrival and countersigned by both resident and warden.",
        "STANDARD_FAIR", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 1.3",
        "Establishes objective baseline documentation protecting students at move-out.",
    ),
    (
        "The Company maintains a strict zero-tolerance policy against workplace harassment and discrimination, providing confidential access to the Internal Complaints Committee for all interns.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_global_05", "Clause 10.1",
        "Essential student workplace safety and anti-harassment protection.",
    ),
    (
        "The student contractor operates as an independent educational content creator; nothing in this Agreement establishes an agency, partnership, or joint venture between the parties.",
        "STANDARD_FAIR", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_notes_04", "Clause 10.1",
        "Standard independent-relationship clarification clause.",
    ),
    (
        "This Agreement may be executed electronically in two counterparts via verified digital signature, each of which constitutes an original binding copy.",
        "STANDARD_FAIR", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 9.1",
        "Standard electronic execution and counterparts clause.",
    ),
    (
        "The University Research Office shall facilitate academic credit transfer, ethics compliance guidance, and laboratory orientation before the student begins experimental work.",
        "STANDARD_FAIR", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01", "Clause 1.2",
        "Clear institutional onboarding and academic credit commitment.",
    ),
    (
        "Sponsor logos, promotional banners, and booth artwork shall be provided in print-ready vector format at least ten (10) days prior to the festival opening ceremony.",
        "STANDARD_FAIR", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02", "Clause 1.3",
        "Reasonable operational deadline for sponsor branding assets.",
    ),
    (
        "Hostel residents shall receive at least 48 hours' advance written notice before routine room maintenance inspections, except in urgent electrical or plumbing emergencies.",
        "STANDARD_FAIR", "low", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06", "Clause 7.2",
        "Protects student privacy and reasonable notice for room entry.",
    ),
    (
        "All personal data collected during student onboarding—such as bank account details and tax IDs—shall be used strictly for payroll and statutory compliance in accordance with data protection laws.",
        "STANDARD_FAIR", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07", "Clause 10.1",
        "Transparent employee data privacy commitment.",
    ),
    (
        "The coding bootcamp shall provide students with downloadable syllabus outlines, weekly mentor office hours, and transparent rubric criteria for earning the completion certificate.",
        "STANDARD_FAIR", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_bootcamp_02", "Clause 1.2",
        "Clear educational deliverables and transparent assessment criteria.",
    ),
    (
        "The host university shall provide visiting exchange students with airport arrival orientation, campus health clinic access, and an assigned international student advisor.",
        "STANDARD_FAIR", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03", "Clause 1.2",
        "Supportive exchange student arrival and welfare provisions.",
    ),
    (
        "Student lab fellows shall complete mandatory chemical and electrical safety training prior to operating shared fabrication machinery.",
        "STANDARD_FAIR", "low", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04", "Clause 1.2",
        "Standard academic laboratory safety prerequisite.",
    ),
    (
        "Neither party may assign its rights or delegate core obligations under this Agreement to an unrelated third party without obtaining the other party's prior written consent.",
        "STANDARD_FAIR", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05", "Clause 10.2",
        "Balanced mutual anti-assignment boilerplate.",
    ),
    (
        "The EdTech platform shall provide clear technical style guidelines and review submitted lesson recordings within five (5) business days with actionable written revision notes.",
        "STANDARD_FAIR", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01", "Clause 1.2",
        "Predictable review turnaround preventing indefinite acceptance delays.",
    ),
    (
        "Campus ambassadors shall receive an official certificate of appreciation and verified LinkedIn recommendation detailing campaign metrics upon completing the semester program.",
        "STANDARD_FAIR", "low", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_ambassador_02", "Clause 1.3",
        "Clear recognition and verifiable credential commitment for student ambassadors.",
    ),
    (
        "The event venue shall provide accessible ramps, emergency first-aid volunteers, and drinking water stations throughout the 36-hour student symposium.",
        "STANDARD_FAIR", "low", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03", "Clause 1.4",
        "Standard event accessibility and student welfare provision.",
    ),
    (
        "A failure or delay by either party in exercising any contractual right shall not operate as a permanent waiver of that right unless confirmed in writing.",
        "STANDARD_FAIR", "low", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06", "Clause 10.2",
        "Standard mutual non-waiver boilerplate.",
    ),
    (
        "Upon satisfactory conclusion of the six-month rotation, the employer shall conduct a documented career transition review providing honest feedback on full-time conversion eligibility.",
        "STANDARD_FAIR", "low", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04", "Clause 10.1",
        "Constructive career mentorship and transparent conversion review commitment.",
    ),
]


# ---------------------------------------------------------------------------
# Non-Substantive Structural / Informational Examples (for structural-filter eval)
# Tuple format:
#   (text, structural_type, domain, source_type, source_group, rationale)
# ---------------------------------------------------------------------------
RAW_STRUCTURAL_SEED: List[Tuple[str, str, str, str, str, str]] = [
    (
        "TECHNOVA SYSTEMS PRIVATE LIMITED\nSTUDENT INTERNSHIP AND TRAINING AGREEMENT",
        "document_header", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship",
        "Top-level agreement title and organization banner; non-substantive header.",
    ),
    (
        "This Internship Agreement (\"Agreement\") is entered into on 15th July 2026, by and between TechNova Systems Private Limited, with its corporate office at Tech Park, Outer Ring Road, Bangalore (\"Company\"), and Aarav Sharma, student of B.Tech Computer Science (\"Intern\").",
        "recital", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship",
        "Introductory party identification recital framing the agreement.",
    ),
    (
        "WHEREAS, the Company agrees to provide practical software engineering training, and the Intern agrees to accept such training under the terms and conditions outlined below:",
        "recital", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship",
        "Background WHEREAS recital; does not impose standalone risk obligations.",
    ),
    (
        "IN WITNESS WHEREOF, the parties hereto have executed this Agreement as of the date first written above.",
        "signature_block", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship",
        "Standard contract execution witness line.",
    ),
    (
        "For TechNova Systems Pvt. Ltd.                For the Intern:\nAuthorised Signatory                          Aarav Sharma (Student)",
        "signature_block", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_technova_internship",
        "Dual-party signature block for execution.",
    ),
    (
        "FICTIONAL TEST DATA — FOR EVALUATION AND TESTING PURPOSES ONLY",
        "disclaimer", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship",
        "Test-data watermark banner; informational disclaimer only.",
    ),
    (
        "Document Reference: CPA/INT/TEST/2027-019 | Effective Date: 1st January 2027",
        "metadata_reference", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship",
        "Document reference number and effective date metadata line.",
    ),
    (
        "DISCLAIMER: This agreement is 100% fictional test data created exclusively for testing contract analysis workflows. All company names, personal names, monetary amounts, and obligations are simulated and have zero legal effect.",
        "disclaimer", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship",
        "Explicit fictional-data disclaimer excluded from risk scoring.",
    ),
    (
        "1. INTERNSHIP DURATION AND WORKING SCHEDULE",
        "section_heading", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship",
        "Standalone numbered section heading divider.",
    ),
    (
        "2. STIPEND AND PAYMENT CONDITIONS",
        "section_heading", "INTERNSHIP", "sample_agreement_extracted", "doc_sample_cloudpulse_internship",
        "Standalone numbered section heading divider.",
    ),
    (
        "Ref: AGS/CAMPUS/2026/0842\nDate: 20th August 2026",
        "metadata_reference", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement",
        "Offer letter reference and issue date metadata.",
    ),
    (
        "To: Pooja Verma\nSubject: Offer of Appointment as Graduate Engineer Trainee (GET)",
        "metadata_reference", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement",
        "Addressee and subject metadata block.",
    ),
    (
        "Sincerely,\nApex Global Solutions Inc.                   Accepted by:\nHR Director                                  Pooja Verma (Candidate)",
        "signature_block", "PLACEMENT_EMPLOYMENT", "sample_agreement_extracted", "doc_sample_apex_placement",
        "Appointment acceptance signature block.",
    ),
    (
        "GREENVALLEY STUDENT RESIDENCES LLP\nCAMPUS RESIDENCY & HOSTEL ROOM ALLOTMENT AGREEMENT",
        "document_header", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel",
        "Hostel residency title header block.",
    ),
    (
        "TERMS AND CONDITIONS OF HOSTEL OCCUPANCY:",
        "section_heading", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel",
        "Introductory heading preceding numbered hostel terms.",
    ),
    (
        "Hostel Warden & Management                   Resident Student:\nGreenValley Residences                       Rohan Patel",
        "signature_block", "HOSTEL_RESIDENCY", "sample_agreement_extracted", "doc_sample_greenvalley_hostel",
        "Hostel warden and resident signature block.",
    ),
    (
        "CAMPUSFEST 2026 - ANNUAL STUDENT TECHNICAL SYMPOSIUM\nCORPORATE SPONSORSHIP AGREEMENT",
        "document_header", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship",
        "Event sponsorship contract title banner.",
    ),
    (
        "For NextGen Brands Corp                       For IEEE Student Committee:\nMarketing Lead                                Lead Student Convener",
        "signature_block", "EVENT_SPONSORSHIP", "sample_agreement_extracted", "doc_sample_campusfest_sponsorship",
        "Sponsor and student convener execution block.",
    ),
    (
        "SKILLPRO EDUCATION TECH SOLUTIONS\nINDEPENDENT STUDENT TUTOR AND CONTENT CREATOR AGREEMENT",
        "document_header", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service",
        "Student service agreement document title.",
    ),
    (
        "This Independent Contractor Agreement is entered into between SkillPro Education Tech Solutions (\"Company\") and Sneha Mukherjee (\"Student Contractor\").",
        "recital", "STUDENT_SERVICE", "sample_agreement_extracted", "doc_sample_skillpro_service",
        "Party identification preamble recital.",
    ),
    (
        "Document ID: HR/CAMPUS/2027/1145 | Version 2.1 | Confidential Draft",
        "metadata_reference", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_fair_06",
        "Document control metadata line.",
    ),
    (
        "Date: 12th February 2027\nFrom: Office of Student Housing & Residential Life\nTo: Incoming Postgraduate Residents",
        "metadata_reference", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_fair_05",
        "Memorandum-style housing header metadata.",
    ),
    (
        "WHEREAS, the University Research Lab operates an interdisciplinary robotics program and wishes to engage a student fellow for the Spring 2027 semester;",
        "recital", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01",
        "Background academic fellowship WHEREAS recital.",
    ),
    (
        "NOW, THEREFORE, in consideration of the mutual covenants and promises herein contained, the parties hereby agree as follows:",
        "recital", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01",
        "Standard transitional contract recital line.",
    ),
    (
        "3. INTELLECTUAL PROPERTY AND RESEARCH RIGHTS",
        "section_heading", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_research_01",
        "Standalone section divider heading.",
    ),
    (
        "4. SECURITY DEPOSIT AND CHECK-OUT PROTOCOL",
        "section_heading", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06",
        "Standalone numbered section heading.",
    ),
    (
        "5. SPONSORSHIP DELIVERABLES AND MEDIA GUIDELINES",
        "section_heading", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02",
        "Standalone event sponsorship section heading.",
    ),
    (
        "6. TERMINATION AND EXIT HANDOVER",
        "section_heading", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_dev_05",
        "Standalone section heading.",
    ),
    (
        "______________________________\nAuthorized Signatory (Campus Hiring)\nDate: 18-Aug-2026",
        "signature_block", "PLACEMENT_EMPLOYMENT", "curated_synthetic", "tpl_place_product_07",
        "Execution signature placeholder block.",
    ),
    (
        "Accepted and Agreed:\n______________________________\nStudent Signature & University Roll Number",
        "signature_block", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08",
        "Student counter-signature execution block.",
    ),
    (
        "ANNEXURE A — SCHEDULE OF PRE-EXISTING STUDENT PROJECTS AND OPEN-SOURCE REPOSITORIES",
        "section_heading", "INTERNSHIP", "curated_synthetic", "tpl_intern_fair_08",
        "Schedule/annexure title banner.",
    ),
    (
        "NOTICE: This sample student housing template is provided for academic benchmarking and software validation only and does not constitute legal advice.",
        "disclaimer", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_coliving_06",
        "Informational non-legal-advice disclaimer.",
    ),
    (
        "Dear Candidate,\nCongratulations on your selection for the Summer 2027 Product Design Internship cohort.",
        "recital", "INTERNSHIP", "curated_synthetic", "tpl_intern_design_04",
        "Offer letter salutation and welcome greeting.",
    ),
    (
        "MEMORANDUM OF UNDERSTANDING — ANNUAL INTER-COLLEGIATE HACKATHON 2027",
        "document_header", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_hackathon_02",
        "Event MOU document banner.",
    ),
    (
        "Reference No.: LAB/FELLOW/2027-09 | Effective Term: Jan 2027 – Jun 2027",
        "metadata_reference", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_lab_04",
        "Fellowship agreement reference and term metadata.",
    ),
    (
        "Witness 1: ________________________          Witness 2: ________________________",
        "signature_block", "HOSTEL_RESIDENCY", "curated_synthetic", "tpl_hostel_pg_03",
        "Witness execution signature lines.",
    ),
    (
        "INTERNATIONAL EXCHANGE CONSORTIUM\nSTUDENT MOBILITY AND SCHOLARSHIP UNDERTAKING",
        "document_header", "GENERAL_STUDENT_AGREEMENT", "curated_synthetic", "tpl_general_exchange_03",
        "Program title header block.",
    ),
    (
        "Subject: Freelance Educational Content Development Contract (DSA Track)",
        "metadata_reference", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_edtech_01",
        "Subject line metadata header.",
    ),
    (
        "DISCLAIMER: All organizational names, fee figures, and student identities in this fixture are synthetic and carry zero binding legal force.",
        "disclaimer", "STUDENT_SERVICE", "curated_synthetic", "tpl_service_freelance_03",
        "Synthetic fixture disclaimer.",
    ),
    (
        "IN WITNESS WHEREOF, the Sponsor and the Student Organizing Committee have affixed their authorized signatures below.",
        "signature_block", "EVENT_SPONSORSHIP", "curated_synthetic", "tpl_event_techfest_03",
        "Closing execution witness statement.",
    ),
]


def _seed_tuples_to_dicts(
    seed_records: Sequence[Tuple[str, str, str, str, str, str, str, str]]
) -> List[Dict[str, Any]]:
    """Convert raw seed tuples into structured candidate dictionaries."""
    items: List[Dict[str, Any]] = []
    for text, category, risk_level, domain, source_type, source_group, clause_ref, rationale in seed_records:
        items.append({
            "text": text.strip(),
            "category": category,
            "risk_level": risk_level,
            "domain": domain,
            "source_type": source_type,
            "source_group": source_group,
            "clause_ref": clause_ref,
            "rationale": rationale.strip(),
        })
    return items


def run_quality_gates(
    raw_records: Sequence[Dict[str, Any]],
    jaccard_threshold: float = NEAR_DUPLICATE_JACCARD_THRESHOLD,
    min_chars: int = MIN_CLAUSE_CHARS,
    max_chars: int = MAX_CLAUSE_CHARS,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Apply deterministic data quality gates:
      1. Schema / label / domain / risk / source_type validation
      2. Character length validation [min_chars, max_chars]
      3. Exact duplicate removal (case/whitespace-normalized SHA-256 hash)
      4. Near-duplicate detection & removal (token Jaccard >= jaccard_threshold)

    Returns (cleaned_records, gate_summary).
    """
    valid_categories = set(CLAUSE_CATEGORIES)
    valid_domains = set(AGREEMENT_DOMAINS)

    schema_rejected: List[Dict[str, Any]] = []
    length_rejected: List[Dict[str, Any]] = []
    exact_duplicates_removed: List[Dict[str, Any]] = []
    near_duplicates_removed: List[Dict[str, Any]] = []

    accepted: List[Dict[str, Any]] = []
    seen_hashes: Dict[str, int] = {}
    accepted_token_sets: List[Tuple[int, Set[str]]] = []

    for idx, rec in enumerate(raw_records):
        text = str(rec.get("text", "")).strip()
        category = rec.get("category")
        risk_level = rec.get("risk_level")
        domain = rec.get("domain")
        source_type = rec.get("source_type")
        source_group = str(rec.get("source_group", "")).strip()
        clause_ref = str(rec.get("clause_ref", "")).strip()
        rationale = str(rec.get("rationale", "")).strip()

        # Gate 1: Schema validation
        schema_errors: List[str] = []
        if category not in valid_categories:
            schema_errors.append(f"invalid category '{category}'")
        if domain not in valid_domains:
            schema_errors.append(f"invalid domain '{domain}'")
        if risk_level not in VALID_RISK_LEVELS:
            schema_errors.append(f"invalid risk_level '{risk_level}'")
        if source_type not in VALID_SOURCE_TYPES:
            schema_errors.append(f"invalid source_type '{source_type}'")
        if not source_group:
            schema_errors.append("missing source_group")
        if not clause_ref:
            schema_errors.append("missing clause_ref")
        if not rationale:
            schema_errors.append("missing rationale")

        if schema_errors:
            schema_rejected.append({
                "raw_index": idx,
                "errors": schema_errors,
                "text_preview": text[:80],
            })
            continue

        # Gate 2: Length validation
        if len(text) < min_chars or len(text) > max_chars:
            length_rejected.append({
                "raw_index": idx,
                "char_length": len(text),
                "text_preview": text[:80],
            })
            continue

        # Gate 3: Exact duplicate removal
        text_hash = compute_exact_hash(text)
        if text_hash in seen_hashes:
            exact_duplicates_removed.append({
                "raw_index": idx,
                "duplicate_of_accepted_index": seen_hashes[text_hash],
                "text_preview": text[:80],
            })
            continue

        # Gate 4: Near-duplicate detection (token Jaccard >= threshold)
        cand_tokens = tokenize_for_jaccard(text)
        matched_near_idx: Optional[int] = None
        matched_sim = 0.0
        for acc_idx, acc_tokens in accepted_token_sets:
            sim = compute_token_jaccard(cand_tokens, acc_tokens)
            if sim >= jaccard_threshold:
                matched_near_idx = acc_idx
                matched_sim = sim
                break

        if matched_near_idx is not None:
            near_duplicates_removed.append({
                "raw_index": idx,
                "near_duplicate_of_accepted_index": matched_near_idx,
                "jaccard_similarity": round(matched_sim, 4),
                "text_preview": text[:80],
            })
            continue

        acc_pos = len(accepted)
        seen_hashes[text_hash] = acc_pos
        accepted_token_sets.append((acc_pos, cand_tokens))
        accepted.append({
            "text": text,
            "category": category,
            "risk_level": risk_level,
            "domain": domain,
            "source_type": source_type,
            "source_group": source_group,
            "clause_ref": clause_ref,
            "rationale": rationale,
        })

    gate_summary = {
        "input_count": len(raw_records),
        "accepted_count": len(accepted),
        "schema_rejected_count": len(schema_rejected),
        "length_rejected_count": len(length_rejected),
        "exact_duplicates_removed_count": len(exact_duplicates_removed),
        "near_duplicates_removed_count": len(near_duplicates_removed),
        "near_duplicate_jaccard_threshold": jaccard_threshold,
        "min_chars": min_chars,
        "max_chars": max_chars,
        "schema_rejected_samples": schema_rejected[:10],
        "length_rejected_samples": length_rejected[:10],
        "exact_duplicate_samples": exact_duplicates_removed[:10],
        "near_duplicate_samples": near_duplicates_removed[:10],
    }
    return accepted, gate_summary


def stratified_group_split(
    records: Sequence[Dict[str, Any]],
    ratios: Optional[Dict[str, float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Deterministic, group-aware stratified 70/15/15 split into train/val/test.
    Every record sharing a `source_group` is assigned atomically to a single split,
    guaranteeing ZERO group leakage and ZERO text overlap across splits while
    balancing per-class proportions across train, val, and test.
    """
    split_ratios = ratios or SPLIT_RATIOS
    splits = ("train", "val", "test")

    # Aggregate groups deterministically
    groups_map: Dict[str, List[int]] = defaultdict(list)
    class_totals: Counter = Counter()
    for idx, rec in enumerate(records):
        groups_map[rec["source_group"]].append(idx)
        class_totals[rec["category"]] += 1

    total_n = len(records)
    target_total = {s: total_n * split_ratios[s] for s in splits}
    target_class = {
        s: {cat: class_totals[cat] * split_ratios[s] for cat in CLAUSE_CATEGORIES}
        for s in splits
    }

    # Precompute class counts per group and sort largest groups first with deterministic hash tie-break
    group_items: List[Tuple[str, List[int], Counter]] = []
    for grp, idxs in groups_map.items():
        grp_cats = Counter(records[i]["category"] for i in idxs)
        group_items.append((grp, idxs, grp_cats))

    group_items.sort(
        key=lambda item: (
            -len(item[1]),
            hashlib.md5(item[0].encode("utf-8")).hexdigest(),
        )
    )

    assigned_total: Dict[str, int] = {s: 0 for s in splits}
    assigned_class: Dict[str, Counter] = {s: Counter() for s in splits}
    group_to_split: Dict[str, str] = {}

    for grp, idxs, grp_cats in group_items:
        best_split = "train"
        best_delta = float("inf")

        for s in splits:
            # Marginal change in global normalized squared error when placing this group in split `s`
            t_tot = max(target_total[s], 1.0)
            before_tot = ((assigned_total[s] - t_tot) / t_tot) ** 2
            after_tot = ((assigned_total[s] + len(idxs) - t_tot) / t_tot) ** 2
            delta = 4.0 * (after_tot - before_tot)

            for cat, cnt in grp_cats.items():
                tgt = target_class[s][cat]
                if tgt > 0:
                    c_before = ((assigned_class[s][cat] - tgt) / tgt) ** 2
                    c_after = ((assigned_class[s][cat] + cnt - tgt) / tgt) ** 2
                    delta += (c_after - c_before)
                    # Strong preference to avoid leaving any class empty in val/test
                    if assigned_class[s][cat] == 0 and cnt > 0:
                        delta -= 0.35

            if delta < best_delta:
                best_delta = delta
                best_split = s

        group_to_split[grp] = best_split
        assigned_total[best_split] += len(idxs)
        assigned_class[best_split].update(grp_cats)

    # Materialize final split-annotated records with deterministic IDs
    annotated: List[Dict[str, Any]] = []
    for seq_idx, rec in enumerate(records, start=1):
        entry = dict(rec)
        entry["id"] = f"CG-{seq_idx:04d}"
        entry["split"] = group_to_split[rec["source_group"]]
        annotated.append(entry)

    # Post-split leakage verification (hard checks)
    split_groups: Dict[str, Set[str]] = {s: set() for s in splits}
    split_hashes: Dict[str, Set[str]] = {s: set() for s in splits}
    split_class_counts: Dict[str, Dict[str, int]] = {
        s: {cat: 0 for cat in CLAUSE_CATEGORIES} for s in splits
    }

    for entry in annotated:
        s = entry["split"]
        split_groups[s].add(entry["source_group"])
        split_hashes[s].add(compute_exact_hash(entry["text"]))
        split_class_counts[s][entry["category"]] += 1

    group_overlap_train_val = len(split_groups["train"] & split_groups["val"])
    group_overlap_train_test = len(split_groups["train"] & split_groups["test"])
    group_overlap_val_test = len(split_groups["val"] & split_groups["test"])

    text_overlap_train_val = len(split_hashes["train"] & split_hashes["val"])
    text_overlap_train_test = len(split_hashes["train"] & split_hashes["test"])
    text_overlap_val_test = len(split_hashes["val"] & split_hashes["test"])

    total_group_overlap = group_overlap_train_val + group_overlap_train_test + group_overlap_val_test
    total_text_overlap = text_overlap_train_val + text_overlap_train_test + text_overlap_val_test

    if total_group_overlap != 0:
        raise RuntimeError(f"Group leakage detected across splits: {total_group_overlap}")
    if total_text_overlap != 0:
        raise RuntimeError(f"Text overlap detected across splits: {total_text_overlap}")

    split_meta = {
        "target_ratios": dict(split_ratios),
        "actual_counts": {s: assigned_total[s] for s in splits},
        "actual_ratios": {
            s: round(assigned_total[s] / total_n, 4) if total_n else 0.0 for s in splits
        },
        "group_counts": {s: len(split_groups[s]) for s in splits},
        "per_class_counts": split_class_counts,
        "leakage_checks": {
            "group_overlap_train_val": group_overlap_train_val,
            "group_overlap_train_test": group_overlap_train_test,
            "group_overlap_val_test": group_overlap_val_test,
            "text_overlap_train_val": text_overlap_train_val,
            "text_overlap_train_test": text_overlap_train_test,
            "text_overlap_val_test": text_overlap_val_test,
            "zero_group_leakage": total_group_overlap == 0,
            "zero_text_overlap": total_text_overlap == 0,
            "all_classes_in_all_splits": all(
                split_class_counts[s][cat] > 0 for s in splits for cat in CLAUSE_CATEGORIES
            ),
        },
    }
    return annotated, split_meta


def _build_structural_records() -> List[Dict[str, Any]]:
    """Build deduplicated non-substantive structural/informational records."""
    seen_hashes: Set[str] = set()
    structural_records: List[Dict[str, Any]] = []
    for text, structural_type, domain, source_type, source_group, rationale in RAW_STRUCTURAL_SEED:
        clean_t = text.strip()
        h = compute_exact_hash(clean_t)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        structural_records.append({
            "id": f"ST-{len(structural_records) + 1:04d}",
            "text": clean_t,
            "structural_type": structural_type,
            "domain": domain,
            "source_type": source_type,
            "source_group": source_group,
            "is_substantive": False,
            "rationale": rationale.strip(),
        })
    return structural_records


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(REQUIRED_CLAUSE_FIELDS))
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in REQUIRED_CLAUSE_FIELDS})


def _generate_dataset_card(report: Dict[str, Any]) -> str:
    """Render an auditable academic DATASET_CARD.md from real computed statistics."""
    cat_dist = report["category_distribution"]
    dom_dist = report["domain_distribution"]
    risk_dist = report["risk_distribution"]
    src_dist = report["source_type_distribution"]
    gates = report["quality_gates"]
    splits = report["split_summary"]
    leakage = splits["leakage_checks"]

    lines = [
        "# ClauseGuard AI — Student Agreement Clause Dataset Card",
        "",
        f"**Dataset Version:** `{report['dataset_version']}`  ",
        f"**Generated At (UTC):** `{report['generated_at_utc']}`  ",
        f"**Total Validated Substantive Clauses:** `{report['total_clauses']}` (expanded from `{report['baseline_model_dataset_size']}` baseline clauses)  ",
        f"**Separate Structural/Informational Examples:** `{report['structural_examples_count']}`  ",
        f"**Distinct Document/Template Source Groups:** `{report['distinct_source_groups']}`  ",
        f"**Active ML Model Retraining Status:** `{report['retraining_status']}`  ",
        "",
        "> **Honest Status Notice:** This dataset has been prepared, validated, and split into leakage-free",
        "> `train` / `val` / `test` partitions for future supervised training and evaluation. The currently",
        "> active baseline classifier in `backend/models/` was trained on the original 125-clause corpus in",
        "> `backend/train_eval_model.py` and has **NOT** been retrained yet (`PENDING_AUTHORIZATION`).",
        "",
        "---",
        "",
        "## 1. Purpose & Scope",
        "",
        "This dataset supports multi-class legal clause categorization and student risk analysis across",
        "six student-facing contract domains: `INTERNSHIP`, `PLACEMENT_EMPLOYMENT`, `HOSTEL_RESIDENCY`,",
        "`EVENT_SPONSORSHIP`, `STUDENT_SERVICE`, and `GENERAL_STUDENT_AGREEMENT`.",
        "",
        "A separate file (`datasets/structural_examples.jsonl`) stores non-substantive document blocks",
        "(headers, recitals, metadata references, section titles, signature blocks, and disclaimers) to",
        "benchmark structural filtering without contaminating substantive clause classification.",
        "",
        "---",
        "",
        "## 2. Schema (`clauses_master.jsonl` & `clauses_master.csv`)",
        "",
        "| Field | Type | Description |",
        "| :--- | :--- | :--- |",
        "| `id` | `string` | Deterministic clause identifier (`CG-0001` …) |",
        "| `text` | `string` | Clean contractual clause text (35–1500 chars) |",
        "| `category` | `string` | One of the 10 `CLAUSE_CATEGORIES` from `backend/config.py` |",
        "| `risk_level` | `string` | Per-clause student risk grade (`high`, `medium`, `low`) |",
        "| `domain` | `string` | One of the 6 `AGREEMENT_DOMAINS` from `backend/config.py` |",
        "| `source_type` | `string` | `curated_synthetic` or `sample_agreement_extracted` |",
        "| `source_group` | `string` | Document/template group ID enforced atomically across splits |",
        "| `clause_ref` | `string` | Clause reference label (e.g., `Clause 3.2`) |",
        "| `rationale` | `string` | Auditable annotation justification for category and risk level |",
        "| `split` | `string` | Partition assignment (`train`, `val`, or `test`) |",
        "",
        "---",
        "",
        "## 3. Category Distribution (All 10 Classes)",
        "",
        "| Category | Total | Train | Val | Test |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for cat in CLAUSE_CATEGORIES:
        total_c = cat_dist.get(cat, 0)
        tr_c = splits["per_class_counts"]["train"].get(cat, 0)
        va_c = splits["per_class_counts"]["val"].get(cat, 0)
        te_c = splits["per_class_counts"]["test"].get(cat, 0)
        lines.append(f"| `{cat}` | {total_c} | {tr_c} | {va_c} | {te_c} |")

    lines.extend([
        "",
        "---",
        "",
        "## 4. Domain, Risk Level & Provenance Breakdown",
        "",
        "### Domains",
        "| Domain | Count |",
        "| :--- | :---: |",
    ])
    for dom in AGREEMENT_DOMAINS:
        lines.append(f"| `{dom}` | {dom_dist.get(dom, 0)} |")

    lines.extend([
        "",
        "### Risk Levels",
        "| Risk Level | Count |",
        "| :--- | :---: |",
    ])
    for rl in ("high", "medium", "low"):
        lines.append(f"| `{rl}` | {risk_dist.get(rl, 0)} |")

    lines.extend([
        "",
        "### Source Provenance",
        "| Source Type | Count |",
        "| :--- | :---: |",
    ])
    for st in ("curated_synthetic", "sample_agreement_extracted"):
        lines.append(f"| `{st}` | {src_dist.get(st, 0)} |")

    lines.extend([
        "",
        "---",
        "",
        "## 5. Automated Quality Gates & Leakage Verification",
        "",
        f"- **Raw Input Records:** `{gates['input_count']}`",
        f"- **Validated Accepted Records:** `{gates['accepted_count']}`",
        f"- **Schema Rejected:** `{gates['schema_rejected_count']}`",
        f"- **Length Rejected (outside {gates['min_chars']}–{gates['max_chars']} chars):** `{gates['length_rejected_count']}`",
        f"- **Exact Duplicates Removed:** `{gates['exact_duplicates_removed_count']}`",
        f"- **Near-Duplicates Removed (Token Jaccard >= {gates['near_duplicate_jaccard_threshold']}):** `{gates['near_duplicates_removed_count']}`",
        f"- **Split Counts:** `train={splits['actual_counts']['train']}` ({splits['actual_ratios']['train']*100:.1f}%), "
        f"`val={splits['actual_counts']['val']}` ({splits['actual_ratios']['val']*100:.1f}%), "
        f"`test={splits['actual_counts']['test']}` ({splits['actual_ratios']['test']*100:.1f}%)",
        f"- **Zero Source-Group Leakage Across Splits:** `{leakage['zero_group_leakage']}`",
        f"- **Zero Text Overlap Across Splits:** `{leakage['zero_text_overlap']}`",
        f"- **All 10 Classes Present in Train/Val/Test:** `{leakage['all_classes_in_all_splits']}`",
        "",
        "---",
        "",
        "## 6. Ethical & Legal Limitations",
        "",
        "- Clauses are synthetic or extracted from fictional sample agreements; no real student PII is included.",
        "- Annotations reflect student-protection educational framing and are not formal legal advice.",
        "- Any future model retraining using these splits must preserve `source_group` boundaries and report held-out `test.jsonl` metrics transparently.",
        "",
    ])
    return "\n".join(lines)


def build_dataset(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Execute the full auditable dataset preparation pipeline and persist all
    artifacts under `datasets/` (or a custom `output_dir`).
    Does NOT retrain any ML model.
    """
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_DATASETS_DIR
    splits_dir = out_dir / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    raw_candidates = _seed_tuples_to_dicts(RAW_CLAUSE_SEED)
    cleaned_clauses, gate_summary = run_quality_gates(raw_candidates)
    annotated_clauses, split_summary = stratified_group_split(cleaned_clauses)
    structural_examples = _build_structural_records()

    # Partition by split
    train_records = [r for r in annotated_clauses if r["split"] == "train"]
    val_records = [r for r in annotated_clauses if r["split"] == "val"]
    test_records = [r for r in annotated_clauses if r["split"] == "test"]

    # Write dataset files
    master_jsonl = out_dir / "clauses_master.jsonl"
    master_csv = out_dir / "clauses_master.csv"
    structural_jsonl = out_dir / "structural_examples.jsonl"
    train_jsonl = splits_dir / "train.jsonl"
    val_jsonl = splits_dir / "val.jsonl"
    test_jsonl = splits_dir / "test.jsonl"
    report_json = out_dir / "dataset_report.json"
    dataset_card_md = out_dir / "DATASET_CARD.md"

    _write_jsonl(master_jsonl, annotated_clauses)
    _write_csv(master_csv, annotated_clauses)
    _write_jsonl(structural_jsonl, structural_examples)
    _write_jsonl(train_jsonl, train_records)
    _write_jsonl(val_jsonl, val_records)
    _write_jsonl(test_jsonl, test_records)

    cat_dist = dict(Counter(r["category"] for r in annotated_clauses))
    dom_dist = dict(Counter(r["domain"] for r in annotated_clauses))
    risk_dist = dict(Counter(r["risk_level"] for r in annotated_clauses))
    src_dist = dict(Counter(r["source_type"] for r in annotated_clauses))
    struct_type_dist = dict(Counter(r["structural_type"] for r in structural_examples))
    char_lengths = [len(r["text"]) for r in annotated_clauses]

    report: Dict[str, Any] = {
        "dataset_version": "2.0.0-prepared",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "retraining_status": "PENDING_AUTHORIZATION",
        "retraining_note": (
            "Expanded dataset is built, validated, and leakage-verified, but the active baseline "
            "TF-IDF + Logistic Regression model in backend/models/ (trained on 125 baseline clauses) "
            "has NOT been retrained yet."
        ),
        "baseline_model_dataset_size": BASELINE_MODEL_DATASET_SIZE,
        "total_clauses": len(annotated_clauses),
        "num_categories": len(cat_dist),
        "min_clauses_per_category": min(cat_dist.values()) if cat_dist else 0,
        "max_clauses_per_category": max(cat_dist.values()) if cat_dist else 0,
        "distinct_source_groups": len({r["source_group"] for r in annotated_clauses}),
        "structural_examples_count": len(structural_examples),
        "structural_type_distribution": struct_type_dist,
        "text_length_stats": {
            "min_chars": min(char_lengths) if char_lengths else 0,
            "max_chars": max(char_lengths) if char_lengths else 0,
            "avg_chars": round(sum(char_lengths) / len(char_lengths), 1) if char_lengths else 0.0,
        },
        "category_distribution": {cat: cat_dist.get(cat, 0) for cat in CLAUSE_CATEGORIES},
        "domain_distribution": {dom: dom_dist.get(dom, 0) for dom in AGREEMENT_DOMAINS},
        "risk_distribution": {rl: risk_dist.get(rl, 0) for rl in ("high", "medium", "low")},
        "source_type_distribution": {
            st: src_dist.get(st, 0) for st in ("curated_synthetic", "sample_agreement_extracted")
        },
        "quality_gates": gate_summary,
        "split_summary": split_summary,
        "artifacts": {
            "clauses_master_jsonl": str(master_jsonl),
            "clauses_master_csv": str(master_csv),
            "structural_examples_jsonl": str(structural_jsonl),
            "train_jsonl": str(train_jsonl),
            "val_jsonl": str(val_jsonl),
            "test_jsonl": str(test_jsonl),
            "dataset_report_json": str(report_json),
            "dataset_card_md": str(dataset_card_md),
        },
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    with open(dataset_card_md, "w", encoding="utf-8") as f:
        f.write(_generate_dataset_card(report))

    return report


def get_dataset_status(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Read `datasets/dataset_report.json` cleanly and return dataset readiness
    metadata alongside an honest `retraining_status: "PENDING_AUTHORIZATION"` notice.
    """
    out_dir = Path(output_dir) if output_dir is not None else DEFAULT_DATASETS_DIR
    report_path = out_dir / "dataset_report.json"

    honest_note = (
        "The expanded clause dataset is prepared and audited, while the active baseline ML "
        "model (trained on 125 clauses in backend/train_eval_model.py) has NOT been retrained yet."
    )

    if not report_path.exists():
        return {
            "status": "NOT_BUILT",
            "dataset_ready": False,
            "retraining_status": "PENDING_AUTHORIZATION",
            "active_model_retrained": False,
            "baseline_model_dataset_size": BASELINE_MODEL_DATASET_SIZE,
            "note": (
                "Expanded dataset artifacts have not been generated yet. Run "
                "`python -m backend.dataset_pipeline` to build datasets/. "
                + honest_note
            ),
            "report_path": str(report_path),
        }

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "status": "ERROR",
            "dataset_ready": False,
            "retraining_status": "PENDING_AUTHORIZATION",
            "active_model_retrained": False,
            "note": f"Failed to read dataset report: {exc}",
            "report_path": str(report_path),
        }

    leakage = report.get("split_summary", {}).get("leakage_checks", {})
    return {
        "status": "READY_FOR_EVALUATION",
        "dataset_ready": True,
        "dataset_version": report.get("dataset_version"),
        "generated_at_utc": report.get("generated_at_utc"),
        "retraining_status": "PENDING_AUTHORIZATION",
        "active_model_retrained": False,
        "baseline_model_dataset_size": report.get("baseline_model_dataset_size", BASELINE_MODEL_DATASET_SIZE),
        "total_clauses": report.get("total_clauses", 0),
        "structural_examples_count": report.get("structural_examples_count", 0),
        "num_categories": report.get("num_categories", 0),
        "distinct_source_groups": report.get("distinct_source_groups", 0),
        "split_counts": report.get("split_summary", {}).get("actual_counts", {}),
        "category_distribution": report.get("category_distribution", {}),
        "domain_distribution": report.get("domain_distribution", {}),
        "leakage_checks": leakage,
        "note": report.get("retraining_note", honest_note),
        "report_path": str(report_path),
    }


if __name__ == "__main__":
    summary = build_dataset()
    print(f"Dataset built successfully: {summary['total_clauses']} clauses across {summary['num_categories']} categories.")
    print(f"Structural examples: {summary['structural_examples_count']}")
    print(f"Distinct source groups: {summary['distinct_source_groups']}")
    print(f"Quality gates: {summary['quality_gates']['exact_duplicates_removed_count']} exact dupes, "
          f"{summary['quality_gates']['near_duplicates_removed_count']} near dupes removed.")
    print(f"Splits: {summary['split_summary']['actual_counts']} | "
          f"Leakage checks: {summary['split_summary']['leakage_checks']}")
    print(f"Retraining status: {summary['retraining_status']}")
