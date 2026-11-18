"""
ClauseGuard AI - Configuration and Design System Tokens
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = BASE_DIR / "backend"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = BACKEND_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
SAMPLES_DIR = BACKEND_DIR / "sample_agreements"
REPORTS_DIR = BACKEND_DIR / "generated_reports"
UPLOADS_DIR = BACKEND_DIR / "uploads"

# Ensure directories exist
for directory in [MODELS_DIR, DATASETS_DIR, SAMPLES_DIR, REPORTS_DIR, UPLOADS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Color Palette (Legal-Tech Design System)
PALETTE = {
    "deep_navy": "#102B35",
    "forest_teal": "#245B52",
    "sage": "#D7E6DF",
    "muted_gold": "#E8C78B",
    "warm_bg": "#F3F5F2",
    "high_risk": "#DC2626",      # Restrained red
    "high_risk_bg": "#FEE2E2",
    "medium_risk": "#D97706",    # Amber
    "medium_risk_bg": "#FEF3C7",
    "low_risk": "#16A34A",       # Green
    "low_risk_bg": "#DCFCE7",
}

# Supported Student Agreement Domains
AGREEMENT_DOMAINS = [
    "INTERNSHIP",
    "PLACEMENT_EMPLOYMENT",
    "HOSTEL_RESIDENCY",
    "EVENT_SPONSORSHIP",
    "STUDENT_SERVICE",
    "GENERAL_STUDENT_AGREEMENT",
]

# Clause Categories
CLAUSE_CATEGORIES = [
    "FINANCIAL_PENALTY_BOND",
    "STIPEND_PAYMENT_TERMS",
    "DEPOSIT_REFUND",
    "TERMINATION_NOTICE",
    "CONFIDENTIALITY_NDA",
    "INTELLECTUAL_PROPERTY",
    "RESTRICTIVE_NON_COMPETE",
    "LIABILITY_INDEMNITY",
    "DISPUTE_JURISDICTION",
    "STANDARD_FAIR",
]

# Risk Level Mappings
RISK_LEVELS = {
    "HIGH": "high",
    "MEDIUM": "medium",
    "LOW": "low",
}
