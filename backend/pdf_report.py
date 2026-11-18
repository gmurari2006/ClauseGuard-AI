"""
ClauseGuard AI - PDF Audit Report Generator
Produces a professional, audit-consistent legal-tech PDF report using ReportLab.

Includes:
  - Branded header banner and consistent running headers/footers with Page X of Y
  - Document details clearly distinguishing extracted blocks vs. classified substantive clauses
  - Risk distribution summary with exact reconciled totals
  - Actual transparent risk score methodology and itemized calculation
  - Complete Student vs. Counterparty balance breakdown with all clauses accounted for
  - Flagged clauses with verbatim quotations, cross-clause relationships, student impact,
    verified legal context vs. automated risk flags, and recommended negotiation actions
  - Missing or unclear protective terms
  - Legal and educational disclaimer
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from xml.sax.saxutils import escape as xml_escape

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

from backend.config import REPORTS_DIR
from backend.rule_engine import calculate_fairness_indicator, compute_risk_score_breakdown


# Brand palette tokens (Deep Navy, Forest Teal, Sage, Muted Gold, Warm BG)
C_NAVY = colors.HexColor("#102B35")
C_TEAL = colors.HexColor("#245B52")
C_SAGE = colors.HexColor("#D7E6DF")
C_SAGE_LIGHT = colors.HexColor("#EDF4F1")
C_GOLD = colors.HexColor("#E8C78B")
C_GOLD_LIGHT = colors.HexColor("#FAF3E3")
C_WARM_BG = colors.HexColor("#F3F5F2")
C_RED = colors.HexColor("#DC2626")
C_RED_BG = colors.HexColor("#FEF2F2")
C_AMBER = colors.HexColor("#D97706")
C_AMBER_BG = colors.HexColor("#FFFBEB")
C_GREEN = colors.HexColor("#16A34A")
C_DARK_GRAY = colors.HexColor("#1E293B")
C_BODY_GRAY = colors.HexColor("#334155")
C_MUTED = colors.HexColor("#64748B")
C_BORDER = colors.HexColor("#CBD5E1")


def _esc(val: Any) -> str:
    """Safely escape text for ReportLab Paragraph XML markup and replace non-WinAnsi glyphs."""
    if val is None:
        return ""
    cleaned = " ".join(str(val).split())
    # Built-in PDF Helvetica lacks the Unicode Rupee glyph (U+20B9); render cleanly as 'Rs. '
    cleaned = cleaned.replace("₹ ", "Rs. ").replace("₹", "Rs. ")
    return xml_escape(cleaned)


class NumberedCanvas(pdf_canvas.Canvas):
    """Two-pass ReportLab canvas that draws running headers (pages 2+) and Page X of Y footers."""

    def __init__(self, *args, doc_filename: str = "Agreement", **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []
        self._doc_filename = doc_filename

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self._draw_chrome(num_pages)
            super().showPage()
        super().save()

    def _draw_chrome(self, total_pages: int):
        page_num = self._pageNumber
        width, height = letter
        margin = 40

        self.saveState()

        # Subtle running header on pages 2+
        if page_num > 1:
            self.setStrokeColor(C_SAGE)
            self.setLineWidth(0.75)
            self.line(margin, height - 34, width - margin, height - 34)
            self.setFont("Helvetica-Bold", 7.5)
            self.setFillColor(C_NAVY)
            self.drawString(margin, height - 28, "CLAUSEGUARD AI  |  Student Agreement Risk Analysis Report")
            self.setFont("Helvetica", 7.5)
            self.setFillColor(C_MUTED)
            short_name = self._doc_filename if len(self._doc_filename) <= 46 else self._doc_filename[:43] + "..."
            self.drawRightString(width - margin, height - 28, short_name)

        # Running footer on every page
        self.setStrokeColor(C_GOLD)
        self.setLineWidth(0.8)
        self.line(margin, 36, width - margin, 36)

        self.setFont("Helvetica", 7.5)
        self.setFillColor(C_MUTED)
        self.drawString(
            margin,
            24,
            "ClauseGuard AI  •  Automated Preliminary Risk Analysis (Educational Use Only — Not Legal Advice)"
        )
        self.setFont("Helvetica-Bold", 7.5)
        self.setFillColor(C_NAVY)
        self.drawRightString(width - margin, 24, f"Page {page_num} of {total_pages}")

        self.restoreState()


def generate_pdf_report(analysis_result: Dict[str, Any], output_filename: str) -> Path:
    """Generate a complete, audit-consistent downloadable PDF report."""
    output_path = REPORTS_DIR / output_filename
    doc_meta = analysis_result.get("document_info", {})
    doc_filename = doc_meta.get("filename", "Agreement")

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=46,
        bottomMargin=50,
        title=f"ClauseGuard AI Risk Report - {doc_filename}",
        author="ClauseGuard AI"
    )

    styles = getSampleStyleSheet()
    page_w = letter[0] - 80  # 532 pt usable content width

    # Typography styles
    title_style = ParagraphStyle(
        "DocTitle", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=18, leading=22,
        textColor=colors.white, alignment=TA_LEFT
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle", parent=styles["Normal"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=C_SAGE, alignment=TA_LEFT
    )
    r_head_style = ParagraphStyle(
        "RHead", parent=subtitle_style, alignment=TA_RIGHT,
        fontName="Helvetica-Bold", textColor=C_GOLD
    )
    r_sub_style = ParagraphStyle(
        "RSub", parent=subtitle_style, alignment=TA_RIGHT
    )
    h1_style = ParagraphStyle(
        "H1Custom", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=12, leading=16,
        textColor=C_NAVY, spaceBefore=14, spaceAfter=6
    )
    h2_style = ParagraphStyle(
        "H2Custom", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=10.5, leading=14,
        textColor=C_NAVY, spaceBefore=4, spaceAfter=4
    )
    body_style = ParagraphStyle(
        "BodyCustom", parent=styles["Normal"],
        fontName="Helvetica", fontSize=8.8, leading=13,
        textColor=C_BODY_GRAY, alignment=TA_LEFT
    )
    body_justify = ParagraphStyle(
        "BodyJustify", parent=body_style, alignment=TA_JUSTIFY
    )
    body_bold = ParagraphStyle(
        "BodyBold", parent=body_style,
        fontName="Helvetica-Bold", textColor=C_NAVY
    )
    small_muted = ParagraphStyle(
        "SmallMuted", parent=styles["Normal"],
        fontName="Helvetica", fontSize=7.8, leading=11,
        textColor=C_MUTED
    )
    quote_style = ParagraphStyle(
        "ContractQuote", parent=styles["Normal"],
        fontName="Helvetica-Oblique", fontSize=8.6, leading=12.5,
        textColor=C_DARK_GRAY, alignment=TA_LEFT
    )
    score_num_style = ParagraphStyle(
        "ScoreNum", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=24, leading=28,
        alignment=TA_CENTER
    )
    score_cat_style = ParagraphStyle(
        "ScoreCat", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=9.2, leading=12.5,
        alignment=TA_CENTER
    )
    score_lbl_style = ParagraphStyle(
        "ScoreLbl", parent=styles["Normal"],
        fontName="Helvetica-Bold", fontSize=7.2, leading=10,
        textColor=C_MUTED, alignment=TA_CENTER
    )

    # ------------------------------------------------------------------
    # Reconcile source data strictly so EVERY count in the PDF matches
    # ------------------------------------------------------------------
    all_extracted = analysis_result.get("clauses", [])
    classified_clauses = [c for c in all_extracted if not c.get("is_heading") and not c.get("is_structural")]
    structural_entries = [c for c in all_extracted if c.get("is_structural") and not c.get("is_heading")]
    heading_blocks = [c for c in all_extracted if c.get("is_heading")]

    high_clauses = [c for c in classified_clauses if c.get("final_risk_level") == "high"]
    med_clauses = [c for c in classified_clauses if c.get("final_risk_level") == "medium"]
    low_clauses = [c for c in classified_clauses if c.get("final_risk_level") == "low"]

    total_extracted = len(all_extracted)
    total_classified = len(classified_clauses)
    total_structural = len(structural_entries)
    total_headings = len(heading_blocks)
    cross_links_count = sum(1 for c in classified_clauses if c.get("related_clauses"))

    missing_items = analysis_result.get("missing_information", [])
    domain = analysis_result.get("domain", "STUDENT_AGREEMENT")

    # Always derive live breakdown from the exact classified clauses & missing items
    risk_breakdown = analysis_result.get("risk_score_breakdown")
    if not risk_breakdown or risk_breakdown.get("high_count") != len(high_clauses) or risk_breakdown.get("low_count") != len(low_clauses):
        risk_breakdown = compute_risk_score_breakdown(classified_clauses, missing_items)

    risk_score = risk_breakdown["final_score"]
    risk_cat = risk_breakdown["category"]

    # Always ensure fairness metrics account for 100% of classified clauses
    fairness = analysis_result.get("fairness", {})
    if (
        "moderate_caution_clause_count" not in fairness
        or (
            fairness.get("unilateral_clause_count", 0)
            + fairness.get("moderate_caution_clause_count", 0)
            + fairness.get("bilateral_clause_count", 0)
        ) != total_classified
    ):
        fairness = calculate_fairness_indicator(classified_clauses, domain)

    story = []

    # ==================================================================
    # 1. BRANDED HEADER BANNER
    # ==================================================================
    header_data = [
        [
            Paragraph("CLAUSEGUARD AI", title_style),
            Paragraph(
                f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
                r_head_style
            )
        ],
        [
            Paragraph("Smart Agreement Analysis &amp; Risk Detection for Students &amp; Educational Institutions", subtitle_style),
            Paragraph("Engine: ML Classifier + Rule Heuristics + Cross-Clause", r_sub_style)
        ]
    ]
    header_table = Table(header_data, colWidths=[332, 200])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 14),
        ("RIGHTPADDING", (0, 0), (-1, -1), 14),
        ("TOPPADDING", (0, 0), (-1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
        ("TOPPADDING", (0, 1), (-1, 1), 2),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 12),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 1), (-1, 1), 3, C_GOLD),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 10))

    # ==================================================================
    # 2. DOCUMENT OVERVIEW & OVERALL RISK SCORE (ZERO CLIPPING)
    # ==================================================================
    if risk_score >= 75:
        score_color = C_RED
        score_bg = C_RED_BG
    elif risk_score >= 40:
        score_color = C_AMBER
        score_bg = C_AMBER_BG
    else:
        score_color = C_GREEN
        score_bg = C_SAGE_LIGHT

    domain_readable = domain.replace("_", " ").title()
    doc_format = doc_meta.get("format", Path(doc_filename).suffix.lstrip(".").upper() or "TXT")

    meta_p = Paragraph(
        f"<b>Document Title:</b> {_esc(doc_filename)}<br/>"
        f"<b>Format &amp; Domain:</b> {_esc(doc_format)}  |  {_esc(domain_readable)}<br/>"
        f"<b>Word Count:</b> {doc_meta.get('word_count', 0):,} words<br/>"
        f"<b>Extracted Document Blocks:</b> {total_extracted} total "
        f"({total_classified} classified clauses + {total_structural} structural/informational + {total_headings} headings)<br/>"
        f"<b>Classified Clauses Scored:</b> <b>{total_classified}</b> "
        f"({len(high_clauses)} High, {len(med_clauses)} Medium, {len(low_clauses)} Low; "
        f"{cross_links_count} with cross-clause links)",
        body_style
    )

    # Stacked inner score cell with generous leading so heading never clips or overlaps
    score_cell_table = Table(
        [
            [Paragraph("OVERALL RISK SCORE", score_lbl_style)],
            [Paragraph(f"<font color='{score_color.hexval()}'>{risk_score} / 100</font>", score_num_style)],
            [Paragraph(f"<font color='{score_color.hexval()}'>{_esc(risk_cat.upper())}</font>", score_cat_style)],
        ],
        colWidths=[168]
    )
    score_cell_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (0, 0), 2),
        ("BOTTOMPADDING", (0, 0), (0, 0), 4),
        ("TOPPADDING", (0, 1), (0, 1), 2),
        ("BOTTOMPADDING", (0, 1), (0, 1), 6),
        ("TOPPADDING", (0, 2), (0, 2), 2),
        ("BOTTOMPADDING", (0, 2), (0, 2), 2),
    ]))

    info_table = Table([[meta_p, score_cell_table]], colWidths=[344, 188])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), C_WARM_BG),
        ("BACKGROUND", (1, 0), (1, 0), score_bg),
        ("BOX", (0, 0), (-1, -1), 1, C_SAGE),
        ("LINEBEFORE", (1, 0), (1, 0), 1.5, score_color),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("RIGHTPADDING", (0, 0), (0, 0), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (1, 0), (1, 0), 8),
        ("RIGHTPADDING", (1, 0), (1, 0), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 10))

    # ==================================================================
    # 3. RISK DISTRIBUTION SUMMARY (RECONCILED EXACT TOTALS)
    # ==================================================================
    story.append(Paragraph("1. Risk Distribution Summary", h1_style))

    denom = max(1, total_classified)
    pct_high = (len(high_clauses) / denom) * 100
    pct_med = (len(med_clauses) / denom) * 100
    pct_low = (len(low_clauses) / denom) * 100

    dist_data = [
        [
            Paragraph("<b>Risk Severity</b>", body_bold),
            Paragraph("<b>Classified Clauses</b>", ParagraphStyle("CAlign", parent=body_bold, alignment=TA_CENTER)),
            Paragraph("<b>Share</b>", ParagraphStyle("CAlign2", parent=body_bold, alignment=TA_CENTER)),
            Paragraph("<b>Preliminary Review Guidance</b>", body_bold),
        ],
        [
            Paragraph("<font color='#DC2626'><b>HIGH RISK (Critical Flag)</b></font>", body_style),
            Paragraph(f"<b>{len(high_clauses)}</b>", ParagraphStyle("D1", parent=body_style, alignment=TA_CENTER)),
            Paragraph(f"{pct_high:.1f}%", ParagraphStyle("D2", parent=body_style, alignment=TA_CENTER)),
            Paragraph("Significant student exposure; seek written amendment before signing.", body_style),
        ],
        [
            Paragraph("<font color='#D97706'><b>MEDIUM RISK (Caution Flag)</b></font>", body_style),
            Paragraph(f"<b>{len(med_clauses)}</b>", ParagraphStyle("D3", parent=body_style, alignment=TA_CENTER)),
            Paragraph(f"{pct_med:.1f}%", ParagraphStyle("D4", parent=body_style, alignment=TA_CENTER)),
            Paragraph("Condition-dependent or unbalanced term; clarify scope in writing.", body_style),
        ],
        [
            Paragraph("<font color='#16A34A'><b>LOW RISK (Standard / Fair)</b></font>", body_style),
            Paragraph(f"<b>{len(low_clauses)}</b>", ParagraphStyle("D5", parent=body_style, alignment=TA_CENTER)),
            Paragraph(f"{pct_low:.1f}%", ParagraphStyle("D6", parent=body_style, alignment=TA_CENTER)),
            Paragraph("Standard administrative, introductory, or balanced covenant.", body_style),
        ],
        [
            Paragraph("<b>TOTAL CLASSIFIED CLAUSES</b>", body_bold),
            Paragraph(f"<b>{total_classified}</b>", ParagraphStyle("DT1", parent=body_bold, alignment=TA_CENTER)),
            Paragraph("<b>100.0%</b>", ParagraphStyle("DT2", parent=body_bold, alignment=TA_CENTER)),
            Paragraph(
                f"<i>Excludes {total_structural} structural/informational entries &amp; {total_headings} headings from {total_extracted} blocks.</i>",
                small_muted
            ),
        ],
    ]
    dist_table = Table(dist_data, colWidths=[148, 86, 62, 236])
    dist_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_SAGE),
        ("BACKGROUND", (0, 4), (-1, 4), C_SAGE_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LINEABOVE", (0, 4), (-1, 4), 1.0, C_TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, 3), [colors.white, C_WARM_BG]),
    ]))
    story.append(dist_table)
    story.append(Spacer(1, 8))

    # ==================================================================
    # 4. RISK SCORE METHODOLOGY (ACTUAL TRANSPARENT CALCULATION)
    # ==================================================================
    rb = risk_breakdown
    r_align = ParagraphStyle("RAlign", parent=body_style, alignment=TA_RIGHT)
    r_bold = ParagraphStyle("RBold", parent=body_bold, alignment=TA_RIGHT)
    cap_note = (
        f"Raw score ({rb['raw_score']}) capped to upper bound {rb['bounds'][1]}."
        if rb["raw_score"] > rb["bounds"][1]
        else (
            f"Raw score ({rb['raw_score']}) floored to lower bound {rb['bounds'][0]}."
            if rb["raw_score"] < rb["bounds"][0]
            else "Within [10, 98] range (no cap applied)."
        )
    )

    score_calc_data = [
        [
            Paragraph("<b>Scoring Component</b>", body_bold),
            Paragraph("<b>Observed Count × Weight</b>", body_bold),
            Paragraph("<b>Contribution</b>", r_bold),
        ],
        [
            Paragraph("Contract Formality Baseline", body_style),
            Paragraph("Fixed starting baseline", body_style),
            Paragraph(f"+{rb['baseline']}", r_align),
        ],
        [
            Paragraph("High-Risk Classified Clauses", body_style),
            Paragraph(f"{rb['high_count']} clauses × +{rb['high_weight']}", body_style),
            Paragraph(f"+{rb['high_points']}", r_align),
        ],
        [
            Paragraph("Automated Rule Engine Flags", body_style),
            Paragraph(f"{rb['rule_flag_count']} triggered flags × +{rb['rule_flag_weight']}", body_style),
            Paragraph(f"+{rb['rule_flag_points']}", r_align),
        ],
        [
            Paragraph("Medium-Risk Classified Clauses", body_style),
            Paragraph(f"{rb['medium_count']} clauses × +{rb['medium_weight']}", body_style),
            Paragraph(f"+{rb['medium_points']}", r_align),
        ],
        [
            Paragraph("Missing Baseline Protective Terms", body_style),
            Paragraph(f"{rb['missing_count']} missing items × +{rb['missing_weight']}", body_style),
            Paragraph(f"+{rb['missing_points']}", r_align),
        ],
        [
            Paragraph("Low-Risk / Standard Clause Offset", body_style),
            Paragraph(f"{rb['low_count']} clauses × {rb['low_weight']}", body_style),
            Paragraph(f"{rb['low_offset']}", r_align),
        ],
        [
            Paragraph("<b>Uncapped Sum → Final Bounded Score</b>", body_bold),
            Paragraph(f"<i>{_esc(cap_note)}</i>", small_muted),
            Paragraph(f"<b>{rb['raw_score']} → {rb['final_score']}/100</b>", r_bold),
        ],
    ]
    score_calc_table = Table(score_calc_data, colWidths=[210, 222, 100])
    score_calc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_SAGE),
        ("BACKGROUND", (0, 7), (-1, 7), C_SAGE_LIGHT),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("LINEABOVE", (0, 7), (-1, 7), 1.0, C_TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, 6), [colors.white, C_WARM_BG]),
    ]))
    story.append(KeepTogether([
        Paragraph("2. Risk Score Methodology &amp; Live Calculation", h1_style),
        Paragraph(
            "The overall risk score (bounded between <b>10</b> and <b>98</b>) is computed deterministically from the "
            "classified substantive clauses, automated safeguard rule flags, and missing baseline protections. "
            "Thresholds: <b>75–98</b> Critical Risk, <b>55–74</b> Elevated Risk, <b>35–54</b> Moderate Caution, <b>10–34</b> Low Risk.",
            body_justify
        ),
        Spacer(1, 5),
        score_calc_table,
    ]))
    story.append(Spacer(1, 8))

    # ==================================================================
    # 5. STUDENT VS. COUNTERPARTY BALANCE ASSESSMENT (ALL CLAUSES RECONCILED)
    # ==================================================================
    story.append(Paragraph("3. Student vs. Counterparty Balance Assessment", h1_style))

    s_pct = fairness.get("student_favor_pct", 50)
    c_pct = fairness.get("counterparty_favor_pct", 50)
    uni_cnt = fairness.get("unilateral_clause_count", 0)
    mod_cnt = fairness.get("moderate_caution_clause_count", 0)
    bil_cnt = fairness.get("bilateral_clause_count", 0)
    s_pts = fairness.get("student_points", 0.0)
    c_pts = fairness.get("counterparty_points", 0.0)
    t_pts = fairness.get("total_points", 0.0)
    f_total = uni_cnt + mod_cnt + bil_cnt

    fairness_p = Paragraph(
        f"<b>Balance Verdict:</b> {_esc(fairness.get('fairness_label', 'Evaluated'))}  "
        f"(<b>{s_pct}% Student / Standard</b> vs. <b>{c_pct}% Counterparty Bias</b>)<br/>"
        f"{_esc(fairness.get('summary_verdict', ''))}<br/><br/>"
        f"<b>Clause Attribution ({f_total} of {total_classified} classified clauses accounted for):</b><br/>"
        f"• <b>{uni_cnt}</b> High-Risk Unilateral / Counterparty-Skewed clauses (weight 3.0 each = <b>{uni_cnt * 3.0:.1f}</b> counterparty pts)<br/>"
        f"• <b>{mod_cnt}</b> Medium-Risk Caution / Partially Unbalanced clauses (weight 1.5 each = <b>{mod_cnt * 1.5:.1f}</b> counterparty pts)<br/>"
        f"• <b>{bil_cnt}</b> Low-Risk Standard / Balanced clauses (weight 1.5 each = <b>{bil_cnt * 1.5:.1f}</b> student/standard pts)<br/>"
        f"<b>Percentage Formula:</b> Counterparty share = {c_pts:.1f} / {t_pts:.1f} "
        f"({(c_pts / max(0.1, t_pts)) * 100:.1f}%, bounded 8%–92%) = <b>{c_pct}%</b>; "
        f"Student/Standard share = 100% − {c_pct}% = <b>{s_pct}%</b>.",
        body_style
    )
    fairness_table = Table([[fairness_p]], colWidths=[page_w])
    fairness_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_WARM_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, C_SAGE),
        ("LINEBEFORE", (0, 0), (0, 0), 3.5, C_TEAL),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
    ]))
    story.append(fairness_table)
    story.append(Spacer(1, 12))

    # ==================================================================
    # 6. FLAGGED CLAUSES DEEP-DIVE (HIGH & MEDIUM)
    # ==================================================================
    story.append(Paragraph("4. Flagged Contract Clauses, Cross-Clause Links &amp; Student Advisory", h1_style))
    story.append(Paragraph(
        "The items below are <b>automated preliminary risk flags</b> produced by ClauseGuard AI's ML classifier, "
        "student-protection rule heuristics, and cross-clause engine. They highlight practical negotiation risks "
        "and general statutory context where relevant, and should <b>not</b> be read as binding legal conclusions.",
        body_justify
    ))
    story.append(Spacer(1, 8))

    flagged = high_clauses + med_clauses
    source_labels = {
        "STATUTORY_RULE": "Automated Rule Flag",
        "COMBINED_EVALUATION": "ML Classifier + Automated Rule Flag",
        "CROSS_CLAUSE_ANALYSIS": "Cross-Clause Relationship Analysis",
        "ML_CLASSIFIER": "ML Classifier",
        "DOCUMENT_STRUCTURE": "Document Structure",
    }

    if not flagged:
        story.append(Paragraph("No high-risk or medium-risk clauses were flagged in this agreement.", body_style))
    else:
        for idx, item in enumerate(flagged, 1):
            risk = (item.get("final_risk_level") or "low").upper()
            cat = (item.get("category") or "GENERAL").replace("_", " ").title().replace(" Nda", " NDA")
            clause_ref = f"Clause {item['clause_number']}" if item.get("clause_number") else f"Clause #{item.get('index', idx)}"
            source_raw = item.get("source", "ML_CLASSIFIER")
            source_readable = source_labels.get(source_raw, source_raw.replace("_", " ").title())
            badge_hex = "#DC2626" if risk == "HIGH" else "#D97706"
            accent_col = C_RED if risk == "HIGH" else C_AMBER

            block: List[Any] = [
                Paragraph(
                    f"<b>{idx}. {_esc(clause_ref)} — {_esc(cat)}</b> &nbsp;|&nbsp; "
                    f"<font color='{badge_hex}'><b>[{risk} RISK]</b></font> &nbsp;|&nbsp; "
                    f"<font color='{C_MUTED.hexval()}'>Detection: {_esc(source_readable)}</font>",
                    h2_style
                )
            ]

            # Cross-clause relationships callout (muted gold accent)
            related_list = item.get("related_clauses", [])
            if related_list:
                rel_text = "  •  ".join(_esc(r.get("display_text", "")) for r in related_list if r.get("display_text"))
                rel_tbl = Table(
                    [[Paragraph(f"<b>Cross-Clause Relationship:</b> {rel_text}", body_style)]],
                    colWidths=[page_w]
                )
                rel_tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), C_GOLD_LIGHT),
                    ("LINEBEFORE", (0, 0), (0, 0), 3, C_GOLD),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]))
                block.append(rel_tbl)
                block.append(Spacer(1, 5))

            # Verbatim contract quotation box (readable, full text preserved)
            quote_tbl = Table(
                [[Paragraph(f"\"{_esc(item.get('original_text', ''))}\"", quote_style)]],
                colWidths=[page_w]
            )
            quote_tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), C_WARM_BG),
                ("LINEBEFORE", (0, 0), (0, 0), 3, accent_col),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            block.append(Paragraph("<b>Original Contract Quotation:</b>", body_bold))
            block.append(Spacer(1, 2))
            block.append(quote_tbl)
            block.append(Spacer(1, 5))

            # Triggered automated rule / statutory context (if any)
            rule_viols = item.get("rule_violations", [])
            if rule_viols:
                v0 = rule_viols[0]
                block.append(Paragraph(
                    f"<b>Automated Flag / Legal Context:</b> {_esc(v0.get('rule_name', ''))} "
                    f"<i>({_esc(v0.get('statute', 'Automated Heuristic'))})</i>",
                    body_style
                ))
                block.append(Spacer(1, 3))

            # Plain-English student impact
            block.append(Paragraph(
                f"<b>Plain-English Student Impact:</b> {_esc(item.get('explanation', ''))}",
                body_justify
            ))
            block.append(Spacer(1, 3))

            # Recommended action
            block.append(Paragraph(
                f"<b>Recommended Action:</b> <font color='{C_TEAL.hexval()}'>{_esc(item.get('recommended_action', ''))}</font>",
                body_justify
            ))
            block.append(Spacer(1, 8))
            block.append(HRFlowable(width="100%", thickness=0.6, color=C_SAGE, spaceBefore=2, spaceAfter=8))

            story.append(KeepTogether(block))

    # ==================================================================
    # 7. MISSING OR UNCLEAR PROTECTIVE TERMS
    # ==================================================================
    story.append(Paragraph("5. Missing or Unclear Protective Terms", h1_style))
    if not missing_items:
        story.append(Paragraph(
            "All checked baseline student protective topics were referenced in this draft agreement.",
            body_style
        ))
    else:
        story.append(Paragraph(
            f"The automated completeness check identified <b>{len(missing_items)}</b> missing or unaddressed protective topic(s):",
            body_style
        ))
        story.append(Spacer(1, 6))
        for m in missing_items:
            m_block = [
                Paragraph(
                    f"• <b>{_esc(m.get('topic', 'Protective Term'))}</b> "
                    f"<font color='{C_MUTED.hexval()}'>[{_esc(str(m.get('severity', 'medium')).upper())} PRIORITY]</font>: "
                    f"{_esc(m.get('description', ''))}",
                    body_style
                ),
                Paragraph(
                    f"&nbsp;&nbsp;&nbsp;<b>Suggested Addition:</b> <font color='{C_TEAL.hexval()}'><i>{_esc(m.get('recommendation', ''))}</i></font>",
                    body_style
                ),
                Spacer(1, 5)
            ]
            story.append(KeepTogether(m_block))

    # ==================================================================
    # 8. LEGAL & EDUCATIONAL DISCLAIMER
    # ==================================================================
    story.append(Spacer(1, 12))
    disclaimer_text = (
        "<b>LEGAL &amp; EDUCATIONAL DISCLAIMER:</b><br/>"
        "ClauseGuard AI is an educational software tool for preliminary contract risk screening and student negotiation literacy. "
        "All risk ratings, cross-clause flags, fairness indices, and statutory context notes in this report are generated by "
        "automated machine learning (TF-IDF + calibrated Logistic Regression) and deterministic text heuristics — they are "
        "<b>automated preliminary risk flags, NOT formal legal conclusions or legal advice</b>. Statutory references (such as "
        "Sections 27, 74, and 124 of the Indian Contract Act, 1872, Sections 17 &amp; 57 of the Copyright Act, 1957, or the "
        "Consumer Protection Act, 2019) depend strictly on governing law, engagement status, and specific factual circumstances; "
        "no contractual bond, forfeiture term, or restriction is automatically enforceable or automatically void solely on the basis "
        "of an automated flag. Students and institutions should consult a qualified legal professional or university legal aid cell "
        "before signing, amending, or contesting any agreement."
    )
    disclaimer_table = Table(
        [[Paragraph(disclaimer_text, ParagraphStyle("Disc", parent=body_justify, fontSize=7.7, leading=10.8, textColor=C_BODY_GRAY))]],
        colWidths=[page_w]
    )
    disclaimer_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), C_WARM_BG),
        ("BOX", (0, 0), (-1, -1), 0.75, C_SAGE),
        ("LINEABOVE", (0, 0), (-1, 0), 2, C_NAVY),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether([disclaimer_table]))

    doc.build(
        story,
        canvasmaker=lambda *args, **kwargs: NumberedCanvas(*args, doc_filename=doc_filename, **kwargs)
    )
    return output_path
