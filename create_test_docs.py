"""
Generate realistic sample PDF, DOCX, and TXT agreement files
for testing ClauseGuard AI's document extraction and analysis.
"""
from pathlib import Path
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from backend.sample_agreements import SAMPLE_AGREEMENTS

TEST_DOCS_DIR = Path("test_documents")
TEST_DOCS_DIR.mkdir(parents=True, exist_ok=True)

def generate_sample_files():
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, textColor=colors.HexColor('#102B35'))
    body_style = ParagraphStyle('BodyStyle', parent=styles['BodyText'], fontSize=10, leading=14, textColor=colors.HexColor('#1E293B'))

    for key, data in SAMPLE_AGREEMENTS.items():
        # 1. TXT File
        txt_path = TEST_DOCS_DIR / data["filename"]
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(data["content"])
        print(f"Created TXT: {txt_path}")

        # 2. DOCX File
        docx_filename = data["filename"].replace(".txt", ".docx")
        docx_path = TEST_DOCS_DIR / docx_filename
        doc = Document()
        doc.add_heading(data["title"], level=1)
        for line in data["content"].split("\n\n"):
            line = line.strip()
            if line:
                if line.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.", "10.")):
                    doc.add_heading(line.split("\n")[0], level=2)
                    rest = "\n".join(line.split("\n")[1:])
                    if rest:
                        doc.add_paragraph(rest)
                else:
                    doc.add_paragraph(line)
        doc.save(str(docx_path))
        print(f"Created DOCX: {docx_path}")

        # 3. PDF File
        pdf_filename = data["filename"].replace(".txt", ".pdf")
        pdf_path = TEST_DOCS_DIR / pdf_filename
        pdf_doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
        story = [Paragraph(f"<b>{data['title']}</b>", title_style), Spacer(1, 14)]
        for line in data["content"].split("\n\n"):
            line = line.strip()
            if line:
                story.append(Paragraph(line.replace("\n", "<br/>"), body_style))
                story.append(Spacer(1, 8))
        pdf_doc.build(story)
        print(f"Created PDF: {pdf_path}")

if __name__ == "__main__":
    generate_sample_files()
