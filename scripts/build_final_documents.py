"""Build complete final documentation DOCX and PDF deliverables."""

from __future__ import annotations

import re
from pathlib import Path

import docx
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt


def convert_md_to_docx(md_path: Path, docx_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    doc = docx.Document()

    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    lines = text.split("\n")
    in_code = False
    code_lines: list[str] = []
    table_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        rows = []
        for line in table_buf:
            line = line.strip()
            if not line.startswith("|") or not line.endswith("|"):
                continue
            parts = [c.strip() for c in line[1:-1].split("|")]
            if all(re.match(r"^:?-+:?$", p) for p in parts if p):
                continue
            rows.append(parts)
        table_buf = []
        if not rows:
            return

        num_cols = max(len(r) for r in rows)
        table = doc.add_table(rows=len(rows), cols=num_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = "Table Grid"
        for r_idx, row in enumerate(rows):
            for c_idx, val in enumerate(row):
                if c_idx < num_cols:
                    cell = table.cell(r_idx, c_idx)
                    cell.text = val
                    if r_idx == 0:
                        for p in cell.paragraphs:
                            for run in p.runs:
                                run.font.bold = True
                                run.font.size = Pt(9.5)
                    else:
                        for p in cell.paragraphs:
                            for run in p.runs:
                                run.font.size = Pt(9)
        doc.add_paragraph()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            table_buf.append(line)
            continue
        else:
            if table_buf:
                flush_table()

        if stripped.startswith("```"):
            if in_code:
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.2)
                run = p.add_run("\n".join(code_lines))
                run.font.name = "Courier New"
                run.font.size = Pt(8.5)
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            continue

        if in_code:
            code_lines.append(line)
            continue

        if not stripped:
            continue

        if stripped.startswith("# "):
            p = doc.add_heading(stripped[2:], level=1)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("#### "):
            doc.add_heading(stripped[5:], level=4)
        elif stripped.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            run = p.add_run(stripped[2:])
            run.font.italic = True
        elif stripped.startswith("- [x] ") or stripped.startswith("- [ ] "):
            doc.add_paragraph(stripped[6:], style="List Bullet")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            doc.add_paragraph(stripped[2:], style="List Bullet")
        elif re.match(r"^\d+\.\s", stripped):
            doc.add_paragraph(re.sub(r"^\d+\.\s", "", stripped), style="List Number")
        else:
            doc.add_paragraph(stripped)

    if table_buf:
        flush_table()

    doc.save(docx_path)
    print(f"Generated DOCX: {docx_path} ({docx_path.stat().st_size:,} bytes)")


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> None:
    """Convert DOCX to PDF using LibreOffice in headless mode if available."""
    import subprocess
    out_dir = pdf_path.parent
    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(out_dir),
        str(docx_path),
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        generated_pdf = out_dir / (docx_path.stem + ".pdf")
        if generated_pdf != pdf_path and generated_pdf.exists():
            generated_pdf.rename(pdf_path)
        print(f"Generated PDF:  {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    except Exception as exc:
        print(f"PDF generation skipped ({exc})")


if __name__ == "__main__":
    report_md = Path("docs/FINAL_PROJECT_REPORT_AND_PLAGIARISM_GUIDE.md")
    report_docx = Path("docs/FINAL_PROJECT_REPORT_AND_PLAGIARISM_GUIDE.docx")
    report_pdf = Path("docs/FINAL_PROJECT_REPORT_AND_PLAGIARISM_GUIDE.pdf")
    convert_md_to_docx(report_md, report_docx)
    convert_docx_to_pdf(report_docx, report_pdf)

    viva_md = Path("docs/FINAL_VIVA_AND_DEMO_GUIDE.md")
    viva_docx = Path("docs/FINAL_VIVA_AND_DEMO_GUIDE.docx")
    viva_pdf = Path("docs/FINAL_VIVA_AND_DEMO_GUIDE.pdf")
    convert_md_to_docx(viva_md, viva_docx)
    convert_docx_to_pdf(viva_docx, viva_pdf)
