"""Convert PBL Markdown reports to professional DOCX and PDF documents with embedded figures and styled tables."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import docx
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor


def set_cell_background(cell, fill_hex: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def set_cell_margins(cell, top: int = 100, bottom: int = 100, left: int = 140, right: int = 140) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)


def set_table_borders(table, color_hex: str = "D3D3D3") -> None:
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="{color_hex}"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="1F4E79"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color_hex}"/>'
        f'<w:insideV w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)


def parse_inline_formatting(paragraph, text: str, base_font_size: float = 10.5, font_name: str = "Calibri", text_color: RGBColor | None = None) -> None:
    """Parse bold, italic, code, and plain text into paragraph runs."""
    pattern = r'(\*\*.*?\*\*|\*.*?\*|`.*?`)'
    parts = re.split(pattern, text)
    for part in parts:
        if not part:
            continue
        run = paragraph.add_run()
        run.font.name = font_name
        run.font.size = Pt(base_font_size)
        if text_color:
            run.font.color.rgb = text_color

        if part.startswith("**") and part.endswith("**") and len(part) >= 4:
            run.text = part[2:-2]
            run.bold = True
        elif part.startswith("*") and part.endswith("*") and len(part) >= 2:
            run.text = part[1:-1]
            run.italic = True
        elif part.startswith("`") and part.endswith("`") and len(part) >= 2:
            run.text = part[1:-1]
            run.font.name = "Consolas"
            run.font.size = Pt(base_font_size - 0.5)
            run.font.color.rgb = RGBColor(180, 40, 40)
        else:
            run.text = part


def convert_markdown_to_docx(md_path: Path, docx_path: Path) -> None:
    base_dir = md_path.parent
    text = md_path.read_text(encoding="utf-8")
    doc = docx.Document()

    # Page setup: Standard Letter/A4 with 0.85 in margins
    for section in doc.sections:
        section.top_margin = Inches(0.85)
        section.bottom_margin = Inches(0.85)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)

    lines = text.split("\n")
    in_code = False
    code_lines: list[str] = []
    table_buf: list[str] = []

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        raw_rows = []
        for line in table_buf:
            line_s = line.strip()
            if not line_s.startswith("|") or not line_s.endswith("|"):
                continue
            cells = [c.strip() for c in line_s[1:-1].split("|")]
            if all(re.match(r"^:?-+:?$", c) for c in cells if c):
                continue
            raw_rows.append(cells)
        table_buf = []
        if not raw_rows:
            return

        # Check if row 0 is blank headers (e.g. metadata table)
        is_blank_header = all(not c for c in raw_rows[0])
        if is_blank_header:
            rows = raw_rows[1:]
            is_metadata_table = True
        else:
            rows = raw_rows
            is_metadata_table = False

        if not rows:
            return

        num_cols = max(len(r) for r in rows)
        table = doc.add_table(rows=len(rows), cols=num_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = True

        if is_metadata_table:
            # Metadata Key-Value Box
            set_table_borders(table, color_hex="E0E0E0")
            for r_idx, row in enumerate(rows):
                for c_idx, val in enumerate(row):
                    if c_idx < num_cols:
                        cell = table.cell(r_idx, c_idx)
                        set_cell_margins(cell, top=70, bottom=70, left=120, right=120)
                        set_cell_background(cell, "F8F9FA" if r_idx % 2 == 0 else "FFFFFF")
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.space_before = Pt(2)
                        parse_inline_formatting(p, val, base_font_size=9.5)
        else:
            # Standard Data Table
            set_table_borders(table, color_hex="D0D7DE")
            for r_idx, row in enumerate(rows):
                is_header = (r_idx == 0)
                for c_idx, val in enumerate(row):
                    if c_idx < num_cols:
                        cell = table.cell(r_idx, c_idx)
                        set_cell_margins(cell, top=80, bottom=80, left=110, right=110)
                        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_after = Pt(2)
                        p.paragraph_format.space_before = Pt(2)

                        if is_header:
                            set_cell_background(cell, "1F4E79")
                            # Bold white text
                            parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', val)
                            for part in parts:
                                if not part:
                                    continue
                                clean = part.replace("**", "").replace("*", "")
                                run = p.add_run(clean)
                                run.font.name = "Calibri"
                                run.font.size = Pt(9.5)
                                run.bold = True
                                run.font.color.rgb = RGBColor(255, 255, 255)
                        else:
                            bg = "F4F6F8" if r_idx % 2 == 1 else "FFFFFF"
                            set_cell_background(cell, bg)
                            parse_inline_formatting(p, val, base_font_size=9.0)

        # Spacing after table
        after_p = doc.add_paragraph()
        after_p.paragraph_format.space_before = Pt(0)
        after_p.paragraph_format.space_after = Pt(4)

    i = 0
    num_lines = len(lines)
    while i < num_lines:
        line = lines[i]
        stripped = line.strip()

        # Table lines
        if stripped.startswith("|") and stripped.endswith("|"):
            table_buf.append(line)
            i += 1
            continue
        else:
            if table_buf:
                flush_table()

        # Code block handling
        if stripped.startswith("```"):
            if in_code:
                # Flush code block
                p = doc.add_paragraph()
                p.paragraph_format.left_indent = Inches(0.3)
                p.paragraph_format.right_indent = Inches(0.3)
                p.paragraph_format.space_before = Pt(4)
                p.paragraph_format.space_after = Pt(6)
                code_text = "\n".join(code_lines)
                run = p.add_run(code_text)
                run.font.name = "Consolas"
                run.font.size = Pt(8.5)
                run.font.color.rgb = RGBColor(40, 40, 40)
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            i += 1
            continue

        if in_code:
            code_lines.append(line)
            i += 1
            continue

        # Skip empty lines
        if not stripped:
            i += 1
            continue

        # Horizontal Rule
        if re.match(r"^---+$", stripped):
            # Add subtle divider line / spacing
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run("―" * 50)
            run.font.color.rgb = RGBColor(200, 205, 210)
            run.font.size = Pt(8)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            i += 1
            continue

        # Markdown Images: ![alt](path)
        img_match = re.match(r"^!\[(.*?)\]\((.*?)\)$", stripped)
        if img_match:
            alt_text, img_rel_path = img_match.group(1), img_match.group(2)
            img_full_path = (base_dir / img_rel_path).resolve()
            if img_full_path.exists():
                p_img = doc.add_paragraph()
                p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_img.paragraph_format.space_before = Pt(10)
                p_img.paragraph_format.space_after = Pt(4)
                run_img = p_img.add_run()
                # Embed picture (max 5.8 inches width to fit neatly on page)
                run_img.add_picture(str(img_full_path), width=Inches(5.8))
            else:
                p_missing = doc.add_paragraph()
                p_missing.add_run(f"[Image not found: {img_rel_path}]").italic = True
            i += 1
            continue

        # Figure / Table Captions (*Figure N. ...* or *Table N. ...*)
        if stripped.startswith("*Figure ") or stripped.startswith("*Table "):
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(2)
            p_cap.paragraph_format.space_after = Pt(10)
            clean_cap = stripped.strip("*")
            run_cap = p_cap.add_run(clean_cap)
            run_cap.font.name = "Calibri"
            run_cap.font.size = Pt(9.0)
            run_cap.italic = True
            run_cap.font.color.rgb = RGBColor(80, 85, 90)
            i += 1
            continue

        # Headings
        if stripped.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(stripped[2:])
            run.font.name = "Calibri"
            run.font.size = Pt(20)
            run.bold = True
            run.font.color.rgb = RGBColor(15, 32, 67)  # Deep Navy
            i += 1
            continue
        elif stripped.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[3:])
            run.font.name = "Calibri"
            run.font.size = Pt(14)
            run.bold = True
            run.font.color.rgb = RGBColor(31, 78, 121)  # Navy Accent
            i += 1
            continue
        elif stripped.startswith("### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[4:])
            run.font.name = "Calibri"
            run.font.size = Pt(12)
            run.bold = True
            run.font.color.rgb = RGBColor(46, 117, 182)  # Blue Accent
            i += 1
            continue
        elif stripped.startswith("#### "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(stripped[5:])
            run.font.name = "Calibri"
            run.font.size = Pt(11)
            run.bold = True
            run.font.color.rgb = RGBColor(60, 60, 60)
            i += 1
            continue

        # Blockquote (> ...)
        if stripped.startswith("> "):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.35)
            p.paragraph_format.right_indent = Inches(0.2)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            parse_inline_formatting(p, stripped[2:], base_font_size=10.0, text_color=RGBColor(50, 60, 70))
            for run in p.runs:
                run.italic = True
            i += 1
            continue

        # Checklists (- [ ] or - [x])
        if stripped.startswith("- [ ] ") or stripped.startswith("- [x] "):
            is_checked = stripped.startswith("- [x] ")
            box = "☑  " if is_checked else "☐  "
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            run_box = p.add_run(box)
            run_box.font.name = "Segoe UI Symbol"
            run_box.font.size = Pt(10.0)
            parse_inline_formatting(p, stripped[6:], base_font_size=10.0)
            i += 1
            continue

        # Bullet List (- or *)
        if stripped.startswith("- ") or stripped.startswith("* "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            parse_inline_formatting(p, stripped[2:], base_font_size=10.0)
            i += 1
            continue

        # Numbered List (1. ...)
        num_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if num_match:
            item_num, item_text = num_match.group(1), num_match.group(2)
            p = doc.add_paragraph(style="List Number")
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(2)
            parse_inline_formatting(p, item_text, base_font_size=10.0)
            i += 1
            continue

        # Standard Paragraph
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        parse_inline_formatting(p, stripped, base_font_size=10.5)
        i += 1

    if table_buf:
        flush_table()

    docx_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(docx_path)
    print(f"Generated DOCX: {docx_path} ({docx_path.stat().st_size:,} bytes)")


def convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> None:
    """Convert DOCX to PDF using headless LibreOffice."""
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
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    generated_pdf = out_dir / (docx_path.stem + ".pdf")
    if generated_pdf != pdf_path and generated_pdf.exists():
        generated_pdf.rename(pdf_path)
    print(f"Generated PDF:  {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


def main():
    root = Path(__file__).resolve().parent.parent

    # PBL-3 Deliverables
    pbl3_md = root / "PBL-3" / "PBL3_Report.md"
    pbl3_docx = root / "PBL-3" / "PBL3_Report.docx"
    pbl3_pdf = root / "PBL-3" / "PBL3_Report.pdf"

    print("Building PBL-3 Report DOCX & PDF...")
    convert_markdown_to_docx(pbl3_md, pbl3_docx)
    convert_docx_to_pdf(pbl3_docx, pbl3_pdf)

    # PBL-2 Deliverables
    pbl2_md = root / "PBL-2" / "PBL2_Report.md"
    pbl2_docx = root / "PBL-2" / "PBL2_Report.docx"
    pbl2_pdf = root / "PBL-2" / "PBL2_Report.pdf"

    print("\nBuilding PBL-2 Report DOCX & PDF...")
    convert_markdown_to_docx(pbl2_md, pbl2_docx)
    convert_docx_to_pdf(pbl2_docx, pbl2_pdf)

    print("\nAll deliverables generated successfully!")


if __name__ == "__main__":
    main()
