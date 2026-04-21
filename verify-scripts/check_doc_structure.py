"""Check headings and appendix structure in the thesis docx."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def dump(docx_path: Path) -> None:
    with zipfile.ZipFile(docx_path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()

    for para in root.iter(f"{{{W}}}p"):
        ppr = para.find(f"{{{W}}}pPr")
        style_name = ""
        if ppr is not None:
            pstyle = ppr.find(f"{{{W}}}pStyle")
            if pstyle is not None:
                style_name = pstyle.get(f"{{{W}}}val", "")
        text = "".join(t.text or "" for t in para.iter(f"{{{W}}}t")).strip()
        if not text:
            continue
        # Print only headings or appendix / chapter-looking paragraphs
        lower = text.lower()
        is_heading_style = style_name.lower().startswith("heading") or style_name.lower().startswith("nadpis")
        looks_like_appendix = lower.startswith("příloha") or lower.startswith("priloha") or lower.startswith("appendix")
        looks_like_chapter = len(text) < 120 and any(text.startswith(prefix) for prefix in [
            "5.", "4.", "6.", "3.", "2.", "1.", "7.", "Kapitola", "Chapter",
        ])
        if is_heading_style or looks_like_appendix or looks_like_chapter:
            print(f"[{style_name or '-'}] {text[:200]}")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("LASTDiplomova_prace_Murcek.docx")
    dump(path)
