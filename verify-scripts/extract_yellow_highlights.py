"""Extract text runs highlighted yellow (w:highlight w:val="yellow") from a .docx."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W}


def extract(docx_path: Path, color: str = "yellow") -> list[str]:
    with zipfile.ZipFile(docx_path) as z:
        with z.open("word/document.xml") as f:
            tree = ET.parse(f)
    root = tree.getroot()

    results: list[str] = []
    for para in root.iter(f"{{{W}}}p"):
        buf: list[str] = []
        for run in para.iter(f"{{{W}}}r"):
            rpr = run.find(f"{{{W}}}rPr")
            if rpr is None:
                continue
            hl = rpr.find(f"{{{W}}}highlight")
            if hl is None or hl.get(f"{{{W}}}val") != color:
                continue
            text = "".join(t.text or "" for t in run.iter(f"{{{W}}}t"))
            if text:
                buf.append(text)
        if buf:
            results.append(" ".join(buf).strip())
    return results


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("LASTDiplomova_prace_Murcek.docx")
    runs = extract(path, "yellow")
    print(f"Found {len(runs)} paragraph(s) with yellow-highlighted text:\n")
    for i, r in enumerate(runs, 1):
        print(f"[{i}] {r}")
        print("-" * 80)
