"""Build rag/data/old_law_sections.json from the BPR&D correspondence tables.

Download the three official PDFs first:
  https://bprd.nic.in/uploads/pdf/COMPARISON%20SUMMARY%20BNS%20to%20IPC%20.pdf
  https://bprd.nic.in/uploads/pdf/Comparison%20summary%20BNSS%20to%20CrPC.pdf
  https://bprd.nic.in/uploads/pdf/Comparison%20Summary%20BSA%20to%20IEA.pdf

Usage:
  uv run python scripts/build_old_law_map.py BNS_IPC.pdf BNSS_CRPC.pdf BSA_IEA.pdf
"""

import argparse
import json
import re
from pathlib import Path

import fitz

OUTPUT = Path(__file__).resolve().parents[1] / "rag" / "data" / "old_law_sections.json"

# (old act, new act, old act name, final old section, source URL, column order new/old)
TABLES = [
    ("IPC", "BNS", "Indian Penal Code, 1860", 511,
     "https://bprd.nic.in/uploads/pdf/COMPARISON%20SUMMARY%20BNS%20to%20IPC%20.pdf", (0, 2)),
    ("CrPC", "BNSS", "Code of Criminal Procedure, 1973", 484,
     "https://bprd.nic.in/uploads/pdf/Comparison%20summary%20BNSS%20to%20CrPC.pdf", (0, 2)),
    ("IEA", "BSA", "Indian Evidence Act, 1872", 167,
     "https://bprd.nic.in/uploads/pdf/Comparison%20Summary%20BSA%20to%20IEA.pdf", (0, 1)),
]


def _cell(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def new_reference(value: str) -> str | None:
    """Normalize a new-act cell such as '189 (8)/ (9)' or 'First proviso to section 22'."""
    value = _cell(value)
    match = re.search(r"\d+(?:\s*\(\s*[0-9a-z]+\s*\))*", value, re.I)
    if not match:
        return None
    reference = re.sub(r"\s+", "", match.group())
    if re.search(r"proviso", value, re.I):
        reference = f"{reference} (proviso)"
    return reference


def old_sections(value: str, final_section: int) -> list[str]:
    """Top-level old sections from cells like '29 and 29A', '3, para 8', '23 Clause-1', '2(c )'."""
    value = _cell(value)
    value = re.sub(r"\b(?:para|clause)\s*-?\s*\d+", " ", value, flags=re.I)
    value = re.sub(r"\([^)]*\)", " ", value)
    sections = []
    for number in re.findall(r"\b(\d+[A-Z]{0,2})\b", value):
        if int(re.match(r"\d+", number).group()) <= final_section and number not in sections:
            sections.append(number)
    return sections


def table_rows(doc: fitz.Document):
    for page in doc:
        for table in page.find_tables().tables:
            for row in table.extract():
                cells = [cell for cell in row if cell is not None] if table.col_count != 4 else row
                if len(cells) >= 4:
                    yield cells


def build_act(path: str, final_section: int, columns: tuple[int, int]) -> dict[str, list[str]]:
    new_col, old_col = columns
    mapping: dict[str, list[str]] = {}
    with fitz.open(path) as doc:
        for cells in table_rows(doc):
            reference = new_reference(cells[new_col] or "")
            if not reference:
                continue
            for section in old_sections(cells[old_col] or "", final_section):
                targets = mapping.setdefault(section, [])
                if reference not in targets:
                    targets.append(reference)
    return dict(sorted(mapping.items(), key=lambda item: (int(re.match(r"\d+", item[0]).group()), item[0])))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bns_ipc")
    parser.add_argument("bnss_crpc")
    parser.add_argument("bsa_iea")
    args = parser.parse_args()
    paths = [args.bns_ipc, args.bnss_crpc, args.bsa_iea]
    acts = {}
    for path, (old_act, new_act, name, final_section, url, columns) in zip(paths, TABLES):
        sections = build_act(path, final_section, columns)
        acts[old_act] = {
            "name": name, "new_act": new_act, "final_section": final_section, "source": url, "sections": sections,
        }
        print(f"{old_act} -> {new_act}: {len(sections)} old sections mapped")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({
        "publisher": "Bureau of Police Research and Development (BPR&D), correspondence tables",
        "acts": acts,
    }, indent=1) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
