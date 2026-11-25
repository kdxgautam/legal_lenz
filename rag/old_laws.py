"""Map repealed IPC, CrPC, and Indian Evidence Act sections to BNS, BNSS, and BSA.

The data comes from the BPR&D correspondence tables; rebuild it with
scripts/build_old_law_map.py.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from rag.statutes import StatuteReference

DATA_PATH = Path(__file__).with_name("data") / "old_law_sections.json"

OLD_ACT_ALIASES = {
    "IPC": r"(?:\bI\.?\s?P\.?\s?C\b\.?|\b(?:Indian\s+)?Penal\s+Code\b(?:,?\s*1860)?)",
    "CrPC": r"(?:\bCr\.?\s?P\.?\s?C\b\.?|\bCode\s+of\s+Criminal\s+Procedure\b(?:,?\s*1973)?)",
    "IEA": r"(?:\bI\.?\s?E\.?\s?A\b\.?|\b(?:Indian\s+)?Evidence\s+Act\b(?:,?\s*1872)?)",
}
SECTION = r"(\d+[A-Z]{0,2})"


@dataclass(frozen=True)
class OldLawReference:
    old_act: str
    old_section: str
    new_act: str
    new_sections: tuple[str, ...]

    @property
    def statute_references(self) -> list[StatuteReference]:
        return list(dict.fromkeys(
            StatuteReference(self.new_act, re.match(r"\d+[A-Z]?", section).group())
            for section in self.new_sections
        ))

    def note(self) -> str:
        if self.new_sections:
            return (
                f"{self.old_act} section {self.old_section} corresponds to {self.new_act} section "
                f"{', '.join(self.new_sections)} (BPR&D correspondence table)."
            )
        return (
            f"{self.old_act} section {self.old_section} has no corresponding {self.new_act} section "
            "in the BPR&D correspondence table."
        )


@cache
def _acts() -> dict:
    return json.loads(DATA_PATH.read_text())["acts"]


def lookup(old_act: str, section: str) -> OldLawReference | None:
    act = _acts()[old_act]
    section = section.upper()
    new_sections = act["sections"].get(section)
    if new_sections is None and int(re.match(r"\d+", section).group()) > act["final_section"]:
        return None
    return OldLawReference(old_act, section, act["new_act"], tuple(new_sections or ()))


def mentioned_old_acts(text: str) -> list[str]:
    return [act for act, alias in OLD_ACT_ALIASES.items() if re.search(alias, text, re.I)]


def extract_old_law_references(text: str) -> list[OldLawReference]:
    references = []
    for act, alias in OLD_ACT_ALIASES.items():
        patterns = (
            rf"\b{SECTION}\s+(?:of\s+(?:the\s+)?)?{alias}",
            rf"{alias}\s*,?\s*(?:section\s+|sec\.?\s*|u/s\.?\s*)?{SECTION}\b",
        )
        for pattern in patterns:
            for section in re.findall(pattern, text, re.I):
                reference = lookup(act, section)
                if reference and reference not in references:
                    references.append(reference)
    return references
