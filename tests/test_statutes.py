from pathlib import Path

import fitz
import pytest

from rag.retriever import extract_statute_reference
from rag.statutes import (
    STATUTES,
    SourceLine,
    StatuteUnit,
    chunk_statute_units,
    extract_statute_structure,
)
from rag.statutes import _looks_like_body_section


PDFS = {
    "BNS": "bhartiya_nyay_sanhita.pdf",
    "BNSS": "the_bharatiya_nagarik_suraksha_sanhita,_2023.pdf",
    "BSA": "bhartiya_sakshya.pdf",
}


def open_statute_pdf(act):
    path = Path("data/pdfs") / PDFS[act]
    if not path.exists():
        pytest.skip(f"Optional statute corpus is unavailable: {path}")
    return fitz.open(path)


@pytest.mark.parametrize("act", ["BNS", "BNSS", "BSA"])
def test_real_statute_structure_matches_expected_counts(act):
    definition = STATUTES[act]
    with open_statute_pdf(act) as document:
        units, report = extract_statute_structure(document, definition)

    assert report.sections_detected == definition.final_section
    assert report.chapters_detected == definition.expected_chapters
    assert report.failed_sections == 0
    assert [unit.section_number for unit in units] == [str(number) for number in range(1, definition.final_section + 1)]


def test_bns_section_303_keeps_legal_metadata_and_subsections():
    with open_statute_pdf("BNS") as document:
        units, _ = extract_statute_structure(document, STATUTES["BNS"])

    section = units[302]
    assert section.section_number == "303"
    assert section.section_title == "Theft"
    assert section.chapter_number == "XVII"
    assert section.chapter_title == "OF OFFENCES AGAINST PROPERTY"
    assert section.page_start == 88
    assert section.page_end == 89
    assert section.subsections[:2] == ("1", "2")


def test_bsa_margin_title_and_bns_irregular_section_are_recovered():
    with open_statute_pdf("BSA") as document:
        bsa, _ = extract_statute_structure(document, STATUTES["BSA"])
    with open_statute_pdf("BNS") as document:
        bns, _ = extract_statute_structure(document, STATUTES["BNS"])

    assert bsa[62].section_title == "Admissibility of electronic records"
    assert bsa[169].section_title == "Repeal and savings"
    assert bns[254].section_number == "255"
    assert bns[254].section_title.startswith("Public servant disobeying")


def test_large_section_splits_without_losing_metadata_or_page_range():
    lines = (
        SourceLine(10, 1, "106A. Long section.—(1) " + "first " * 220),
        SourceLine(11, 1, "(2) " + "second " * 220),
    )
    unit = StatuteUnit(
        "Example Act, 2023",
        "EA",
        2023,
        "IV",
        "EXAMPLE",
        "106A",
        "Long section",
        "\n".join(line.text for line in lines),
        10,
        11,
        ("1", "2"),
        lines,
    )

    chunks = chunk_statute_units([unit])

    assert len(chunks) > 2
    assert {chunk["metadata"]["section_number"] for chunk in chunks} == {"106A"}
    assert {chunk["metadata"]["chapter_number"] for chunk in chunks} == {"IV"}
    assert {chunk["page"] for chunk in chunks} == {10, 11}
    assert [chunk["metadata"]["chunk_index"] for chunk in chunks] == list(range(1, len(chunks) + 1))


@pytest.mark.parametrize(
    ("question", "act", "section"),
    [
        ("What does Section 303 of BNS say?", "BNS", "303"),
        ("Section 106 BNSS", "BNSS", "106"),
        ("section 63 of BSA", "BSA", "63"),
        ("BNS 303", "BNS", "303"),
        ("Bharatiya Nyaya Sanhita section 303", "BNS", "303"),
    ],
)
def test_exact_statute_reference_formats(question, act, section):
    reference = extract_statute_reference(question)
    assert reference.act_short_name == act
    assert reference.section_number == section


def test_unrelated_number_is_not_a_statute_reference():
    assert extract_statute_reference("What happened in India in 2023?") is None
    assert extract_statute_reference("Explain section 303") is None


def test_alphanumeric_section_follows_base_section():
    definition = STATUTES["BNS"]
    arrangement = {"106A": {"section_title": "Inserted offence"}}
    assert _looks_like_body_section("106A", "Inserted offence.—Text", "106", definition, arrangement)
