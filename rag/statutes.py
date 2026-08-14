from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class StatuteDefinition:
    act_name: str
    short_name: str
    year: int
    effective_from: str
    expected_pages: int
    expected_chapters: int
    final_section: int

    def metadata(self) -> dict:
        return {
            "act_name": self.act_name,
            "act_short_name": self.short_name,
            "act_year": self.year,
            "effective_from": self.effective_from,
            "jurisdiction": "India",
            "source_type": "official_pdf",
        }


STATUTES = {
    "BNS": StatuteDefinition("Bharatiya Nyaya Sanhita, 2023", "BNS", 2023, "2024-07-01", 112, 20, 358),
    "BNSS": StatuteDefinition(
        "Bharatiya Nagarik Suraksha Sanhita, 2023", "BNSS", 2023, "2024-07-01", 279, 39, 531
    ),
    "BSA": StatuteDefinition("Bharatiya Sakshya Adhiniyam, 2023", "BSA", 2023, "2024-07-01", 47, 12, 170),
}


@dataclass(frozen=True)
class SourceLine:
    page: int
    y: float
    text: str


@dataclass(frozen=True)
class StatuteUnit:
    act_name: str
    act_short_name: str
    act_year: int
    chapter_number: str | None
    chapter_title: str | None
    section_number: str
    section_title: str | None
    text: str
    page_start: int
    page_end: int
    subsections: tuple[str, ...] = ()
    source_lines: tuple[SourceLine, ...] = field(default=(), repr=False, compare=False)


@dataclass
class StatuteReport:
    pages_processed: int
    chapters_detected: int = 0
    sections_detected: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    failed_sections: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class StatuteReference:
    act_short_name: str
    section_number: str


CHAPTER_RE = re.compile(r"^CHAPTER\s*([IVXLCDM]+)$", re.I)
SECTION_RE = re.compile(r"^(\d+[A-Z]?)\.\s*(.*)$", re.I)
SUBSECTION_RE = re.compile(r"^\((\d+[A-Z]?)\)\s*", re.I)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _noise(value: str) -> bool:
    upper = value.upper()
    return (
        not value
        or bool(re.fullmatch(r"\d+", value))
        or bool(re.fullmatch(r"[_—–-]+", value))
        or "THE GAZETTE OF INDIA EXTRAORDINARY" in upper
        or upper.startswith("[PART II")
        or upper.startswith("SEC. 1]")
        or upper == "SECTIONS"
    )


def _blocks(doc: fitz.Document) -> tuple[list[SourceLine], dict[int, list[SourceLine]]]:
    central = []
    margins: dict[int, list[SourceLine]] = {}
    for page_number, page in enumerate(doc, start=1):
        width = page.rect.width
        for block in page.get_text("blocks", sort=True):
            x0, y0, x1, _, text = block[:5]
            lines = [_clean(line) for line in text.splitlines()]
            if x1 > width * 0.2 and x0 < width * 0.8:
                for offset, line in enumerate(lines):
                    if not _noise(line):
                        central.append(SourceLine(page_number, y0 + offset * 0.1, line))
            else:
                margin = _clean(" ".join(line for line in lines if not _noise(line)))
                if margin:
                    margins.setdefault(page_number, []).append(SourceLine(page_number, y0, margin))
    return central, margins


def _chapter_title(lines: list[SourceLine], start: int) -> str | None:
    title = []
    for line in lines[start : start + 4]:
        if CHAPTER_RE.match(line.text) or SECTION_RE.match(line.text) or line.text.upper().startswith("PART"):
            break
        if line.text.upper() == line.text and len(line.text) > 2:
            title.append(line.text)
        elif title:
            break
    return " ".join(title) or None


def _arrangement(lines: list[SourceLine], body_start: int) -> dict[str, dict]:
    if not any("ARRANGEMENT OF SECTIONS" in line.text.upper() for line in lines[:body_start]):
        return {}
    result: dict[str, dict] = {}
    chapter_number = chapter_title = None
    pending_number = None
    pending_title = ""

    def save() -> None:
        nonlocal pending_number, pending_title
        if pending_number:
            result[pending_number] = {
                "section_title": pending_title.rstrip(".").strip() or None,
                "chapter_number": chapter_number,
                "chapter_title": chapter_title,
            }
        pending_number = None
        pending_title = ""

    for index, line in enumerate(lines[:body_start]):
        chapter = CHAPTER_RE.match(line.text)
        if chapter:
            save()
            chapter_number = chapter.group(1).upper()
            chapter_title = _chapter_title(lines, index + 1)
            continue
        section = SECTION_RE.match(line.text)
        if section and int(re.match(r"\d+", section.group(1)).group()) <= 1000:
            save()
            pending_number = section.group(1).upper()
            pending_title = section.group(2)
            if pending_title.endswith("."):
                save()
        elif pending_number:
            pending_title += " " + line.text
            if pending_title.endswith("."):
                save()
    save()
    return result


def _margin_title(margins: dict[int, list[SourceLine]], line: SourceLine) -> str | None:
    candidates = [item for item in margins.get(line.page, []) if abs(item.y - line.y) < 45]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs(item.y - line.y)).text.rstrip(".")


def _inline_side_title(lines: list[SourceLine]) -> str | None:
    nearby = [
        line.text
        for line in lines[1:]
        if line.page == lines[0].page
        and 0 <= line.y - lines[0].y < 5
        and not re.match(r"^\d+ of \d+\.$", line.text, re.I)
    ]
    return " ".join(nearby).rstrip(".") or None


def _subsection(line: str) -> str | None:
    match = SUBSECTION_RE.match(line)
    if not match and SECTION_RE.match(line):
        match = re.search(r"(?:\.\s*|[—–-])\((\d+[A-Z]?)\)\s*", line, re.I)
    return match.group(1).upper() if match else None


def _looks_like_body_section(
    number: str,
    tail: str,
    previous: str | None,
    definition: StatuteDefinition,
    arrangement: dict[str, dict],
) -> bool:
    numeric = int(re.match(r"\d+", number).group())
    previous_match = re.match(r"(\d+)([A-Z]?)", previous or "0")
    previous_numeric = int(previous_match.group(1))
    previous_suffix = previous_match.group(2)
    suffix = number.removeprefix(str(numeric))
    follows = numeric == previous_numeric + 1 or (
        numeric == previous_numeric and suffix and (not previous_suffix or suffix > previous_suffix)
    )
    if not follows or numeric > definition.final_section:
        return False
    if not arrangement:
        return True
    expected = arrangement.get(number, {}).get("section_title")
    if not expected:
        return False
    actual_words = re.findall(r"[a-z0-9]+", re.sub(r"^[.—–-]+", "", tail).casefold())
    expected_words = re.findall(r"[a-z0-9]+", expected.casefold())
    prefix = min(2, len(expected_words))
    return actual_words[:prefix] == expected_words[:prefix]


def extract_statute_structure(
    doc: fitz.Document, definition: StatuteDefinition
) -> tuple[list[StatuteUnit], StatuteReport]:
    lines, margins = _blocks(doc)
    report = StatuteReport(pages_processed=doc.page_count)
    try:
        body_start = next(index for index, line in enumerate(lines) if "BE it enacted" in line.text)
    except StopIteration:
        body_start = 0
        report.warnings.append("Enactment marker not found; parsed the full document.")
    arrangement = _arrangement(lines, body_start)
    body = lines[body_start + 1 :]
    units: list[StatuteUnit] = []
    current: list[SourceLine] = []
    current_number = None
    current_chapter = current_chapter_title = None
    previous_number = None

    def save() -> None:
        nonlocal current, current_number
        if not current_number or not current:
            return
        metadata = arrangement.get(current_number, {})
        title = metadata.get("section_title") or _margin_title(margins, current[0]) or _inline_side_title(current)
        chapter = metadata.get("chapter_number") or current_chapter
        chapter_title = metadata.get("chapter_title") or current_chapter_title
        text = "\n".join(line.text for line in current).strip()
        subsections = tuple(label for line in current if (label := _subsection(line.text)))
        units.append(
            StatuteUnit(
                definition.act_name,
                definition.short_name,
                definition.year,
                chapter,
                chapter_title,
                current_number,
                title,
                text,
                current[0].page,
                current[-1].page,
                subsections,
                tuple(current),
            )
        )
        current = []
        current_number = None

    for index, line in enumerate(body):
        upper = line.text.upper()
        if (
            upper.startswith("THE FIRST SCHEDULE")
            or re.fullmatch(r"THE\s*SCHEDULE", upper)
            or upper == "STATEMENT OF OBJECTS AND REASONS"
        ):
            save()
            break
        if re.fullmatch(r"PART\s*[IVXLCDM]+", upper):
            save()
            continue
        chapter = CHAPTER_RE.match(line.text)
        if chapter:
            save()
            current_chapter = chapter.group(1).upper()
            current_chapter_title = _chapter_title(body, index + 1)
            continue
        section = SECTION_RE.match(line.text)
        if section:
            number = section.group(1).upper()
            if _looks_like_body_section(number, section.group(2), previous_number, definition, arrangement):
                save()
                current_number = number
                current = [line]
                previous_number = number
                continue
        if current_number:
            current.append(line)
    save()

    chapters = {unit.chapter_number for unit in units if unit.chapter_number}
    report.chapters_detected = len(chapters)
    report.sections_detected = len(units)
    found = {int(re.match(r"\d+", unit.section_number).group()) for unit in units}
    missing = [number for number in range(1, definition.final_section + 1) if number not in found]
    for previous, current in zip(sorted(found), sorted(found)[1:]):
        if current > previous + 1:
            report.warnings.append(f"Section sequence jumped from {previous} to {current}.")
    if missing:
        report.failed_sections = len(missing)
        report.warnings.append(f"Missing sections: {', '.join(map(str, missing))}.")
    if doc.page_count != definition.expected_pages:
        report.warnings.append(f"Expected {definition.expected_pages} pages, found {doc.page_count}.")
    if report.chapters_detected != definition.expected_chapters:
        report.warnings.append(
            f"Expected {definition.expected_chapters} chapters, found {report.chapters_detected}."
        )
    return units, report


def _part_pages(lines: list[SourceLine], part: str, start: int = 0) -> tuple[int, int, int]:
    full = "\n".join(line.text for line in lines)
    offset = full.find(part, start)
    if offset < 0:
        return lines[0].page, lines[-1].page, start
    end = offset + len(part)
    positions = []
    cursor = 0
    for line in lines:
        positions.append((cursor, cursor + len(line.text), line.page))
        cursor += len(line.text) + 1
    pages = [page for left, right, page in positions if right >= offset and left <= end]
    return min(pages), max(pages), offset


def chunk_statute_units(units: list[StatuteUnit]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    for unit in units:
        groups: list[tuple[str | None, list[SourceLine]]] = []
        for line in unit.source_lines:
            subsection = _subsection(line.text)
            if subsection:
                groups.append((subsection, [line]))
            elif groups:
                groups[-1][1].append(line)
            else:
                groups.append((None, [line]))
        if len(groups) > 1 and groups[0][0] is None:
            header = groups.pop(0)[1]
            groups[0][1][:0] = header

        packed: list[tuple[list[str], list[SourceLine]]] = []
        for label, group_lines in groups:
            group_text = "\n".join(line.text for line in group_lines)
            if packed and len("\n".join(line.text for line in packed[-1][1])) + len(group_text) + 1 <= 1000:
                if label:
                    packed[-1][0].append(label)
                packed[-1][1].extend(group_lines)
            else:
                packed.append(([label] if label else [], list(group_lines)))

        chunk_index = 0
        for labels, group_lines in packed:
            group_text = "\n".join(line.text for line in group_lines)
            parts = [group_text] if len(group_text) <= 1000 else splitter.split_text(group_text)
            search_from = 0
            for part in parts:
                chunk_index += 1
                page_start, page_end, found_at = _part_pages(group_lines, part, max(0, search_from - 200))
                search_from = max(search_from, found_at + len(part))
                chunks.append(
                    {
                        "page": page_start,
                        "article": None,
                        "content": part,
                        "metadata": {
                            "act_name": unit.act_name,
                            "act_short_name": unit.act_short_name,
                            "act_year": unit.act_year,
                            "chapter_number": unit.chapter_number,
                            "chapter_title": unit.chapter_title,
                            "section_number": unit.section_number,
                            "section_title": unit.section_title,
                            "subsection": labels[0] if len(labels) == 1 else None,
                            "page_end": page_end,
                            "chunk_index": chunk_index,
                        },
                    }
                )
    return chunks
