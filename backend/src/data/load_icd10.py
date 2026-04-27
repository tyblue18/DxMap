"""Load the CMS-published ICD-10-CM code set.

CMS publishes the ICD-10-CM tabular as a fixed-width text file. The format is
well-documented in the FY release notes. We parse out (code, description)
tuples; everything else (chapter headers, includes/excludes notes, etc.) is
ignored for the MVP but is the natural place to extend.

Where to get the file:
  https://www.cms.gov/medicare/coding-billing/icd-10-codes
  Download the most recent "ICD-10-CM Code Descriptions in Tabular Order" zip.
  Extract icd10cm_codes_YYYY.txt (or similar). Place it at data/icd10cm.txt.

The file format is simple:
  <code><spaces><description>
where code is left-justified and the description starts at a fixed column.
We're permissive about whitespace because the format has shifted slightly
across yearly releases.
"""

from __future__ import annotations

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
ICD10_FILE = DATA_DIR / "icd10cm.txt"


def load_icd10_codes() -> list[dict[str, str]]:
    """Return a list of {code, description} dicts."""
    if not ICD10_FILE.exists():
        raise FileNotFoundError(
            f"{ICD10_FILE} not found. See scripts/fetch_icd10.sh and the README "
            "Quickstart section."
        )

    codes: list[dict[str, str]] = []
    with ICD10_FILE.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Code is the first whitespace-delimited token; description is the rest.
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            code, description = parts[0].strip(), parts[1].strip()
            # ICD-10-CM codes are alphanumeric, between 3-7 chars, no internal periods
            # in the CMS tabular format (period is added at display time after position 3).
            if not (3 <= len(code) <= 7) or not code[0].isalpha():
                continue
            codes.append({"code": _format_code(code), "description": description})
    return codes


def _format_code(raw: str) -> str:
    """CMS publishes codes without the decimal; standard display is X##.### format.

    E.g. 'A0101' -> 'A01.01'. The decimal goes after the third character.
    """
    if len(raw) <= 3:
        return raw
    return f"{raw[:3]}.{raw[3:]}"
