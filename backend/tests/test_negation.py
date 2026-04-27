"""Tests for negation detection. Run with: pytest backend/tests"""

import pytest

from src.coder.negation import find_negated_spans, is_span_negated


@pytest.mark.parametrize(
    "note,should_be_negated",
    [
        ("Patient denies chest pain.", "chest pain"),
        ("No history of diabetes.", "diabetes"),
        ("Negative for hypertension.", "hypertension"),
    ],
)
def test_finds_negated_entities(note, should_be_negated):
    spans = find_negated_spans(note)
    found_texts = [s.text.lower() for s in spans]
    assert any(should_be_negated in t for t in found_texts), (
        f"Expected {should_be_negated!r} in negated entities, got {found_texts}"
    )


def test_does_not_negate_positive_mention():
    note = "Patient has type 2 diabetes."
    spans = find_negated_spans(note)
    # No negated entities expected
    assert all("diabetes" not in s.text.lower() for s in spans)


def test_overlap_check():
    from src.coder.schema import TextSpan

    negated = [TextSpan(start=10, end=20, text="diabetes")]
    assert is_span_negated(15, 25, negated) is True
    assert is_span_negated(0, 9, negated) is False
    assert is_span_negated(20, 30, negated) is False  # no overlap, end is exclusive
