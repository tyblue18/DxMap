"""Tests for negation handling. Run with: pytest backend/tests

Negation is defended in three places, and the tests below reflect how much work
each layer actually does:

  1. Prefix filter          (retrieval._is_useful_query)   — carries real load
  2. Context-window check   (retrieval._entity_is_negated) — carries most load
  3. NegEx                  (negation.find_negated_spans)  — near-zero coverage

Layer 3 is the one the architecture nominally rests on, and it is the weakest:
NegEx operates on named entities, and en_core_web_sm recognises virtually no
clinical term as a named entity. See test_negex_clinical_coverage below.
"""

import pytest

from src.coder.negation import find_negated_spans, is_span_negated
from src.coder.retrieval import _entity_is_negated, _extract_query_entities


# --- Layer 3: NegEx ---------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known limitation, not a regression: NegEx flags negated *named entities*, "
        "but en_core_web_sm tags no clinical term as an entity, so doc.ents is empty "
        "and find_negated_spans returns []. This is why the retrieval-layer filters "
        "below exist. Swapping in scispaCy (en_core_sci_sm) should make these pass — "
        "if this test XPASSes, that swap happened and the xfail should be removed."
    ),
)
@pytest.mark.parametrize(
    "note,should_be_negated",
    [
        ("Patient denies chest pain.", "chest pain"),
        ("No history of diabetes.", "diabetes"),
        ("Negative for hypertension.", "hypertension"),
    ],
)
def test_negex_clinical_coverage(note, should_be_negated):
    spans = find_negated_spans(note)
    found_texts = [s.text.lower() for s in spans]
    assert any(should_be_negated in t for t in found_texts), (
        f"Expected {should_be_negated!r} in negated entities, got {found_texts}"
    )


def test_negex_pipe_is_wired_correctly():
    """NegEx itself works — it just has nothing clinical to work on.

    Distinguishes "the negex pipe is misconfigured" from "en_core_web_sm found
    no entities", which is the actual cause of test_negex_clinical_coverage.
    """
    spans = find_negated_spans("Patient denies travelling to Boston.")
    assert any("boston" in s.text.lower() for s in spans), (
        "NegEx failed to flag a negated entity it can actually see (GPE); "
        "the pipe is misconfigured, not merely starved of clinical entities."
    )


def test_does_not_negate_positive_mention():
    note = "Patient has type 2 diabetes."
    spans = find_negated_spans(note)
    assert all("diabetes" not in s.text.lower() for s in spans)


# --- Layers 1 + 2: retrieval-side filtering ---------------------------------


@pytest.mark.parametrize(
    "note,excluded",
    [
        # Layer 2 case: spaCy emits "chest pain" as a bare chunk. The cue
        # ("denies") sits outside the chunk, so the prefix filter cannot see it.
        ("Patient denies chest pain. Reports mild fatigue.", "chest pain"),
        # The README's canonical sub-chunk case: "no family history" is caught by
        # the prefix filter, but "sudden cardiac death" is emitted separately and
        # only the context window catches it.
        (
            "No family history of sudden cardiac death. Presents with palpitations.",
            "sudden cardiac death",
        ),
        ("Denies shortness of breath. Complains of knee swelling.", "shortness of breath"),
    ],
)
def test_negated_findings_are_not_issued_as_queries(note, excluded):
    """Regression test for the bug where negated chunks reached retrieval.

    _extract_query_entities gated the context-window check behind a non-empty
    NegEx result. Because NegEx returns [] for clinical text, the check never
    ran and negated findings were issued as live retrieval queries — which is
    the precise failure that pollutes the candidate set with codes for
    ruled-out conditions.
    """
    queries = [q.lower() for q in _extract_query_entities(note, negated_spans=[])]
    assert excluded.lower() not in queries, (
        f"{excluded!r} is negated but was issued as a retrieval query: {queries}"
    )


@pytest.mark.parametrize(
    "note,kept",
    [
        ("Patient denies chest pain. Reports mild fatigue.", "mild fatigue"),
        ("No family history of sudden cardiac death. Presents with palpitations.", "palpitations"),
    ],
)
def test_positive_findings_survive_negation_filtering(note, kept):
    """The filters must not be so aggressive that live diagnoses are dropped."""
    queries = [q.lower() for q in _extract_query_entities(note, negated_spans=[])]
    assert kept.lower() in queries, f"expected {kept!r} to survive, got {queries}"


def test_context_window_does_not_cross_clause_boundary():
    """A cue in a *previous* sentence must not negate the current one."""
    note = "No chest pain on arrival. Patient has type 2 diabetes mellitus."
    assert _entity_is_negated("type 2 diabetes mellitus", note, []) is False


# --- Span overlap arithmetic ------------------------------------------------


def test_overlap_check():
    from src.coder.schema import TextSpan

    negated = [TextSpan(start=10, end=20, text="diabetes")]
    assert is_span_negated(15, 25, negated) is True
    assert is_span_negated(0, 9, negated) is False
    assert is_span_negated(20, 30, negated) is False  # no overlap, end is exclusive
