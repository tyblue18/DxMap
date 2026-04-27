"""Negation detection.

Real clinical notes are full of phrases like "no history of diabetes" or
"denies chest pain". A naive retrieval system will match these against
diabetes/chest-pain codes and produce false positives. This module identifies
negation scopes so downstream stages can exclude them.

We use negspaCy (NegEx algorithm by Chapman et al.) which is a rule-based
system specifically designed for clinical negation. It is lightweight,
deterministic, and well-validated on clinical text.

Production note: NegEx misses some negations and over-flags others. A robust
system would combine this with an LLM-based double-check for ambiguous cases.
For an MVP this is sufficient and dramatically improves precision.
"""

from __future__ import annotations

from functools import lru_cache

import spacy
from negspacy.negation import Negex  # noqa: F401  (registers the pipe)

from .schema import TextSpan


@lru_cache(maxsize=1)
def _get_nlp():
    """Lazy-load the spaCy pipeline. Cached because loading is slow."""
    nlp = spacy.load("en_core_web_sm")
    # Add negation detection. The default termset works for general clinical text.
    if "negex" not in nlp.pipe_names:
        nlp.add_pipe(
            "negex",
            config={"chunk_prefix": ["no", "denies", "without", "negative for"]},
        )
    return nlp


def find_negated_spans(note: str) -> list[TextSpan]:
    """Return character spans for every entity flagged as negated.

    Note: spaCy's default English model has a limited clinical vocabulary.
    For production, consider scispaCy (en_core_sci_sm) which has better
    biomedical entity recognition. We use en_core_web_sm here for ease of setup.
    """
    nlp = _get_nlp()
    doc = nlp(note)

    negated: list[TextSpan] = []
    for ent in doc.ents:
        if getattr(ent._, "negex", False):
            negated.append(
                TextSpan(start=ent.start_char, end=ent.end_char, text=ent.text)
            )
    return negated


def is_span_negated(span_start: int, span_end: int, negated_spans: list[TextSpan]) -> bool:
    """Check if a candidate justification span overlaps any negated entity span."""
    for neg in negated_spans:
        # Overlap if the spans share any character range
        if not (span_end <= neg.start or span_start >= neg.end):
            return True
    return False
