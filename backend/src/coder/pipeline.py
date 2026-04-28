"""End-to-end pipeline: note in, ranked code suggestions out.

This is the orchestrator. Each stage is a simple function call, which makes
the pipeline easy to test in isolation and easy to extend (e.g., adding a
hierarchical-code-rollup stage between rerank and calibration).
"""

from __future__ import annotations

import time

from .calibration import calibrate, load_calibrator
from .negation import find_negated_spans
from .rerank import rerank
from .retrieval import retrieve_decomposed
from .schema import CodingRequest, CodingResponse, CodeSuggestion


def run(request: CodingRequest) -> CodingResponse:
    t0 = time.perf_counter()

    # Stage 1: identify negated spans up front so retrieval and rerank can use them.
    negated = find_negated_spans(request.note)

    # Stage 2 + 3: hybrid retrieval, then LLM rerank with span attribution.
    # negated spans are passed to retrieval so NegEx-detected entities are
    # excluded from queries in addition to the prefix-based filter.
    icd_candidates = retrieve_decomposed(
        request.note, code_system="ICD-10-CM", negated_spans=negated
    )
    icd_suggestions = rerank(
        request.note, icd_candidates, negated, code_system="ICD-10-CM"
    )

    cpt_suggestions: list[CodeSuggestion] = []
    if request.include_cpt:
        cpt_candidates = retrieve_decomposed(
            request.note, code_system="CPT", negated_spans=negated
        )
        cpt_suggestions = rerank(
            request.note, cpt_candidates, negated, code_system="CPT"
        )

    # Stage 4: calibrate confidences. Falls through silently if no calibrator is fit yet.
    calibrator = load_calibrator()
    for s in icd_suggestions + cpt_suggestions:
        s.calibrated_confidence = calibrate(s.raw_confidence, calibrator)
        s.needs_human_review = s.calibrated_confidence < 0.5

    # Truncate to top_k (rerank may return up to 5 by prompt design, but defensive)
    icd_suggestions = icd_suggestions[: request.top_k]
    cpt_suggestions = cpt_suggestions[: request.top_k]

    return CodingResponse(
        icd10_suggestions=icd_suggestions,
        cpt_suggestions=cpt_suggestions,
        negated_phrases=negated,
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )
