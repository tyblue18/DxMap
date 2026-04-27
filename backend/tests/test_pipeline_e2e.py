"""End-to-end pipeline tests with the LLM call mocked.

These tests verify the orchestration logic without making real API calls,
which keeps the test suite fast and free.
"""

from unittest.mock import patch

import pytest

from src.coder.schema import CodeCandidate, CodingRequest


@pytest.fixture
def fake_candidates():
    return [
        CodeCandidate(
            code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            retrieval_score=0.9,
        ),
        CodeCandidate(
            code="I10",
            description="Essential (primary) hypertension",
            retrieval_score=0.7,
        ),
    ]


@pytest.fixture
def fake_llm_response():
    return {
        "suggestions": [
            {
                "code": "E11.9",
                "confidence": 0.85,
                "rationale": "Note states 'type 2 diabetes' explicitly.",
                "spans": [{"start": 19, "end": 34, "text": "type 2 diabetes"}],
            }
        ]
    }


def test_pipeline_returns_response(fake_candidates, fake_llm_response):
    from src.coder.pipeline import run

    note = "Patient has known type 2 diabetes managed with metformin."

    with (
        patch("src.coder.pipeline.retrieve", return_value=fake_candidates),
        patch("src.coder.rerank._call_llm", return_value=fake_llm_response),
    ):
        response = run(CodingRequest(note=note, include_cpt=False))

    assert len(response.icd10_suggestions) == 1
    assert response.icd10_suggestions[0].code == "E11.9"
    assert response.icd10_suggestions[0].justification_spans
    assert response.latency_ms >= 0


def test_pipeline_drops_hallucinated_codes(fake_candidates):
    from src.coder.pipeline import run

    bad_response = {
        "suggestions": [
            {
                "code": "Z99.99",  # Not in candidate list
                "confidence": 0.9,
                "rationale": "Hallucinated.",
                "spans": [],
            }
        ]
    }

    with (
        patch("src.coder.pipeline.retrieve", return_value=fake_candidates),
        patch("src.coder.rerank._call_llm", return_value=bad_response),
    ):
        response = run(CodingRequest(note="Some note text here.", include_cpt=False))

    # Hallucinated code should be dropped
    assert all(s.code != "Z99.99" for s in response.icd10_suggestions)


def test_pipeline_handles_llm_failure(fake_candidates):
    from src.coder.pipeline import run

    with (
        patch("src.coder.pipeline.retrieve", return_value=fake_candidates),
        patch("src.coder.rerank._call_llm", side_effect=RuntimeError("API down")),
    ):
        response = run(CodingRequest(note="Some clinical note text.", include_cpt=False))

    # Should fall back to retrieval order with low confidence + human review flag
    assert len(response.icd10_suggestions) > 0
    assert all(s.needs_human_review for s in response.icd10_suggestions)
