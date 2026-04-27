"""Pydantic models defining the contracts between pipeline stages and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TextSpan(BaseModel):
    """A character span within the source note."""

    start: int = Field(..., description="Character offset (inclusive)")
    end: int = Field(..., description="Character offset (exclusive)")
    text: str = Field(..., description="The literal substring; redundant with offsets but useful for debugging")


class CodeCandidate(BaseModel):
    """A code retrieved from the corpus before reranking."""

    code: str
    description: str
    code_system: Literal["ICD-10-CM", "CPT"] = "ICD-10-CM"
    retrieval_score: float
    retrieval_method: Literal["bm25", "dense", "rrf"] = "rrf"


class CodeSuggestion(BaseModel):
    """A reranked code with attribution and confidence."""

    code: str
    description: str
    code_system: Literal["ICD-10-CM", "CPT"]
    rank: int
    raw_confidence: float = Field(..., ge=0.0, le=1.0, description="LLM-reported confidence")
    calibrated_confidence: float = Field(..., ge=0.0, le=1.0, description="After isotonic calibration")
    justification_spans: list[TextSpan]
    rationale: str = Field(..., description="One-sentence explanation from the LLM")
    needs_human_review: bool


class CodingRequest(BaseModel):
    note: str = Field(..., min_length=20, max_length=20_000)
    include_cpt: bool = True
    top_k: int = Field(default=5, ge=1, le=20)


class CodingResponse(BaseModel):
    icd10_suggestions: list[CodeSuggestion]
    cpt_suggestions: list[CodeSuggestion]
    negated_phrases: list[TextSpan] = Field(
        default_factory=list,
        description="Spans identified as negated; useful for UI to dim them",
    )
    pipeline_version: str = "0.1.0"
    latency_ms: int


class EvalExample(BaseModel):
    """One row in the held-out eval set."""

    id: str
    note: str
    gold_icd10: list[str] = Field(default_factory=list, description="Ground-truth ICD-10-CM codes")
    gold_cpt: list[str] = Field(default_factory=list)
    notes: str | None = Field(None, description="Coder's reasoning, for review")
