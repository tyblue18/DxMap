"""LLM reranking with span attribution.

The retrieval step gives us ~20 plausible candidate codes. The LLM's job is to:

  1. Look at the actual note in context
  2. Decide which candidates are genuinely supported by the note
  3. For each kept code, point to the exact substring(s) of the note that justify it
  4. Self-report a confidence score

We use structured output (JSON mode) so the response is parseable and the
schema is enforced. The prompt is intentionally explicit about negation —
even though we filter negated spans at the retrieval stage, the LLM gets a
second chance to catch anything missed.

Provider-agnostic: this module supports OpenAI and Anthropic via simple
adapters. Swap providers by setting LLM_PROVIDER env var.
"""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from typing import Any

logger = logging.getLogger(__name__)

from .negation import is_span_negated
from .schema import CodeCandidate, CodeSuggestion, TextSpan

RERANK_SYSTEM_PROMPT = """You are an expert medical coder. Your job is to review a clinical note and a list of candidate ICD-10-CM or CPT codes, and decide which codes are actually supported by the note.

Rules you MUST follow:
1. Only assign a code if it accurately represents what the note documents. Do NOT assign codes for conditions that are negated ("no chest pain"), ruled out ("rules out PE"), or mentioned only as family history. The following are valid current diagnoses and SHOULD be assigned when documented:
   - Personal-history codes (Z85.x, Z86.x, Z87.x) correctly represent resolved prior conditions — e.g. Z85.3 is the right code for a cancer surveillance visit after completed breast cancer treatment.
   - Follow-up encounter codes (Z08, Z09) are correct when the visit purpose is post-treatment surveillance.
   - "Old" or chronic-sequela codes such as I25.2 (Old myocardial infarction) represent an ongoing ICD-10 diagnosis from a prior event and must be coded when the note documents that prior event.
2. For every code you keep, point to the EXACT substring(s) of the note that justify it. Use character offsets (start and end indices into the original note string).
3. Self-report your confidence as a number between 0 and 1. Use 0.9+ only when the documentation is unambiguous. Use <0.5 when you are uncertain or when human review would be appropriate.
4. If a candidate code is NOT supported, omit it from your output entirely.
5. Do not invent codes. Only return codes from the candidate list.
6. When the note contains an explicit provider diagnosis (phrases like "diagnosed with X", "assessment: X", "impression: X", "diagnosis: X"), prefer the corresponding diagnosis code over individual symptom codes (R-codes). Symptom codes (e.g. R05.1 cough, R30.0 dysuria, R00.2 palpitations) are only appropriate when no diagnosis has been established. If the note says "diagnosed with community-acquired pneumonia", code J18.9, not R05.1. If it says "diagnosis: UTI", code N39.0, not R30.0. If it says "diagnosed with panic disorder", code F41.0, not R00.2 or I47.x.
7. When multiple candidates represent the same condition at different specificity levels, always choose the MOST SPECIFIC code the documentation supports. Key examples:
   - CKD stage: use N18.31–N18.5 when the provider documents a stage; N18.9 (unspecified) only when no stage is given.
   - MDD severity: PHQ-9 5–9 = mild (F32.0/F33.0), 10–14 = moderate (F32.1/F33.1), 15–27 = severe without psychosis (F32.2/F33.2). Use the severity-specific code when a PHQ-9 score is documented.
   - Anemia etiology: D63.1 (anemia in CKD) over D64.9 (unspecified) when CKD is the documented cause; D63.8 when another chronic disease is the cause.
   - Hyperparathyroidism: N25.81 (secondary hyperparathyroidism of renal origin) when CKD causes elevated PTH. E21.1 (secondary hyperparathyroidism, NEC) is reserved for non-renal causes only.
   - Laterality: always prefer the side-specific code (e.g. M17.12 left knee, M17.11 right knee) over the unspecified code (M17.10) when the note names the affected side.

Output a single JSON object with key "suggestions" containing an array. Each entry must have:
  - code: string (exactly as in the candidate list)
  - confidence: float in [0,1]
  - rationale: one short sentence (≤ 25 words) explaining why
  - spans: array of objects with {start: int, end: int, text: string}

Return at most 5 codes per call.
"""

USER_TEMPLATE = """Clinical note:
\"\"\"
{note}
\"\"\"

Candidate codes (you may keep, reorder, or drop any):
{candidates}

Return JSON only."""


def _format_candidates(candidates: list[CodeCandidate]) -> str:
    return "\n".join(f"  {c.code}: {c.description}" for c in candidates)


# ---------- Provider adapters ----------


def _call_openai(system: str, user: str) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    return json.loads(resp.choices[0].message.content)


def _call_anthropic(system: str, user: str) -> dict[str, Any]:
    import anthropic

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user + "\n\nReturn ONLY a JSON object, no prose."}],
    )
    text = resp.content[0].text
    # Defensive: strip code fences if the model adds them despite instructions
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def _gemini_retry_delay(exc: Exception) -> float | None:
    """Return the server-suggested retry delay (seconds) from a 429 ClientError, or None."""
    try:
        # ClientError is constructed as (status_code, response_json, response)
        response_json = exc.args[1] if len(exc.args) > 1 else {}
        for detail in response_json.get("error", {}).get("details", []):
            if detail.get("@type", "").endswith("RetryInfo"):
                return float(detail["retryDelay"].rstrip("s"))
    except Exception:  # noqa: BLE001
        pass
    return None


def _gemini_is_daily_quota(exc: Exception) -> bool:
    """Return True if the 429 is a per-day hard cap — retrying won't help."""
    try:
        response_json = exc.args[1] if len(exc.args) > 1 else {}
        violations = (
            response_json.get("error", {})
            .get("details", [{}])[1]
            .get("violations", [])
        )
        return all("PerDay" in v.get("quotaId", "") for v in violations)
    except Exception:  # noqa: BLE001
        return False


def _call_gemini(system: str, user: str) -> dict[str, Any]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=0.0,
    )

    max_retries = 3
    last_exc: Exception
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
                contents=user,
                config=config,
            )
            return json.loads(resp.text)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            is_last = attempt == max_retries
            if _gemini_is_daily_quota(exc):
                logger.error(
                    "_call_gemini: daily quota exhausted — retrying won't help. Error: %s",
                    exc,
                )
                break
            suggested = _gemini_retry_delay(exc)
            delay = suggested if suggested is not None else [5, 15, 30][min(attempt, 2)]
            if is_last:
                logger.error(
                    "_call_gemini all %d attempts exhausted. Final error: %s\n%s",
                    max_retries + 1,
                    exc,
                    traceback.format_exc(),
                )
            else:
                logger.warning(
                    "_call_gemini attempt %d/%d failed (retry in %.0fs): %s",
                    attempt + 1,
                    max_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
    raise last_exc


def _call_llm(system: str, user: str) -> dict[str, Any]:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "openai":
        return _call_openai(system, user)
    if provider == "anthropic":
        return _call_anthropic(system, user)
    if provider == "gemini":
        return _call_gemini(system, user)
    raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


# ---------- Main reranker ----------


def _retrieval_passthrough(
    candidates: list[CodeCandidate],
    code_system: str,
) -> list[CodeSuggestion]:
    """Return candidates as-is using their raw retrieval score as confidence."""
    return [
        CodeSuggestion(
            code=c.code,
            description=c.description,
            code_system=c.code_system,
            rank=i + 1,
            raw_confidence=c.retrieval_score,
            calibrated_confidence=c.retrieval_score,
            justification_spans=[],
            rationale="",
            needs_human_review=True,
        )
        for i, c in enumerate(candidates[:5])
    ]


def rerank(
    note: str,
    candidates: list[CodeCandidate],
    negated_spans: list[TextSpan],
    code_system: str = "ICD-10-CM",
) -> list[CodeSuggestion]:
    """Rerank candidates using an LLM and produce span-attributed suggestions."""
    if not candidates:
        return []

    if os.getenv("SKIP_RERANK", "false").lower() == "true":
        return _retrieval_passthrough(candidates, code_system)

    print(
        f"[rerank] {code_system} — {len(candidates)} candidates received: "
        + ", ".join(f"{c.code}({c.retrieval_score:.4f})" for c in candidates)
    )

    shown = candidates[:10]
    code_to_desc = {c.code: c.description for c in shown}

    print(
        f"[rerank] {code_system} — {len(shown)} shown to LLM: "
        + ", ".join(f"{c.code}: {c.description}" for c in shown)
    )

    user_msg = USER_TEMPLATE.format(
        note=note,
        candidates=_format_candidates(shown),
    )

    try:
        response = _call_llm(RERANK_SYSTEM_PROMPT, user_msg)
        print(f"[rerank] {code_system} — raw LLM response: {response}")
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "_call_llm failed (provider=%s): %s\n%s",
            os.getenv("LLM_PROVIDER", "openai"),
            exc,
            traceback.format_exc(),
        )
        # If the LLM call fails, fall back to retrieval order with low confidence.
        # This is a deliberate design choice: it's better to return *something*
        # than to 500 the API. The low confidence triggers the human-review flag.
        return [
            CodeSuggestion(
                code=c.code,
                description=c.description,
                code_system=c.code_system,
                rank=i + 1,
                raw_confidence=0.3,
                calibrated_confidence=0.3,
                justification_spans=[],
                rationale=f"LLM unavailable ({type(exc).__name__}); retrieval-order fallback",
                needs_human_review=True,
            )
            for i, c in enumerate(candidates[:5])
        ]

    suggestions: list[CodeSuggestion] = []
    for rank, item in enumerate(response.get("suggestions", []), start=1):
        code = item.get("code", "").strip()
        if code not in code_to_desc:
            # LLM hallucinated a code not in the candidate list — drop it.
            continue

        # Validate spans against the actual note text and the negation filter.
        validated_spans: list[TextSpan] = []
        for s in item.get("spans", []):
            try:
                start, end = int(s["start"]), int(s["end"])
            except (KeyError, ValueError, TypeError):
                continue
            if not (0 <= start < end <= len(note)):
                continue
            actual_text = note[start:end]
            # If the span overlaps a negated entity, skip it — this code shouldn't fire.
            if is_span_negated(start, end, negated_spans):
                continue
            validated_spans.append(TextSpan(start=start, end=end, text=actual_text))

        # If after negation filtering we have no spans left, this code is suspect.
        # Lower its confidence rather than dropping (the user can still see it for review).
        confidence = float(item.get("confidence", 0.5))
        if not validated_spans:
            confidence = min(confidence, 0.4)

        suggestions.append(
            CodeSuggestion(
                code=code,
                description=code_to_desc[code],
                code_system=code_system,  # type: ignore[arg-type]
                rank=rank,
                raw_confidence=confidence,
                calibrated_confidence=confidence,  # Calibrated downstream
                justification_spans=validated_spans,
                rationale=item.get("rationale", "")[:200],
                needs_human_review=confidence < 0.5,
            )
        )

    return suggestions
