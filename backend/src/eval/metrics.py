"""Span-level F1 for code attribution.

A predicted span is a "true positive" if it overlaps any gold span by at least
``overlap_threshold`` characters. Partial credit is the standard approach for
clinical NER evaluation; strict-match F1 is overly punishing for cases where
the model picks a slightly different span boundary than the human.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int

    def overlaps(self, other: "Span", min_chars: int = 1) -> bool:
        overlap_start = max(self.start, other.start)
        overlap_end = min(self.end, other.end)
        return overlap_end - overlap_start >= min_chars


def span_f1(predicted: list[Span], gold: list[Span], min_chars: int = 1) -> dict[str, float]:
    if not predicted and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not predicted:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for p in predicted if any(p.overlaps(g, min_chars) for g in gold))
    fp = len(predicted) - tp
    fn = sum(1 for g in gold if not any(p.overlaps(g, min_chars) for p in predicted))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def span_f1_for_example(
    predicted_spans_per_code: dict[str, list[Span]],
    gold_spans_per_code: dict[str, list[Span]],
) -> float:
    """Macro-average span F1 across the codes that appear in both predicted and gold."""
    common = set(predicted_spans_per_code) & set(gold_spans_per_code)
    if not common:
        return 0.0
    f1s = [
        span_f1(predicted_spans_per_code[c], gold_spans_per_code[c])["f1"]
        for c in common
    ]
    return sum(f1s) / len(f1s)
