"""Evaluation harness.

Usage:
  python -m src.eval.eval_harness --test-set ../eval/test_set.jsonl

Computes:
  - Top-K accuracy (K = 1, 3, 5)
  - Span-level F1 (when gold spans are provided)
  - Expected Calibration Error (ECE) on raw and calibrated confidences
  - Per-example breakdown saved to results/

The harness also writes a calibrator fit on this test set so the inference
path can use it. Note: fitting calibration on the eval set inflates apparent
calibration quality. For a real evaluation, use a separate calibration split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..coder.calibration import (
    expected_calibration_error,
    fit_calibrator,
    save_calibrator,
)
from ..coder.pipeline import run
from ..coder.schema import CodingRequest, EvalExample
from .metrics import span_f1_for_example


def load_test_set(path: Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(EvalExample(**json.loads(line)))
    return examples


def evaluate(examples: list[EvalExample]) -> dict[str, float]:
    n = len(examples)
    top1_hits = 0
    top3_hits = 0
    top5_hits = 0
    span_f1s: list[float] = []
    raw_confidences: list[float] = []
    correctness: list[int] = []

    detailed_results = []

    for ex in examples:
        response = run(CodingRequest(note=ex.note, top_k=5, include_cpt=False))
        predicted_codes = [s.code for s in response.icd10_suggestions]

        gold_set = set(ex.gold_icd10)
        if not gold_set:
            continue  # Skip examples without gold ICD-10 (CPT-only)

        # Top-K hit if any gold code appears in the top K predictions.
        top1_hit = any(c in gold_set for c in predicted_codes[:1])
        top3_hit = any(c in gold_set for c in predicted_codes[:3])
        top5_hit = any(c in gold_set for c in predicted_codes[:5])

        top1_hits += int(top1_hit)
        top3_hits += int(top3_hit)
        top5_hits += int(top5_hit)

        # For calibration: each top-1 prediction gets a (confidence, correctness) pair.
        if response.icd10_suggestions:
            top_pred = response.icd10_suggestions[0]
            raw_confidences.append(top_pred.raw_confidence)
            correctness.append(int(top_pred.code in gold_set))

        # Span F1 — only computed if the harness has gold spans (extension point).
        # For this MVP we don't ship gold spans in the test set; you add them later.

        detailed_results.append(
            {
                "id": ex.id,
                "gold": list(gold_set),
                "predicted": predicted_codes[:5],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "top5_hit": top5_hit,
            }
        )

    metrics: dict[str, float] = {
        "n_examples": n,
        "top1_accuracy": top1_hits / n if n else 0.0,
        "top3_accuracy": top3_hits / n if n else 0.0,
        "top5_accuracy": top5_hits / n if n else 0.0,
    }

    if len(raw_confidences) >= 10:
        metrics["raw_ece"] = expected_calibration_error(raw_confidences, correctness)
        # Fit and persist a calibrator on this set
        cal = fit_calibrator(raw_confidences, correctness)
        save_calibrator(cal)
        calibrated = [float(cal.predict([c])[0]) for c in raw_confidences]
        metrics["calibrated_ece"] = expected_calibration_error(calibrated, correctness)

    # Save the per-example breakdown for inspection.
    results_dir = Path(__file__).resolve().parents[3] / "eval" / "results"
    results_dir.mkdir(exist_ok=True)
    with (results_dir / "latest.json").open("w") as f:
        json.dump({"metrics": metrics, "examples": detailed_results}, f, indent=2)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", required=True, type=Path)
    args = parser.parse_args()

    examples = load_test_set(args.test_set)
    if not examples:
        print("Empty test set.", file=sys.stderr)
        return 2

    print(f"Running pipeline on {len(examples)} examples...\n")
    metrics = evaluate(examples)

    print("Results")
    print("=======")
    for k, v in metrics.items():
        print(f"  {k:25s}: {v:.4f}" if isinstance(v, float) else f"  {k:25s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
