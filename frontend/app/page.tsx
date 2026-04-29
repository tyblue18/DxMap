"use client";

import { useState, useMemo, useCallback } from "react";

type TextSpan = { start: number; end: number; text: string };

type CodeSuggestion = {
  code: string;
  description: string;
  code_system: "ICD-10-CM" | "CPT";
  rank: number;
  raw_confidence: number;
  calibrated_confidence: number;
  justification_spans: TextSpan[];
  rationale: string;
  needs_human_review: boolean;
};

type CodingResponse = {
  icd10_suggestions: CodeSuggestion[];
  cpt_suggestions: CodeSuggestion[];
  negated_phrases: TextSpan[];
  pipeline_version: string;
  latency_ms: number;
};

const SAMPLE_NOTE = `65-year-old male presents for follow-up of type 2 diabetes mellitus and hypertension. Glycemic control suboptimal with HbA1c of 8.2. Blood pressure today 152/94. Continues metformin 1000mg BID, lisinopril 20mg daily. Will increase lisinopril to 40mg and add empagliflozin 10mg.`;

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Page() {
  const [note, setNote] = useState(SAMPLE_NOTE);
  const [response, setResponse] = useState<CodingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeSpans, setActiveSpans] = useState<TextSpan[]>([]);

  const onSubmit = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const res = await fetch(`${API_BASE}/code`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ note, include_cpt: true, top_k: 5 }),
      });
      if (!res.ok) {
        const detail = await res.text();
        throw new Error(`${res.status}: ${detail}`);
      }
      const data: CodingResponse = await res.json();
      setResponse(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, [note]);

  return (
    <main className="min-h-screen bg-paper">
      <header className="border-b border-ink/10 bg-paper-raised">
        <div className="mx-auto max-w-6xl px-6 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <img
              src="/DxMap.png"
              alt="DxMap logo"
              className="h-9 w-auto"
            />
            <div>
              <h1 className="font-serif text-2xl text-ink tracking-tight">
                DxMap
              </h1>
              <p className="text-xs text-ink-muted mt-0.5 font-mono">
                ICD-10-CM &middot; CPT &middot; Span attribution &middot; Calibrated confidence
              </p>
            </div>
          </div>
          <a
            href="https://github.com/tyblue18/clinical-coder"
            className="text-xs text-ink-muted hover:text-ink transition-colors font-mono"
          >
            github →
          </a>
        </div>
      </header>

      <div className="mx-auto max-w-6xl px-6 py-10 grid grid-cols-1 lg:grid-cols-2 gap-10">
        {/* LEFT: input */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-ink-muted uppercase tracking-wider">
              Clinical note
            </h2>
            <span className="text-xs text-ink-subtle font-mono">
              {note.length} chars
            </span>
          </div>

          <NoteWithHighlights
            note={note}
            onChange={setNote}
            negatedSpans={response?.negated_phrases ?? []}
            activeSpans={activeSpans}
            disabled={loading}
          />

          <div className="mt-4 flex gap-3">
            <button
              onClick={onSubmit}
              disabled={loading || note.length < 20}
              className="px-5 py-2 bg-accent text-paper-raised text-sm font-medium hover:bg-accent-hover transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {loading ? "Analyzing…" : "Suggest codes"}
            </button>
            <button
              onClick={() => {
                setNote("");
                setResponse(null);
                setError(null);
              }}
              disabled={loading}
              className="px-4 py-2 text-sm text-ink-muted hover:text-ink transition-colors"
            >
              Clear
            </button>
          </div>

          {error && (
            <div className="mt-4 p-3 border border-warn/30 bg-warn/5 text-sm text-warn font-mono">
              {error}
            </div>
          )}
        </section>

        {/* RIGHT: results */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-sm font-medium text-ink-muted uppercase tracking-wider">
              Suggestions
            </h2>
            {response && (
              <span className="text-xs text-ink-subtle font-mono">
                {response.latency_ms}ms
              </span>
            )}
          </div>

          {!response && !loading && (
            <div className="border border-dashed border-ink/15 p-8 text-center text-ink-subtle text-sm">
              Paste a note and click <span className="text-ink">Suggest codes</span>
              <br />
              <span className="text-xs font-mono mt-1 inline-block">
                Hover suggestions to highlight justifications
              </span>
            </div>
          )}

          {loading && <SkeletonResults />}

          {response && (
            <div className="space-y-5">
              <ResultGroup
                title="ICD-10-CM diagnoses"
                suggestions={response.icd10_suggestions}
                onHover={setActiveSpans}
              />
              <ResultGroup
                title="CPT procedures"
                suggestions={response.cpt_suggestions}
                onHover={setActiveSpans}
              />

              {response.icd10_suggestions.length === 0 &&
                response.cpt_suggestions.length === 0 && (
                  <div className="text-sm text-ink-muted">
                    No codes suggested. The note may be too brief or describe a
                    condition not covered by the indexed code set.
                  </div>
                )}
            </div>
          )}
        </section>
      </div>

      <footer className="border-t border-ink/10 mt-16 py-6">
        <div className="mx-auto max-w-6xl px-6 text-xs text-ink-subtle font-mono">
          Demo project. Not a medical device. Not validated for clinical use.
        </div>
      </footer>
    </main>
  );
}

// ---------- Sub-components ----------

function NoteWithHighlights({
  note,
  onChange,
  negatedSpans,
  activeSpans,
  disabled,
}: {
  note: string;
  onChange: (v: string) => void;
  negatedSpans: TextSpan[];
  activeSpans: TextSpan[];
  disabled: boolean;
}) {
  // We render two stacked layers: a rendered overlay with highlighting, and
  // the actual textarea on top. When the note has results, we show the
  // highlighted view in a read-only block with an "Edit" affordance, because
  // mixing live highlighting + editing in a textarea is fiddly and out of
  // scope for the MVP.
  const hasHighlights = negatedSpans.length > 0 || activeSpans.length > 0;

  if (!hasHighlights) {
    return (
      <textarea
        value={note}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        rows={14}
        className="w-full p-4 border border-ink/15 bg-paper-raised text-ink leading-relaxed font-serif text-[15px] focus:outline-none focus:border-accent transition-colors resize-y"
        placeholder="Paste a clinical encounter note here..."
      />
    );
  }

  return (
    <div className="border border-ink/15 bg-paper-raised">
      <div className="p-4 leading-relaxed font-serif text-[15px] text-ink whitespace-pre-wrap">
        <RenderedNote
          note={note}
          negatedSpans={negatedSpans}
          activeSpans={activeSpans}
        />
      </div>
      <div className="flex justify-between border-t border-ink/10 px-4 py-2 bg-paper">
        <span className="text-xs text-ink-subtle">
          <span className="span-highlight">highlighted</span> = justifies a
          suggestion &middot;{" "}
          <span className="span-negated">struck</span> = negated
        </span>
        <button
          onClick={() => {
            const t = note;
            onChange("");
            setTimeout(() => onChange(t), 10);
          }}
          className="text-xs text-ink-muted hover:text-ink"
        >
          Edit
        </button>
      </div>
    </div>
  );
}

function RenderedNote({
  note,
  negatedSpans,
  activeSpans,
}: {
  note: string;
  negatedSpans: TextSpan[];
  activeSpans: TextSpan[];
}) {
  // Merge all spans, label each, sort by start, and render text segments interleaved.
  type LabeledSpan = TextSpan & { type: "negated" | "active" };
  const all: LabeledSpan[] = useMemo(() => {
    const merged: LabeledSpan[] = [
      ...negatedSpans.map((s) => ({ ...s, type: "negated" as const })),
      ...activeSpans.map((s) => ({ ...s, type: "active" as const })),
    ];
    return merged.sort((a, b) => a.start - b.start);
  }, [negatedSpans, activeSpans]);

  const segments: React.ReactNode[] = [];
  let cursor = 0;
  for (let i = 0; i < all.length; i++) {
    const span = all[i];
    if (span.start < cursor) continue; // skip overlapping spans for simplicity
    if (span.start > cursor) segments.push(note.slice(cursor, span.start));
    const text = note.slice(span.start, span.end);
    segments.push(
      <span
        key={`${span.start}-${span.end}-${i}`}
        className={span.type === "negated" ? "span-negated" : "span-highlight active"}
      >
        {text}
      </span>
    );
    cursor = span.end;
  }
  if (cursor < note.length) segments.push(note.slice(cursor));

  return <>{segments}</>;
}

function ResultGroup({
  title,
  suggestions,
  onHover,
}: {
  title: string;
  suggestions: CodeSuggestion[];
  onHover: (spans: TextSpan[]) => void;
}) {
  if (suggestions.length === 0) return null;
  return (
    <div className="fade-up">
      <h3 className="text-xs uppercase tracking-wider text-ink-subtle mb-2 font-mono">
        {title}
      </h3>
      <ul className="space-y-2">
        {suggestions.map((s) => (
          <li
            key={s.code}
            onMouseEnter={() => onHover(s.justification_spans)}
            onMouseLeave={() => onHover([])}
            className="border border-ink/10 bg-paper-raised p-3 hover:border-accent/40 transition-colors cursor-default"
          >
            <div className="flex items-baseline justify-between gap-3">
              <div className="flex items-baseline gap-3 min-w-0">
                <span className="font-mono text-sm font-medium text-accent">
                  {s.code}
                </span>
                <span className="text-sm text-ink truncate">{s.description}</span>
              </div>
              <ConfidenceBadge
                confidence={s.calibrated_confidence}
                needsReview={s.needs_human_review}
              />
            </div>
            {s.rationale && (
              <p className="mt-1.5 text-xs text-ink-muted italic">{s.rationale}</p>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function ConfidenceBadge({
  confidence,
  needsReview,
}: {
  confidence: number;
  needsReview: boolean;
}) {
  const pct = Math.round(confidence * 100);
  if (needsReview) {
    return (
      <span className="shrink-0 text-[11px] uppercase tracking-wide text-review font-mono">
        review · {pct}%
      </span>
    );
  }
  return (
    <span className="shrink-0 text-[11px] tracking-wide text-ink-muted font-mono">
      {pct}%
    </span>
  );
}

function SkeletonResults() {
  return (
    <div className="space-y-2">
      {[0, 1, 2].map((i) => (
        <div
          key={i}
          className="h-14 border border-ink/10 bg-paper-raised animate-pulse"
          style={{ animationDelay: `${i * 80}ms` }}
        />
      ))}
    </div>
  );
}
